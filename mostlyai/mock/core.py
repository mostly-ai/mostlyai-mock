# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
from collections import deque
from collections.abc import AsyncGenerator
from enum import Enum
from io import StringIO
from typing import Any, Literal

import dateutil.parser
import litellm
import pandas as pd
import tenacity
from pydantic import BaseModel, Field, RootModel, create_model, field_validator, model_validator
from tqdm.asyncio import tqdm

litellm.suppress_debug_info = True


class LLMOutputFormat(str, Enum):
    JSON = "JSON"
    CSV = "CSV"


class LLMConfig(BaseModel):
    model: str = "openai/gpt-4.1-nano"
    api_key: str | None = None
    temperature: float = 1.0
    top_p: float = 0.95


class MockConfig(RootModel[dict[str, "TableConfig"]]):
    root: dict[str, TableConfig] = Field(..., min_length=1)

    @field_validator("root")
    @classmethod
    def validate_consistency_of_relationships(cls, tables: dict[str, TableConfig]) -> dict[str, TableConfig]:
        for table_name, table_config in tables.items():
            if not table_config.foreign_keys:
                continue

            for fk in table_config.foreign_keys:
                if fk.referenced_table not in tables:
                    raise ValueError(
                        f"Foreign key violation in table '{table_name}': "
                        f"Referenced table '{fk.referenced_table}' does not exist"
                    )

                referenced_config = tables[fk.referenced_table]
                if not referenced_config.primary_key:
                    raise ValueError(
                        f"Foreign key violation in table '{table_name}': "
                        f"Referenced table '{fk.referenced_table}' has no primary key defined"
                    )

                if fk.column not in table_config.columns:
                    raise ValueError(
                        f"Foreign key violation in table '{table_name}': "
                        f"Column '{fk.column}' does not exist in the schema"
                    )

                fk_field = table_config.columns[fk.column]
                pk_field = referenced_config.columns[referenced_config.primary_key]
                if fk_field.dtype != pk_field.dtype:
                    raise ValueError(
                        f"Foreign key violation in table '{table_name}': "
                        f"Column '{fk.column}' type '{fk_field.dtype.value}' does not match "
                        f"referenced primary key '{referenced_config.primary_key}' type '{pk_field.dtype.value}'"
                    )

        return tables

    @model_validator(mode="after")
    def validate_no_circular_dependencies(self) -> MockConfig:
        child_to_parents = {}
        for table_name, table_config in self.root.items():
            child_to_parents[table_name] = [fk.referenced_table for fk in table_config.foreign_keys]
        visited = set()

        def detect_cycle(table_name: str, path: list[str]) -> None:
            if table_name in path:
                cycle_start = path.index(table_name)
                cycle = path[cycle_start:] + [table_name]
                if len(cycle) > 2:  # len(cycle) == 2 means self-referencing table, which is allowed
                    raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}.")
            if table_name in visited:
                return
            visited.add(table_name)
            path.append(table_name)
            for parent in child_to_parents[table_name]:
                detect_cycle(parent, path)
            path.pop()

        for table_name in child_to_parents:
            detect_cycle(table_name, [])

        return self

    @model_validator(mode="after")
    def ensure_values_are_not_provided_for_primary_key(self) -> MockConfig:
        for table_name, table_config in self.root.items():
            for column_name, column_config in table_config.columns.items():
                if column_name == table_config.primary_key and column_config.values:
                    raise ValueError(
                        f"Values cannot be provided for primary key column '{column_name}' in table '{table_name}'"
                    )
        return self

    @model_validator(mode="after")
    def ensure_primary_key_is_string_dtype(self) -> MockConfig:
        for table_name, table_config in self.root.items():
            if table_config.primary_key:
                column_config = table_config.columns[table_config.primary_key]
                if column_config.dtype not in [DType.STRING]:
                    raise ValueError(
                        f"Primary key column '{table_config.primary_key}' in table '{table_name}' must be one of the following types:"
                        f" {[DType.STRING.value]}"
                    )
        return self

    def get_dependency_mappings(self) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
        child_to_parents = {}
        parent_to_children = {}

        for table_name in self.root:
            child_to_parents[table_name] = set()
            parent_to_children[table_name] = set()

        for table_name, table_config in self.root.items():
            if table_config.foreign_keys:
                for fk in table_config.foreign_keys:
                    referenced_table = fk.referenced_table
                    child_to_parents[table_name].add(referenced_table)
                    parent_to_children[referenced_table].add(table_name)

        root_tables = []
        for table_name, parents in child_to_parents.items():
            if not parents or parents == {table_name}:  # no dependencies or only self-dependency
                root_tables.append(table_name)
        return child_to_parents, parent_to_children, root_tables


class TableConfig(BaseModel):
    prompt: str = ""
    columns: dict[str, ColumnConfig] = Field(..., min_length=1)
    primary_key: str | None = None
    foreign_keys: list[ForeignKeyConfig] = Field(default_factory=list)


class ColumnConfig(BaseModel):
    prompt: str = ""
    dtype: DType
    values: list[Any] = Field(default_factory=list)

    @model_validator(mode="before")
    def set_default_dtype(cls, data):
        if isinstance(data, dict):
            if "dtype" not in data:
                if data.get("values"):
                    data["dtype"] = DType.CATEGORY
                else:
                    data["dtype"] = DType.STRING
        return data

    @model_validator(mode="after")
    def ensure_values_are_unique(self) -> ColumnConfig:
        if self.values:
            if len(self.values) != len(set(self.values)):
                raise ValueError("Values must be unique")
        return self

    @model_validator(mode="after")
    def ensure_values_are_provided_for_category_dtype(self) -> ColumnConfig:
        if self.dtype == DType.CATEGORY and not self.values:
            raise ValueError("At least one value must be provided when dtype is 'category'")
        return self

    @model_validator(mode="after")
    def override_values_for_boolean_dtype(self) -> ColumnConfig:
        if self.dtype == DType.BOOLEAN:
            self.values = [True, False]
        return self

    @model_validator(mode="after")
    def harmonize_values_with_dtypes(self) -> ColumnConfig:
        if self.values:
            cast_fn, convertible_to = {
                DType.INTEGER: (int, "integers"),
                DType.FLOAT: (float, "floats"),
                DType.STRING: (str, "strings"),
                DType.CATEGORY: (lambda c: c, "categories"),
                DType.BOOLEAN: (bool, "booleans"),
                DType.DATE: (str, "strings"),
                DType.DATETIME: (str, "strings"),
            }[self.dtype]
            try:
                self.values = [cast_fn(c) if pd.notna(c) else None for c in self.values]
            except ValueError:
                raise ValueError(
                    f"All values must be convertible to {convertible_to} when dtype is '{self.dtype.value}'"
                )
        return self


class DType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    CATEGORY = "category"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"


class ForeignKeyConfig(BaseModel):
    column: str
    referenced_table: str
    prompt: str | None = None


async def _sample_table(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    foreign_keys: list[ForeignKeyConfig],
    primary_keys: dict[str, str],
    data: dict[str, pd.DataFrame],
    sample_size: int | None,
    previous_rows_size: int,
    non_context_size: int | None,
    n_workers: int,
    llm_config: LLMConfig,
) -> pd.DataFrame:
    table_rows_generator = _create_table_rows_generator(
        name=name,
        prompt=prompt,
        columns=columns,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        data=data,
        sample_size=sample_size,
        previous_rows_size=previous_rows_size,
        non_context_size=non_context_size,
        n_workers=n_workers,
        llm_config=llm_config,
    )
    table_rows_generator = tqdm(table_rows_generator, desc=f"Generating rows for table `{name}`".ljust(45))
    table_df = await _convert_table_rows_generator_to_df(table_rows_generator=table_rows_generator, columns=columns)
    return table_df


def _sample_table_sync(*args, **kwargs) -> pd.DataFrame:
    return asyncio.run(_sample_table(*args, **kwargs))


def _create_system_prompt(llm_output_format: LLMOutputFormat) -> str:
    return f"""
You are a specialized data generator designed to create highly realistic, contextually appropriate data based on schema definitions.

Your task is to:

1. Generate data that strictly adheres to the provided schema constraints (data types, ranges, formats)
2. Ensure logical consistency across related tables and foreign key relationships
3. Create contextually appropriate values that reflect real-world patterns and distributions
4. Produce diverse, non-repetitive data that avoids obvious patterns
5. Respect uniqueness constraints and other data integrity rules
6. When enriching existing data, ensure that new values are consistent with existing values
7. Return well-formatted {llm_output_format.value} output that can be directly parsed
8. Don't use markdown formatting

For numeric fields, generate realistic distributions rather than random values. For text fields, create contextually \
appropriate content. For dates and timestamps, ensure logical chronology. Always maintain referential integrity \
across tables.

When enriching existing data, carefully analyze the patterns and relationships in the existing columns \
to generate compatible and realistic values for the missing columns.
"""


def _create_table_prompt(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    primary_keys: dict[str, str],
    batch_idx: int,
    batch_size: int | None,
    foreign_keys: list[ForeignKeyConfig],
    existing_data: pd.DataFrame | None,
    context_data: pd.DataFrame | None,
    non_context_data: dict[str, pd.DataFrame] | None,
    previous_rows: list[dict] | None,
    llm_output_format: LLMOutputFormat,
) -> str:
    # add table prompt
    prompt = f"# {prompt}\n\n"

    # define table
    prompt += f"## Target Table: `{name}`\n\n"

    target_primary_key = primary_keys[name]
    prompt += f"### Target Table Primary Key: `{target_primary_key}`\n\n"

    # add columns specifications
    prompt += "### Target Table Column Specifications:\n\n"
    column_specifications = {
        name: config.model_dump(exclude_defaults=True, exclude_unset=True, exclude_none=True)
        for name, config in columns.items()
    }
    if existing_data is not None:
        # do not generate values for columns that already exist in existing data
        column_specifications = {
            column: spec for column, spec in column_specifications.items() if column not in existing_data.columns
        }
    prompt += f"{json.dumps(column_specifications, indent=2)}\n\n"

    # add previous rows as context to help the LLM generate consistent data
    has_previous_rows_section = False
    if previous_rows:
        has_previous_rows_section = True
        prompt += f"\n## Previous `{len(previous_rows)}` Rows of Target Table `{name}`:\n\n"
        prompt += f"{json.dumps(previous_rows, indent=2)}\n\n"

    # add existing data to augment
    has_existing_data_section = False
    if existing_data is not None:
        has_existing_data_section = True
        prompt += f"\n## Existing Data of Target Table `{name}` to Augment:\n\n"
        prompt += f"{existing_data.to_json(orient='records', date_format='iso', indent=2)}\n\n"

    # define self referencing foreign keys
    has_self_referencing_foreign_keys_section = False
    self_referencing_foreign_keys = [fk for fk in foreign_keys if fk.referenced_table == name]
    if self_referencing_foreign_keys:
        has_self_referencing_foreign_keys_section = True
        prompt += f"## Self Referencing Foreign Keys in Target Table `{name}`\n\n"
        for fk in self_referencing_foreign_keys:
            prompt += f"### Primary Key Column: `{target_primary_key}`\n\n"

            prompt += f"### Foreign Key Column: `{fk.column}`\n\n"

            prompt += f"### Description of the Relationship: `{fk.prompt}`\n\n"

    foreign_keys = [fk for fk in foreign_keys if fk.referenced_table != name]  # exclude self-dependency going forward

    # add context table name, primary key and data
    has_context_table_section = False
    if foreign_keys:
        has_context_table_section = True
        assert context_data is not None
        fk = foreign_keys[0]
        prompt += f"## Context Table: `{fk.referenced_table}`\n\n"

        prompt += f"### Context Table Primary Key: `{primary_keys[fk.referenced_table]}`\n\n"

        prompt += f"### Foreign Key Column in Target Table `{name}`: `{fk.column}`\n\n"

        prompt += f"### Description of the Relationship: `{fk.prompt}`\n\n"

        prompt += "### Context Table Data:\n\n"
        prompt += f"{context_data.to_json(orient='records', date_format='iso', indent=2)}\n\n"

    # add non-context table names, primary keys and data
    has_non_context_tables_section = False
    if foreign_keys and len(foreign_keys) > 1:
        has_non_context_tables_section = True
        for fk in foreign_keys[1:]:
            assert non_context_data is not None
            assert fk.referenced_table in non_context_data
            prompt += f"## Non-Context Table: `{fk.referenced_table}`\n\n"

            prompt += f"### Non-Context Table Primary Key: `{primary_keys[fk.referenced_table]}`\n\n"

            prompt += f"### Foreign Key Column in Target Table `{name}`: `{fk.column}`\n\n"

            prompt += f"### Description of the Relationship: `{fk.prompt}`\n\n"

            prompt += "### Non-Context Table Data:\n\n"
            prompt += (
                f"{non_context_data[fk.referenced_table].to_json(orient='records', date_format='iso', indent=2)}\n\n"
            )

    # add instructions
    prompt += "\n## Instructions:\n\n"

    verb = "generate" if existing_data is None else "augment"

    n_rows = None
    if existing_data is not None:
        n_rows = len(existing_data)
    elif not foreign_keys and not self_referencing_foreign_keys:
        assert batch_size is not None
        n_rows = batch_size

    prompt += f"{verb.capitalize()} data for the Target Table `{name}`.\n\n"
    if n_rows is not None:
        prompt += f"Number of data rows to {verb}: `{n_rows}`.\n\n"

    if target_primary_key is not None:
        prompt += f"Add prefix to all values of Target Table Primary Key. The prefix is 'B{batch_idx}-'."
        prompt += " There is one exception: if primary keys are in existing data, don't add prefix to them."
        prompt += "\n\n"

    if has_context_table_section:
        assert foreign_keys
        prompt += f"Target Table Foreign Key column `{foreign_keys[0].column}` may only contain values from `Context Table Data`."
        if has_previous_rows_section:
            prompt += " Never use values from `Previous Rows of Target Table` section."
        prompt += " Respect the `Description of the Relationship` of `Context Table` section to understand the relationship, in particular the number of rows to generate."
        prompt += "\n\n"

    if has_self_referencing_foreign_keys_section:
        prompt += "Target Table Self Referencing Foreign Key columns defined in `Self Referencing Foreign Keys` must be consistent with the `Target Table Primary Key`."
        prompt += " Respect the `Description of the Relationship` of `Self Referencing Foreign Keys` section to understand the relationship."
        prompt += "\n\n"

    if has_non_context_tables_section:
        assert len(foreign_keys) > 1
        prompt += "All other Target Table Foreign Key columns may only contain values from `Non-Context Table Data` of relevant `Non-Context Table` sections."
        prompt += " Respect the `Description of the Relationship` of relevant `Non-Context Table` section to understand the relationship."
        prompt += "\n\n"

    if has_existing_data_section:
        assert existing_data is not None
        prompt += (
            f"You are given existing data for the `{name}` table and asked to generate "
            f"values for the missing columns. The existing data contains column(s): {list(existing_data.columns)}. "
            f"You need to generate values for column(s): {list(columns.keys() - existing_data.columns)}. "
            f"Ensure that the generated values are contextually appropriate and consistent with the existing data. "
            f"Use the existing columns' values to inform the generation of new values. "
            f"Don't generate new rows, only augment the existing data.\n\n"
        )

    if has_previous_rows_section:
        assert previous_rows is not None
        prompt += (
            f"{verb.capitalize()} new rows that maintain consistency with the previous rows where appropriate. "
            "Don't copy previous rows in the output. "
            "Don't pay attention to the number of previous rows; there might have been more generated than provided.\n\n"
        )

    prompt += f"Do not use code to {verb} the data.\n\n"

    prompt += f"Return data as a {llm_output_format.value} string."
    if llm_output_format == LLMOutputFormat.JSON:
        prompt += " The JSON string should have 'rows' key at the top level."
        prompt += " The value of 'rows' key should be a list of JSON objects."
        prompt += " Each JSON object should have column names as keys and values as column values."
    else:  # llm_output_format == LLMOutputFormat.CSV
        prompt += " The CSV string should have a header row with column names."
        prompt += " The CSV string should have a data row for each row to be generated."
        prompt += " The CSV string should have a newline character at the end of each row."
        prompt += " Each value in the CSV string should be enclosed in double quotes."

    if existing_data is not None:
        prompt += f" Only include the following columns in the {llm_output_format.value} string: {list(columns.keys() - existing_data.columns)}."

    if llm_output_format == LLMOutputFormat.CSV and batch_size > 10:
        prompt += " Additionally, add column called `_ROW_IDX` that is a counter from 1 to the number of rows generated so far within current batch."

    prompt += "\n"
    return prompt


def _completion_with_retries(*args, **kwargs):
    n_attempts = 3

    def print_on_retry(_):
        print(" * Calling LLM again... * ", end="", flush=True)

    # try up to 3 times, print a message to the user on each retry
    retryer = tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(n_attempts), reraise=True, before_sleep=print_on_retry
    )
    return retryer(litellm.acompletion, *args, **kwargs)


async def _yield_rows_from_json_chunks_stream(response: litellm.CustomStreamWrapper) -> AsyncGenerator[dict]:
    def buffer_to_row(buffer: list[str]) -> dict:
        return json.loads("".join(buffer))

    # starting with dirty buffer is to handle the `{"rows": []}` case
    buffer = list("garbage")
    rows_json_started = False
    in_row_json = False
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        for char in delta:
            buffer.append(char)
            if char == "{" and not rows_json_started:
                # {"rows": [{"name": "Jo\}h\{n"}]}
                # *                                 <- start of rows json stream
                rows_json_started = True
            elif char == "{" and not in_row_json:
                # {"rows": [{"name": "Jo\}h\{n"}]}
                #           *                       <- start of single row json stream
                buffer = list("{")
                in_row_json = True
            elif char == "}":
                # {"rows": [{"name": "Jo\}h\{n"}]}
                #                        *     * *  <- any of these
                try:
                    row = buffer_to_row(buffer)
                except Exception:
                    # in case of any error, silently drop the row
                    continue
                finally:
                    buffer = list()
                    in_row_json = False
                yield row


async def _yield_rows_from_csv_chunks_stream(response: litellm.CustomStreamWrapper) -> AsyncGenerator[dict]:
    def buffer_to_row(buffer: list[str]) -> list[str]:
        return pd.read_csv(StringIO("".join(buffer)), header=None).astype(str).iloc[0].to_list()

    buffer = list()
    header = None
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        for char in delta:
            buffer.append(char)
            if char == "\n":
                try:
                    row = buffer_to_row(buffer)
                except Exception:
                    # in case of any error, silently drop the row
                    continue
                finally:
                    buffer = list()
                if header is None:
                    # column1,column2,column3\n
                    #                        ** <- end of header row
                    header = row
                else:
                    # value_1,value_2,value_3\n
                    #                        ** <- end of data row
                    yield dict(zip(header, row))
    if buffer:
        # last row might not finish with a newline, in which case the buffer would not be empty here
        try:
            last_row = buffer_to_row(buffer)
            yield dict(zip(header, last_row))
        except Exception:
            # in case of any error, silently drop the row
            pass


def _create_structured_output_schema(
    columns: dict[str, ColumnConfig], existing_data: pd.DataFrame | None
) -> type[BaseModel]:
    def create_annotation(column_config: ColumnConfig) -> type:
        if column_config.values or column_config.dtype is DType.CATEGORY:
            return Literal[tuple(column_config.values)]  # type: ignore
        return {
            DType.INTEGER: int | None,
            DType.FLOAT: float | None,
            DType.STRING: str | None,
            DType.BOOLEAN: bool | None,
            # response_format has limited support for JSON Schema features
            # thus we represent dates and datetimes as strings
            DType.DATE: str | None,
            DType.DATETIME: str | None,
        }[column_config.dtype]

    fields = {}
    for column_name, column_config in columns.items():
        if existing_data is not None and column_name in existing_data.columns:
            continue  # skip columns that already exist in existing data
        annotation = create_annotation(column_config)
        fields[column_name] = (annotation, Field(...))
    TableRow = create_model("TableRow", **fields)
    TableRows = create_model("TableRows", rows=(list[TableRow], ...))
    return TableRows


async def _worker(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    foreign_keys: list[ForeignKeyConfig],
    primary_keys: dict[str, str],
    previous_rows: deque[dict],
    batch_queue: asyncio.Queue,
    result_queue: asyncio.Queue,
    retry_queue: asyncio.Queue,
    n_workers: int,
    llm_output_format: LLMOutputFormat,
    llm_config: LLMConfig,
):
    try:
        while True:
            do_repeat_task = False

            # get task from the batch_queue
            batch_idx, task = await batch_queue.get()
            if task is None:
                # no more tasks for the worker; break the loop
                batch_queue.task_done()
                break

            # deconstruct task
            batch_size = task["batch_size"]
            existing_batch = task.get("existing_batch")
            context_batch = task.get("context_batch")
            non_context_batch = task.get("non_context_batch")

            # resolve columns to generate
            generated_columns = set(columns.keys())
            if existing_batch is not None:
                generated_columns = generated_columns - set(existing_batch.columns)

            # construct schema for Structured Outputs (applies to JSON LLMOutputFormat only)
            structured_output_schema = None
            if llm_output_format == LLMOutputFormat.JSON:
                structured_output_schema = _create_structured_output_schema(
                    columns=columns, existing_data=existing_batch
                )

            # construct litellm kwargs
            litellm_kwargs = {
                "temperature": llm_config.temperature,
                "top_p": llm_config.top_p,
                "model": llm_config.model,
                "api_key": llm_config.api_key,
                "stream": True,
            }

            # construct messages
            system_prompt = _create_system_prompt(llm_output_format)
            user_prompt = _create_table_prompt(
                name=name,
                prompt=prompt,
                columns=columns,
                batch_idx=batch_idx,
                batch_size=batch_size,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                existing_data=existing_batch,
                context_data=context_batch,
                non_context_data=non_context_batch,
                previous_rows=list(previous_rows),
                llm_output_format=llm_output_format,
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

            if generated_columns:
                # make LLM call
                response = await _completion_with_retries(
                    messages=messages, response_format=structured_output_schema, **litellm_kwargs
                )
                yield_rows_from_chunks_stream = {
                    LLMOutputFormat.JSON: _yield_rows_from_json_chunks_stream,
                    LLMOutputFormat.CSV: _yield_rows_from_csv_chunks_stream,
                }[llm_output_format]
                rows_stream = yield_rows_from_chunks_stream(response)
            else:
                # skip roundtrip to LLM in case all columns are provided in existing data
                assert existing_batch is not None

                async def _yield_empty_rows(n_rows: int) -> AsyncGenerator[dict]:
                    for _ in range(n_rows):
                        yield {}

                rows_stream = _yield_empty_rows(len(existing_batch))

            # we first generate all rows in the batch, in order to run consistency checks
            rows_generated_part = []
            async for row_generated_part in rows_stream:
                # remove internal columns, if exist
                row_generated_part = {k: v for k, v in row_generated_part.items() if k in generated_columns}

                if set(row_generated_part.keys()) != generated_columns:
                    if context_batch is not None or existing_batch is not None:
                        # in case of linked tables and data enrichment, it's critical that all rows have expected columns
                        print(" * Malformed row, repeating batch... * ", end="", flush=True)
                        do_repeat_task = True
                        break
                    else:
                        # in case of flat tables generation, each row is independent, therefore we only skip the invalid row
                        continue
                rows_generated_part.append(row_generated_part)

            # at least some valid rows are expected per batch, repeat the batch otherwise
            if len(rows_generated_part) == 0:
                print(" * No valid rows were generated, repeating batch... * ", end="", flush=True)
                do_repeat_task = True

            # in case of data enrichment, check that all rows were completed successfully
            if existing_batch is not None and len(rows_generated_part) != len(existing_batch):
                print(" * Some rows were not enriched successfully, repeating batch... * ", end="", flush=True)
                do_repeat_task = True

            if do_repeat_task:
                # allow 10 retries across all workers before propagating the exception to the orchestrator
                await retry_queue.put(1)
                if retry_queue.qsize() < 10:
                    # put task back to the front of the batch queue
                    await batch_queue.put((batch_idx, task))
                else:
                    # inform the orchestrator that max retries were reached
                    raise RuntimeError(
                        "Too many malformed batches were generated. "
                        "Consider changing the model in order to make generation more stable."
                    )

                # mark current task as done
                batch_queue.task_done()
                continue

            # collapse existing and generated parts into coherent rows
            rows = []
            for row_idx, row_generated_part in enumerate(rows_generated_part):
                row_existing_part = existing_batch.iloc[row_idx].to_dict() if existing_batch is not None else {}
                row = {**row_generated_part, **row_existing_part}
                # keep columns order according to user's spec
                row = {column: row[column] for column in columns.keys()}
                rows.append(row)

            # track previous rows for improved data consistency, in case of sequential generation
            if n_workers == 1:
                previous_rows.extend(rows)

            # put rows to the result queue and mark current task as done
            await result_queue.put((batch_idx, rows))
            batch_queue.task_done()
    except Exception as e:
        # propagate any exception through the result queue
        await result_queue.put((batch_idx, e))
        raise


async def _create_table_rows_generator(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    foreign_keys: list[ForeignKeyConfig],
    primary_keys: dict[str, str],
    data: dict[str, pd.DataFrame],
    sample_size: int | None,
    previous_rows_size: int,
    non_context_size: int | None,
    n_workers: int,
    llm_config: LLMConfig,
) -> AsyncGenerator[dict]:
    batch_size = 20  # generate 20 root table rows at a time

    def supports_structured_outputs(model: str) -> bool:
        model = model.removeprefix("litellm_proxy/")
        supported_params = litellm.get_supported_openai_params(model=model) or []
        return "response_format" in supported_params and litellm.supports_response_schema(model)

    llm_output_format = LLMOutputFormat.JSON if supports_structured_outputs(llm_config.model) else LLMOutputFormat.CSV

    previous_rows = deque(maxlen=previous_rows_size)

    # derive data for augmentation
    existing_data: pd.DataFrame | None = None
    if name in data:
        existing_data = data[name]
        sample_size = len(existing_data)
        batch_size = 10  # augment 10 root table rows at a time

    # derive context data (if first foreign key is present) and harmonize sample size accordingly
    context_data: pd.DataFrame | None = None
    context_batches: list[pd.DataFrame] | None = None
    if foreign_keys and foreign_keys[0].referenced_table != name:  # self-dependency is not considered as context
        context_table_name = foreign_keys[0].referenced_table
        assert context_table_name in data
        context_data = data[context_table_name]
        batch_size = 1  # generate 1 sequence at a time
        sample_size = len(context_data)
        context_batches = [context_data.iloc[i : i + batch_size] for i in range(0, len(context_data), batch_size)]

    # derive non-context data (if more than one foreign key is present)
    non_context_data: dict[str, pd.DataFrame] = {}
    if foreign_keys and len(foreign_keys) > 1:
        assert non_context_size is not None
        for fk in foreign_keys[1:]:
            if fk.referenced_table == name:  # self-dependency is not considered as non-context
                continue
            non_context_table_name = fk.referenced_table
            assert non_context_table_name in data
            non_context_data[non_context_table_name] = data[non_context_table_name]

    # calculate batch_sizes
    assert sample_size is not None, "sample_size should have been filled by this point"
    n_total_batches = len(context_batches) if context_batches is not None else math.ceil(sample_size / batch_size)
    batch_sizes = [batch_size] * n_total_batches
    if context_batches is None:
        # optimise the last batch size for flat tables
        # +2 because LLM may not always count the rows correctly
        batch_sizes[-1] = sample_size - sum(batch_sizes[:-1]) + 2

    # initialize queues for async communication
    batch_queue = asyncio.PriorityQueue()
    result_queue = asyncio.Queue()
    retry_queue = asyncio.Queue()

    # populate batch queue
    for batch_idx in range(n_total_batches):
        context_batch = context_batches[batch_idx] if context_batches is not None else None

        # pick existing rows for current batch
        existing_batch: pd.DataFrame | None = None
        if existing_data is not None:
            if context_batch is None:
                # progressively pick portions of existing data in case of root tables
                assert batch_size is not None
                existing_batch = existing_data.iloc[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            else:
                # pick existing rows that match current context batch
                assert foreign_keys is not None
                context_table_name, foreign_key = foreign_keys[0].referenced_table, foreign_keys[0].column
                context_primary_key = primary_keys[context_table_name]
                existing_batch = existing_data[existing_data[foreign_key].isin(context_batch[context_primary_key])]
            if existing_batch.empty:
                existing_batch = None

        # sample candidate rows from non-context tables for current batch
        non_context_batch: dict[str, pd.DataFrame] | None = None
        if non_context_data:
            non_context_batch = {
                table_name: df.sample(frac=1.0).head(non_context_size) for table_name, df in non_context_data.items()
            }

        task = {
            "batch_size": batch_sizes[batch_idx],
            "existing_batch": existing_batch,
            "context_batch": context_batch,
            "non_context_batch": non_context_batch,
        }
        await batch_queue.put((batch_idx, task))

    # initialize workers
    n_workers = min(n_total_batches, n_workers)
    workers = [
        asyncio.create_task(
            _worker(
                name=name,
                prompt=prompt,
                columns=columns,
                foreign_keys=foreign_keys,
                primary_keys=primary_keys,
                previous_rows=previous_rows,
                batch_queue=batch_queue,
                result_queue=result_queue,
                retry_queue=retry_queue,
                n_workers=n_workers,
                llm_output_format=llm_output_format,
                llm_config=llm_config,
            )
        )
        for _ in range(n_workers)
    ]

    n_completed_batches = 0
    n_yielded_sequences = 0
    while n_yielded_sequences < sample_size:
        if n_completed_batches >= n_total_batches:
            assert context_data is None, "n_total_batches is fixed for linked tables"
            assert existing_data is None, "n_total_batches is fixed for data enrichment"
            # LLMs may not generate exactly the number of rows requested
            # in case of flat tables, we still accept such incomplete batches,
            # but that means we may need to generate more batches to reach the sample size
            # +2 because LLM may not always count the rows correctly
            n_total_batches += 1
            task = {
                "batch_size": sample_size - n_yielded_sequences + 2,
            }
            await batch_queue.put((n_total_batches, task))
        batch_idx, result = await result_queue.get()
        if isinstance(result, Exception):
            # if an exception is raised by any worker, cancel all workers and raise that exception
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers)
            raise result
        rows = result
        for row_idx, row in enumerate(rows):
            yield (batch_idx, row)
            if context_batches is None or row_idx == len(rows) - 1:
                # in case of flat table, each row is considered a single sequence
                # in case of linked table, all rows are considered a single sequence
                # NOTE: this assumes that we generate a single sequence per batch
                n_yielded_sequences += 1
            if n_yielded_sequences >= sample_size:
                break
        n_completed_batches += 1
        result_queue.task_done()

    # gracefully shutdown workers
    await batch_queue.join()
    for _ in workers:
        await batch_queue.put((n_total_batches + 1, None))
    await asyncio.gather(*workers)


def _align_series_dtypes_with_column_config(series: pd.Series, column_config: ColumnConfig) -> pd.Series:
    series = series.copy()
    if column_config.dtype in [DType.DATE, DType.DATETIME]:

        def harmonize_datetime(x: Any):
            try:
                return dateutil.parser.parse(str(x))
            except Exception:
                return pd.NaT

        series = pd.to_datetime(series.apply(harmonize_datetime), errors="coerce")
    elif column_config.dtype is DType.INTEGER:
        series = pd.to_numeric(series, errors="coerce", downcast="integer").astype("int64[pyarrow]")
    elif column_config.dtype is DType.FLOAT:
        series = pd.to_numeric(series, errors="coerce").astype("double[pyarrow]")
    elif column_config.dtype is DType.BOOLEAN:
        series = series.map(lambda x: True if str(x).lower() == "true" else x)
        series = series.map(lambda x: False if str(x).lower() == "false" else x)
        series = pd.to_numeric(series, errors="coerce").astype("boolean[pyarrow]")
    elif column_config.dtype is DType.CATEGORY:
        series = pd.Categorical(series, categories=column_config.values)
    else:
        series = series.astype("string[pyarrow]")
    return series


async def _convert_table_rows_generator_to_df(
    table_rows_generator: AsyncGenerator[dict],
    columns: dict[str, ColumnConfig],
) -> pd.DataFrame:
    def align_df_dtypes_with_mock_dtypes(df: pd.DataFrame, columns: dict[str, ColumnConfig]) -> pd.DataFrame:
        df = df.copy()
        for column_name, column_config in columns.items():
            df[column_name] = _align_series_dtypes_with_column_config(df[column_name], column_config)
        return df

    # consume entire generator
    items = [{"batch_idx": batch_idx, "row": row} async for batch_idx, row in table_rows_generator]
    # sort items by batch_idx to maintain order (relevant especially for keeping the order of existing data)
    items = sorted(items, key=lambda x: x["batch_idx"])
    # extract rows and convert to DataFrame
    rows = [item["row"] for item in items]
    df = pd.DataFrame(rows)
    # harmonize dtypes
    df = align_df_dtypes_with_mock_dtypes(df, columns)
    return df


def _harmonize_tables(tables: dict[str, dict], existing_data: dict[str, pd.DataFrame] | None) -> dict[str, dict]:
    def _infer_dtype(series: pd.Series) -> DType:
        if pd.api.types.is_integer_dtype(series):
            return DType.INTEGER
        elif pd.api.types.is_float_dtype(series):
            return DType.FLOAT
        elif pd.api.types.is_datetime64_dtype(series):
            return DType.DATETIME
        elif pd.api.types.is_bool_dtype(series):
            return DType.BOOLEAN
        else:
            return DType.STRING

    if existing_data is None:
        return tables

    tables = tables.copy()
    for table_name, existing_table in existing_data.items():
        table_config = tables.setdefault(table_name, {})

        # prepend column configs for existing data columns, that are not specified in the mock config
        column_configs = table_config.setdefault("columns", {})
        existing_column_configs = {
            existing_column: {"dtype": _infer_dtype(existing_table[existing_column])}
            for existing_column in existing_table.columns
            if existing_column not in column_configs
        }
        column_configs = {**existing_column_configs, **column_configs}

        # primary keys are always strings
        primary_key = table_config.get("primary_key", None)
        if primary_key is not None:
            column_configs[primary_key]["dtype"] = DType.STRING

        table_config["columns"] = column_configs
    return tables


def _harmonize_sample_size(sample_size: int | dict[str, int], config: MockConfig) -> dict[str, int]:
    _, _, root_tables = config.get_dependency_mappings()

    if isinstance(sample_size, int):
        sample_size = {table_name: sample_size for table_name in root_tables}

    for table_name in root_tables:
        if table_name not in sample_size or sample_size[table_name] is None:
            # set default sample size for missing or None sample sizes
            sample_size[table_name] = 4
        # clamp sample_size to [1, inf)
        sample_size[table_name] = max(1, sample_size[table_name])

    return sample_size


def _harmonize_existing_data(
    existing_data: dict[str, pd.DataFrame] | None, mock_config: MockConfig
) -> dict[str, pd.DataFrame]:
    if existing_data is None:
        return {}

    # by this point, mock config should have been validated, so we can assume that all tables in existing_data are defined in the mock config
    assert set(mock_config.root.keys()).issuperset(existing_data.keys())

    for existing_table_name, existing_table in existing_data.items():
        existing_table_config = mock_config.root[existing_table_name]

        for existing_column in existing_table.columns:
            existing_column_config = existing_table_config.columns[existing_column]

            # ensure that the existing data has compatible dtypes with the column config
            original_series = existing_table[existing_column]
            coerced_series = _align_series_dtypes_with_column_config(original_series, existing_column_config)
            n_original_na = original_series.isna().sum()
            n_coerced_na = coerced_series.isna().sum()
            if n_original_na != n_coerced_na:
                raise ValueError(
                    f"Coercion of existing data column '{existing_column}' in table '{existing_table_name}' resulted in data loss. "
                    f"Ensure that the existing data is consistent with the mock configuration."
                )

            # ensure that the existing data has values allowed by the column config
            if existing_column_config.values:
                if not set(existing_table[existing_column].unique()).issubset(existing_column_config.values):
                    raise ValueError(
                        f"Existing data column '{existing_column}' in table '{existing_table_name}' has values disallowed by the column config. "
                        f"Ensure that the existing data is consistent with the mock configuration."
                    )

        # ensure that the existing data has unique primary keys
        existing_table_primary_key = existing_table_config.primary_key
        if existing_table_primary_key is not None:
            if not existing_table[existing_table_primary_key].is_unique:
                raise ValueError(
                    f"Existing data table '{existing_table_name}' has non-unique primary key column '{existing_table_primary_key}'. "
                    f"Ensure that the primary key is unique."
                )

            existing_table[existing_column] = coerced_series

    return existing_data


def _build_execution_plan(config: MockConfig) -> list[str]:
    child_to_parents, parent_to_children, root_tables = config.get_dependency_mappings()

    execution_plan = []
    bfs_queue = list(root_tables)
    processed = set()

    while bfs_queue:
        table_name = bfs_queue.pop(0)
        if table_name in processed:
            continue

        # ensure all parents are processed before processing this table
        unprocessed_parents = []
        for parent in child_to_parents[table_name]:
            if parent not in processed and parent != table_name:  # exclude self-dependency
                unprocessed_parents.append(parent)
        if unprocessed_parents:
            bfs_queue.extend(unprocessed_parents)
            bfs_queue.append(table_name)
            continue

        execution_plan.append(table_name)
        processed.add(table_name)

        for child in parent_to_children[table_name]:
            if child not in bfs_queue and child not in processed:
                bfs_queue.append(child)
    return execution_plan


def sample(
    *,
    tables: dict[str, dict],
    sample_size: int | dict[str, int] = 4,
    existing_data: dict[str, pd.DataFrame] | None = None,
    model: str = "openai/gpt-4.1-nano",
    api_key: str | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    n_workers: int = 10,
    return_type: Literal["auto", "dict"] = "auto",
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Generate synthetic data from scratch or enrich existing data with new columns.

    While faker and numpy are useful to create fake data, this utility is unique as it allows
    the creation of coherent, realistic multi-table tabular mock data
    or the enrichment of existing datasets with new, context-aware columns.

    It is particularly useful for quickly simulating production-like datasets for testing or prototyping purposes.
    It is advised to limit mocking to small datasets for performance reasons (rows * cols < 100).
    It might take a couple of minutes for bigger datasets.

    Args:
        tables (dict[str, dict]): The table specifications to generate mock data for. See examples for usage.
        sample_size (int | dict[str, int]): The number of rows to generate for each subject table.
            If a single integer is provided, the same number of rows will be generated for each subject table.
            If a dictionary is provided, the number of rows to generate for each subject table can be specified individually.
            Default is 4. Ignored if existing_data is provided. Ignored for non-root tables.
            If a table has a foreign key, the sample size is determined by the corresponding foreign key prompt. If nothing specified, a few rows per parent record are generated.
        existing_data (dict[str, pd.DataFrame] | None): Existing data to augment. If provided, the sample_size argument is ignored.
            Default is None.
        model (str): The LiteLLM chat completion model to be used.
            Examples include:
            - `openai/gpt-4.1-nano` (default; fast, and smart)
            - `openai/gpt-4.1-mini` (slower, but smarter)
            - `openai/gpt-4.1` (slowest, but smartest)
            - `gemini/gemini-2.0-flash`
            - `gemini/gemini-2.5-flash-preview-04-17`
            - 'groq/gemma2-9b-it`
            - `groq/llama-3.3-70b-versatile`
            - `anthropic/claude-3-7-sonnet-latest`
            See https://docs.litellm.ai/docs/providers/ for more options.
        api_key (str | None): The API key to use for the LLM. If not provided, LiteLLM will take it from the environment variables.
        temperature (float): The temperature to use for the LLM. Default is 1.0.
        top_p (float): The top-p value to use for the LLM. Default is 0.95.
        n_workers (int): The number of concurrent workers making the LLM calls. Default is 10. The value is clamped to the range [1, 10].
            If n_workers is 1, the generation of batches becomes sequential and certain features for better data consistency are enabled.
        return_type (Literal["auto", "dict"]): The format of the returned data. Default is "auto".

    Returns:
        - pd.DataFrame: A single DataFrame containing the generated mock data, if only one table is provided.
        - dict[str, pd.DataFrame]: A dictionary containing the generated mock data for each table, if multiple tables are provided.

    Example of generating mock data for a single table (without PK):
    ```python
    from mostlyai import mock

    tables = {
        "guests": {
            "prompt": "Guests of an Alpine ski hotel in Austria",
            "columns": {
                "nationality": {"prompt": "2-letter code for the nationality", "dtype": "string"},
                "name": {"prompt": "first name and last name of the guest", "dtype": "string"},
                "gender": {"dtype": "category", "values": ["male", "female"]},
                "age": {"prompt": "age in years; min: 18, max: 80; avg: 25", "dtype": "integer"},
                "date_of_birth": {"prompt": "date of birth", "dtype": "date"},
                "checkin_time": {"prompt": "the check in timestamp of the guest; may 2025", "dtype": "datetime"},
                "is_vip": {"prompt": "is the guest a VIP", "dtype": "boolean"},
                "price_per_night": {"prompt": "price paid per night, in EUR", "dtype": "float"},
                "room_number": {"prompt": "room number", "dtype": "integer", "values": [101, 102, 103, 201, 202, 203, 204]}
            },
        }
    }
    df = mock.sample(tables=tables, sample_size=10, model="openai/gpt-4.1-nano")
    ```

    Example of generating mock data for multiple tables (with PK/FK relationships):
    ```python
    from mostlyai import mock

    tables = {
        "customers": {
            "prompt": "Customers of a hardware store",
            "columns": {
                "customer_id": {"prompt": "the unique id of the customer", "dtype": "string"},
                "name": {"prompt": "first name and last name of the customer", "dtype": "string"},
            },
            "primary_key": "customer_id",  # single string; no composite keys allowed; primary keys must have string dtype
        },
        "warehouses": {
            "prompt": "Warehouses of a hardware store",
            "columns": {
                "warehouse_id": {"prompt": "the unique id of the warehouse", "dtype": "string"},
                "name": {"prompt": "the name of the warehouse", "dtype": "string"},
            },
            "primary_key": "warehouse_id",
        },
        "orders": {
            "prompt": "Orders of a Customer",
            "columns": {
                "customer_id": {"prompt": "the customer id for that order", "dtype": "string"},
                "warehouse_id": {"prompt": "the warehouse id for that order", "dtype": "string"},
                "order_id": {"prompt": "the unique id of the order", "dtype": "string"},
                "text": {"prompt": "order text description", "dtype": "string"},
                "amount": {"prompt": "order amount in USD", "dtype": "float"},
            },
            "primary_key": "order_id",
            "foreign_keys": [
                {
                    "column": "customer_id",
                    "referenced_table": "customers",
                    "prompt": "each customer has anywhere between 2 and 3 orders",
                },
                {
                    "column": "warehouse_id",
                    "referenced_table": "warehouses",
                },
            ],
        },
        "items": {
            "prompt": "Items in an Order",
            "columns": {
                "item_id": {"prompt": "the unique id of the item", "dtype": "string"},
                "order_id": {"prompt": "the order id for that item", "dtype": "string"},
                "name": {"prompt": "the name of the item", "dtype": "string"},
                "price": {"prompt": "the price of the item in USD", "dtype": "float"},
            },
            "foreign_keys": [
                {
                    "column": "order_id",
                    "referenced_table": "orders",
                    "prompt": "each order has between 1 and 2 items",
                }
            ],
        },
    }
    data = mock.sample(tables=tables, sample_size=2, model="openai/gpt-4.1")
    df_customers = data["customers"]
    df_warehouses = data["warehouses"]
    df_orders = data["orders"]
    df_items = data["items"]
    ```

    Example of enriching a single dataframe:
    ```python
    from mostlyai import mock
    import pandas as pd

    tables = {
        "patients": {
            "prompt": "Patients of a hospital in Finland",
            "columns": {
                "full_name": {"prompt": "first name and last name of the patient", "dtype": "string"},
                "date_of_birth": {"prompt": "date of birth", "dtype": "date"},
                "place_of_birth": {"prompt": "place of birth", "dtype": "string"},
            },
        },
    }
    existing_df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "gender": ["male", "male", "female", "female"],
    })
    enriched_df = mock.sample(
        tables=tables,
        existing_data={"patients": existing_df},
        model="openai/gpt-4.1-nano"
    )
    enriched_df
    ```

    Example of enriching / augmenting an existing dataset:
    ```python
    from mostlyai import mock
    import pandas as pd

    tables = {
        "customers": {
            "prompt": "Customers of a hardware store",
            "columns": {
                "customer_id": {"prompt": "the unique id of the customer", "dtype": "string"},
                "name": {"prompt": "first name and last name of the customer", "dtype": "string"},
                "email": {"prompt": "email address of the customer", "dtype": "string"},
                "phone": {"prompt": "phone number of the customer", "dtype": "string"},
                "loyalty_level": {"dtype": "category", "values": ["bronze", "silver", "gold", "platinum"]},
            },
            "primary_key": "customer_id",
        },
        "orders": {
            "prompt": "Orders of a Customer",
            "columns": {
                "order_id": {"prompt": "the unique id of the order", "dtype": "string"},
                "customer_id": {"prompt": "the customer id for that order", "dtype": "string"},
                "order_date": {"prompt": "the date when the order was placed", "dtype": "date"},
                "total_amount": {"prompt": "order amount in USD", "dtype": "float"},
                "status": {"dtype": "category", "values": ["pending", "shipped", "delivered", "cancelled"]},
            },
            "primary_key": "order_id",
            "foreign_keys": [
                {
                    "column": "customer_id",
                    "referenced_table": "customers",
                    "prompt": "each customer has anywhere between 1 and 3 orders",
                },
            ],
        },
    }
    existing_customers = pd.DataFrame({
        "customer_id": [101, 102, 103],
        "name": ["John Davis", "Maria Garcia", "Wei Chen"],
    })
    existing_orders = pd.DataFrame({
        "order_id": ["ORD-001", "ORD-002"],
        "customer_id": [101, 101],
    })
    data = mock.sample(
        tables=tables,
        existing_data={
            "customers": existing_customers,
            "orders": existing_orders,
        },
        model="openai/gpt-4.1-nano"
    )
    df_customers = data["customers"]
    df_orders = data["orders"]
    ```
    """

    tables: dict[str, TableConfig] = _harmonize_tables(tables, existing_data)
    config = MockConfig(tables)

    llm_config = LLMConfig(model=model, api_key=api_key, temperature=temperature, top_p=top_p)

    sample_size: dict[str, int] = _harmonize_sample_size(sample_size, config)
    primary_keys = {table_name: table_config.primary_key for table_name, table_config in config.root.items()}

    n_workers = max(min(n_workers, 10), 1)

    execution_plan: list[str] = _build_execution_plan(config)

    data: dict[str, pd.DataFrame] = _harmonize_existing_data(existing_data, config) or {}

    # synchronous `sample` function makes independent calls to asynchronous `_sample_table` function
    # in order to avoid conflicts with potentially existing event loop (e.g. in Jupyter environment),
    # a new thread is spawned for each call to `_sample_table`
    # NOTE: initialize executor only once, doing that inside the loop might lead to deadlocks
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for table_name in execution_plan:
            table_config = config.root[table_name]
            future = executor.submit(
                _sample_table_sync,
                name=table_name,
                prompt=table_config.prompt,
                columns=table_config.columns,
                foreign_keys=table_config.foreign_keys,
                primary_keys=primary_keys,
                data=data,
                sample_size=sample_size.get(table_name),
                previous_rows_size=10,  # present 10 previously generated rows to the LLM
                non_context_size=10,  # pick 10 rows to choose from for each non-context foreign key
                n_workers=n_workers,
                llm_config=llm_config,
            )
            df = future.result()
            data[table_name] = df

    return next(iter(data.values())) if len(data) == 1 and return_type == "auto" else data
