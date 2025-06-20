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

import itertools
import json
from collections import deque
from collections.abc import Generator
from enum import Enum
from typing import Any, Literal

import litellm
import pandas as pd
import tenacity
from pydantic import BaseModel, Field, RootModel, create_model, field_validator, model_validator
from tqdm import tqdm

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
                        f"Column '{fk.column}' type '{fk_field.dtype}' does not match "
                        f"referenced primary key '{referenced_config.primary_key}' type '{pk_field.dtype}'"
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


def _sample_table(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    foreign_keys: list[ForeignKeyConfig] | None,
    primary_keys: dict[str, str] | None,
    data: dict[str, pd.DataFrame],
    sample_size: int,
    batch_size: int,
    previous_rows_size: int,
    non_context_size: int | None,
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
        batch_size=batch_size,
        previous_rows_size=previous_rows_size,
        non_context_size=non_context_size,
        llm_config=llm_config,
    )
    table_rows_generator = tqdm(table_rows_generator, desc=f"Generating rows for table `{name}`".ljust(45))
    table_df = _convert_table_rows_generator_to_df(table_rows_generator=table_rows_generator, columns=columns)
    return table_df


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
7. Return well-formatted {llm_output_format} output that can be directly parsed
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
    primary_keys: dict[str, str] | None,
    batch_size: int | None,
    foreign_keys: list[ForeignKeyConfig] | None,
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

    prompt += f"### Target Table Primary Key: `{primary_keys[name]}`\n\n"

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
            prompt += f"### Primary Key Column: `{primary_keys[name]}`\n\n"

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
        prompt += f"Number of rows to {verb}: `{n_rows}`.\n\n"

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

    prompt += f"Return data as a {llm_output_format} string."
    if llm_output_format == LLMOutputFormat.JSON:
        prompt += " The JSON string should have 'rows' key at the top level."
        prompt += " The value of 'rows' key should be a list of JSON objects."
        prompt += " Each JSON object should have column names as keys and values as column values."
    else:  # llm_output_format == LLMOutputFormat.CSV
        prompt += " The CSV string should have a header row with column names."
        prompt += " The CSV string should have a data row for each row to be generated."
        prompt += " The CSV string should have a newline character at the end of each row."

    if existing_data is not None:
        prompt += f" Only include the following columns in the {llm_output_format} string: {list(columns.keys() - existing_data.columns)}."
    prompt += "\n"
    return prompt


def _create_table_rows_generator(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    foreign_keys: list[ForeignKeyConfig] | None,
    primary_keys: dict[str, str] | None,
    data: dict[str, pd.DataFrame],
    sample_size: int,
    batch_size: int,
    previous_rows_size: int,
    non_context_size: int | None,
    llm_config: LLMConfig,
) -> Generator[dict]:
    def supports_structured_outputs(model: str) -> bool:
        supported_params = litellm.get_supported_openai_params(model=model) or []
        return "response_format" in supported_params and litellm.supports_response_schema(model)

    def create_structured_output_schema(
        columns: dict[str, ColumnConfig], existing_data: pd.DataFrame | None
    ) -> type[BaseModel]:
        def create_annotation(column_config: ColumnConfig) -> type:
            if column_config.values or column_config.dtype is DType.CATEGORY:
                return Literal[tuple(column_config.values)]
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

    def yield_rows_from_json_chunks_stream(response: litellm.CustomStreamWrapper) -> Generator[dict]:
        # starting with dirty buffer is to handle the `{"rows": []}` case
        buffer = "garbage"
        rows_json_started = False
        in_row_json = False
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue
            for char in delta:
                buffer += char
                if char == "{" and not rows_json_started:
                    # {"rows": [{"name": "Jo\}h\{n"}]}
                    # *                                 <- start of rows json stream
                    rows_json_started = True
                elif char == "{" and not in_row_json:
                    # {"rows": [{"name": "Jo\}h\{n"}]}
                    #           *                       <- start of single row json stream
                    buffer = "{"
                    in_row_json = True
                elif char == "}":
                    # {"rows": [{"name": "Jo\}h\{n"}]}
                    #                        *     * *  <- any of these
                    try:
                        row = json.loads(buffer)
                        yield row
                        buffer = ""
                        in_row_json = False
                    except json.JSONDecodeError:
                        continue

    def yield_rows_from_csv_chunks_stream(response: litellm.CustomStreamWrapper) -> Generator[dict]:
        buffer = ""
        header = None
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue
            for char in delta:
                buffer += char
                if char == "\n":
                    if header is None:
                        # column1,column2,column3\n
                        #                        ** <- end of header row
                        header = buffer[:-1].split(",")
                        buffer = ""
                    else:
                        # value_1,value_2,value_3\n
                        #                        ** <- end of data row
                        row = buffer[:-1].split(",")
                        yield {header[i]: row[i] for i in range(len(header))}
                        buffer = ""

    def batch_infinitely(data: pd.DataFrame | None) -> Generator[pd.DataFrame | None]:
        while True:
            if data is None:
                yield None
            else:
                for i in range(0, len(data), batch_size):
                    yield data.iloc[i : i + batch_size]

    def completion_with_retries(*args, **kwargs):
        n_attempts = 3

        def print_on_retry(_):
            print(" * Calling LLM again... * ", end="", flush=True)

        # try up to 3 times, print a message to the user on each retry
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(n_attempts), reraise=True, before_sleep=print_on_retry
        )
        return retryer(litellm.completion, *args, **kwargs)

    # derive data for augmentation
    existing_data: pd.DataFrame | None = None
    if name in data:
        existing_data = data[name]
        sample_size = len(existing_data)

    # derive context data (if first foreign key is present) and harmonize sample size accordingly
    context_data: pd.DataFrame | None = None
    if foreign_keys and foreign_keys[0].referenced_table != name:  # self-dependency is not considered as context
        context_table_name = foreign_keys[0].referenced_table
        assert context_table_name in data
        context_data = data[context_table_name]
        batch_size = 1  # generate one sequence at a time
        sample_size = len(context_data)

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

    litellm_kwargs = {
        "temperature": llm_config.temperature,
        "top_p": llm_config.top_p,
        "model": llm_config.model,
        "api_key": llm_config.api_key,
        "stream": True,
    }

    batch_idx = 0
    yielded_sequences = 0
    previous_rows = deque(maxlen=previous_rows_size)
    context_batch_generator = batch_infinitely(context_data)
    context_batch = None
    do_repeat_previous_batch = False
    n_malformed_batches = 0
    llm_output_format = LLMOutputFormat.CSV
    while True:  # iterate over batches
        # pick next context batch or repeat the previous one
        context_batch = context_batch if do_repeat_previous_batch else next(context_batch_generator)
        n_malformed_batches += int(do_repeat_previous_batch)
        do_repeat_previous_batch = False

        # fallback to JSON with Structured Outputs if CSV generation fails too often
        if (
            llm_output_format == LLMOutputFormat.CSV
            and supports_structured_outputs(model=llm_config.model)
            and n_malformed_batches >= 3
        ):
            print(" * Falling back to JSON with Structured Outputs... * ", end="", flush=True)
            llm_output_format = LLMOutputFormat.JSON

        # raise error if generation is not stable
        if n_malformed_batches >= 10:
            raise RuntimeError(
                "Too many malformed batches were generated. Consider changing the model in order to make generation more stable."
            )

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

        if context_batch is None:
            # for root tables, scale down batch size in order to prevent excessive generations
            remaining_rows = sample_size - yielded_sequences
            if batch_size >= remaining_rows:
                batch_size = remaining_rows + 2  # +2 because LLM may not always count the rows correctly

        structured_output_schema = None
        if llm_output_format == LLMOutputFormat.JSON:
            structured_output_schema = create_structured_output_schema(columns=columns, existing_data=existing_batch)

        system_prompt = _create_system_prompt(llm_output_format)
        user_prompt = _create_table_prompt(
            name=name,
            prompt=prompt,
            columns=columns,
            primary_keys=primary_keys,
            batch_size=batch_size,
            foreign_keys=foreign_keys,
            existing_data=existing_batch,
            context_data=context_batch,
            non_context_data=non_context_batch,
            previous_rows=list(previous_rows),
            llm_output_format=llm_output_format,
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        n_existing_columns = len(existing_batch.columns) if existing_batch is not None else 0
        n_columns_to_generate = len(columns) - n_existing_columns
        if n_columns_to_generate > 0:
            response = completion_with_retries(
                messages=messages, response_format=structured_output_schema, **litellm_kwargs
            )
            yield_rows_from_chunks_stream = {
                LLMOutputFormat.JSON: yield_rows_from_json_chunks_stream,
                LLMOutputFormat.CSV: yield_rows_from_csv_chunks_stream,
            }[llm_output_format]
            rows_stream = yield_rows_from_chunks_stream(response)
        else:
            # skip roundtrip to LLM in case all columns are provided in existing data
            rows_stream = itertools.repeat({})

        rows = []
        for row_idx, row_generated_part in enumerate(rows_stream):  # iterate over rows in current batch
            row_existing_part = existing_batch.iloc[row_idx].to_dict() if existing_batch is not None else {}
            row = {**row_existing_part, **row_generated_part}
            if set(row.keys()) != set(columns.keys()):
                # mismatch of columns in a row suggests that entire batch is malformed; repeat the same batch in the next iteration
                print(" * Malformed row, repeating batch... * ", end="", flush=True)
                do_repeat_previous_batch = True
                break
            row = {column: row[column] for column in columns.keys()}  # keep columns order according to user's spec
            yield row
            if context_batch is None:
                # each subject row is considered a single sequence
                yielded_sequences += 1
                if yielded_sequences >= sample_size:
                    return  # move to next table
            rows.append(row)

        if do_repeat_previous_batch:
            continue

        previous_rows.extend(rows)

        if context_batch is not None:
            # for each context_batch, full sequences are generated
            yielded_sequences += len(context_batch)
            if yielded_sequences >= sample_size:
                return  # move to next table

        batch_idx += 1


def _convert_table_rows_generator_to_df(
    table_rows_generator: Generator[dict],
    columns: dict[str, ColumnConfig],
) -> pd.DataFrame:
    def align_df_dtypes_with_mock_dtypes(df: pd.DataFrame, columns: dict[str, ColumnConfig]) -> pd.DataFrame:
        for column_name, column_config in columns.items():
            if column_config.dtype in [DType.DATE, DType.DATETIME]:
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif column_config.dtype is DType.INTEGER:
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce", downcast="integer").astype(
                    "int64[pyarrow]"
                )
            elif column_config.dtype is DType.FLOAT:
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("double[pyarrow]")
            elif column_config.dtype is DType.BOOLEAN:
                df[column_name] = df[column_name].map(lambda x: True if str(x).lower() == "true" else x)
                df[column_name] = df[column_name].map(lambda x: False if str(x).lower() == "false" else x)
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("boolean[pyarrow]")
            elif column_config.dtype is DType.CATEGORY:
                df[column_name] = pd.Categorical(df[column_name], categories=column_config.values)
            else:
                df[column_name] = df[column_name].astype("string[pyarrow]")
        return df

    df = pd.DataFrame(list(table_rows_generator))
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
        column_configs = table_config.setdefault("columns", {})
        existing_column_configs = {
            existing_column: {"dtype": _infer_dtype(existing_table[existing_column])}
            for existing_column in existing_table.columns
            if existing_column not in column_configs
        }
        column_configs = {**existing_column_configs, **column_configs}
        table_config["columns"] = column_configs
    return tables


def _harmonize_sample_size(sample_size: int | dict[str, int], config: MockConfig) -> dict[str, int]:
    if isinstance(sample_size, int):
        return {table_name: sample_size for table_name in config.root}

    if sample_size.keys() != config.root.keys():
        raise ValueError(f"Sample size keys must match table names: {sample_size.keys()} != {config.root.keys()}")
    return sample_size


def _build_execution_plan(config: MockConfig) -> list[str]:
    def build_dependency_mappings(config: MockConfig) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
        child_to_parents = {}
        parent_to_children = {}

        for table_name in config.root:
            child_to_parents[table_name] = set()
            parent_to_children[table_name] = set()

        for table_name, table_config in config.root.items():
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

    child_to_parents, parent_to_children, root_tables = build_dependency_mappings(config)

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
            Default is 4. Ignored if existing_data is provided.
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
                "customer_id": {"prompt": "the unique id of the customer", "dtype": "integer"},
                "name": {"prompt": "first name and last name of the customer", "dtype": "string"},
            },
            "primary_key": "customer_id",  # single string; no composite keys allowed
        },
        "warehouses": {
            "prompt": "Warehouses of a hardware store",
            "columns": {
                "warehouse_id": {"prompt": "the unique id of the warehouse", "dtype": "integer"},
                "name": {"prompt": "the name of the warehouse", "dtype": "string"},
            },
            "primary_key": "warehouse_id",
        },
        "orders": {
            "prompt": "Orders of a Customer",
            "columns": {
                "customer_id": {"prompt": "the customer id for that order", "dtype": "integer"},
                "warehouse_id": {"prompt": "the warehouse id for that order", "dtype": "integer"},
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
                "customer_id": {"prompt": "the unique id of the customer", "dtype": "integer"},
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
                "customer_id": {"prompt": "the customer id for that order", "dtype": "integer"},
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

    execution_plan: list[str] = _build_execution_plan(config)

    data: dict[str, pd.DataFrame] = existing_data or {}

    for table_name in execution_plan:
        table_config = config.root[table_name]
        df = _sample_table(
            name=table_name,
            prompt=table_config.prompt,
            columns=table_config.columns,
            foreign_keys=table_config.foreign_keys,
            primary_keys=primary_keys,
            data=data,
            sample_size=sample_size[table_name],
            batch_size=30,  # generate 30 root table rows at a time
            previous_rows_size=10,  # present 10 previously generated rows to the LLM
            non_context_size=10,  # pick 10 rows to choose from for each non-context foreign key
            llm_config=llm_config,
        )
        data[table_name] = df

    return next(iter(data.values())) if len(data) == 1 and return_type == "auto" else data
