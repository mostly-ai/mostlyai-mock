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

import json
from collections import deque
from collections.abc import Generator
from enum import Enum
from typing import Any, Literal, Type

import litellm
import pandas as pd
from pydantic import BaseModel, Field, RootModel, create_model, field_validator, model_validator
from tqdm import tqdm

SYSTEM_PROMPT = f"""
You are a specialized synthetic data generator designed to create
highly realistic, contextually appropriate data based on schema definitions. Your task is to:

1. Generate data that strictly adheres to the provided schema constraints (data types, ranges, formats)
2. Ensure logical consistency across related tables and foreign key relationships
3. Create contextually appropriate values that reflect real-world patterns and distributions
4. Produce diverse, non-repetitive data that avoids obvious patterns
5. Respect uniqueness constraints and other data integrity rules
6. Return well-formatted JSON output that can be directly parsed.
7. Don't use markdown formatting.

For numeric fields, generate realistic distributions rather than random values. For text fields, create contextually \
appropriate content. For dates and timestamps, ensure logical chronology. Always maintain referential integrity \
across tables.
"""


class LLMConfig(BaseModel):
    model: str = "openai/gpt-4.1-nano"
    api_key: str | None = None
    temperature: float = 1.0
    top_p: float = 0.95


class MockConfig(RootModel[dict[str, "TableConfig"]]):
    root: dict[str, TableConfig] = Field(..., min_items=1)

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
    columns: dict[str, ColumnConfig] = Field(..., min_items=1)
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
    generated_data: dict[str, pd.DataFrame] | None,
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
        generated_data=generated_data,
        sample_size=sample_size,
        batch_size=batch_size,
        previous_rows_size=previous_rows_size,
        non_context_size=non_context_size,
        llm_config=llm_config,
    )
    table_rows_generator = tqdm(table_rows_generator, desc=f"Generating rows for table `{name}`".ljust(45))
    table_df = _convert_table_rows_generator_to_df(table_rows_generator=table_rows_generator, columns=columns)
    return table_df


def _create_table_prompt(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    primary_keys: dict[str, str] | None,
    batch_size: int | None,
    foreign_keys: list[ForeignKeyConfig] | None,
    context_data: pd.DataFrame | None,
    non_context_data: dict[str, pd.DataFrame] | None,
    previous_rows: list[dict] | None,
) -> str:
    # add table prompt
    prompt = f"# {prompt}\n\n"

    # define table
    prompt += f"## Table: {name}\n\n"

    prompt += f"## Table Primary Key: `{primary_keys[name]}`\n\n"

    # add columns specifications
    prompt += "## Columns Specifications:\n\n"
    prompt += f"{json.dumps({name: config.model_dump() for name, config in columns.items()}, indent=2)}\n\n"

    # add previous rows as context to help the LLM generate consistent data
    if previous_rows:
        prompt += f"\n## Previous {len(previous_rows)} Rows:\n\n"
        prompt += f"{json.dumps(previous_rows, indent=2)}\n\n"

    # define foreign keys
    if foreign_keys:
        prompt += "## Foreign Keys:\n\n"
        prompt += f"{json.dumps([fk.model_dump() for fk in foreign_keys], indent=2)}\n\n"

    # add context table name, primary key and data
    if foreign_keys and foreign_keys[0].referenced_table != name:  # self-dependency is not considered as context
        assert context_data is not None
        fk = foreign_keys[0]
        prompt += f"## Context Table: `{fk.referenced_table}`\n\n"

        prompt += f"## Context Table Primary Key: `{primary_keys[fk.referenced_table]}`\n\n"

        prompt += f"## Context Table Data:\n\n"
        prompt += f"{context_data.to_json(orient='records', indent=2)}\n\n"

    # add non-context table names, primary keys and data
    if foreign_keys and len(foreign_keys) > 1:
        for fk in foreign_keys[1:]:
            if fk.referenced_table == name:  # self-dependency is not considered as non-context
                continue
            assert non_context_data is not None
            assert fk.referenced_table in non_context_data
            prompt += f"## Non-Context Table: `{fk.referenced_table}`\n\n"

            prompt += f"## Non-Context Table Primary Key: `{primary_keys[fk.referenced_table]}`\n\n"

            prompt += f"## Non-Context Table Data:\n\n"
            prompt += f"{non_context_data[fk.referenced_table].to_json(orient='records', indent=2)}\n\n"

    # add instructions
    prompt += "\n## Instructions:\n\n"
    if not foreign_keys:
        assert batch_size is not None
        prompt += f"Generate {batch_size} rows for the `{name}` table.\n\n"
    else:
        prompt += (
            f"Generate data for the `{name}` table. "
            f"The first Foreign Key column from Foreign Keys section may only contain values from Context Table Data. "
            f"The following Foreign Key columns from Foreign Keys section (if exists) may only contain values from Non-Context Table Data sections. "
            f"If either relevant Context Table Data or Non-Context Table Data is not present, this means that table has self-dependency. "
            f"In this case, ensure that the generated foreign keys are consistent with generated primary keys of the table. "
            f"Pay attention to prompt of the Foreign Key column to understand the relationship.\n\n"
        )

    if previous_rows:
        prompt += (
            "Generate new rows that maintain consistency with the previous rows where appropriate. "
            "Don't copy previous rows in the output. "
            "Don't pay attention to the number of previous rows; there might have been more generated than provided.\n\n"
        )
    prompt += f"Do not use code to generate the data.\n\n"
    prompt += f"Return the full data as a JSON string.\n"

    return prompt


def _create_table_rows_generator(
    *,
    name: str,
    prompt: str,
    columns: dict[str, ColumnConfig],
    foreign_keys: list[ForeignKeyConfig] | None,
    primary_keys: dict[str, str] | None,
    generated_data: dict[str, pd.DataFrame] | None,
    sample_size: int,
    batch_size: int,
    previous_rows_size: int,
    non_context_size: int | None,
    llm_config: LLMConfig,
) -> Generator[dict]:
    def create_table_response_format(columns: dict[str, ColumnConfig]) -> BaseModel:
        def create_annotation(column_config: ColumnConfig) -> Type:
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

    def batch_infinitely(data: pd.DataFrame | None) -> Generator[pd.DataFrame | None]:
        while True:
            if data is None:
                yield None
            else:
                for i in range(0, len(data), batch_size):
                    yield data.iloc[i : i + batch_size]

    # ensure model supports response_format and json schema
    supported_params = litellm.get_supported_openai_params(model=llm_config.model)
    assert "response_format" in supported_params
    assert litellm.supports_response_schema(llm_config.model), (
        "The model does not support structured output / JSON mode."
    )

    # derive context data (if first foreign key is present) and harmonize sample size accordingly
    context_data: pd.DataFrame | None = None
    if foreign_keys and foreign_keys[0].referenced_table != name:  # self-dependency is not considered as context
        context_table_name = foreign_keys[0].referenced_table
        assert generated_data is not None
        assert context_table_name in generated_data
        context_data = generated_data[context_table_name]
        sample_size = len(context_data)

    # derive non-context data (if more than one foreign key is present)
    non_context_data: dict[str, pd.DataFrame] = {}
    if foreign_keys and len(foreign_keys) > 1:
        assert generated_data is not None
        assert non_context_size is not None
        for fk in foreign_keys[1:]:
            if fk.referenced_table == name:  # self-dependency is not considered as non-context
                continue
            non_context_table_name = fk.referenced_table
            assert non_context_table_name in generated_data
            non_context_data[non_context_table_name] = generated_data[non_context_table_name]

    litellm_kwargs = {
        "response_format": create_table_response_format(columns=columns),
        "temperature": llm_config.temperature,
        "top_p": llm_config.top_p,
        "model": llm_config.model,
        "api_key": llm_config.api_key,
        "stream": True,
    }

    yielded_sequences = 0
    previous_rows = deque(maxlen=previous_rows_size)
    for context_batch in batch_infinitely(context_data):
        non_context_batch = (
            {table_name: df.sample(frac=1.0).head(non_context_size) for table_name, df in non_context_data.items()}
            if non_context_data
            else None
        )
        prompt = _create_table_prompt(
            name=name,
            prompt=prompt,
            columns=columns,
            primary_keys=primary_keys,
            batch_size=batch_size,
            foreign_keys=foreign_keys,
            context_data=context_batch,
            non_context_data=non_context_batch,
            previous_rows=list(previous_rows),
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        response = litellm.completion(messages=messages, **litellm_kwargs)
        rows_stream = yield_rows_from_json_chunks_stream(response)

        while True:
            try:
                row = next(rows_stream)
            except StopIteration:
                break  # move to next batch
            previous_rows.append(row)
            yield row
            if context_batch is None:
                # each subject row is considered a single sequence
                yielded_sequences += 1
                if yielded_sequences >= sample_size:
                    return  # move to next table
        if context_batch is not None:
            # for each context_batch, full sequences are generated
            yielded_sequences += len(context_batch)
            if yielded_sequences >= sample_size:
                return  # move to next table


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
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce").astype("boolean[pyarrow]")
            elif column_config.dtype is DType.CATEGORY:
                df[column_name] = pd.Categorical(df[column_name], categories=column_config.values)
            else:
                df[column_name] = df[column_name].astype("string[pyarrow]")
        return df

    df = pd.DataFrame(list(table_rows_generator))
    df = align_df_dtypes_with_mock_dtypes(df, columns)
    return df


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
    sample_size: int | dict[str, int] = 10,
    model: str = "openai/gpt-4.1-nano",
    api_key: str | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    return_type: Literal["auto", "dict"] = "auto",
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Generate mock data by prompting an LLM.

    Args:
        tables (dict[str, dict]): The table specifications to generate mock data for. See examples for usage.
        sample_size (int | dict[str, int]): The number of rows to generate for each subject table.
            If a single integer is provided, the same number of rows will be generated for each subject table.
            If a dictionary is provided, the number of rows to generate for each subject table can be specified
            individually.
            Default is 10.
        model (str): The LiteLLM chat completion model to be used. Requires support for structured output / JSON mode.
            Examples include:
            - `openai/gpt-4.1-nano` (default)
            - `openai/gpt-4.1-mini`
            - `openai/gpt-4.1`
            - `gemini/gemini-2.0-flash`
            - `gemini/gemini-2.5-flash-preview-04-17`
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

    Example of single table (without PK):
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

    Example of multiple tables (with PK/FK relationships):
    ```python
    from mostlyai import mock

    tables = {
        "customers": {
            "prompt": "Customers of a hardware store",
            "columns": {
                "customer_id": {"prompt": "the unique id of the customer", "dtype": "integer"},
                "name": {"prompt": "first name and last name of the customer", "dtype": "string"},
            },
            "primary_key": "customer_id",
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
    """

    config = MockConfig(tables)
    llm_config = LLMConfig(model=model, api_key=api_key, temperature=temperature, top_p=top_p)

    sample_size = _harmonize_sample_size(sample_size, config)
    primary_keys = {table_name: table_config.primary_key for table_name, table_config in config.root.items()}

    execution_plan: list[str] = _build_execution_plan(config)

    data: dict[str, pd.DataFrame] = {}

    for table_name in execution_plan:
        table_config = config.root[table_name]
        df = _sample_table(
            name=table_name,
            prompt=table_config.prompt,
            columns=table_config.columns,
            foreign_keys=table_config.foreign_keys,
            primary_keys=primary_keys,
            generated_data=data,
            sample_size=sample_size[table_name],
            batch_size=30,  # generate 30 root table rows at a time
            previous_rows_size=10,  # present 10 previously generated rows to the LLM
            non_context_size=10,  # pick 10 rows to choose from for each non-context foreign key
            llm_config=llm_config,
        )
        data[table_name] = df

    return next(iter(data.values())) if len(data) == 1 and return_type == "auto" else data
