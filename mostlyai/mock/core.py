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

import datetime
import json
import os
import time
import uuid
from typing import Literal, get_origin
from collections.abc import Generator

import litellm
import pandas as pd
from pydantic import BaseModel, Field, RootModel, create_model, field_validator
from tqdm import tqdm

# configure logfire if installed and LOGFIRE_TOKEN is set
if logfire_token := os.environ.get("LOGFIRE_TOKEN"):
    try:
        import logfire

        logfire.configure(token=logfire_token, console=None)
        logfire.instrument_openai()
    except ImportError:
        pass

SYSTEM_PROMPT = """
You are a specialized synthetic data generator designed to create
highly realistic, contextually appropriate data based on schema definitions. Your task is to:

1. Generate data that strictly adheres to the provided schema constraints (data types, ranges, formats)
2. Ensure logical consistency across related tables and foreign key relationships
3. Create contextually appropriate values that reflect real-world patterns and distributions
4. Produce diverse, non-repetitive data that avoids obvious patterns
5. Respect uniqueness constraints and other data integrity rules
6. Return well-formatted JSON outputs that can be directly parsed

For numeric fields, generate realistic distributions rather than random values. For text fields, create contextually \
appropriate content. For dates and timestamps, ensure logical chronology. Always maintain referential integrity \
across tables.
"""


class LLMConfig(BaseModel):
    model: str
    api_key: str | None = None


class ForeignKeyConfig(BaseModel):
    column: str
    referenced_table: str
    description: str | None = None


class TableConfig(BaseModel):
    data_schema: type[BaseModel]
    primary_key: str | None = None
    foreign_keys: list[ForeignKeyConfig] = Field(default_factory=list, min_length=0, max_length=1)


class MockConfig(RootModel[dict[str, TableConfig]]):
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

                if fk.column not in table_config.data_schema.model_fields:
                    raise ValueError(
                        f"Foreign key violation in table '{table_name}': "
                        f"Column '{fk.column}' does not exist in the schema"
                    )

                fk_field = table_config.data_schema.model_fields[fk.column]
                pk_field = referenced_config.data_schema.model_fields[referenced_config.primary_key]
                if fk_field.annotation != pk_field.annotation:
                    raise ValueError(
                        f"Foreign key violation in table '{table_name}': "
                        f"Column '{fk.column}' type '{fk_field.annotation}' does not match "
                        f"referenced primary key '{referenced_config.primary_key}' type '{pk_field.annotation}'"
                    )

        return tables


def _sample_table(
    *,
    table_name: str,
    table_config: TableConfig,
    sample_size: int | None,
    context_data: pd.DataFrame | None,
    temperature: float,
    top_p: float,
    batch_size: int,
    previous_rows_size: int,
    llm_config: LLMConfig,
) -> pd.DataFrame:
    assert (sample_size is None) != (context_data is None), (
        "Exactly one of sample_size or context_data must be provided"
    )
    if sample_size is None:
        sample_size = len(context_data)
    table_rows_generator = _create_table_rows_generator(
        table_name=table_name,
        table_config=table_config,
        sample_size=sample_size,
        context_data=context_data,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        previous_rows_size=previous_rows_size,
        llm_config=llm_config,
    )
    table_rows_generator = tqdm(table_rows_generator, desc=f"Generating rows for table `{table_name}`".ljust(45))
    table_df = _convert_table_rows_generator_to_df(table_rows_generator=table_rows_generator, table_config=table_config)
    return table_df


def _create_table_prompt(
    *,
    table_name: str,
    table_schema: type[BaseModel],
    batch_size: int | None,
    foreign_keys: list[ForeignKeyConfig] | None,
    context_data: pd.DataFrame | None,
    previous_rows: list[dict],
) -> str:
    if batch_size is not None:
        assert foreign_keys is None
        assert context_data is None
    else:
        assert foreign_keys is not None
        assert context_data is not None

    table_description = table_schema.__doc__

    # add description
    prompt = f"# {table_description}\n\n"

    # define table
    prompt += f"## Table: {table_name}\n\n"

    # add columns specifications
    prompt += "## Columns Specifications:\n\n"
    prompt += f"{json.dumps(table_schema.model_json_schema()['properties'], indent=2)}\n\n"

    # define foreign keys
    if foreign_keys is not None:
        prompt += "## Foreign Keys:\n\n"
        prompt += f"{json.dumps([fk.model_dump() for fk in foreign_keys], indent=2)}\n\n"

    # add context key data
    if context_data is not None:
        prompt += "## Context Foreign Key Data:\n\n"
        prompt += f"{context_data.to_json(orient='records', indent=2)}\n\n"

    # add previous rows as context to help the LLM generate consistent data
    if previous_rows:
        prompt += "\n## Previous Rows:\n\n"
        prompt += json.dumps(previous_rows, indent=2)

    # add instructions
    prompt += "\n## Instructions:\n\n"
    if batch_size is not None:
        prompt += f"Generate {batch_size} rows for the `{table_name}` table.\n\n"
    else:
        prompt += f"Generate rows for the `{table_name}` table\n\n"
    if previous_rows:
        prompt += "Generate new rows that maintain consistency with the previous rows where appropriate.\n\n"
    prompt += "Do not use code to generate the data. Return the full data in the JSON array.\n"

    return prompt


def _create_table_rows_generator(
    *,
    table_name: str,
    table_config: TableConfig,
    sample_size: int,
    temperature: float,
    top_p: float,
    context_data: pd.DataFrame | None,
    batch_size: int,
    previous_rows_size: int,
    llm_config: LLMConfig,
) -> Generator[dict]:
    def create_table_response_format(table_schema: type[BaseModel]) -> BaseModel:
        def create_compatible_pydantic_model(model: type[BaseModel]) -> type[BaseModel]:
            # response_format has limited support for pydantic features
            # so we need to convert the model to a compatible one
            # - date / time / datetime / UUID -> str
            # - remove any constraints (e.g. gt/ge/lt/le)
            fields = {}
            for field_name, field in model.model_fields.items():
                annotation = field.annotation
                if field.annotation in [datetime.date, datetime.time, datetime.datetime, uuid.UUID]:
                    annotation = str
                fields[field_name] = (annotation, Field(...))
            return create_model("TableRow", **fields)

        TableRow = create_compatible_pydantic_model(table_schema)
        TableRows = create_model("TableRows", rows=(list[TableRow], ...))
        return TableRows

    def yield_rows_from_chunks_stream(stream: litellm.CustomStreamWrapper) -> Generator[dict]:
        # starting with dirty buffer is to handle the `{"rows": []}` case
        buffer = "garbage"
        rows_json_started = False
        in_row_json = False
        for chunk in stream:
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

    # ensure model supports response_format
    supported_params = litellm.get_supported_openai_params(model=llm_config.model)
    assert "response_format" in supported_params

    # ensure json schema is supported
    assert litellm.supports_response_schema(llm_config.model)

    litellm_kwargs = {
        "response_format": create_table_response_format(table_schema=table_config.data_schema),
        "temperature": temperature,
        "top_p": top_p,
        "model": llm_config.model,
        "api_key": llm_config.api_key,
        "stream": True,
    }

    yielded_sequences = 0
    previous_rows = []
    for context_batch in batch_infinitely(context_data):
        prompt_kwargs = {
            "table_name": table_name,
            "table_schema": table_config.data_schema,
            "batch_size": batch_size if context_batch is None else None,
            "foreign_keys": table_config.foreign_keys if context_batch is not None else None,
            "context_data": context_batch if context_batch is not None else None,
            "previous_rows": previous_rows,
        }
        prompt = _create_table_prompt(**prompt_kwargs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        litellm_result = litellm.completion(messages=messages, **litellm_kwargs)
        rows_stream = yield_rows_from_chunks_stream(litellm_result)

        batch_rows = []
        while True:
            try:
                row = next(rows_stream)
            except StopIteration:
                break  # move to next batch
            batch_rows.append(row)
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

        previous_rows = batch_rows[:previous_rows_size]


def _convert_table_rows_generator_to_df(
    table_rows_generator: Generator[dict], table_config: TableConfig
) -> pd.DataFrame:
    def coerce_pandas_dtypes_to_pydantic_model(df: pd.DataFrame, model: type[BaseModel]) -> pd.DataFrame:
        for field_name, field in model.model_fields.items():
            if field.annotation in [datetime.date, datetime.datetime]:
                # datetime.date, datetime.datetime -> datetime64[ns] / datetime64[ns, tz]
                df[field_name] = pd.to_datetime(df[field_name], errors="coerce")
            elif field.annotation is datetime.time:
                # datetime.time -> object
                df[field_name] = pd.to_datetime(df[field_name], format="%H:%M:%S", errors="coerce").dt.time
            elif field.annotation in [int, float]:
                # int -> int64[pyarrow], float -> double[pyarrow]
                df[field_name] = pd.to_numeric(df[field_name], errors="coerce", dtype_backend="pyarrow")
            elif field.annotation is bool:
                # bool -> bool
                df[field_name] = df[field_name].astype(bool)
            elif get_origin(field.annotation) == Literal:
                # Literal -> category
                df[field_name] = pd.Categorical(df[field_name])
            else:
                # other -> string[pyarrow]
                df[field_name] = df[field_name].astype("string[pyarrow]")
        return df

    df = pd.DataFrame(list(table_rows_generator))
    df = coerce_pandas_dtypes_to_pydantic_model(df, table_config.data_schema)
    return df


def _harmonize_sample_size(sample_size: int | dict[str, int], config: MockConfig) -> dict[str, int]:
    if isinstance(sample_size, int):
        return {table_name: sample_size for table_name in config.root}

    if sample_size.keys() != config.root.keys():
        raise ValueError(f"Sample size keys must match table names: {sample_size.keys()} != {config.root.keys()}")
    return sample_size


def sample(
    *,
    tables: dict[str, dict],
    sample_size: int | dict[str, int] = 10,
    model: str = "openai/gpt-4.1-nano",
    api_key: str | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    
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
        model (str): The model to use for the LLM. Default is "openai/gpt-4.1-nano". This will be passed to LiteLLM.
        api_key (str | None): The API key to use for the LLM. If not provided, LiteLLM will take it from the environment variables.
        temperature (float): The temperature to use for the LLM. Default is 1.0.
        top_p (float): The top-p value to use for the LLM. Default is 0.95.

    Returns:
        - pd.DataFrame: A single DataFrame containing the generated mock data, if only one table is provided.
        - dict[str, pd.DataFrame]: A dictionary containing the generated mock data for each table, if multiple tables are provided.

    Example of single table (without PK):
    ```python
    import datetime
    from pydantic import BaseModel, Field
    from typing import Literal
    from mostlyai import mock

    class Guest(BaseModel):
        '''Guests of an Alpine ski hotel in Austria'''

        nationality: str = Field(description="2-letter code for the nationality")
        name: str = Field(description="first name and last name of the guest")
        gender: Literal["male", "female"]
        age: int = Field(description="age in years; min: 18, max: 80; avg: 25")
        date_of_birth: datetime.date = Field(description="date of birth")
        checkin_time: datetime.datetime = Field(description="the check in timestamp of the guest; may 2025")
        is_vip: bool = Field(description="is the guest a VIP")
        price_per_night: float = Field(description="price paid per night, in EUR")

    tables = {
        "guests": {
            "data_schema": Guest,
        }
    }
    df = mock.sample(tables=tables, sample_size=10)
    ```

    Example of multiple tables (with PK/FK relationships):
    ```python
    from pydantic import BaseModel, Field
    from mostlyai import mock

    class Guest(BaseModel):
        '''Guests of an Alpine ski hotel in Austria'''

        id: int = Field(description="the unique id of the guest")
        name: str = Field(description="first name and last name of the guest")
        
    class Purchases(BaseModel):
        '''Purchases of a Guest during their stay'''

        guest_id: int = Field(description="the guest id for that purchase")
        purchase_id: str = Field(description="the unique id of the purchase")
        text: str = Field(description="purchase text description")
        amount: float = Field(description="purchase amount in EUR")

    tables = {
        "guest": {
            "data_schema": Guest,
            "primary_key": "id",
        },
        "purchases": {
            "data_schema": Purchases,
            "foreign_keys": [{"column": "guest_id", "referenced_table": "guest", "description": "each guest has anywhere between 1 and 5 purchases"}],
        },
    }
    data = mock.sample(tables=tables, sample_size=5)
    df_guests = data["guest"]
    df_purchases = data["purchases"]
    ```
    """

    config = MockConfig(tables)

    sample_size = _harmonize_sample_size(sample_size, config)

    dfs = {}
    for table_name, table_config in config.root.items():
        if len(dfs) == 0:
            # subject table
            df = _sample_table(
                table_name=table_name,
                table_config=table_config,
                sample_size=sample_size[table_name],
                context_data=None,
                temperature=temperature,
                top_p=top_p,
                batch_size=20,  # generate 20 subjects at a time
                previous_rows_size=5,
                llm_config=LLMConfig(model=model, api_key=api_key),
            )
        elif len(dfs) == 1:
            # sequence table
            df = _sample_table(
                table_name=table_name,
                table_config=table_config,
                sample_size=None,
                context_data=next(iter(dfs.values())),
                temperature=temperature,
                top_p=top_p,
                batch_size=1,  # generate one sequence at a time
                previous_rows_size=5,
                llm_config=LLMConfig(model=model, api_key=api_key),
            )
        else:
            raise RuntimeError("Only 1 or 2 table setups are supported for now")
        dfs[table_name] = df

    return dfs if len(dfs) > 1 else next(iter(dfs.values()))
