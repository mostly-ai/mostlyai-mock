## Engine Reference

### mostlyai.mock.sample

```python
sample(
    *,
    tables,
    sample_size=4,
    existing_data=None,
    model="openai/gpt-4.1-nano",
    api_key=None,
    temperature=1.0,
    top_p=0.95,
    n_workers=10,
    return_type="auto"
)

```

Generate synthetic data from scratch or enrich existing data with new columns.

While faker and numpy are useful to create fake data, this utility is unique as it allows the creation of coherent, realistic multi-table tabular mock data or the enrichment of existing datasets with new, context-aware columns.

It is particularly useful for quickly simulating production-like datasets for testing or prototyping purposes. It is advised to limit mocking to small datasets for performance reasons (rows * cols < 1000). It might take a couple of minutes for bigger datasets.

Parameters:

| Name | Type | Description | Default | | --- | --- | --- | --- | | `tables` | `dict[str, dict]` | The table specifications to generate mock data for. See examples for usage. Note: Avoid using double quotes (") and other special characters in column names. | *required* | | `sample_size` | `int | dict[str, int]` | The number of rows to generate for each subject table. If a single integer is provided, the same number of rows will be generated for each subject table. If a dictionary is provided, the number of rows to generate for each subject table can be specified individually. Default is 4. Ignored if existing_data is provided. Ignored for non-root tables. If a table has a foreign key, the sample size is determined by the corresponding foreign key prompt. If nothing specified, a few rows per parent record are generated. | `4` | | `existing_data` | `dict[str, DataFrame] | None` | Existing data to augment. If provided, the sample_size argument is ignored. Default is None. | `None` | | `model` | `str` | The LiteLLM chat completion model to be used. Examples include: - openai/gpt-4.1-nano (default; fast, and smart) - openai/gpt-4.1-mini (slower, but smarter) - openai/gpt-4.1 (slowest, but smartest) - gemini/gemini-2.0-flash - gemini/gemini-2.5-flash-preview-04-17 - 'groq/gemma2-9b-it-groq/llama-3.3-70b-versatile-anthropic/claude-3-7-sonnet-latest`See https://docs.litellm.ai/docs/providers/ for more options. |`'openai/gpt-4.1-nano'`| |`api_key`|`str | None`| The API key to use for the LLM. If not provided, LiteLLM will take it from the environment variables. |`None`| |`temperature`|`float`| The temperature to use for the LLM. Default is 1.0. |`1.0`| |`top_p`|`float`| The top-p value to use for the LLM. Default is 0.95. |`0.95`| |`n_workers`|`int`| The number of concurrent workers making the LLM calls. Default is 10. The value is clamped to the range [1, 10]. If n_workers is 1, the generation of batches becomes sequential and certain features for better data consistency are enabled. |`10`| |`return_type`|`Literal['auto', 'dict']`| The format of the returned data. Default is "auto". |`'auto'\` |

Returns:

| Type | Description | | --- | --- | | `DataFrame | dict[str, DataFrame]` | pd.DataFrame: A single DataFrame containing the generated mock data, if only one table is provided. | | `DataFrame | dict[str, DataFrame]` | dict\[str, pd.DataFrame\]: A dictionary containing the generated mock data for each table, if multiple tables are provided. |

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
