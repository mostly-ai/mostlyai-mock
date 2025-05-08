# Synthetic Mock Data 🔮

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-mock/) [![stats](https://pepy.tech/badge/mostlyai-mock)](https://pypi.org/project/mostlyai-mock/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-mock) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-mock)

<!-- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-mock) -->

Create data out of nothing. Prompt LLMs for Tabular Data.

## Getting Started

1. Install the latest version of the `mostlyai-mock` python package.

```bash
pip install -U mostlyai-mock
```

2. Setup the API key of your LLM endpoint (if not done yet)

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
# os.environ["GEMINI_API_KEY"] = "your-api-key"
# os.environ["GROQ_API_KEY"] = "your-api-key"
```

Note: You will need to obtain your API key directly from the LLM service provider (e.g. for Open AI from [here](https://platform.openai.com/api-keys)). The LLM endpoint will be determined by the chosen `model` when making calls to `mock.sample`.

3. Create your first basic synthetic table from scratch

```python
from mostlyai import mock

tables = {
    "guests": {
        "description": "Guests of an Alpine ski hotel in Austria",
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
df = mock.sample(
    tables=tables,  # provide table and column definitions
    sample_size=10,  # generate 10 records
    model="openai/gpt-4.1-nano",  # select the LLM model (optional)
)
df
```

4. Create your first multi-table synthetic dataset

```python
from mostlyai import mock

tables = {
    "customers": {
        "description": "Customers of a hardware store",
        "columns": {
            "customer_id": {"prompt": "the unique id of the customer", "dtype": "integer"},
            "name": {"prompt": "first name and last name of the customer", "dtype": "string"},
        },
        "primary_key": "customer_id",
    },
    "orders": {
        "description": "Orders of a Customer",
        "columns": {
            "customer_id": {"prompt": "the customer id for that order", "dtype": "integer"},
            "order_id": {"prompt": "the unique id of the order", "dtype": "string"},
            "text": {"prompt": "order text description", "dtype": "string"},
            "amount": {"prompt": "order amount in USD", "dtype": "float"},
        },
        "primary_key": "order_id",
        "foreign_keys": [
            {
                "column": "customer_id",
                "referenced_table": "customers",
                "description": "each customer has anywhere between 1 and 3 orders",
            }
        ],
    },
    "items": {
        "description": "Items in an Order",
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
                "description": "each order has between 2 and 5 items",
            }
        ],
    },
}
data = mock.sample(
    tables=tables, 
    sample_size=2, 
    model="openai/gpt-4.1"
)
print(data["customers"])
print(data["orders"])
print(data["items"])
```
