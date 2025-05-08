# Synthetic Mock Data 🔮

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-mock/) [![stats](https://pepy.tech/badge/mostlyai-mock)](https://pypi.org/project/mostlyai-mock/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-mock) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-mock)

<!-- ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-mock) -->

Create data out of nothing. Prompt LLMs for Tabular Data.

## Installation

The latest release of `mostlyai-mock` can be installed via pip:

```bash
pip install -U mostlyai-mock
```

Note: An API key to a LLM endpoint, with structured response, is required. It is recommended to set such a key as an environment variable (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.). Alternatively, the key needs to be passed to every call to the library itself via the parameter `api_key`.

## Quick Start

### Single Table

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
df = mock.sample(tables=tables, sample_size=10, model="openai/gpt-4.1-nano")
print(df)
```

### Multiple Tables

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
data = mock.sample(tables=tables, sample_size=2, model="openai/gpt-4.1")
df_customers = data["customers"]
df_orders = data["orders"]
df_items = data["items"]
print(df_customers)
print(df_orders)
print(df_items)
```
