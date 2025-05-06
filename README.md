# Synthetic Mock Data 🔮

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-mock/) [![stats](https://pepy.tech/badge/mostlyai-mock)](https://pypi.org/project/mostlyai-mock/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-mock) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-mock) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mostlyai-mock)

Create data out of nothing. Prompt LLMs for Tabular Data.

## Installation

The latest release of `mostlyai-mock` can be installed via pip:

```bash
pip install -U mostlyai-mock
```

Note: An API key to a LLM endpoint, with structured response, is required. It is recommended to set such a key as an environment variable (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.). Alternatively, the key needs to be passed to every call to the library iteself via the parameter `api_key`.

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
    "guests": {
        "description": "Guests of an Alpine ski hotel in Austria",
        "columns": {
            "id": {"prompt": "the unique id of the guest", "dtype": "integer"},
            "name": {"prompt": "first name and last name of the guest", "dtype": "string"},
        },
        "primary_key": "id",
    },
    "purchases": {
        "description": "Purchases of a Guest during their stay",
        "columns": {
            "guest_id": {"prompt": "the guest id for that purchase", "dtype": "integer"},
            "purchase_id": {"prompt": "the unique id of the purchase", "dtype": "string"},
            "text": {"prompt": "purchase text description", "dtype": "string"},
            "amount": {"prompt": "purchase amount in EUR", "dtype": "float"},
        },
        "foreign_keys": [
            {
                "column": "guest_id",
                "referenced_table": "guests",
                "description": "each guest has anywhere between 1 and 10 purchases",
            }
        ],
    },
}
data = mock.sample(tables=tables, sample_size=5, model="openai/gpt-4.1-nano")
df_guests = data["guests"]
df_purchases = data["purchases"]
print(df_guests)
print(df_purchases)
```
