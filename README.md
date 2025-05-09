# Synthetic Mock Data 🔮

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-mock/) [![stats](https://pepy.tech/badge/mostlyai-mock)](https://pypi.org/project/mostlyai-mock/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-mock) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-mock)

Create data out of nothing. Prompt LLMs for Tabular Data.

## Key Features

* A light-weight python client for prompting LLMs for mixed-type tabular data
* Select from a range of LLM endpoints, that provide structured output
* Supports single-table as well as multi-table scenarios.
* Supports variety of data types: `string`, `categorical`, `integer`, `float`, `boolean`, `date`, and `datetime`.
* Specify context, distributions and rules via dataset-, table- or column-level prompts.
* Tailor the diversity and realism of your generated data via temperature and top_p.

## Getting Started

1. Install the latest version of the `mostlyai-mock` python package.

```bash
pip install -U mostlyai-mock
```

2. Set the API key of your LLM endpoint (if not done yet)

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
print(df)
#   nationality            name  gender  age date_of_birth        checkin_time  is_vip  price_per_night  room_number
# 0          AT     Anna Müller  female   29    1994-09-15 2025-01-05 14:30:00    True            350.0          101
# 1          DE  Johann Schmidt    male   45    1978-11-20 2025-01-06 16:45:00   False            250.0          102
# 2          CH      Lara Meier  female   32    1991-04-12 2025-01-05 12:00:00    True            400.0          103
# 3          IT     Marco Rossi    male   38    1985-02-25 2025-01-07 09:15:00   False            280.0          201
# 4          FR   Claire Dupont  female   24    2000-07-08 2025-01-07 11:20:00   False            220.0          202
# 5          AT    Felix Gruber    male   52    1972-01-10 2025-01-06 17:50:00    True            375.0          203
# 6          DE   Sophie Becker  female   27    1996-03-30 2025-01-08 08:30:00   False            230.0          204
# 7          CH      Max Keller    male   31    1992-05-16 2025-01-09 14:10:00   False            290.0          101
# 8          IT  Giulia Bianchi  female   36    1988-08-19 2025-01-05 15:55:00    True            410.0          102
# 9          FR    Louis Martin    male   44    1980-12-05 2025-01-07 10:40:00   False            270.0          103
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
                "description": "each customer has anywhere between 2 and 3 orders",
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
                "description": "each order has between 1 and 2 items",
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
#    customer_id            name
# 0            1  Michael Torres
# 1            2      Elaine Kim
print(data["orders"])
#    customer_id        order_id                                               text  amount
# 0            1  ORD20240612001        Home office desk and ergonomic chair bundle  412.95
# 1            1  ORD20240517322               Wireless noise-cancelling headphones  226.49
# 2            1  ORD20240430307         Smart LED desk lamp with USB charging port   69.99
# 3            2  ORD20240614015            Eco-friendly bamboo kitchen utensil set   39.95
# 4            2  ORD20240528356  Air fryer with digital touch screen, 5-quart c...  129.99
# 5            2  ORD20240510078          Double-walled glass coffee mugs, set of 4    48.5
print(data["items"])
#         item_id        order_id                                       name   price
# 0   ITEM100001A  ORD20240612001                Ergonomic Mesh Office Chair  179.99
# 1   ITEM100001B  ORD20240612001                Adjustable Home Office Desk  232.96
# 2   ITEM100002A  ORD20240517322       Wireless Noise-Cancelling Headphones  226.49
# 3   ITEM100003A  ORD20240430307                        Smart LED Desk Lamp   59.99
# 4   ITEM100003B  ORD20240430307  USB Charging Cable (Desk Lamp Compatible)    10.0
# 5   ITEM100004A  ORD20240614015                       Bamboo Cooking Spoon   13.49
# 6   ITEM100004B  ORD20240614015                      Bamboo Slotted Turner   12.99
# 7   ITEM100005A  ORD20240528356         Digital Air Fryer (5-Quart, Black)  115.99
# 8   ITEM100005B  ORD20240528356     Silicone Liner for Air Fryer (5-Quart)   13.99
# 9   ITEM100006A  ORD20240510078      Double-Walled Glass Coffee Mug (12oz)   13.75
# 10  ITEM100006B  ORD20240510078       Double-Walled Glass Coffee Mug (8oz)   11.25
```
