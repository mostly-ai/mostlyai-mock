# LLM-generated Mock Data 🔮

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

3. Create your first basic mock table from scratch

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
df = mock.sample(
    tables=tables,   # provide table and column definitions
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

4. Create your first multi-table mock dataset

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
data = mock.sample(
    tables=tables,
    sample_size=2,
    model="openai/gpt-4.1"
)
print(data["customers"])
#    customer_id             name
# 0            1  Matthew Carlson
# 1            2       Priya Shah
print(data["warehouses"])
#    warehouse_id                        name
# 0             1    Central Distribution Hub
# 1             2  Northgate Storage Facility
print(data["orders"])
#    customer_id  warehouse_id   order_id                                               text  amount
# 0            1             2  ORD-10294  3-tier glass shelving units, expedited deliver...  649.25
# 1            1             1  ORD-10541  Office desk chairs, set of 6, with assembly se...   824.9
# 2            1             1  ORD-10802  Executive standing desk, walnut finish, standa...   519.0
# 3            2             1  ORD-11017  Maple conference table, cable management inclu...  1225.5
# 4            2             2  ORD-11385  Set of ergonomic task chairs, black mesh, stan...  767.75
print(data["items"])
#      item_id   order_id                                        name   price
# 0  ITM-80265  ORD-10294         3-Tier Tempered Glass Shelving Unit   409.0
# 1  ITM-80266  ORD-10294  Brushed Aluminum Shelf Brackets (Set of 4)  240.25
# 2  ITM-81324  ORD-10541              Ergonomic Mesh-Back Desk Chair   132.5
# 3  ITM-81325  ORD-10541  Professional Office Chair Assembly Service    45.0
# 4  ITM-82101  ORD-10802      Executive Standing Desk, Walnut Finish   469.0
# 5  ITM-82102  ORD-10802         Desk Installation and Setup Service    50.0
# 6  ITM-83391  ORD-11017             Maple Conference Table, 10-Seat  1125.5
# 7  ITM-83392  ORD-11017       Integrated Table Cable Management Kit   100.0
# 8  ITM-84311  ORD-11385            Ergonomic Task Chair, Black Mesh  359.25
# 9  ITM-84312  ORD-11385                   Standard Delivery Service    48.5
```

6. Create your first self-referencing mock table

```python
from mostlyai import mock

tables = {
    "employees": {
        "prompt": "Employees of a company",
        "columns": {
            "employee_id": {"prompt": "the unique id of the employee", "dtype": "integer"},
            "name": {"prompt": "first name and last name of the president", "dtype": "string"},
            "boss_id": {"prompt": "the id of the boss of the employee", "dtype": "integer"},
            "role": {"prompt": "the role of the employee", "dtype": "string"},
        },
        "primary_key": "employee_id",
        "foreign_keys": [
            {
                "column": "boss_id",
                "referenced_table": "employees",
                "prompt": "each boss has at most 3 employees",
            },
        ],
    }
}
df = mock.sample(tables=tables, sample_size=10, model="openai/gpt-4.1")
print(df)
#    employee_id             name  boss_id                      role
# 0            1  Sandra Phillips     <NA>                 President
# 1            2      Marcus Tran        1   Chief Financial Officer
# 2            3    Ava Whittaker        1  Chief Technology Officer
# 3            4    Sophie Martin        1  Chief Operations Officer
# 4            5      Chad Nelson        2           Finance Manager
# 5            6     Ethan Glover        2         Senior Accountant
# 6            7   Kimberly Ortiz        2         Junior Accountant
# 7            8     Lucas Romero        3                IT Manager
# 8            9      Priya Desai        3    Lead Software Engineer
# 9           10    Felix Bennett        3    Senior Systems Analyst
```

7. Enrich existing data with additional columns

```python
from mostlyai import mock
import pandas as pd

tables = {
    "guests": {
        "prompt": "Guests of an Alpine ski hotel in Austria",
        "columns": {
            "gender": {"dtype": "category", "values": ["male", "female"]},
            "age": {"prompt": "age in years; min: 18, max: 80; avg: 25", "dtype": "integer"},
            "room_number": {"prompt": "room number", "dtype": "integer"},
            "is_vip": {"prompt": "is the guest a VIP", "dtype": "boolean"},
        },
        "primary_key": "guest_id",
    }
}
existing_guests = pd.DataFrame({
    "guest_id": [1, 2, 3],
    "name": ["Anna Schmidt", "Marco Rossi", "Sophie Dupont"],
    "nationality": ["DE", "IT", "FR"],
})
df = mock.sample(
    tables=tables,
    existing_data={"guests": existing_guests},
    model="openai/gpt-4.1-nano"
)
print(df)
#    guest_id           name nationality  gender  age  room_number is_vip
# 0         1   Anna Schmidt          DE  female   29          101   True
# 1         2    Marco Rossi          IT    male   34          102  False
# 2         3  Sophie Dupont          FR  female   27          103  False
```

## MCP Server

This repo comes with MCP Server. It can be easily consumed by any MCP Client by providing the following configuration:

```json
{
  "mcpServers": {
      "mostlyai-mock-mcp": {
          "command": "uvx",
          "args": ["--from", "mostlyai-mock", "mcp-server"],
          "env": {
              "OPENAI_API_KEY": "PROVIDE YOUR KEY",
              "GEMINI_API_KEY": "PROVIDE YOUR KEY",
              "GROQ_API_KEY": "PROVIDE YOUR KEY",
              "ANTHROPIC_API_KEY": "PROVIDE YOUR KEY"
          }
      }
  }
}
```

For example:
- in Claude Desktop, go to "Settings" > "Developer" > "Edit Config" and paste the above into `claude_desktop_config.json`
- in Cursor, go to "Settings" > "Cursor Settings" > "MCP" > "Add new global MCP server" and paste the above into `mcp.json`

Troubleshooting:
1. If the MCP Client fails to detect the MCP Server, provide the absolute path in the `command` field, for example: `/Users/johnsmith/.local/bin/uvx`
2. To debug MCP Server issues, you can use MCP Inspector by running: `npx @modelcontextprotocol/inspector -- uvx --from mostlyai-mock mcp-server`
3. In order to develop locally, modify the configuration by replacing `"command": "uv"` (or use the full path to `uv` if needed) and `"args": ["--directory", "/Users/johnsmith/mostlyai-mock", "run", "mcp-server"]`
