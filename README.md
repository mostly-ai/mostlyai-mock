# Synthetic Mock Data 🔮

[![Documentation](https://img.shields.io/badge/docs-latest-green)](https://mostly-ai.github.io/mostlyai-mock/) [![stats](https://pepy.tech/badge/mostlyai-mock)](https://pypi.org/project/mostlyai-mock/) ![license](https://img.shields.io/github/license/mostly-ai/mostlyai-mock) ![GitHub Release](https://img.shields.io/github/v/release/mostly-ai/mostlyai-mock)

Use LLMs to generate any Tabular Data towards your needs. Create from scratch, expand existing datasets, or enrich tables with new columns. Your prompts, your rules, your data.

## Key Features

* A light-weight python client for prompting LLMs for mixed-type tabular data.
* Select from a wide range of LLM endpoints and LLM models.
* Supports single-table as well as multi-table scenarios.
* Supports variety of data types: `string`, `categorical`, `integer`, `float`, `boolean`, `date`, and `datetime`.
* Specify context, distributions and rules via dataset-, table- or column-level prompts.
* Create from scratch or enrich existing datasets with new columns and/or rows.
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
#   nationality                 name  gender  age date_of_birth        checkin_time is_vip  price_per_night  room_number
# 0          FR          Jean Dupont    male   29    1994-03-15 2025-01-10 14:30:00  False            150.0          101
# 1          DE         Anna Schmidt  female   34    1989-07-22 2025-01-11 16:45:00   True            200.0          201
# 2          IT          Marco Rossi    male   45    1979-11-05 2025-01-09 10:15:00  False            180.0          102
# 3          AT         Laura Gruber  female   28    1996-02-19 2025-01-12 09:00:00  False            165.0          202
# 4          CH         David Müller    male   37    1987-08-30 2025-01-08 17:20:00   True            210.0          203
# 5          NL  Sophie van den Berg  female   22    2002-04-12 2025-01-10 12:00:00  False            140.0          103
# 6          GB         James Carter    male   31    1992-09-10 2025-01-11 11:30:00  False            155.0          204
# 7          BE        Lotte Peeters  female   26    1998-05-25 2025-01-09 15:45:00  False            160.0          201
# 8          DK        Anders Jensen    male   33    1990-12-03 2025-01-12 08:15:00   True            220.0          202
# 9          ES         Carlos Lopez    male   38    1985-06-14 2025-01-10 18:00:00  False            170.0          203
```

4. Create your first multi-table mock dataset

```python
from mostlyai import mock

tables = {
    "customers": {
        "prompt": "Customers of a hardware store",
        "columns": {
            "customer_id": {"prompt": "the unique id of the customer", "dtype": "string"},
            "name": {"prompt": "first name and last name of the customer", "dtype": "string"},
        },
        "primary_key": "customer_id",
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
        "primary_key": "item_id",
    },
}
data = mock.sample(
    tables=tables,
    sample_size=2,
    model="openai/gpt-4.1",
    n_workers=1,
)
print(data["customers"])
#   customer_id             name
# 0   B0-100235  Danielle Rogers
# 1   B0-100236       Edward Kim
print(data["warehouses"])
#   warehouse_id                          name
# 0       B0-001  Downtown Distribution Center
# 1       B0-002     Westside Storage Facility
print(data["orders"])
#   customer_id warehouse_id    order_id                                               text   amount
# 0   B0-100235       B0-002  B0-3010021  Office furniture replenishment - desks, chairs...  1268.35
# 1   B0-100235       B0-001  B0-3010022  Bulk stationery order: printer paper, notebook...    449.9
# 2   B0-100235       B0-001  B0-3010023  Electronics restock: monitors and wireless key...    877.6
# 3   B0-100236       B0-001  B1-3010021  Monthly cleaning supplies: disinfectant, trash...   314.75
# 4   B0-100236       B0-002  B1-3010022  Breakroom essentials restock: coffee, tea, and...   182.45
print(data["items"])
#      item_id    order_id                                   name   price
# 0  B0-200501  B0-3010021                  Ergonomic Office Desk  545.99
# 1  B0-200502  B0-3010021              Mesh Back Executive Chair   399.5
# 2  B1-200503  B0-3010022   Multipack Printer Paper (500 sheets)  129.95
# 3  B1-200504  B0-3010022             Spiral Notebooks - 12 Pack   59.99
# 4  B2-200505  B0-3010023               27" LED Computer Monitor  489.95
# 5  B2-200506  B0-3010023            Wireless Ergonomic Keyboard  387.65
# 6  B3-200507  B1-3010021  Industrial Disinfectant Solution (5L)  148.95
# 7  B3-200508  B1-3010021  Commercial Trash Liners - Case of 100    84.5
# 8  B4-200509  B1-3010022        Premium Ground Coffee (2lb Bag)   74.99
# 9  B4-200510  B1-3010022         Bottled Spring Water (24 Pack)   34.95
```

6. Create your first self-referencing mock table

```python
from mostlyai import mock

tables = {
    "employees": {
        "prompt": "Employees of a company",
        "columns": {
            "employee_id": {"prompt": "the unique id of the employee; sequential", "dtype": "string"},
            "name": {"prompt": "first name and last name of the president", "dtype": "string"},
            "boss_id": {"prompt": "the id of the boss of the employee", "dtype": "string"},
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
#   employee_id              name boss_id                   role
# 0        B0-1      Patricia Lee    <NA>              President
# 1        B0-2  Edward Rodriguez    B0-1       VP of Operations
# 2        B0-3      Maria Cortez    B0-1          VP of Finance
# 3        B0-4     Thomas Nguyen    B0-1       VP of Technology
# 4        B0-5        Rachel Kim    B0-2     Operations Manager
# 5        B0-6     Jeffrey Patel    B0-2      Supply Chain Lead
# 6        B0-7      Olivia Smith    B0-2  Facilities Supervisor
# 7        B0-8      Brian Carter    B0-3     Accounting Manager
# 8        B0-9   Lauren Anderson    B0-3      Financial Analyst
# 9       B0-10   Santiago Romero    B0-3     Payroll Specialist
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
#   guest_id           name nationality  gender  age  room_number is_vip
# 0        1   Anna Schmidt          DE  female   30          102  False
# 1        2    Marco Rossi          IT    male   27          215   True
# 2        3  Sophie Dupont          FR  female   22          108  False
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
