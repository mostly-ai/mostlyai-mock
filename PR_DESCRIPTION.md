# Add support for enriching existing data

This PR adds a new feature to `mostlyai-mock` that allows users to enrich existing data with generated columns.

## Features

- Added a new `existing_data` parameter to the `sample` function that accepts a dictionary of existing DataFrames
- Implemented logic to identify missing columns and generate only those columns
- Enhanced the prompting to ensure generated values are consistent with existing data
- Updated system prompt to better handle enrichment scenarios
- Added documentation with examples for both single-table and multi-table enrichment

## Use cases

This feature is useful for:
- Adding synthetic columns to real data while preserving relationships
- Generating plausible values for missing columns
- Extending existing datasets with new attributes
- Creating consistent test data that incorporates real-world patterns

## Example

```python
from mostlyai import mock
import pandas as pd

# Define the schema
tables = {
    "guests": {
        "prompt": "Guests of an Alpine ski hotel in Austria",
        "columns": {
            "guest_id": {"prompt": "the unique id of the guest", "dtype": "integer"},
            "name": {"prompt": "first name and last name of the guest", "dtype": "string"},
            "nationality": {"prompt": "2-letter code for the nationality", "dtype": "string"},
            "gender": {"dtype": "category", "values": ["male", "female"]},
            "age": {"prompt": "age in years; min: 18, max: 80; avg: 25", "dtype": "integer"},
            "room_number": {"prompt": "room number", "dtype": "integer"},
            "is_vip": {"prompt": "is the guest a VIP", "dtype": "boolean"},
        },
        "primary_key": "guest_id",
    }
}

# Create existing data with some columns already filled
existing_df = pd.DataFrame({
    "guest_id": [1, 2, 3],
    "name": ["Anna Schmidt", "Marco Rossi", "Sophie Dupont"],
    "nationality": ["DE", "IT", "FR"],
})

# Enrich the existing data with additional columns
enriched_df = mock.sample(
    tables=tables, 
    existing_data={"guests": existing_df},
    model="openai/gpt-4.1-nano"
)
```

## Implementation Notes

1. **Existing data handling**:
   - The existing data is passed as a dictionary of DataFrames, indexed by table name
   - For each table, we check which columns already exist and which need to be generated
   - Only missing columns are generated using the LLM
   - The process ensures both structural and semantic consistency with existing values

2. **Table generation ordering**:
   - The implementation respects the dependency order established by foreign keys
   - Existing data is incorporated before generated data to maintain referential integrity
   - Foreign key relationships work both with enriched tables and newly generated tables

3. **Row-by-row processing for enrichment**:
   - When enriching existing data, each row is processed individually 
   - This allows the LLM to generate context-appropriate values based on existing column values
   - The approach ensures that relationships between columns (e.g., age and date of birth) are maintained

4. **Prompt adaptation**:
   - Special prompts are created for enrichment scenarios to guide the LLM
   - The prompts include existing column values as context for each row
   - The system prompt has been enhanced to better handle enrichment tasks 