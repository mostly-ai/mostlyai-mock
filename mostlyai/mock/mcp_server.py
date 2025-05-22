import os
import tempfile

import pandas as pd
from fastmcp import FastMCP

from mostlyai import mock

SAMPLE_MOCK_TOOL_DESCRIPTION = f"""
Generate mock data by prompting an LLM.

This tool is a proxy to the `mostlyai.mock.sample` function, but returns a dictionary of paths to the generated CSV files.

Present the result nicely to the user, in Markdown format. Example:

Mock data can be found under the following paths:
- `/tmp/tmpl41bwa6n/players.csv`
- `/tmp/tmpl41bwa6n/seasons.csv`

== mostlyai.mock.sample DocString ==
{mock.sample.__doc__}
"""

mcp = FastMCP(name="MostlyAI Mock MCP Server")


def _store_locally(data: dict[str, pd.DataFrame]) -> dict[str, str]:
    temp_dir = tempfile.mkdtemp()
    locations = {}
    for table_name, df in data.items():
        csv_path = os.path.join(temp_dir, f"{table_name}.csv")
        df.to_csv(csv_path, index=False)
        locations[table_name] = csv_path
    return locations


@mcp.tool(description=SAMPLE_MOCK_TOOL_DESCRIPTION)
def mock_data(
    *,
    tables: dict[str, dict],
    sample_size: int,
    model: str = "openai/gpt-4.1-nano",
    api_key: str | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> dict[str, str]:
    data = mock.sample(
        tables=tables,
        sample_size=sample_size,
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        return_type="dict",
    )
    locations = _store_locally(data)
    return locations


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
