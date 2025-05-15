import os
import tempfile
import zipfile

import requests
from fastmcp import Context, FastMCP

from mostlyai import mock

SAMPLE_MOCK_TOOL_DESCRIPTION = f"""
It is proxy to the `mostlyai.mock.sample` function.

This function returns an URL to the generated CSV bundle (as ZIP file).
Print this URL in Markdown format, so user can easily download the data.

What comes after the `=============================` is the documentation of the `mostlyai.mock.sample` function.

=============================
{mock.sample.__doc__}
"""

mcp = FastMCP(name="MostlyAI Mock MCP Server")


def _upload_to_0x0st(data: dict) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "mock_data.zip")
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for table_name, df in data.items():
                csv_path = os.path.join(temp_dir, f"{table_name}.csv")
                df.to_csv(csv_path, index=False)
                zip_file.write(csv_path, arcname=f"{table_name}.csv")

        with open(zip_path, "rb") as f:
            response = requests.post(
                "https://0x0.st",
                files={"file": f},
                data={"expires": "24", "secret": ""},
                headers={"User-Agent": "MockData/1.0"},
            )

        if response.status_code == 200:
            url = response.text.strip()
            return url
        else:
            raise Exception(f"Failed to upload ZIP: HTTP {response.status_code}")


@mcp.tool(description=SAMPLE_MOCK_TOOL_DESCRIPTION)
def sample_mock_data(
    *,
    tables: dict[str, dict],
    sample_size: int,
    model: str = "openai/gpt-4.1-nano",
    api_key: str | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    ctx: Context,
) -> str:
    # Notes:
    # 1. Returning DataFrames directly results in converting them into truncated string.
    # 2. The logs / progress bars are not propagated to the MCP Client. There is a dedicated API to do that (e.g. `ctx.info(...)`)
    # 3. MCP Server inherits only selected environment variables (PATH, USER...); one way to pass LLM keys is through client configuration (`mcpServers->env`)
    # 4. Some MCP Clients, e.g. Cursor, do not like Unions or Optionals in type hints
    ctx.info(f"Generating mock data for `{len(tables)}` tables")
    data = mock.sample(
        tables=tables,
        sample_size=sample_size,
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        return_type="dict",
    )
    ctx.info(f"Generated mock data for `{len(tables)}` tables")
    url = _upload_to_0x0st(data)
    return url


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
