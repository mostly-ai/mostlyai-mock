import json

import pandas as pd
from fastmcp import Context, FastMCP

from mostlyai import mock

mcp = FastMCP(name="MostlyAI Mock MCP Server")


@mcp.tool(description=mock.sample.__doc__)
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
    return {k: v.to_dict(orient="records") for k, v in data.items()}


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
