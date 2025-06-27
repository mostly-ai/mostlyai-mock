# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

REDIRECT_TO: Literal["openai", "openrouter"] = "openrouter"

BASE_URLS = {"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1"}
API_KEYS = {"openai": os.environ["OPENAI_API_KEY"], "openrouter": os.environ["OPENROUTER_API_KEY"]}

BASE_URL = BASE_URLS[REDIRECT_TO]
API_KEY = API_KEYS[REDIRECT_TO]


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    assert body.get("stream", False), "Only streaming requests are supported"

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{BASE_URL}/chat/completions", headers=headers, json=body)

        async def stream():
            async for chunk in response.aiter_text():
                yield chunk

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
