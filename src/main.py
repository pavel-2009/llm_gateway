from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from llm_gateway.adapter.contracts import GenerateResponse
from llm_gateway.core.config import ExtractedReview
from llm_gateway.llm_gateway import LLMGateway


class PromptRequest(BaseModel):
    prompt: str


app = FastAPI(title="LLM Gateway")
gateway = LLMGateway()


@app.post("/generate")
async def generate(request: PromptRequest) -> GenerateResponse:
    return await gateway.generate(request.prompt)


@app.post("/extract")
async def extract(request: PromptRequest) -> ExtractedReview:
    return await gateway.extract(request.prompt)


@app.post("/stream")
async def stream(request: PromptRequest) -> StreamingResponse:
    async def chunks() -> AsyncIterator[str]:
        async for chunk in gateway.stream(request.prompt):
            yield chunk

    return StreamingResponse(chunks(), media_type="text/plain")
