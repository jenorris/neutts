"""Preserve a second voice endpoint without another GPU model copy."""
from contextlib import asynccontextmanager
import os

import httpx
import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get('NEUTTS_UPSTREAM', 'http://127.0.0.1:8290')
VOICE = os.environ.get('NEUTTS_PROXY_VOICE', 'secondary')


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(300, connect=5),
                                 trust_env=False) as client:
        app.state.client = client
        yield


app = FastAPI(lifespan=lifespan)


@app.post('/v1/audio/speech')
@app.post('/audio/speech')
async def speech(request: Request):
    client = app.state.client
    try:
        response = await client.send(client.build_request(
            'POST', f'/internal/voices/{VOICE}/v1/audio/speech',
            content=await request.body(), headers={'Content-Type': 'application/json'}), stream=True)
    except httpx.HTTPError:
        return JSONResponse({'detail': 'Shared voice worker unavailable'}, status_code=503)

    async def chunks():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()
    return StreamingResponse(chunks(), status_code=response.status_code,
                             media_type=response.headers.get('Content-Type', 'audio/pcm'))


@app.get('/v1/audio/voices')
@app.get('/audio/voices')
async def voices():
    return {'object': 'list', 'data': [{'id': 'clone', 'object': 'voice'}]}


@app.get('/health')
async def health():
    try:
        response = await app.state.client.get(f'/internal/voices/{VOICE}/health')
        return JSONResponse(response.json(), status_code=response.status_code)
    except httpx.HTTPError:
        return JSONResponse({'ok': False}, status_code=503)
