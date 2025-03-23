# app/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import time
from app.rag import process_query
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from app.config import UV_CONFIG

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG Inference API")
app.state.limiter = limiter

# Instrumentare Prometheus
Instrumentator().instrument(app).expose(app)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: list
    timing: dict

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

@app.post("/answer", response_model=QueryResponse)
#@limiter.limit("10/minute")
async def answer_question(query_req: QueryRequest):
    start_time = time.time()
    try:
        result = process_query(query_req.query)
        result["timing"]["total"] = time.time() - start_time
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(UV_CONFIG.APP, host=UV_CONFIG.HOST, port=UV_CONFIG.PORT, workers=UV_CONFIG.WORKERS)
