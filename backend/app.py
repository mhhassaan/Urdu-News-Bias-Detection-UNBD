import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from services.predictor import predict
from services.validator import is_urdu
from services.url_extractor import extract_text_from_url
from services.llm_service import explain_bias, rewrite_unbiased

app = FastAPI(title="Urdu Bias Detection API")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    text: Optional[str] = Field(default=None, max_length=100_000)
    url: Optional[str] = Field(default=None, max_length=2_048)

@app.get("/")
async def root():
    return {"message": "Urdu Bias Detection API is running"}

@app.post("/predict")
async def classify(data: RequestData):
    input_text = ""
    
    if data.url:
        input_text = await run_in_threadpool(extract_text_from_url, data.url)
        if not input_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the provided URL")
    elif data.text:
        input_text = data.text
    else:
        raise HTTPException(status_code=400, detail="No input provided. Please provide 'text' or 'url'.")

    # Validate if it's Urdu
    if not is_urdu(input_text):
        raise HTTPException(status_code=400, detail="The input text does not appear to be in Urdu.")

    try:
        result = await run_in_threadpool(predict, input_text)
        result["text_preview"] = (
            input_text[:200] + "..." if len(input_text) > 200 else input_text
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        print(f"Prediction error: {e.__class__.__name__}")
        raise HTTPException(
            status_code=500,
            detail="The model could not complete this prediction.",
        ) from e

class LLMRequest(BaseModel):
    sentence: str

class LLMExplainRequest(BaseModel):
    data: dict

@app.post("/explain")
async def explain(req: LLMExplainRequest):
    try:
        explanation = await run_in_threadpool(explain_bias, req.data)
        return {"explanation": explanation}
    except Exception as e:
        print(f"[API] Error in /explain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rewrite")
async def rewrite(data: LLMRequest):
    try:
        rewritten = await run_in_threadpool(rewrite_unbiased, data.sentence)
        return {"rewritten": rewritten}
    except Exception as e:
        print(f"[API] Error in /rewrite: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
