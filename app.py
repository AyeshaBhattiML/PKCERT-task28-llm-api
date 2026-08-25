
# ============================================================
# TASK 28 - PART C
# Production LLM Inference API
# FastAPI + Hugging Face Remote Inference + Render
# ============================================================

import asyncio
import json
import logging
import os
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field, field_validator

from starlette.exceptions import HTTPException as StarletteHTTPException

from huggingface_hub import InferenceClient


# ============================================================
# 1. STRUCTURED JSON LOGGING
# ============================================================

class JSONFormatter(logging.Formatter):
    """
    Format application logs as JSON objects.
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        }

        return json.dumps(log_record)


logger = logging.getLogger("task28_api")

logger.setLevel(logging.INFO)

handler = logging.StreamHandler()

handler.setFormatter(
    JSONFormatter()
)

logger.handlers.clear()

logger.addHandler(handler)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

# Models remain on Hugging Face.
# Render does NOT download or store model weights.

SENTIMENT_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

SUMMARIZATION_MODEL = (
    "sshleifer/distilbart-cnn-6-6"
)

GENERATION_MODEL = "distilgpt2"


# ============================================================
# 3. ENVIRONMENT CONFIGURATION
# ============================================================

# Hugging Face token is stored securely in Render
# Environment Variables.

HF_TOKEN = os.getenv("HF_TOKEN")

# Render provides PORT automatically.
# Local development falls back to port 8000.

PORT = int(
    os.getenv("PORT", "8000")
)


# ============================================================
# 4. HUGGING FACE CLIENT
# ============================================================

hf_client = None


def initialize_huggingface_client():
    """
    Initialize the Hugging Face remote inference client.

    The token is read from the environment and is never
    hard-coded into the source code.
    """

    global hf_client

    if not HF_TOKEN:
        logger.warning(
            "HF_TOKEN is not configured."
        )
        return

    hf_client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN
    )

    logger.info(
        "Hugging Face remote inference client initialized."
    )


# ============================================================
# 5. FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting Task 28 LLM Inference API."
    )

    logger.info(
        "Remote Hugging Face inference enabled."
    )

    initialize_huggingface_client()

    yield

    logger.info(
        "Shutting down Task 28 LLM Inference API."
    )


# ============================================================
# 6. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Task 28 LLM Inference API",
    description=(
        "Production-style API for sentiment analysis, "
        "abstractive summarization, and causal text generation "
        "using Hugging Face remote inference."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# 7. CORS MIDDLEWARE
# ============================================================

# Allows Streamlit, HTML, JavaScript, or another frontend
# to communicate with the API.

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# 8. STANDARD ERROR RESPONSE
# ============================================================

class ErrorResponse(BaseModel):
    """
    Standard structure for API error responses.
    """

    error: str

    message: str

    status_code: int


# ============================================================
# 9. HTTP EXCEPTION HANDLER
# ============================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
):

    return JSONResponse(
        status_code=exc.status_code,

        content={
            "error": "HTTP_ERROR",
            "message": str(exc.detail),
            "status_code": exc.status_code
        }
    )


# ============================================================
# 10. VALIDATION ERROR HANDLER
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):

    return JSONResponse(
        status_code=422,

        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "status_code": 422,
            "details": exc.errors()
        }
    )


# ============================================================
# 11. VALUE ERROR HANDLER
# ============================================================

@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):

    return JSONResponse(
        status_code=400,

        content={
            "error": "BAD_REQUEST",
            "message": str(exc),
            "status_code": 400
        }
    )


# ============================================================
# 12. GENERAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
):

    logger.exception(
        "Unhandled application exception."
    )

    return JSONResponse(
        status_code=500,

        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": (
                "An unexpected internal server error occurred."
            ),
            "status_code": 500
        }
    )


# ============================================================
# 13. SENTIMENT REQUEST SCHEMA
# ============================================================

class SentimentRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for sentiment."
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Text cannot be empty."
            )

        return value


# ============================================================
# 14. SENTIMENT RESPONSE SCHEMA
# ============================================================

class SentimentResponse(BaseModel):

    label: str

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )


# ============================================================
# 15. SUMMARIZATION REQUEST SCHEMA
# ============================================================

class SummarizationRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Text document to summarize."
    )

    max_tokens: int = Field(
        default=80,
        ge=10,
        le=256,
        description="Maximum summary tokens."
    )

    min_tokens: int = Field(
        default=20,
        ge=0,
        le=128,
        description="Minimum summary tokens."
    )

    num_beams: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of beam-search candidates."
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Text cannot be empty."
            )

        return value


# ============================================================
# 16. SUMMARIZATION RESPONSE SCHEMA
# ============================================================

class SummarizationResponse(BaseModel):

    summary: str


# ============================================================
# 17. GENERATION REQUEST SCHEMA
# ============================================================

class GenerationRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Initial text prompt."
    )

    max_tokens: int = Field(
        default=50,
        ge=1,
        le=128,
        description="Maximum generated tokens."
    )

    do_sample: bool = Field(
        default=False,
        description="Enable probabilistic sampling."
    )

    num_beams: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of beam-search candidates."
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Top-k sampling parameter."
    )

    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability."
    )

    temperature: float | None = Field(
        default=None,
        gt=0.0,
        le=2.0,
        description="Sampling temperature."
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Prompt cannot be empty."
            )

        return value


# ============================================================
# 18. GENERATION RESPONSE SCHEMA
# ============================================================

class GenerationResponse(BaseModel):

    generated_text: str


# ============================================================
# 19. HEALTH CHECK
# ============================================================

@app.get(
    "/healthz",
    response_model=dict
)
async def health_check():

    return {
        "status": "healthy",
        "service": "Task 28 LLM Inference API",
        "inference": "Hugging Face remote",
        "model_loading": "remote",
        "device": "Hugging Face"
    }


# ============================================================
# 20. SENTIMENT ENDPOINT
# ============================================================

@app.post(
    "/api/v1/sentiment",
    response_model=SentimentResponse
)
async def sentiment_analysis(
    request: SentimentRequest
):

    if hf_client is None:
        raise ValueError(
            "HF_TOKEN is not configured."
        )

    start_time = time.perf_counter()

    # Run the synchronous Hugging Face client in a worker
    # thread so the FastAPI event loop remains responsive.

    result = await asyncio.to_thread(
        hf_client.text_classification,
        request.text,
        model=SENTIMENT_MODEL,
        top_k=1
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Sentiment request completed: %.4f seconds",
        latency
    )

    return SentimentResponse(
        label=result[0].label,
        score=result[0].score
    )


# ============================================================
# 21. SUMMARIZATION ENDPOINT
# ============================================================

@app.post(
    "/api/v1/summarize",
    response_model=SummarizationResponse
)
async def summarize_text(
    request: SummarizationRequest
):

    if hf_client is None:
        raise ValueError(
            "HF_TOKEN is not configured."
        )

    start_time = time.perf_counter()

    # Remote summarization is performed by Hugging Face.
    # No summarization model is stored inside the container.

    result = await asyncio.to_thread(
        hf_client.summarization,
        request.text,
        model=SUMMARIZATION_MODEL,
        truncation="longest_first",
        generate_parameters={
            "max_new_tokens": request.max_tokens,
            "min_new_tokens": request.min_tokens,
            "num_beams": request.num_beams
        }
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Summarization request completed: %.4f seconds",
        latency
    )

    return SummarizationResponse(
        summary=result.generated_text
    )


# ============================================================
# 22. TEXT GENERATION ENDPOINT
# ============================================================

@app.post(
    "/api/v1/generate",
    response_model=GenerationResponse
)
async def generate_text(
    request: GenerationRequest
):

    if hf_client is None:
        raise ValueError(
            "HF_TOKEN is not configured."
        )

    start_time = time.perf_counter()

    # Basic generation configuration.

    generation_parameters = {
        "max_new_tokens": request.max_tokens,
        "do_sample": request.do_sample,
        "return_full_text": True
    }

    # Beam search configuration.

    if request.num_beams > 1:

        generation_parameters["num_beams"] = (
            request.num_beams
        )

    # Sampling parameters are only applied when sampling
    # has explicitly been enabled.

    if request.do_sample:

        if request.top_k is not None:

            generation_parameters["top_k"] = (
                request.top_k
            )

        if request.top_p is not None:

            generation_parameters["top_p"] = (
                request.top_p
            )

        if request.temperature is not None:

            generation_parameters["temperature"] = (
                request.temperature
            )

    # Hugging Face text_generation returns a string by default.

    generated_text = await asyncio.to_thread(
        hf_client.text_generation,
        request.prompt,
        model=GENERATION_MODEL,
        **generation_parameters
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Generation request completed: %.4f seconds",
        latency
    )

    return GenerationResponse(
        generated_text=generated_text
    )


# ============================================================
# 23. STARTUP INFORMATION
# ============================================================

logger.info(
    "Task 28 Part C API configuration complete."
)

logger.info(
    "Production configuration: Hugging Face remote inference."
)

logger.info(
    "Render-compatible lightweight deployment enabled."
)
