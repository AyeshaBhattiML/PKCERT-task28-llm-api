# TASK 28 - PART C (copied api)
# Production LLM Inference API
# FastAPI + Lazy Model Loading + Docker/Cloud Deployment

import asyncio
import gc
import json
import logging
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


# 1. STRUCTURED JSON LOGGING

class JSONFormatter(logging.Formatter):
    """
    Format application logs as JSON objects.
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(
                record,
                self.datefmt
            ),
            "level": record.levelname,
            "message": record.getMessage()
        }

        return json.dumps(log_record)


logger = logging.getLogger("task28_api")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger.handlers.clear()
logger.addHandler(handler)


# 2. MODEL CONFIGURATION

SENTIMENT_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

SUMMARIZATION_MODEL = (
    "sshleifer/distilbart-cnn-6-6"
)

GENERATION_MODEL = "distilgpt2"


# 3. DEVICE CONFIGURATION

device = 0 if torch.cuda.is_available() else -1

logger.info(
    "Inference device: %s",
    "GPU" if device == 0 else "CPU"
)


# 4. LAZY MODEL STATE

sentiment_pipeline = None

summarization_tokenizer = None
summarization_model = None

generation_tokenizer = None
generation_model = None

summarization_device = "cpu"
generation_device = "cpu"

# Indicates whether the API itself is ready.
# Models are intentionally NOT loaded at startup.
api_ready = True

# Only one model is kept in memory at a time.
model_lock = Lock()


# 5. THREAD POOL

# Model inference is CPU/GPU intensive and should not block
# the FastAPI event loop.

inference_executor = ThreadPoolExecutor(
    max_workers=2
)


# 6. MODEL MEMORY MANAGEMENT

def unload_all_models():
    """
    Release all currently loaded models.

    Only one model is kept in memory at a time in order to
    reduce RAM consumption on free cloud infrastructure.
    """

    global sentiment_pipeline
    global summarization_tokenizer
    global summarization_model
    global generation_tokenizer
    global generation_model

    logger.info(
        "Releasing currently loaded models."
    )

    sentiment_pipeline = None
    summarization_tokenizer = None
    summarization_model = None
    generation_tokenizer = None
    generation_model = None

    # Force Python garbage collection.
    gc.collect()

    # Release unused CUDA memory when a GPU is available.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Model memory cleanup completed."
    )


# 7. SENTIMENT MODEL LOADER

def load_sentiment_model():
    """
    Load sentiment model only when required.
    """

    global sentiment_pipeline

    with model_lock:

        if sentiment_pipeline is not None:
            return

        logger.info(
            "Loading sentiment analysis model."
        )

        unload_all_models()

        sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device
        )

        logger.info(
            "Sentiment analysis model loaded successfully."
        )


# 8. SUMMARIZATION MODEL LOADER

def load_summarization_model():
    """
    Load summarization model only when required.
    """

    global summarization_tokenizer
    global summarization_model
    global summarization_device

    with model_lock:

        if summarization_model is not None:
            return

        logger.info(
            "Loading summarization model."
        )

        unload_all_models()

        summarization_tokenizer = (
            AutoTokenizer.from_pretrained(
                SUMMARIZATION_MODEL
            )
        )

        summarization_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                SUMMARIZATION_MODEL,
                low_cpu_mem_usage=True
            )
        )

        summarization_model.eval()

        if torch.cuda.is_available():

            summarization_model = (
                summarization_model.to("cuda")
            )

            summarization_device = "cuda"

        else:

            summarization_device = "cpu"

        logger.info(
            "Summarization model loaded successfully."
        )

        logger.info(
            "Summarization device: %s",
            summarization_device
        )


# 9. GENERATION MODEL LOADER

def load_generation_model():
    """
    Load causal generation model only when required.
    """

    global generation_tokenizer
    global generation_model
    global generation_device

    with model_lock:

        if generation_model is not None:
            return

        logger.info(
            "Loading causal generation model."
        )

        unload_all_models()

        generation_tokenizer = (
            AutoTokenizer.from_pretrained(
                GENERATION_MODEL
            )
        )

        generation_model = (
            AutoModelForCausalLM.from_pretrained(
                GENERATION_MODEL,
                low_cpu_mem_usage=True
            )
        )

        generation_model.eval()

        # GPT-2 does not define a padding token.
        if generation_tokenizer.pad_token is None:

            generation_tokenizer.pad_token = (
                generation_tokenizer.eos_token
            )

        if torch.cuda.is_available():

            generation_model = (
                generation_model.to("cuda")
            )

            generation_device = "cuda"

        else:

            generation_device = "cpu"

        logger.info(
            "Causal text generation model loaded successfully."
        )

        logger.info(
            "Generation device: %s",
            generation_device
        )


# 10. FASTAPI LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Models are lazy-loaded instead of loading all models during
    application startup.

    This significantly reduces cloud startup memory usage.
    """

    logger.info(
        "Starting Task 28 LLM Inference API."
    )

    logger.info(
        "Lazy model loading enabled."
    )

    logger.info(
        "Models will be loaded only when their endpoints are used."
    )

    yield

    logger.info(
        "Shutting down Task 28 LLM Inference API."
    )

    unload_all_models()

    inference_executor.shutdown(
        wait=True
    )

    logger.info(
        "Inference executor shut down."
    )


# 11. FASTAPI APPLICATION

app = FastAPI(
    title="Task 28 LLM Inference API",
    description=(
        "Production-style API for sentiment analysis, "
        "abstractive summarization, and causal text generation."
    ),
    version="1.0.0",
    lifespan=lifespan
)

print("FastAPI application initialized.")


# 12. CORS

# Allows the future Streamlit / HTML frontend to communicate
# with the cloud API.

app.add_middleware(
    CORSMiddleware,

    # Allow frontend applications to communicate with the API.
    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)


# 13. STANDARD ERROR RESPONSE

class ErrorResponse(BaseModel):
    """
    Standard structure for API error responses.
    """

    error: str

    message: str

    status_code: int


# 14. HTTP EXCEPTION HANDLER

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
):
    """
    Handle HTTP exceptions using a standardized JSON format.
    """

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": str(exc.detail),
            "status_code": exc.status_code
        }
    )


# 15. VALIDATION ERROR HANDLER

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    Handle Pydantic/FastAPI request validation errors.

    Returns HTTP 422.
    """

    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "status_code": 422,
            "details": exc.errors()
        }
    )


# 16. VALUE ERROR HANDLER

@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):
    """
    Handle application-level ValueError exceptions.

    Returns HTTP 400.
    """

    return JSONResponse(
        status_code=400,
        content={
            "error": "BAD_REQUEST",
            "message": str(exc),
            "status_code": 400
        }
    )


# 17. GENERAL EXCEPTION HANDLER

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
):
    """
    Handle unexpected internal application errors.

    Returns HTTP 500.
    """

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


# 18. SENTIMENT REQUEST

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


# 19. SENTIMENT RESPONSE

class SentimentResponse(BaseModel):

    label: str

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )


# 20. SUMMARIZATION REQUEST

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
        le=512,
        description="Maximum summary tokens."
    )

    min_tokens: int = Field(
        default=20,
        ge=0,
        le=256,
        description="Minimum summary tokens."
    )

    num_beams: int = Field(
        default=4,
        ge=1,
        le=10,
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


# 21. SUMMARIZATION RESPONSE

class SummarizationResponse(BaseModel):

    summary: str


# 22. GENERATION REQUEST

class GenerationRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Initial text prompt."
    )

    max_tokens: int = Field(
        default=50,
        ge=1,
        le=512,
        description="Maximum number of generated tokens."
    )

    do_sample: bool = Field(
        default=False,
        description="Enable probabilistic sampling."
    )

    num_beams: int = Field(
        default=1,
        ge=1,
        le=10,
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


# 23. GENERATION RESPONSE

class GenerationResponse(BaseModel):

    generated_text: str


# 24. HEALTH CHECK

@app.get(
    "/healthz",
    response_model=dict
)
async def health_check():

    return {
        "status": "healthy" if api_ready else "starting",
        "service": "Task 28 LLM Inference API",
        "model_loading": "lazy",
        "device": (
            "GPU"
            if torch.cuda.is_available()
            else "CPU"
        )
    }


# 25. SENTIMENT INFERENCE

def run_sentiment_inference(text: str):

    # Load the model only when the sentiment endpoint is used.
    load_sentiment_model()

    result = sentiment_pipeline(text)

    return result


# 26. SENTIMENT ENDPOINT

@app.post(
    "/api/v1/sentiment",
    response_model=SentimentResponse
)
async def sentiment_analysis(
    request: SentimentRequest
):

    start_time = time.perf_counter()

    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(
        inference_executor,
        run_sentiment_inference,
        request.text
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Sentiment request completed: %.4f seconds",
        latency
    )

    return SentimentResponse(
        label=result[0]["label"],
        score=result[0]["score"]
    )


# 27. SUMMARIZATION INFERENCE

def run_summarization_inference(request: SummarizationRequest):
    result = hf_client.summarization(
        request.text,
        model=SUMMARIZATION_MODEL
    )

    if isinstance(result, dict):
        return result.get(
            "summary_text",
            result.get("summary", "")
        )

    return str(result)


# 28. SUMMARIZATION ENDPOINT

@app.post(
    "/api/v1/summarize",
    response_model=SummarizationResponse
)
async def summarize_text(
    request: SummarizationRequest
):

    start_time = time.perf_counter()

    loop = asyncio.get_running_loop()

    summary = await loop.run_in_executor(
        inference_executor,
        run_summarization_inference,
        request
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Summarization request completed: %.4f seconds",
        latency
    )

    return SummarizationResponse(
        summary=summary
    )


# 29. GENERATION INFERENCE

def run_generation_inference(request: GenerationRequest):
    result = hf_client.text_generation(
        request.prompt,
        model=GENERATION_MODEL,
        max_new_tokens=request.max_tokens,
        do_sample=request.do_sample,
        num_beams=request.num_beams
    )

    return str(result)


# 30. GENERATION ENDPOINT

@app.post(
    "/api/v1/generate",
    response_model=GenerationResponse
)
async def generate_text(
    request: GenerationRequest
):

    start_time = time.perf_counter()

    loop = asyncio.get_running_loop()

    generated_text = await loop.run_in_executor(
        inference_executor,
        run_generation_inference,
        request
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Generation request completed: %.4f seconds",
        latency
    )

    return GenerationResponse(
        generated_text=generated_text
    )

# 30. GENERATION ENDPOINT

@app.post(
    "/api/v1/generate",
    response_model=GenerationResponse
)
async def generate_text(
    request: GenerationRequest
):

    start_time = time.perf_counter()

    loop = asyncio.get_running_loop()

    generated_text = await loop.run_in_executor(
        inference_executor,
        run_generation_inference,
        request
    )

    latency = time.perf_counter() - start_time

    logger.info(
        "Generation request completed: %.4f seconds",
        latency
    )

    return GenerationResponse(
        generated_text=generated_text
    )


# 31. STARTUP MESSAGE

logger.info(
    "Task 28 Part C API configuration complete."
)

logger.info(
    "Production configuration: lazy model loading enabled."
)
