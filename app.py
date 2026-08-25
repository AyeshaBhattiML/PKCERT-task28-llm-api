# TASK 28 - copy of finalized Part-B API
# Production LLM Inference API with FastAPI

# 1. IMPORTS

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
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
    AutoModelForCausalLM
)

# 2. STRUCTURED JSON LOGGING

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
handler.setFormatter(
    JSONFormatter()
)
logger.handlers.clear()
logger.addHandler(handler)

# 3. MODEL CONFIGURATION

SENTIMENT_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)
SUMMARIZATION_MODEL = (
    "sshleifer/distilbart-cnn-6-6"
)
GENERATION_MODEL = "distilgpt2"

# 4. GLOBAL MODEL REFERENCES

sentiment_pipeline = None
summarization_tokenizer = None
summarization_model = None
generation_tokenizer = None
generation_model = None
summarization_device = "cpu"
generation_device = "cpu"

# 5. DEVICE CONFIGURATION

device = 0 if torch.cuda.is_available() else -1
logger.info(
    "Inference device: %s",
    "GPU" if device == 0 else "CPU"
)

# 6. THREAD POOL
# Heavy model inference can block the FastAPI event loop.
# A small thread pool allows inference to run separately.

inference_executor = ThreadPoolExecutor(
    max_workers=2
)

# 7. MODEL LOADING

def load_models():
    """
    Load all pretrained models once during application startup.
    """
    global sentiment_pipeline
    global summarization_tokenizer
    global summarization_model
    global generation_tokenizer
    global generation_model
    global summarization_device
    global generation_device

    # Sentiment Model

    logger.info(
        "Loading sentiment analysis model."
    )
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=device
    )
    logger.info(
        "Sentiment analysis model loaded successfully."
    )
    # Summarization Model

    logger.info(
        "Loading summarization model."
    )
    summarization_tokenizer = (
        AutoTokenizer.from_pretrained(
            SUMMARIZATION_MODEL
        )
    )
    summarization_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARIZATION_MODEL
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

    # Causal Text Generation Model

    logger.info(
        "Loading causal generation model."
    )
    generation_tokenizer = (
        AutoTokenizer.from_pretrained(
            GENERATION_MODEL
        )
    )
    generation_model = (
        AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL
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


# 8. FASTAPI LIFESPAN

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Models are loaded once when the application starts.
    """

    logger.info(
        "Starting Task 28 LLM Inference API."
    )

    load_models()

    logger.info(
        "All models initialized successfully."
    )

    yield

    logger.info(
        "Shutting down Task 28 LLM Inference API."
    )

    inference_executor.shutdown(
        wait=True
    )

    logger.info(
        "Inference executor shut down."
    )


# 9. FASTAPI APPLICATION

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


# 10. CORS MIDDLEWARE

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# 11. STANDARD ERROR RESPONSE

class ErrorResponse(BaseModel):
    """
    Standard structure for API error responses.
    """

    error: str

    message: str

    status_code: int


# 12. HTTP EXCEPTION HANDLER

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


# 13. VALIDATION ERROR HANDLER

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


# 14. VALUE ERROR HANDLER

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


# 15. GENERAL EXCEPTION HANDLER

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


# 16. SENTIMENT REQUEST SCHEMA

class SentimentRequest(BaseModel):
    """
    Request schema for sentiment analysis.
    """

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


# 17. SENTIMENT RESPONSE SCHEMA

class SentimentResponse(BaseModel):
    """
    Response schema for sentiment analysis.
    """

    label: str

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )


# 18. SUMMARIZATION REQUEST SCHEMA

class SummarizationRequest(BaseModel):
    """
    Request schema for abstractive summarization.
    """

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


# 19. SUMMARIZATION RESPONSE SCHEMA

class SummarizationResponse(BaseModel):
    """
    Response schema for summarization.
    """

    summary: str


# 20. GENERATION REQUEST SCHEMA

class GenerationRequest(BaseModel):
    """
    Request schema for causal text generation.
    """

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


# 21. GENERATION RESPONSE SCHEMA

class GenerationResponse(BaseModel):
    """
    Response schema for causal generation.
    """

    generated_text: str


# 22. HEALTH CHECK ENDPOINT

@app.get(
    "/healthz",
    response_model=dict
)
async def health_check():
    """
    Return the health status of the API service.
    """

    return {
        "status": "healthy",
        "service": "Task 28 LLM Inference API"
    }


# 23. SENTIMENT INFERENCE FUNCTION

def run_sentiment_inference(text: str):
    """
    Run sentiment model inference.
    """

    return sentiment_pipeline(text)


# 24. SENTIMENT API ENDPOINT

@app.post(
    "/api/v1/sentiment",
    response_model=SentimentResponse
)
async def sentiment_analysis(
    request: SentimentRequest
):
    """
    Perform sentiment analysis.
    """

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


# 25. SUMMARIZATION INFERENCE FUNCTION

def run_summarization_inference(
    request: SummarizationRequest
):
    """
    Run summarization model inference.
    """

    # Tokenize and truncate input to the model context limit.
    inputs = summarization_tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Move tensors to GPU when available.
    if summarization_device == "cuda":

        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

    # Disable gradient calculation for inference.
    with torch.no_grad():

        summary_ids = summarization_model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            min_new_tokens=request.min_tokens,
            num_beams=request.num_beams,
            do_sample=False
        )

    # Convert generated tokens back into text.
    summary = summarization_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary


# 26. SUMMARIZATION API ENDPOINT

@app.post(
    "/api/v1/summarize",
    response_model=SummarizationResponse
)
async def summarize_text(
    request: SummarizationRequest
):
    """
    Perform abstractive text summarization.
    """

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


# 27. GENERATION INFERENCE FUNCTION

def run_generation_inference(
    request: GenerationRequest
):
    """
    Run causal language model inference.
    """

    # Tokenize prompt and truncate to model context limit.
    inputs = generation_tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Move tensors to GPU when available.
    if generation_device == "cuda":

        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

    # Basic generation configuration.
    generation_kwargs = {
        "max_new_tokens": request.max_tokens,
        "do_sample": request.do_sample,
        "num_beams": request.num_beams,
        "pad_token_id": generation_tokenizer.eos_token_id
    }

    # Add sampling parameters only when sampling is enabled.
    if request.do_sample:

        if request.top_k is not None:

            generation_kwargs["top_k"] = request.top_k

        if request.top_p is not None:

            generation_kwargs["top_p"] = request.top_p

        if request.temperature is not None:

            generation_kwargs["temperature"] = (
                request.temperature
            )

    # Run model inference without gradients.
    with torch.no_grad():

        output_ids = generation_model.generate(
            **inputs,
            **generation_kwargs
        )

    # Decode generated tokens.
    generated_text = generation_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return generated_text


# 28. GENERATION API ENDPOINT

@app.post(
    "/api/v1/generate",
    response_model=GenerationResponse
)
async def generate_text(
    request: GenerationRequest
):
    """
    Generate text using the causal language model.
    """

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


# 29. STARTUP MESSAGE

logger.info(
    "Task 28 Part B API configuration complete."
)