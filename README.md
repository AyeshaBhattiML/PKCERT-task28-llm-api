# PKCERT Task 28 – Production LLM Serving & Full-Stack Model Deployment

A production-oriented NLP microservice developed as part of the **PKCERT AI & Software Development Internship**.

This project demonstrates how pretrained Transformer-based NLP models can be exposed through a structured **FastAPI REST API**, containerized using **Docker**, orchestrated with **Docker Compose**, and accessed through an interactive **Streamlit frontend**.

---

## Project Overview

Task 28 focuses on building an end-to-end machine learning serving architecture supporting three NLP capabilities:

* **Sentiment Analysis**
* **Abstractive Text Summarization**
* **Causal Text Generation**

The system follows a client-server architecture in which the Streamlit frontend communicates with a FastAPI backend. The backend performs inference using Hugging Face remote inference services.

### Architecture

```text
┌──────────────────────────┐
│    Streamlit Frontend    │
│  Interactive Web Client  │
└────────────┬─────────────┘
             │ HTTP Requests
             ▼
┌──────────────────────────┐
│      FastAPI Backend     │
│     REST Microservice    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Hugging Face Inference   │
│         Service          │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     NLP Model Results    │
│ Sentiment / Summary /    │
│       Generation         │
└──────────────────────────┘
```

---

## Technologies Used

| Technology     | Purpose                         |
| -------------- | ------------------------------- |
| Python 3.11    | Application development         |
| FastAPI        | REST API development            |
| Pydantic       | Request and response validation |
| Hugging Face   | Remote model inference          |
| DistilBERT     | Sentiment analysis              |
| DistilBART     | Abstractive summarization       |
| DistilGPT-2    | Causal text generation          |
| Docker         | Application containerization    |
| Docker Compose | Local container orchestration   |
| Streamlit      | Interactive frontend            |
| Git & GitHub   | Version control                 |
| PowerShell     | Local development and testing   |

---

## NLP Services

### 1. Sentiment Analysis

Uses a pretrained **DistilBERT** model to classify input text and return the predicted sentiment with a confidence score.

**Endpoint:**

```text
POST /api/v1/sentiment
```

Example request:

```json
{
  "text": "I really enjoyed working on this project."
}
```

Example response:

```json
{
  "label": "POSITIVE",
  "score": 0.99
}
```

---

### 2. Abstractive Text Summarization

Uses **DistilBART** to generate a concise summary of an input document.

**Endpoint:**

```text
POST /api/v1/summarize
```

Supported parameters include:

* Maximum output tokens
* Minimum output tokens
* Number of beams

---

### 3. Causal Text Generation

Uses **DistilGPT-2** to generate a continuation based on a user-provided prompt.

**Endpoint:**

```text
POST /api/v1/generate
```

The service supports configurable generation parameters including:

* Maximum tokens
* Temperature
* Top-k
* Top-p
* Sampling
* Beam search

---

## REST API

The FastAPI service exposes the following endpoints:

| Method | Endpoint            | Description              |
| ------ | ------------------- | ------------------------ |
| GET    | `/healthz`          | API health check         |
| POST   | `/api/v1/sentiment` | Sentiment classification |
| POST   | `/api/v1/summarize` | Text summarization       |
| POST   | `/api/v1/generate`  | Text generation          |

Interactive API documentation is automatically provided by FastAPI through Swagger UI.

```text
http://localhost:8000/docs
```

---

## Validation & Error Handling

Pydantic schemas are used to validate API requests and responses.

The API validates:

* Empty input text
* Maximum input length
* Token limits
* Beam-search parameters
* Top-k values
* Top-p values
* Temperature values

Standardized error handling was implemented for validation errors, HTTP errors, bad requests, and unexpected server exceptions.

---

## Asynchronous Request Processing

The API uses FastAPI's asynchronous architecture to avoid blocking the main event loop during synchronous inference/network operations.

Blocking Hugging Face requests are executed through worker threads, allowing the service to handle concurrent requests more efficiently.

CORS middleware is also configured to allow communication between the frontend and backend.

---

## Health Monitoring

A dedicated health endpoint was implemented:

```text
GET /healthz
```

Example response:

```json
{
  "status": "healthy",
  "service": "Task 28 LLM Inference API",
  "inference": "Hugging Face remote",
  "model_loading": "remote",
  "device": "Hugging Face"
}
```

---

## Docker

The application is containerized using Docker.

The container configuration includes:

* Python 3.11 environment
* Application dependencies
* Uvicorn server
* Environment variables
* Health checking
* Configurable application port
* Non-root execution
* Resource limits

Docker Compose was used for local orchestration.

The local configuration was tested with:

```text
CPU Limit: 2 CPUs
Memory Limit: 3 GB
```

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/AyeshaBhattiML/<repository-name>.git
```

```bash
cd <repository-name>
```

### 2. Build the Docker image

```bash
docker compose build
```

### 3. Start the application

```bash
docker compose up -d
```

### 4. Check the container

```bash
docker compose ps
```

### 5. Check API health

Open:

```text
http://localhost:8000/healthz
```

### 6. Open API documentation

```text
http://localhost:8000/docs
```

---

## Streamlit Frontend

An interactive Streamlit interface was developed for communicating with the FastAPI backend.

The frontend provides:

* Backend status checking
* Sentiment analysis
* Text summarization
* Text generation
* Generation parameter controls
* Response visualization
* User-facing error messages

Run the frontend with:

```bash
streamlit run frontend.py
```

The frontend can communicate with the local API through:

```text
http://localhost:8000
```

---

## Performance Benchmarking

The API was tested using both single-client and concurrent multi-client workloads.

### Single-Client Results

| Metric              |       Result |
| ------------------- | -----------: |
| Total Requests      |           20 |
| Successful Requests |           20 |
| Success Rate        |         100% |
| Average Latency     |    710.99 ms |
| P50                 |    395.52 ms |
| P90                 |    826.87 ms |
| P99                 |   4693.55 ms |
| Throughput          | 1.41 req/sec |

### Concurrent Results

| Metric              |        Result |
| ------------------- | ------------: |
| Total Requests      |            20 |
| Successful Requests |            20 |
| Success Rate        |          100% |
| Average Latency     |     945.91 ms |
| P50                 |     991.61 ms |
| P90                 |    1256.81 ms |
| P99                 |    1436.92 ms |
| Throughput          | 12.85 req/sec |

The concurrent benchmark achieved higher throughput while maintaining a 100% request success rate.

---

## Docker Resource Monitoring

The Docker container was monitored over 20 samples.

| Resource             |    Result |
| -------------------- | --------: |
| Average CPU          |     0.67% |
| Maximum CPU          |    10.50% |
| Average RAM          |     1.80% |
| Maximum RAM          |     2.08% |
| Maximum Observed RAM | 63.87 MiB |

The low memory consumption is consistent with the remote inference architecture because model weights are not stored inside the application container.

---

## Engineering Challenges

### Model Resource Requirements

Running multiple Transformer models locally can significantly increase memory consumption.

**Solution:**
Remote Hugging Face inference was used to keep the Docker container lightweight and reduce local model memory requirements.

### Blocking Network Operations

Synchronous inference requests can block an asynchronous web server.

**Solution:**
Blocking operations were executed through worker threads so that the FastAPI event loop could remain responsive.

### Free Cloud Deployment Restrictions

Public cloud deployment was investigated, but available free container-hosting options required billing verification.

**Solution:**
The application was fully tested locally using Docker, Docker Compose, API health checks, frontend integration, load testing, and resource monitoring rather than claiming an incomplete public deployment.

---

## Project Structure

```text
Task-28/
│
├── app.py
├── app-local.py
├── frontend.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── benchmark.py
├── resource_monitor.py
├── locustfile.py
├── .gitignore
│
└── README.md
```

---

## Key Learning Outcomes

This project provided practical experience in:

* Pretrained Transformer model integration
* NLP inference pipelines
* FastAPI microservice development
* Asynchronous request handling
* Pydantic validation
* REST API design
* Docker containerization
* Docker Compose orchestration
* Streamlit frontend development
* Git and GitHub workflows
* API health monitoring
* Load and latency testing
* CPU and RAM monitoring
* Deployment architecture evaluation

---

## Internship

**Program:** PKCERT AI & Software Development Internship
**Task:** Task 28 – Production LLM Serving, Asynchronous Microservices & Full-Stack Model Deployment
**Developer:** Ayesha Bhatti
**Date:** August 2026

---

