# Production Deployment, REST API & Inference

## Overview
The platform supports multi-channel production deployment: FastAPI REST service, Docker containerization, unified CLI, and official BioDCASE competition submission generation.

---

## 1. Unified Master CLI (`cli.py`)

Run platform workflows from a single entrypoint:

```bash
# Execute end-to-end data processing & feature extraction
python cli.py pipeline

# Train and benchmark all 15 ML models
python cli.py train

# Execute scientific evaluation & generate figures
python cli.py evaluate

# Predict bird count for a single audio file or directory
python cli.py predict --input data/raw/dev_aviary_1/chunk_000/rec_d1_00_00_01.000000.wav

# Generate official BioDCASE submission file
python cli.py submit --data-dir data/raw
```

---

## 2. FastAPI REST Service (`src/inference/api.py`)

Start the REST API server:

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

### Endpoints Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck endpoint confirming API operational status |
| `GET` | `/version` | Project version and seed metadata |
| `GET` | `/model_info` | Active model details and registered model catalog |
| `GET` | `/feature_info` | Active feature extractors and registry catalog |
| `POST` | `/predict` | Predict bird count for an uploaded `.wav`/`.mp3` audio file |

Access interactive Swagger UI documentation at: `http://localhost:8000/docs`.

---

## 3. Docker Deployment (`Dockerfile`, `docker-compose.yml`)

Run the application inside a lightweight Python 3.10 multi-stage container with audio system dependencies (`libsndfile1`, `ffmpeg`):

```bash
# Build and start containerized REST API service
docker-compose up --build -d

# Verify operational status
curl http://localhost:8000/health
```

---

## 4. BioDCASE Submission Generator (`src/inference/submission.py`)
Generates standardized competition submission CSVs under `submissions/submission_v1.0.0_YYYYMMDD_HHMMSS.csv` with MD5 integrity checksums.
