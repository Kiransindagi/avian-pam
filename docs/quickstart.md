# Avian PAM Platform - Quickstart Guide

## 1. Prerequisites & Environment Setup

Ensure Python 3.10+ and `ffmpeg` / `libsndfile` are installed on your machine.

```bash
# Clone repository
git clone https://github.com/Kiransindagi/avian-pam.git
cd avian-pam

# Virtual environment creation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production and development dependencies
pip install -r requirements.txt
```

---

## 2. Running Command Line Workflows (`cli.py`)

Execute platform tasks through the unified `cli.py` interface:

```bash
# 1. Execute end-to-end processing pipeline
python cli.py pipeline

# 2. Train and benchmark all ML models
python cli.py train

# 3. Perform statistical evaluation and generate figures
python cli.py evaluate

# 4. Predict bird abundance for a single recording
python cli.py predict --input data/raw/dev_aviary_1/chunk_000/rec_d1_00_00_01.000000.wav

# 5. Generate official BioDCASE competition submission CSV
python cli.py submit --data-dir data/raw
```

---

## 3. Docker REST API Deployment

```bash
# Build and run API container
docker-compose up --build -d

# Verify API health
curl http://localhost:8000/health
```
