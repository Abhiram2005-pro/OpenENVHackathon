# 🛠️ OpenEnv — Data Pipeline Debugger

> An AI agent environment where LLM agents debug and fix broken data pipelines.
> Built for the [OpenEnv Hackathon](https://huggingface.co/openenv).

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://huggingface.co/openenv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Overview

**Data Pipeline Debugger** is an OpenEnv-compatible environment where AI agents must identify and fix broken pandas data pipelines. The agent receives structured observations (data previews, schema info, error messages) and submits Python code to progressively fix the data — earning rewards based on how accurately it matches the expected output.

This environment models a **real-world data engineering problem** that occurs daily in industry, making it highly practical and novel in the agent evaluation space.

---

## 🧩 Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `fix_csv_encoding` | 🟢 Easy | 10 | Fix broken CSV with wrong delimiters, bad emails, OCR digit errors |
| `fix_schema_errors` | 🟡 Medium | 15 | Correct data type mismatches, null handling, boolean parsing |
| `optimize_pipeline` | 🔴 Hard | 20 | Fix logic bugs in a regional sales aggregation pipeline |

---

## 🔄 Environment API

All interactions follow the standard OpenEnv interface:

```python
from environment import DataPipelineEnv, Action

env = DataPipelineEnv()

# Start an episode
observation = env.reset(task_name="fix_csv_encoding")

# Agent takes actions
result = env.step(Action(code="df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)"))

print(result.reward)  # e.g. 0.25
print(result.done)    # False / True
```

### Observation (what the agent sees)
```
data_preview   — first 5 rows of current DataFrame
schema_info    — column names, dtypes, null counts
error_message  — error from last action (if any)
step_number    — current step
task_name      — which task is active
```

### Action (what the agent does)
```
code — Python pandas code string operating on variable `df`
```

### Reward
```
+1.0    — task fully solved (score ≥ 0.95)
0–1.0   — partial credit per column correctly fixed
-0.1    — action caused an exception
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info and status |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks and schemas |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Get current environment state |
| `GET` | `/grader` | Grade the completed episode |
| `POST` | `/baseline` | Run the LLM baseline agent |

---

## 📊 Grading

Graders are **fully deterministic** and based on DataFrame comparison:

- **Easy** — Correctness score + efficiency bonus (fewer steps = bonus)
- **Medium** — Correctness score − regression penalty − error penalty
- **Hard** — Correctness score + progress bonus − error rate penalty

Final scores are in range **[0.0, 1.0]**.

---

## 🚀 Running Locally

```bash
# 1. Clone and enter directory
git clone <your-repo-url>
cd openenv-data-pipeline

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env with your API credentials (see API_KEY_SETUP.md)

# 5. Start the server
uvicorn server:app --reload --port 7860

# 6. Open API docs
# http://localhost:7860/docs
```

---

## 🤖 Running the Inference Agent

```bash
python inference.py
```

The inference agent uses OpenAI API client to generate pandas fix code for each task.

**Required environment variables:**
- `API_BASE_URL` — LLM API endpoint (e.g., https://api.openai.com/v1)
- `MODEL_NAME` — Model identifier (e.g., gpt-4, meta-llama/Llama-2-7b-chat-hf)
- `HF_TOKEN` — API authentication token

See [API_KEY_SETUP.md](API_KEY_SETUP.md) for detailed steps to obtain these credentials.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker

```bash
docker build -t openenv-data-pipeline .
docker run -p 7860:7860 --env-file .env openenv-data-pipeline
```

---

## 📁 Project Structure

```
openenv-data-pipeline/
├── environment.py      # Core OpenEnv environment (tasks, step, reward logic)
├── graders.py          # Deterministic episode graders (per-task scoring)
├── baseline.py         # LLM baseline agent (GPT-4o-mini)
├── server.py           # FastAPI REST server
├── openenv.yaml        # Environment metadata specification
├── Dockerfile          # Docker container for deployment
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed to git)
├── tasks/              # Task definitions package
└── tests/
    └── test_env.py     # pytest test suite (35+ tests)
```

---

## 🏆 Competition Highlights

- ✅ **Real-world utility** — Data pipeline debugging is a universal problem
- ✅ **Deterministic graders** — 100% reproducible scores via DataFrame comparison
- ✅ **Partial credit rewards** — Fine-grained feedback per column
- ✅ **3 difficulty levels** — Easy → Medium → Hard with increasing complexity
- ✅ **LLM-compatible** — Clean text-based observations and code actions
- ✅ **Fully tested** — Comprehensive test suite included
- ✅ **Dockerized** — Ready for Hugging Face Spaces deployment

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
