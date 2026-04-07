"""
server.py — FastAPI web server exposing the OpenEnv Data Pipeline Debugger as a REST API.

Endpoints:
  POST /reset        — Start a new episode
  POST /step         — Take one action
  GET  /state        — Get current environment state
  GET  /tasks        — List all tasks with schemas
  GET  /grader       — Score the current completed episode
  POST /baseline     — Run the inference agent and return all scores
"""

import os
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from environment import DataPipelineEnv, Action, Observation, StepResult
from graders import grade_episode

load_dotenv()

app = FastAPI(
    title="OpenEnv — Data Pipeline Debugger",
    description=(
        "An AI agent environment where agents debug and fix broken data pipelines. "
        "Supports 3 tasks of increasing difficulty. Graders are fully deterministic."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session server)
env = DataPipelineEnv()


# ─── Request / Response Models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "fix_csv_encoding"

class StepRequest(BaseModel):
    code: str

class GraderResponse(BaseModel):
    task: str
    score: float
    reason: str
    breakdown: dict


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "name": "OpenEnv — Data Pipeline Debugger",
        "version": "1.0.0",
        "status": "running",
        "tasks": DataPipelineEnv.TASKS,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"]
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/tasks", tags=["Environment"])
def list_tasks():
    """List all available tasks with their action and observation schemas."""
    try:
        for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
            try:
                with open("openenv.yaml", "r", encoding=enc) as f:
                    config = yaml.safe_load(f)
                return config
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise FileNotFoundError("Could not decode openenv.yaml")
    except FileNotFoundError:
        return {
            "tasks": [
                {"name": "fix_csv_encoding", "difficulty": "easy", "max_steps": 10},
                {"name": "fix_schema_errors", "difficulty": "medium", "max_steps": 15},
                {"name": "optimize_pipeline", "difficulty": "hard", "max_steps": 20},
            ],
            "action_space": {"type": "structured", "fields": [{"name": "code", "type": "string"}]},
            "observation_space": {
                "type": "structured",
                "fields": ["data_preview", "error_message", "schema_info", "step_number", "task_name"]
            }
        }


@app.post("/reset", response_model=Observation, tags=["Environment"])
def reset(request: ResetRequest):
    """Start a new episode for the specified task."""
    valid_tasks = DataPipelineEnv.TASKS
    if request.task_name not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{request.task_name}'. Choose from: {valid_tasks}"
        )
    observation = env.reset(task_name=request.task_name)
    return observation


@app.post("/step", response_model=StepResult, tags=["Environment"])
def step(request: StepRequest):
    """Take one action in the environment."""
    if env.current_df is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    action = Action(code=request.code)
    result = env.step(action)
    return result


@app.get("/state", tags=["Environment"])
def state():
    """Get the full current state of the environment."""
    if env.current_df is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    return env.state()


@app.get("/grader", response_model=GraderResponse, tags=["Grader"])
def grader():
    """
    Grade the current completed episode.
    Returns a deterministic score 0.0 → 1.0 with full breakdown.
    """
    if env.current_df is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset and run an episode first."
        )
    result = grade_episode(env.current_task, env.episode_history)
    return GraderResponse(
        task=env.current_task,
        score=result["score"],
        reason=result["reason"],
        breakdown=result["breakdown"]
    )


@app.post("/baseline", tags=["Baseline"])
def run_baseline():
    """
    Run the inference agent across all 3 tasks and return scores.
    Requires API_BASE_URL, MODEL_NAME, and HF_TOKEN in .env
    """
    try:
        from inference import run_task
        from environment import DataPipelineEnv
        
        tasks = DataPipelineEnv.TASKS
        scores = {}
        
        for task in tasks:
            result = run_task(task)
            scores[task] = result["score"]
        
        return {
            "status": "success",
            "scores": scores,
            "average_score": round(sum(scores.values()) / len(scores), 4)
        }
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Inference module import failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
