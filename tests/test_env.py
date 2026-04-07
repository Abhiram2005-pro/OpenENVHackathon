"""
tests/test_env.py — pytest test suite for the OpenEnv Data Pipeline Debugger.

Covers:
  - Environment reset and observation structure
  - Step execution and reward computation
  - Task completion
  - Grader correctness
  - Server endpoints (integration tests using TestClient)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import DataPipelineEnv, Action, Observation, StepResult
from graders import grade_episode, GRADERS


# ─── Environment Tests ─────────────────────────────────────────────────────────

class TestEnvironmentReset:
    def test_reset_easy_task(self):
        env = DataPipelineEnv()
        obs = env.reset(task_name="fix_csv_encoding")
        assert isinstance(obs, Observation)
        assert obs.task_name == "fix_csv_encoding"
        assert obs.step_number == 0
        assert env.current_df is not None
        assert env.expected_df is not None

    def test_reset_medium_task(self):
        env = DataPipelineEnv()
        obs = env.reset(task_name="fix_schema_errors")
        assert obs.task_name == "fix_schema_errors"
        assert env.max_steps == 15

    def test_reset_hard_task(self):
        env = DataPipelineEnv()
        obs = env.reset(task_name="optimize_pipeline")
        assert obs.task_name == "optimize_pipeline"
        assert env.max_steps == 20

    def test_observation_has_required_fields(self):
        env = DataPipelineEnv()
        obs = env.reset(task_name="fix_csv_encoding")
        assert hasattr(obs, "data_preview")
        assert hasattr(obs, "error_message")
        assert hasattr(obs, "schema_info")
        assert hasattr(obs, "step_number")
        assert hasattr(obs, "task_name")

    def test_reset_clears_history(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        env.step(Action(code="df['age'] = 1"))
        env.reset(task_name="fix_csv_encoding")
        assert len(env.episode_history) == 0
        assert env.step_num == 0
        assert not env.done


class TestEnvironmentStep:
    def test_valid_action_increments_step(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        result = env.step(Action(code="df['age'] = 0"))
        assert isinstance(result, StepResult)
        assert result.observation.step_number == 1

    def test_invalid_code_returns_negative_reward(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        result = env.step(Action(code="this_is_invalid_code!!!"))
        assert result.reward == -0.1
        assert result.info["error"] != ""

    def test_reward_clamped_between_neg1_and_1(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        result = env.step(Action(code="df['age'] = 0"))
        assert -1.0 <= result.reward <= 1.0

    def test_done_after_max_steps(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")  # max_steps=10
        for _ in range(10):
            result = env.step(Action(code="pass"))
        assert result.done is True

    def test_step_after_done_returns_early(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        env.done = True
        result = env.step(Action(code="df['age'] = 0"))
        assert result.reward == 0.0
        assert result.done is True

    def test_episode_history_recorded(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        env.step(Action(code="df['age'] = 0"))
        env.step(Action(code="df['salary'] = 0"))
        assert len(env.episode_history) == 2
        assert "action" in env.episode_history[0]
        assert "reward" in env.episode_history[0]


class TestScoringLogic:
    def test_initial_score_less_than_1(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        score = env._compute_score()
        assert 0.0 <= score < 1.0

    def test_perfect_fix_gives_max_score_easy(self):
        """Directly set df = expected_df and check score is ~1.0"""
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        env.current_df = env.expected_df.copy()
        score = env._compute_score()
        assert score >= 0.95

    def test_state_contains_correct_keys(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        state = env.state()
        for key in ["task", "step", "max_steps", "done", "score", "data_preview", "episode_history"]:
            assert key in state


# ─── Grader Tests ──────────────────────────────────────────────────────────────

class TestGraders:
    def _run_perfect_episode(self, task_name: str):
        """Run an episode where we directly assign the expected df to cheat."""
        env = DataPipelineEnv()
        env.reset(task_name=task_name)
        # Use a code that sets df equal to expected_df
        # We do this by encoding expected values directly
        code = "df = df.copy()"  # no-op first step
        env.step(Action(code=code))
        env.current_df = env.expected_df.copy()
        # Force done
        env.done = True
        return env.episode_history

    def test_grade_easy_task_returns_score(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        env.step(Action(code="pass"))
        result = grade_episode("fix_csv_encoding", env.episode_history)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_grade_medium_task_returns_score(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_schema_errors")
        env.step(Action(code="pass"))
        result = grade_episode("fix_schema_errors", env.episode_history)
        assert 0.0 <= result["score"] <= 1.0

    def test_grade_hard_task_returns_score(self):
        env = DataPipelineEnv()
        env.reset(task_name="optimize_pipeline")
        env.step(Action(code="pass"))
        result = grade_episode("optimize_pipeline", env.episode_history)
        assert 0.0 <= result["score"] <= 1.0

    def test_unknown_task_returns_zero(self):
        result = grade_episode("nonexistent_task", [])
        assert result["score"] == 0.0

    def test_grader_breakdown_has_required_keys(self):
        env = DataPipelineEnv()
        env.reset(task_name="fix_csv_encoding")
        env.step(Action(code="pass"))
        result = grade_episode("fix_csv_encoding", env.episode_history)
        assert "breakdown" in result
        assert "reason" in result
        assert "correctness_score" in result["breakdown"]

    def test_all_graders_registered(self):
        assert "fix_csv_encoding" in GRADERS
        assert "fix_schema_errors" in GRADERS
        assert "optimize_pipeline" in GRADERS


# ─── FastAPI Server Integration Tests ──────────────────────────────────────────

class TestServerEndpoints:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from server import app
        self.client = TestClient(app)

    def test_root_returns_200(self):
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_tasks_endpoint(self):
        response = self.client.get("/tasks")
        assert response.status_code == 200

    def test_reset_valid_task(self):
        response = self.client.post("/reset", json={"task_name": "fix_csv_encoding"})
        assert response.status_code == 200
        data = response.json()
        assert data["task_name"] == "fix_csv_encoding"
        assert data["step_number"] == 0

    def test_reset_invalid_task(self):
        response = self.client.post("/reset", json={"task_name": "nonexistent"})
        assert response.status_code == 400

    def test_step_after_reset(self):
        self.client.post("/reset", json={"task_name": "fix_csv_encoding"})
        response = self.client.post("/step", json={"code": "df['age'] = 0"})
        assert response.status_code == 200
        data = response.json()
        assert "reward" in data
        assert "done" in data

    def test_step_without_reset_returns_400(self):
        from server import env
        env.current_df = None
        response = self.client.post("/step", json={"code": "df['age'] = 0"})
        assert response.status_code == 400

    def test_grader_after_episode(self):
        self.client.post("/reset", json={"task_name": "fix_csv_encoding"})
        self.client.post("/step", json={"code": "df['age'] = 0"})
        response = self.client.get("/grader")
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_state_after_steps(self):
        self.client.post("/reset", json={"task_name": "fix_schema_errors"})
        self.client.post("/step", json={"code": "df['price'] = 0"})
        response = self.client.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert data["step"] == 1
        assert data["task"] == "fix_schema_errors"
