"""
graders.py — Deterministic episode-level graders for OpenEnv Data Pipeline Debugger.

Each grader receives the episode history and returns a final score 0.0 → 1.0.
Graders are fully deterministic (no LLM, no randomness).
"""

from environment import DataPipelineEnv, Action


# ─── Base Grader ───────────────────────────────────────────────────────────────

class BaseGrader:
    """Base class for all task graders."""

    task_name: str = ""

    def grade(self, episode_history: list) -> dict:
        """
        Grade a completed episode.

        Args:
            episode_history: list of dicts from env.episode_history

        Returns:
            dict with keys: score (float 0-1), reason (str), breakdown (dict)
        """
        raise NotImplementedError


# ─── Easy Task Grader: fix_csv_encoding ────────────────────────────────────────

class FixCSVEncodingGrader(BaseGrader):
    """
    Grades the fix_csv_encoding (easy) task.

    Runs a fresh episode replay and evaluates final DataFrame against expected.
    Also awards bonus for fixing it in fewer steps.
    """

    task_name = "fix_csv_encoding"

    def grade(self, episode_history: list) -> dict:
        env = DataPipelineEnv()
        env.reset(task_name=self.task_name)

        final_score = 0.0
        steps_used = 0

        for entry in episode_history:
            action = Action(code=entry["action"])
            result = env.step(action)
            steps_used += 1
            if result.done:
                break

        final_score = env._compute_score()

        # Efficiency bonus: fewer steps = small bonus (up to 0.1 extra cap at 1.0)
        efficiency_bonus = 0.0
        if final_score >= 0.95:
            efficiency_bonus = max(0.0, (env.max_steps - steps_used) / env.max_steps) * 0.1

        total = min(1.0, round(final_score + efficiency_bonus, 4))

        return {
            "score": total,
            "reason": f"Final DataFrame match: {round(final_score, 4)} | Steps used: {steps_used}/{env.max_steps} | Efficiency bonus: {round(efficiency_bonus, 4)}",
            "breakdown": {
                "correctness_score": round(final_score, 4),
                "steps_used": steps_used,
                "max_steps": env.max_steps,
                "efficiency_bonus": round(efficiency_bonus, 4),
                "task": self.task_name
            }
        }


# ─── Medium Task Grader: fix_schema_errors ─────────────────────────────────────

class FixSchemaErrorsGrader(BaseGrader):
    """
    Grades the fix_schema_errors (medium) task.

    Additionally checks that no previously correct columns were broken during fixing.
    Penalizes regression (fixing one column while breaking another).
    """

    task_name = "fix_schema_errors"

    def grade(self, episode_history: list) -> dict:
        env = DataPipelineEnv()
        env.reset(task_name=self.task_name)

        scores_per_step = []
        steps_used = 0
        error_steps = 0

        for entry in episode_history:
            action = Action(code=entry["action"])
            result = env.step(action)
            steps_used += 1
            scores_per_step.append(env._compute_score())
            if entry.get("error"):
                error_steps += 1
            if result.done:
                break

        final_score = env._compute_score()

        # Regression penalty: if score ever went backward significantly
        regression_penalty = 0.0
        for i in range(1, len(scores_per_step)):
            drop = scores_per_step[i - 1] - scores_per_step[i]
            if drop > 0.1:
                regression_penalty += drop * 0.5

        regression_penalty = min(0.2, round(regression_penalty, 4))

        # Error penalty: agent threw errors
        error_penalty = min(0.1, round(error_steps * 0.02, 4))

        total = max(0.0, min(1.0, round(final_score - regression_penalty - error_penalty, 4)))

        return {
            "score": total,
            "reason": (
                f"Final score: {round(final_score, 4)} | "
                f"Regression penalty: -{regression_penalty} | "
                f"Error penalty: -{error_penalty} | "
                f"Steps: {steps_used}"
            ),
            "breakdown": {
                "correctness_score": round(final_score, 4),
                "regression_penalty": regression_penalty,
                "error_penalty": error_penalty,
                "error_steps": error_steps,
                "steps_used": steps_used,
                "task": self.task_name
            }
        }


# ─── Hard Task Grader: optimize_pipeline ───────────────────────────────────────

class OptimizePipelineGrader(BaseGrader):
    """
    Grades the optimize_pipeline (hard) task.

    Evaluates:
    1. Correctness — does the final output match expected aggregation?
    2. Progress slope — did the agent consistently improve or thrash?
    3. Error rate — how often did the agent throw exceptions?
    """

    task_name = "optimize_pipeline"

    def grade(self, episode_history: list) -> dict:
        env = DataPipelineEnv()
        env.reset(task_name=self.task_name)

        scores_per_step = []
        steps_used = 0
        error_steps = 0

        for entry in episode_history:
            action = Action(code=entry["action"])
            result = env.step(action)
            steps_used += 1
            scores_per_step.append(env._compute_score())
            if entry.get("error"):
                error_steps += 1
            if result.done:
                break

        final_score = env._compute_score()

        # Progress quality: positive slope = good, thrashing = bad
        progress_bonus = 0.0
        if len(scores_per_step) >= 2:
            net_progress = scores_per_step[-1] - scores_per_step[0]
            if net_progress > 0:
                progress_bonus = min(0.1, round(net_progress * 0.2, 4))

        # Error rate penalty
        error_rate = error_steps / max(1, steps_used)
        error_penalty = min(0.15, round(error_rate * 0.3, 4))

        total = max(0.0, min(1.0, round(final_score + progress_bonus - error_penalty, 4)))

        return {
            "score": total,
            "reason": (
                f"Final score: {round(final_score, 4)} | "
                f"Progress bonus: +{progress_bonus} | "
                f"Error penalty: -{error_penalty} | "
                f"Steps: {steps_used} | Error rate: {round(error_rate, 2)}"
            ),
            "breakdown": {
                "correctness_score": round(final_score, 4),
                "progress_bonus": progress_bonus,
                "error_penalty": error_penalty,
                "error_steps": error_steps,
                "error_rate": round(error_rate, 4),
                "steps_used": steps_used,
                "task": self.task_name
            }
        }


# ─── Grader Registry ───────────────────────────────────────────────────────────

GRADERS = {
    "fix_csv_encoding": FixCSVEncodingGrader(),
    "fix_schema_errors": FixSchemaErrorsGrader(),
    "optimize_pipeline": OptimizePipelineGrader(),
}


def grade_episode(task_name: str, episode_history: list) -> dict:
    """
    Public function to grade an episode for a given task.

    Args:
        task_name: One of 'fix_csv_encoding', 'fix_schema_errors', 'optimize_pipeline'
        episode_history: The env.episode_history list after an episode

    Returns:
        dict with score, reason, breakdown
    """
    if task_name not in GRADERS:
        return {
            "score": 0.0,
            "reason": f"Unknown task: {task_name}",
            "breakdown": {}
        }
    return GRADERS[task_name].grade(episode_history)
