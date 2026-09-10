"""Trajectory data structures and storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class TokenMetrics:
    """Per-token metrics for a trajectory."""

    token_id: int
    token_text: str
    token_index: int
    probability: float
    log_probability: float
    entropy: float
    top_k_token_ids: list[int] = field(default_factory=list)
    top_k_token_texts: list[str] = field(default_factory=list)
    top_k_probabilities: list[float] = field(default_factory=list)
    probability_margin: float = 0.0
    cumulative_log_prob: float = 0.0
    has_logits: bool = False
    has_hidden_state: bool = False
    has_attention: bool = False


@dataclass
class Trajectory:
    """A complete generation trajectory."""

    sample_id: str
    dataset: str
    model_id: str
    model_version: str
    experiment_id: str
    question: str
    ground_truth: str
    prompt: str
    prompt_token_count: int
    generated_text: str
    extracted_answer: str
    is_correct: bool
    generation_tokens: int
    temperature: float
    seed: int
    prompt_template: str
    token_metrics: list[TokenMetrics] = field(default_factory=list)
    available_signals: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrajectoryStorage:
    """Efficient storage for trajectory data using Parquet + DuckDB."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories_dir = self.base_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.base_dir / "trajectories.duckdb"
        self._conn = duckdb.connect(str(self._db_path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectory_metadata (
                sample_id VARCHAR,
                dataset VARCHAR,
                model_id VARCHAR,
                model_version VARCHAR,
                experiment_id VARCHAR,
                question VARCHAR,
                ground_truth VARCHAR,
                prompt VARCHAR,
                prompt_token_count INTEGER,
                generated_text VARCHAR,
                extracted_answer VARCHAR,
                is_correct BOOLEAN,
                generation_tokens INTEGER,
                temperature DOUBLE,
                seed INTEGER,
                prompt_template VARCHAR,
                available_signals VARCHAR,
                metadata VARCHAR,
                parquet_path VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (sample_id, experiment_id)
            )
        """)

    def store(self, trajectory: Trajectory) -> None:
        """Store a trajectory as Parquet + metadata in DuckDB."""
        token_data = {
            "token_id": [m.token_id for m in trajectory.token_metrics],
            "token_text": [m.token_text for m in trajectory.token_metrics],
            "token_index": [m.token_index for m in trajectory.token_metrics],
            "probability": [m.probability for m in trajectory.token_metrics],
            "log_probability": [m.log_probability for m in trajectory.token_metrics],
            "entropy": [m.entropy for m in trajectory.token_metrics],
            "probability_margin": [m.probability_margin for m in trajectory.token_metrics],
            "cumulative_log_prob": [m.cumulative_log_prob for m in trajectory.token_metrics],
        }

        table = pa.Table.from_pydict(token_data)
        parquet_path = self.trajectories_dir / f"{trajectory.sample_id}.parquet"
        pq.write_table(table, str(parquet_path))

        self._conn.execute(
            """
            INSERT OR REPLACE INTO trajectory_metadata
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                trajectory.sample_id,
                trajectory.dataset,
                trajectory.model_id,
                trajectory.model_version,
                trajectory.experiment_id,
                trajectory.question,
                trajectory.ground_truth,
                trajectory.prompt,
                trajectory.prompt_token_count,
                trajectory.generated_text,
                trajectory.extracted_answer,
                trajectory.is_correct,
                trajectory.generation_tokens,
                trajectory.temperature,
                trajectory.seed,
                trajectory.prompt_template,
                json.dumps(trajectory.available_signals),
                json.dumps(trajectory.metadata),
                str(parquet_path),
            ],
        )

    def store_batch(self, trajectories: list[Trajectory]) -> None:
        for traj in trajectories:
            self.store(traj)

    def load_trajectory(self, sample_id: str) -> Optional[Trajectory]:
        rows = self._conn.execute(
            "SELECT * FROM trajectory_metadata WHERE sample_id = ?", [sample_id]
        ).fetchall()
        if not rows:
            return None
        return self._row_to_trajectory(rows[0])

    def load_parquet(self, sample_id: str) -> Optional[pq.Table]:
        parquet_path = self.trajectories_dir / f"{sample_id}.parquet"
        if parquet_path.exists():
            return pq.read_table(str(parquet_path))
        return None

    def get_experiment_trajectories(
        self, experiment_id: str
    ) -> list[dict[str, Any]]:
        """Load all trajectory summaries for an experiment."""
        rows = self._conn.execute(
            "SELECT * FROM trajectory_metadata WHERE experiment_id = ?",
            [experiment_id],
        ).fetchall()
        cols = [
            d[0] for d in self._conn.execute(
                "SELECT * FROM trajectory_metadata LIMIT 0"
            ).description
        ]
        return [dict(zip(cols, row)) for row in rows]

    def load_all_token_metrics(self, experiment_id: str) -> dict[str, list[TokenMetrics]]:
        """Load token metrics for all trajectories in an experiment."""
        result = {}
        rows = self._conn.execute(
            "SELECT sample_id FROM trajectory_metadata WHERE experiment_id = ?",
            [experiment_id],
        ).fetchall()
        for (sample_id,) in rows:
            table = self.load_parquet(sample_id)
            if table is not None:
                metrics = []
                for i in range(len(table)):
                    row = {col: table.column(col)[i].as_py() for col in table.column_names}
                    metrics.append(
                        TokenMetrics(
                            token_id=row["token_id"],
                            token_text=row["token_text"],
                            token_index=row["token_index"],
                            probability=row["probability"],
                            log_probability=row["log_probability"],
                            entropy=row["entropy"],
                            probability_margin=row["probability_margin"],
                            cumulative_log_prob=row["cumulative_log_prob"],
                        )
                    )
                result[sample_id] = metrics
        return result

    def load_all_token_arrays(self, experiment_id: str) -> dict[str, dict[str, np.ndarray]]:
        """Load token metrics as numpy arrays for efficient computation."""
        raw = self.load_all_token_metrics(experiment_id)
        result = {}
        for sample_id, metrics in raw.items():
            if not metrics:
                continue
            result[sample_id] = {
                "token_ids": np.array([m.token_id for m in metrics]),
                "probabilities": np.array([m.probability for m in metrics]),
                "log_probabilities": np.array([m.log_probability for m in metrics]),
                "entropies": np.array([m.entropy for m in metrics]),
                "margins": np.array([m.probability_margin for m in metrics]),
                "cumulative_log_probs": np.array([m.cumulative_log_prob for m in metrics]),
                "token_indices": np.array([m.token_index for m in metrics]),
            }
        return result

    def get_correctness_labels(self, experiment_id: str) -> dict[str, bool]:
        rows = self._conn.execute(
            "SELECT sample_id, is_correct FROM trajectory_metadata WHERE experiment_id = ?",
            [experiment_id],
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count(self, experiment_id: str) -> int:
        result = self._conn.execute(
            "SELECT COUNT(*) FROM trajectory_metadata WHERE experiment_id = ?",
            [experiment_id],
        ).fetchone()
        return result[0] if result else 0

    def clear(self, experiment_id: str) -> None:
        rows = self._conn.execute(
            "SELECT parquet_path FROM trajectory_metadata WHERE experiment_id = ?",
            [experiment_id],
        ).fetchall()
        self._conn.execute(
            "DELETE FROM trajectory_metadata WHERE experiment_id = ?",
            [experiment_id],
        )
        for (parquet_path,) in rows:
            p = Path(parquet_path)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    def summary(self, experiment_id: str) -> dict[str, Any]:
        total = self.count(experiment_id)
        correct = self._conn.execute(
            "SELECT COUNT(*) FROM trajectory_metadata WHERE experiment_id = ? AND is_correct = true",
            [experiment_id],
        ).fetchone()[0]
        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    def _row_to_trajectory(self, row) -> Trajectory:
        # Get column names
        cols = [d[0] for d in self._conn.execute("SELECT * FROM trajectory_metadata LIMIT 0").description]
        data = dict(zip(cols, row))

        # Load token metrics from parquet
        token_metrics = []
        table = self.load_parquet(data["sample_id"])
        if table is not None:
            for i in range(len(table)):
                trow = {col: table.column(col)[i].as_py() for col in table.column_names}
                token_metrics.append(
                    TokenMetrics(
                        token_id=trow["token_id"],
                        token_text=trow["token_text"],
                        token_index=trow["token_index"],
                        probability=trow["probability"],
                        log_probability=trow["log_probability"],
                        entropy=trow["entropy"],
                        probability_margin=trow["probability_margin"],
                        cumulative_log_prob=trow["cumulative_log_prob"],
                    )
                )

        return Trajectory(
            sample_id=data["sample_id"],
            dataset=data["dataset"],
            model_id=data["model_id"],
            model_version=data["model_version"],
            experiment_id=data["experiment_id"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            prompt=data["prompt"],
            prompt_token_count=data["prompt_token_count"],
            generated_text=data["generated_text"],
            extracted_answer=data["extracted_answer"],
            is_correct=data["is_correct"],
            generation_tokens=data["generation_tokens"],
            temperature=data["temperature"],
            seed=data["seed"],
            prompt_template=data["prompt_template"],
            token_metrics=token_metrics,
            available_signals=json.loads(data["available_signals"]) if data["available_signals"] else [],
            metadata=json.loads(data["metadata"]) if data["metadata"] else {},
        )

    def close(self) -> None:
        self._conn.close()
