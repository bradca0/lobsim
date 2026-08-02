"""Experiment-protocol tests: the seed split, result serialisation, and provenance.

The seed split is a scientific claim, not plumbing -- if training and test seeds ever overlap,
every out-of-sample number in the README silently becomes in-sample. It is asserted here rather
than trusted to a comment.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from lobsim import experiment
from lobsim.agents.fqi import FQIConfig, QModel
from lobsim.experiment import (
    TEST_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    progress,
    provenance,
    read_json,
    stage,
    write_json,
)
from lobsim.features import N_FEATURES
from lobsim.policies import AGENT_SIZE, load_policy


class TestSeedProtocol:
    def test_the_three_splits_are_pairwise_disjoint(self) -> None:
        train, validation, test = set(TRAIN_SEEDS), set(VALIDATION_SEEDS), set(TEST_SEEDS)
        assert train & validation == set()
        assert train & test == set()
        assert validation & test == set()

    def test_no_split_contains_duplicates(self) -> None:
        for split in (TRAIN_SEEDS, VALIDATION_SEEDS, TEST_SEEDS):
            assert len(set(split)) == len(split)

    def test_the_splits_are_large_enough_to_say_anything(self) -> None:
        assert len(TRAIN_SEEDS) >= 100
        assert len(VALIDATION_SEEDS) >= 30
        assert len(TEST_SEEDS) >= 100


class TestSerialisation:
    def test_round_trip_through_disk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(experiment, "RAW_DIR", tmp_path)
        write_json("demo.json", {"value": 1.5, "name": "x"})
        payload = read_json("demo.json")
        assert payload["value"] == 1.5
        assert payload["name"] == "x"

    def test_numpy_types_survive_serialisation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results are full of numpy scalars and arrays; json cannot encode them unaided."""
        monkeypatch.setattr(experiment, "RAW_DIR", tmp_path)
        write_json(
            "numpy.json",
            {
                "int": np.int64(7),
                "float": np.float64(2.5),
                "array": np.arange(3, dtype=np.float64),
            },
        )
        payload = read_json("numpy.json")
        assert payload["int"] == 7
        assert payload["float"] == 2.5
        assert payload["array"] == [0.0, 1.0, 2.0]

    def test_unserialisable_objects_are_rejected_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(experiment, "RAW_DIR", tmp_path)
        with pytest.raises(TypeError, match="cannot serialise"):
            write_json("bad.json", {"obj": object()})

    def test_every_result_carries_its_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A number in the README must always be traceable to the code that produced it."""
        monkeypatch.setattr(experiment, "RAW_DIR", tmp_path)
        path = write_json("stamped.json", {"value": 1})
        payload = json.loads(path.read_text())
        assert set(payload["_provenance"]) >= {"git_revision", "python", "platform", "machine"}

    def test_a_missing_result_file_is_an_actionable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(experiment, "RAW_DIR", tmp_path)
        with pytest.raises(SystemExit, match="make reproduce"):
            read_json("absent.json")

    def test_provenance_reports_a_revision(self) -> None:
        assert provenance()["git_revision"]


class TestReporting:
    def test_stage_wraps_without_swallowing_exceptions(self) -> None:
        with pytest.raises(RuntimeError), stage("demo"):
            raise RuntimeError("boom")

    def test_stage_runs_its_body(self) -> None:
        seen = []
        with stage("demo"):
            seen.append(1)
        assert seen == [1]

    def test_progress_does_not_raise_at_the_boundaries(self) -> None:
        progress(0, 10)
        progress(9, 10)
        progress(5, 10, every=1)


class TestPolicyLoading:
    def test_a_saved_policy_round_trips_into_a_working_factory(self, tmp_path: Path) -> None:
        config = FQIConfig()
        model = QModel(config, N_FEATURES)
        model.fit(
            np.random.default_rng(0).normal(size=(200, N_FEATURES)),
            np.random.default_rng(1).integers(0, 16, size=200),
            np.random.default_rng(2).normal(size=200),
        )
        path = tmp_path / "policy.pkl"
        with path.open("wb") as handle:
            pickle.dump({"model": model, "config": config}, handle)

        factory, info = load_policy(path)
        agent = factory()
        assert info["label"] == config.label()
        assert info["feature_groups"] == list(config.feature_groups)
        assert agent.size == AGENT_SIZE  # type: ignore[attr-defined]
        assert agent.epsilon == 0.0  # type: ignore[attr-defined]
