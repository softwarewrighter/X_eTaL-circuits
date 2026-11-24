"""Experiment files (experiments/*.toml) load, and bad ones are refused by name."""

from pathlib import Path

import pytest

from circuits_adapter import experiment, model_revisions

INDUCTION = Path(__file__).resolve().parents[2] / "experiments" / "induction.toml"


def test_the_induction_experiment_loads():
    e = experiment.load(INDUCTION)
    assert (e.model, e.length) == ("gpt2", 16)
    assert e.revision == model_revisions.GPT2_REVISION
    assert e.finding_seeds == tuple(range(10))
    assert e.heldout_seeds == tuple(range(100, 110))
    assert "zero-ablation" in e.interventions


def edited(tmp_path: Path, old: str, new: str) -> Path:
    text = INDUCTION.read_text()
    assert old in text
    p = tmp_path / "e.toml"
    p.write_text(text.replace(old, new))
    return p


@pytest.mark.parametrize(
    "old,new,says",
    [
        ('heldout_seeds = ["100"', 'heldout_seeds = ["9"', "both a finding seed"),
        ('metric = ["logit-difference"]', 'metric = ["accuracy"]', "metric"),
        ('length = ["16"]', 'length = ["sixteen"]', "length"),
        ('length = ["16"]', "length = [16]", "list of strings"),
        ('corruption = ["fresh-second-copy"]\n', "", "missing corruption"),
        ('heldout_seeds = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109"]',
         "heldout_seeds = []", "both be given"),
    ],
)  # fmt: skip
def test_bad_experiments_are_refused(tmp_path, old, new, says):
    with pytest.raises(experiment.ExperimentError, match=says):
        experiment.load(edited(tmp_path, old, new))
