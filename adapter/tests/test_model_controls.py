"""Controls on GPT-2 Small (needs the model extra and the weights)."""

import numpy as np
import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def seed0():
    from pathlib import Path

    from circuits_adapter import ablate, controls, experiment, model

    m = model.load("gpt2")
    exp = experiment.load(Path(__file__).resolve().parents[2] / "experiments" / "induction.toml")
    return controls.Seed(m, exp.length, 0, ablate.twin_means(m, exp), k=5)


def test_a_change_of_length_zero_has_no_effect(seed0):
    import torch

    from circuits_adapter import controls

    hook = (controls.z_name(5), controls.add_to_head(1, torch.zeros(5, 33, 64)))
    ld = controls.batch_ld(seed0.m, seed0.clean, seed0.clean, seed0.corrupt, 16, 5, [hook])
    assert np.all(ld == seed0.ld["clean"])


def test_5_1_s_mean_ablation_beats_its_controls_on_seed_0(seed0):
    """The real effect (about 4.6) against five random changes of the same length."""
    from circuits_adapter import ablate

    real = seed0.ld["clean"] - ablate.mean_ld(
        seed0.m, seed0.clean, seed0.clean, seed0.corrupt, 16,
        [(ablate.z_name(5), ablate.set_head(1, seed0.means[5][:, 1, :]))],
    )  # fmt: skip
    assert real > seed0.control("mean", 5, 1).max()
