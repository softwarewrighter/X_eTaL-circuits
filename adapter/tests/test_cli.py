import tomllib

import pytest

from circuits_adapter import cli


def test_cli_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_capture_arguments_and_defaults():
    a = cli.build_parser().parse_args(["capture", "--out", "runs/x"])
    assert (a.model, a.seed, a.length, a.text, a.layers, a.device) == (
        "gpt2", 0, 50, None, None, "cpu",
    )  # fmt: skip
    assert not a.safetensors


def test_layers_parse_as_a_comma_list():
    a = cli.build_parser().parse_args(["capture", "--out", "o", "--layers", "0,5,11"])
    assert a.layers == [0, 5, 11]


def test_text_and_length_are_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["capture", "--out", "o", "--text", "hi", "--length", "4"])


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 2
    assert "capture" in capsys.readouterr().out


@pytest.mark.model
def test_capture_command_writes_a_capture(tmp_path):
    out = tmp_path / "cap"
    args = ["capture", "--out", str(out), "--length", "4", "--layers", "0,1", "--safetensors"]
    assert cli.main(args) == 0
    meta = tomllib.loads((out / "capture.toml").read_text())
    assert meta["attn_shape"] == ["2", "12", "9", "9"]
    assert (out / "capture.safetensors").exists()


@pytest.mark.model
def test_capture_command_with_text(tmp_path):
    out = tmp_path / "cap"
    assert cli.main(["capture", "--out", str(out), "--text", "Hello world", "--layers", "0"]) == 0
    meta = tomllib.loads((out / "capture.toml").read_text())
    assert meta["prompt"] == ["Hello world"]
    assert meta["prompt_kind"] == ["text"]
    assert meta["tokens"][1:] == ["Hello", " world"]


def test_baseline_arguments():
    args = cli.build_parser().parse_args(
        ["baseline", "--experiment", "e.toml", "--seed", "3", "--out", "runs/b"]
    )
    assert (args.command, args.experiment, args.seed, args.out) == (
        "baseline",
        "e.toml",
        3,
        "runs/b",
    )
    assert cli.COMMANDS["baseline"] is cli.run_baseline


def test_ablate_arguments_and_seed_lists():
    from circuits_adapter import experiment

    args = cli.build_parser().parse_args(["ablate", "--experiment", "e.toml", "--out", "r"])
    assert (args.command, args.seeds) == ("ablate", "finding")
    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    exp = experiment.load(root / "experiments" / "induction.toml")
    assert cli.seeds_of(exp, "heldout") == exp.heldout_seeds
    assert cli.seeds_of(exp, "3,5") == (3, 5)


def test_patch_arguments():
    args = cli.build_parser().parse_args(
        ["patch", "--experiment", "e.toml", "--seeds", "heldout", "--out", "r"]
    )
    assert (args.command, args.seeds) == ("patch", "heldout")
    assert cli.COMMANDS["patch"] is cli.run_patch


def test_controls_arguments():
    args = cli.build_parser().parse_args(["controls", "--experiment", "e.toml", "--out", "r"])
    assert (args.command, args.seeds) == ("controls", "finding")
    assert cli.COMMANDS["controls"] is cli.run_controls


def test_pathpatch_arguments():
    args = cli.build_parser().parse_args(["pathpatch", "--experiment", "e.toml", "--out", "r"])
    assert (args.command, args.seeds) == ("pathpatch", "finding")
    assert cli.COMMANDS["pathpatch"] is cli.run_pathpatch


def test_attribute_and_choose_arguments():
    args = cli.build_parser().parse_args(["attribute", "--experiment", "e.toml", "--out", "r"])
    assert (args.command, args.seeds) == ("attribute", "finding")
    assert cli.COMMANDS["attribute"] is cli.run_attribute
    args = cli.build_parser().parse_args(["choose", "--edges", "x.toml", "--runs", "d"])
    assert (args.edges, args.runs) == ("x.toml", "d")
    assert cli.COMMANDS["choose"] is cli.run_choose


def test_choose_prints_the_chosen_edges(capsys):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    edges_file = root / "experiments" / "induction-edges.toml"
    runs = root / "fixtures" / "gpt2-induction-attribution"
    assert cli.main(["choose", "--edges", str(edges_file), "--runs", str(runs)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 400 and "4.11 5.1.k" in lines
