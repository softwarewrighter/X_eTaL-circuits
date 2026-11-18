import pytest

import circuits_adapter
from circuits_adapter import cli


def test_package_has_version():
    assert circuits_adapter.__version__ == "0.1.0"


def test_cli_runs_without_arguments():
    assert cli.main([]) == 0


def test_cli_version_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
