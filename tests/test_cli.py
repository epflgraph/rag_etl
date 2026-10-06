import logging

import pytest

from rag_etl.cli import build_parser, main


@pytest.fixture(autouse=True)
def _fresh_root_logger():
    logger = logging.getLogger()
    handlers = logger.handlers[:]
    logger.handlers.clear()
    yield
    logger.handlers = handlers


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "rag-etl" in capsys.readouterr().out


def test_no_command_fails():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_run_requires_project():
    with pytest.raises(SystemExit) as e:
        main(["run"])
    assert e.value.code == 2


def test_run_flags_parse():
    args = build_parser().parse_args(["run", "--project", "COM202", "--dry", "--refresh", "materialize"])
    assert args.command == "run"
    assert args.project == "COM202"
    assert args.dry is True
    assert args.refresh == "materialize"


def test_unknown_project_errors(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "--project", "NOPE"])
    assert e.value.code == 2
    assert "Unknown project code" in capsys.readouterr().err


def test_known_project_runs(monkeypatch):
    ran = []

    class FakeProject:
        def run(self):
            ran.append(True)

    monkeypatch.setattr("rag_etl.cli.BaseProject", type("FakeBase", (), {"from_code": classmethod(lambda cls, code: FakeProject())}))

    main(["run", "--project", "COM202"])

    assert ran == [True]
