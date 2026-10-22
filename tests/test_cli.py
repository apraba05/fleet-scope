import json

import pytest

from fleetscope.cli import main
from tests.conftest import SAMPLE_ROOT


def test_demo_end_to_end(tmp_path, capsys):
    lake = tmp_path / "lake"
    main(
        [
            *("demo", "--lake", str(lake), "--vehicles", "8", "--minutes", "5", "--no-fsync"),
            *("--comma2k19", str(SAMPLE_ROOT), "--clones", "3", "--budget-gb", "0.5"),
        ]
    )
    out = capsys.readouterr().out
    local = json.loads(out[: out.index("}\n") + 1])
    assert local["exact"] and local["lake_rows"] > 0
    assert "diverse:" in out and "tp_labels" in out
    for table in ("telemetry", "scenarios", "trip_metrics", "selection", "ground_truth"):
        assert (lake / table).exists(), table

    capsys.readouterr()
    main(["sql", "--lake", str(lake), "SELECT count(DISTINCT source) AS n FROM telemetry"])
    assert capsys.readouterr().out.split()[-1] == "3"  # synthetic, comma2k19, its clones


def test_no_source_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["local", "--lake", str(tmp_path), "--vehicles", "0"])


def test_sql_error_is_reported_without_traceback(tmp_path):
    with pytest.raises(SystemExit, match="SQL error"):
        main(["sql", "--lake", str(tmp_path), "SELECT no_such_column FROM telemetry"])
