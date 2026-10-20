"""Exercise batch exit paths in cmd.exe without installing packages or downloading weights."""

import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="requires Windows cmd.exe")
@pytest.mark.parametrize("name", ["init", "check", "demo", "pretrained"])
def test_batch_launchers_pause_and_preserve_status(tmp_path, name):
    source = (Path(__file__).resolve().parents[1] / "scripts" / f"{name}.bat").read_text()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    target = scripts / f"{name}.bat"
    lock = tmp_path / "uv.lock"
    for case in ("success", "invalid_backend", "missing_lock", "missing_uv", "command_error"):
        lock.write_text("")
        if case == "missing_lock":
            lock.unlink()
        # Replace external commands only; keep actual batch control flow and pause.
        stub = re.sub(r"(?m)^(\s*)uv [^\n]+", rf"\1cmd /d /c exit {7 if case == 'command_error' else 0}", source)
        stub = stub.replace("where uv >nul 2>&1", f"cmd /d /c exit {1 if case == 'missing_uv' else 0}")
        target.write_text(stub)
        backend = "invalid" if case == "invalid_backend" else "cpu"
        result = subprocess.run(
            ["cmd", "/d", "/c", str(target), backend],
            input=b"\n", capture_output=True, timeout=10,
        )
        expected = 0
        if case in {"invalid_backend", "missing_lock"}:
            expected = 1
        if case == "missing_uv" and name in {"init", "check"}:
            expected = 1
        if case == "command_error":
            expected = 1 if name in {"init", "check"} else 7
        assert result.returncode == expected, result.stdout + result.stderr
        marker = f"Script finished with exit code {expected} (0 = success).".encode()
        assert result.stdout.count(marker) == 1
        # pause prints a localized prompt after the result; do not assume its language.
        assert result.stdout.split(marker)[1].strip()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows cmd.exe")
@pytest.mark.parametrize("name", ["retrain_head", "crossval"])
def test_cached_head_batch_launchers(tmp_path, name):
    source = (Path(__file__).resolve().parents[1] / "scripts" / f"{name}.bat").read_text()
    directory = tmp_path / "scripts"
    directory.mkdir()
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "features.pt").write_text("stub only")
    (tmp_path / "profiles.json").write_text("stub")
    for code in (0, 7):
        stub = re.sub(r"(?m)^uv [^\n]+", f"echo HEAD_COMMAND\ncmd /d /c exit {code}", source)
        stub = stub.replace("where uv >nul 2>&1", "cmd /d /c exit 0")
        script = directory / f"{name}.bat"
        script.write_text(stub)
        result = subprocess.run(["cmd", "/d", "/c", str(script), str(tmp_path / "features.pt"),
                                 "low", str(tmp_path / f"run{code}"), str(tmp_path / f"splits{code}"),
                                 str(tmp_path / "profiles.json")],
                                input=b"\n", capture_output=True, timeout=10)
        assert result.returncode == code, result.stdout + result.stderr
        assert result.stdout.count(b"HEAD_COMMAND") == (2 if name == "crossval" and code == 0 else 1)
        assert f"Script finished with exit code {code} (0 = success).".encode() in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="requires Windows cmd.exe")
@pytest.mark.parametrize("name", ["validate_data", "temporal"])
def test_data_and_image_launchers_pause_on_success_and_failure(tmp_path, name):
    source = (Path(__file__).resolve().parents[1] / "scripts" / f"{name}.bat").read_text()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (tmp_path / "uv.lock").touch()
    (tmp_path / "dataset.json").write_text("{}")
    for code in (0, 7):
        stub = re.sub(r"(?m)^uv [^\n]+", f"cmd /d /c exit {code}", source)
        stub = stub.replace("where uv >nul 2>&1", "cmd /d /c exit 0")
        script = scripts / f"{name}.bat"
        script.write_text(stub)
        args = [str(tmp_path), str(tmp_path / "report")] if name == "validate_data" else ["cv"]
        result = subprocess.run(["cmd", "/d", "/c", str(script), *args],
                                input=b"\n", capture_output=True, timeout=10)
        assert result.returncode == code, result.stdout + result.stderr
        marker = f"Script finished with exit code {code} (0 = success).".encode()
        assert result.stdout.count(marker) == 1
        assert result.stdout.split(marker)[1].strip()
