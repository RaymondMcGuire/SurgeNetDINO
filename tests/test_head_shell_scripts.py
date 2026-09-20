"""Run shell launcher control flow with a stub uv; no models or dependencies installed."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="POSIX bash launcher test")
@pytest.mark.parametrize("name", ["train", "retrain_head", "crossval"])
@pytest.mark.parametrize("code", [0, 7])
def test_launchers_propagate_command_exit_and_arguments(tmp_path, name, code):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts" / f"{name}.sh", scripts / f"{name}.sh")
    (tmp_path / "uv.lock").write_text("")
    source = tmp_path / "source with spaces.dat"
    source.write_text("stub input")
    binary = tmp_path / "bin"
    binary.mkdir()
    stub = binary / "uv"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$STUB_LOG"\nexit "$STUB_CODE"\n')
    stub.chmod(0o755)
    log = tmp_path / "calls.txt"
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "STUB_LOG": str(log), "STUB_CODE": str(code)}
    output = tmp_path / "run with spaces"
    if name == "train":
        args = [str(source), str(output)]
    elif name == "retrain_head":
        args = [str(source), "low", str(output), "cpu"]
    else:
        args = [str(source), str(output), str(tmp_path / "splits"), "low", "", "cpu"]
    result = subprocess.run(["bash", str(scripts / f"{name}.sh"), *args],
                            capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == code, result.stdout + result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == (2 if name in {"train", "crossval"} and code == 0 else 1)
    assert "--locked" in calls[0] and str(source) in calls[0]
    assert f"--output-dir {output / 'cache' if name == 'train' else output}" in calls[0]
    if name != "train":
        assert "--extra cpu" in calls[0] and "--device cpu" in calls[0]
