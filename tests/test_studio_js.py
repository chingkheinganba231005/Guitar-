"""Runs the JavaScript tests of Stem Studio's playback worklet (needs Node)."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_playback_worklet():
    res = subprocess.run(["node", str(Path(__file__).parent / "js" / "worklet.test.cjs")], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stdout + res.stderr
