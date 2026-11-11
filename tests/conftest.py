"""Test setup: the simulator's meshes (carried in assets/cf2), and the repo root on sys.path for Prove's --target imports."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "cf2"
os.environ.setdefault("MJ_DRONES_CF2_ASSETS", str(ASSETS))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_sessionstart(session):
    if not (Path(os.environ["MJ_DRONES_CF2_ASSETS"]) / "cf2_0.obj").exists():
        pytest.exit("simulator meshes missing: assets/cf2 should hold them (or run scripts/fetch_sim_assets.sh and set MJ_DRONES_CF2_ASSETS)", 2)
