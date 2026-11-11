#!/bin/sh
# Fetch the simulator's source at the pinned commit into .work/sim-src, so its mesh files exist.
# The installed wheel of mj-drones-gym omits assets/cf2/*.obj; the tests point MJ_DRONES_CF2_ASSETS here.
# The commit is the one pyproject.toml pins. Re-running is safe.
set -eu
SHA=ca170dc6760d422e47ef46fb3af46f8cd20410eb
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEST="$HERE/.work/sim-src"
mkdir -p "$DEST"
cd "$DEST"
[ -d .git ] || git init -q .
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/tau-intelligence/MuJoCo-drones-gym.git
if [ "$(git rev-parse -q --verify HEAD 2>/dev/null || true)" != "$SHA" ]; then
  GIT_TERMINAL_PROMPT=0 git fetch -q --depth 1 origin "$SHA"
  git checkout -q "$SHA"
fi
test "$(git rev-parse HEAD)" = "$SHA"
echo "simulator at $SHA; meshes: $(ls "$DEST/multi_drone_mujoco/assets/cf2"/*.obj | wc -l) files"
