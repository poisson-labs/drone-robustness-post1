#!/bin/sh
# The study's code reads the sixteen trained policies from .work/4d/runs (session 4D-B's raw output directory, kept out of
# git in the study repository). This repository carries them in data/pi0. Running this once links the two, so the frozen
# tests' real-checkpoint case runs and scripts that take --runs-root find the default. Re-running is safe.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$HERE/.work/4d"
if [ -e "$HERE/.work/4d/runs" ] && [ ! -L "$HERE/.work/4d/runs" ]; then
  echo ".work/4d/runs exists and is not a link; leaving it alone" >&2
  exit 1
fi
ln -sfn ../../data/pi0 "$HERE/.work/4d/runs"
echo ".work/4d/runs -> data/pi0 ($(ls "$HERE/data/pi0" | wc -l | tr -d ' ') policies)"
