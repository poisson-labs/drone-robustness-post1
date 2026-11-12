# Where does a trained drone controller break?

The code, the stored flights and the trained policies behind post 1 of Poisson Labs' Drone Robustness Study. Sixteen
reinforcement-learning policies were trained to fly a simulated Crazyflie along a figure-eight in calm air, then flown
through wind gusts from 16 directions at 2 to 16 m/s, at two points on the loop. A cascaded PID built on the Crazyflie
firmware's structure and gains and a tuned geometric controller flew the same gusts on identical air, along with two
references: a controller handed the true wind, and a privileged oracle. With a re-check on fresh seeds, that is 174,080
flights, and for every direction the study found v50, the gust speed at which half of the flights crash.

**The short version.** The policies break between 8.3 and 11.5 m/s depending on direction, and they are hardest hit by
headwinds from below. Their average sits above the PID in 8 of 16 directions, below it in 7 and level to two decimals in 1,
and below the geometric controller in all 16. All 53,825 crashes on the main grid, across the twenty flyers, were the drone
leaving a 0.5 m tube around the point it was tracking; none tumbled or hit the floor first. Before anything was drawn, every
published number was reproduced from the raw flights by an independent re-implementation that imports none of this code.

- The post: *Where does a trained drone controller break?* <!-- P1c-site: the post's URL on poissonlabs.ai -->
- The numbers: [`RESULTS-P1.md`](RESULTS-P1.md), with its check sheet [`CHECK-P1.md`](CHECK-P1.md)
- The example, *Where does my controller break?*: [`examples/where-does-my-controller-break/`](examples/where-does-my-controller-break/),
  walked through on the docs <!-- P1c-site: the docs example page's URL -->

Everything here is in simulation: one drone model, one air model, one trajectory and these gusts. See *What this does not show* below.

This repository is the part of the study's code that post 1 rests on, cut from the study's private repository on 2026-11-10
with a fresh history. The study continues: its second post retrains policies on the failures found for them and rechecks the
result, and that code, its tests and its data stay private until then. A few modules and test files held both posts' code; each
of those is carried with the second post's part removed and says so at its top, and the kept lines are unchanged. Two lines of
the stack were changed so a clone works on its own: the Crazyflie meshes are read from `assets/cf2` here, where the study
fetched them from the simulator's source. Nothing else in the code differs from what flew.

## What is here

| Path | What |
|---|---|
| `src/drone_study/` | The simulation stack (vehicle, air model, gust, turbulence and sensor tapes, rate loop, mixer and motors, sensing, crash rules, episode loop), the two classical controllers with the wind-only reference and the feasibility oracle, the calm-air training and gating code (reward, PPO settings, training environment, the bar, the yaw gate, the motor line), the boundary sweep, section 9's estimator and the writer of `RESULTS-P1.md` |
| `data/pi0/` | The sixteen trained policies, π0: each lineage's checkpoint at B0 = 5M steps, its training manifest and its learning curve, as the study flew them |
| `data/4d-lineages.json`, `data/yaw-weight.json`, `data/tuning-pick.json` | The lineage file (the calm-air readings of every seed tried, and which passed), the heading-weight pick, and the geometric controller's gain pick |
| `data/rows/` | The stored flights: 60 Prove runs, 174,080 episodes (Zone B with its 16 m/s extension, 168,960, and the anchor slice, 5,120), one row per episode, as the study flew them; `data/rows-hashes.json` is the hash of every file, recorded when they were flown |
| `RESULTS-P1.md`, `CHECK-P1.md` | Every number of post 1 (N1 to N10), as the writer regenerates them, and the sheet the independent checker worked from, which also states the rules the code follows where the spec left a choice |
| `tests/` | The frozen tests the stack, the controllers, the training code and the boundary sweep were built against |
| `scripts/` | What flew and verified the rows, trained and gated the policies, and picked the gains |
| `examples/` | The example: fly your own controller through the same gusts and draw its map beside the study's |
| `assets/cf2/` | The Crazyflie meshes the simulator needs (MIT, see `THIRD_PARTY_NOTICES.md`) |

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/). From a clone:

```console
uv sync                  # the stack, the controllers, Prove, MuJoCo: enough to fly the classical controllers and reproduce the numbers
uv sync --group gate     # adds PyTorch and Stable-Baselines3 (pinned): needed to fly or train the policies
```

Every dependency is pinned in `pyproject.toml` and `uv.lock`. The simulator, `mj-drones-gym`, is pinned by commit; Prove is
`poisson-prove==0.1.2`, the version that produced the results. Nothing else is fetched: the meshes are in `assets/cf2`.

## Reproduce the numbers from the stored flights

`RESULTS-P1.md` is written by one command from the stored rows, the lineage file and the gain pick. First check that the rows
are the ones the study flew, then regenerate and compare:

```console
uv run python scripts/check_rows.py
uv run python -m drone_study.boundary.writer --store data/rows --bars data/4d-lineages.json --tuning data/tuning-pick.json --out out
shasum -a 256 out/RESULTS-P1.md RESULTS-P1.md
```

Both hashes are `db173c86fbcffe86290ba67a68fbd572f992602d120292dbb92139e97d405bec`.
 `CHECK-P1.md` states each number's
definition and the rules the code follows where the spec left a choice, so the numbers can also be recomputed without
importing anything of ours, which is what the independent check did: 174,827 numbers compared, all agree.

To re-fly the rows rather than read them, `scripts/fly_post1.py --store STORE --runs-root data/pi0` flies all twenty flyers
through Zone B and the anchor slice (about 24 core-hours; it needs the `gate` group for the policies), and
`scripts/verify_pairing.py` checks that every flyer met the same seeds in every cell. The classical controllers reproduce
their stored rows bit for bit; see the example's `compare_with_study.py`.

## Run the example

```console
uv run python examples/where-does-my-controller-break/where_does_my_controller_break.py --flyer pid --quick --jobs 2
uv run python examples/where-does-my-controller-break/where_does_my_controller_break.py --flyer my_controller:rollout --quick --jobs 2
```

The first flies the cascaded PID through a headwind and a tailwind on the study's seeds (1,056 episodes, a few minutes on two
cores) and draws its v50 beside the stored maps, in `boundary-pid.svg` beside the script; a second run reads what Prove stored
and flies nothing. The second does the same for the stand-in controller in `my_controller.py`, which is yours to replace: a
controller is a class with `reset()` and `act(view)` that returns a thrust and three body rates, wrapped with `rollout_for` and
named to the script as `module:name`. Give each version of your controller its own `--store` and `--out`: the script looks
runs up by the flyer's name, so after an edit a run into the same store reads the old flights and flies nothing. Without `--quick` the script flies the eight level directions and draws a full ring;
`--all` flies the study's whole grid of sixteen. `--flyer pi0-7` flies one of the sixteen policies (it needs the `gate` group).
`compare_with_study.py --flyer pid` checks, row by row, that what you flew equals what the study stored. The script's
docstring has the rest.

## The policies

`data/pi0/nominal-<seed>/checkpoint-5000000.zip` is a Stable-Baselines3 PPO checkpoint; `manifest.json` beside it records the
reward it trained on, every PPO setting, its seed and its reset log; `curve.csv` is its learning curve. The sixteen seeds are
1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 18, 19, 20 and 21: of 21 seeds tried, four failed the yaw gate and one the motor line
(`data/4d-lineages.json` holds every seed's readings). The study's code reads the policies from `.work/4d/runs`;
`scripts/link_pi0.sh` links that path to `data/pi0`.

To fly one: `--flyer pi0-7` in the example, or `DRONE_STUDY_CHECKPOINT=data/pi0/nominal-7/checkpoint-5000000.zip` with
the Prove target `drone_study.training.targets:policy`. To train one the study's way, `scripts/train_run.py --role nominal
--nominal-seed 1 --k-yaw 0.032 --budget 5000000 --checkpoints 5000000 OUT` (about eleven hours on one core). Training
gives a new policy; the sixteen here are the ones the study flew, and their hashes are in their manifests.

## Tests

```console
uv sync --group gate --group dev
scripts/link_pi0.sh
uv run pytest
```

1,499 cases, all passing, in about six minutes on a laptop. They are the frozen tests the stack, the controllers, the calm-air
training and gating code and the boundary sweep were built against (one file per invariant or rule, with the numbers each takes
from the spec beside it), plus 3B's own checks in `tests_3b/` (`uv run pytest tests_3b`).
The case that loads the sixteen real checkpoints needs the `gate` group and the link above; without them it is skipped.

## What this does not show

- Anything about real flight. A boundary in m/s here does not carry to a real Crazyflie, a larger vehicle, another task or
  another kind of policy. Above about 6 m/s of airspeed the air model is an extrapolation.
- That learned control is worse or better than classical control. These policies never saw wind in training, on purpose,
  so breaking in wind is expected of them. The controllers show how hard the wind is.
- That the feasibility oracle is a controller anyone could fly, or that its ring is the true limit of flight. It reads what no
  real drone knows.
- That a direction with no crash on the grid is safe, or that any flyer is robust. Twenty-four episodes a cell cannot show it.
- A test between a policy and a controller. There is none: the study's analysis plan, frozen before any gust was flown,
  defines no such comparison, and none was added afterwards.
- That the map holds on other trajectories, timings or air models. One trajectory, two timings, one air model.
- Anything about retraining a policy on its failures, or about finding failures by search. That is the study's second post,
  and its code and data are not in this repository.

## Licence and citation

The code, the stored flights and the trained policies are under the MIT licence (`LICENSE`). The simulator and the
Crazyflie meshes carry their own MIT notices, in `THIRD_PARTY_NOTICES.md`. To cite, see `CITATION.cff`.
