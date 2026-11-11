# Third-party notices

This repository's own code, data and trained policies are under the MIT licence in `LICENSE`. It also carries, or is built on, the following.

## MuJoCo-Drones-Gym (the simulator)

`mj-drones-gym`, by tau-intelligence, at commit `ca170dc6760d422e47ef46fb3af46f8cd20410eb`
(https://github.com/tau-intelligence/MuJoCo-drones-gym). The study pins it by commit in `pyproject.toml` and builds its
Crazyflie model through the simulator's own `BaseAviary`. At that commit the simulator declares the **MIT licence** in its
`pyproject.toml` (`license = {text = "MIT"}`) and in its README; it has no `LICENSE` file at its root and no copyright line
for its own code. No file of the simulator's code is copied here. The physical parameters of the vehicle are credited by
the simulator to Förster 2015.

## Crazyflie 2 meshes (`assets/cf2/`)

The mesh files in `assets/cf2/` are the simulator's `multi_drone_mujoco/assets/cf2/*.obj`, carried here because the
installed wheel of the simulator omits them. They are under the **MIT licence, copyright (c) 2024 whoenig**; the licence file
is carried beside them as `assets/cf2/LICENSE`. The simulator's code says they were copied from MuJoCo Menagerie's
`bitcraze_crazyflie_2` model (https://github.com/google-deepmind/mujoco_menagerie).

## MuJoCo

MuJoCo 3.17.0 (https://github.com/google-deepmind/mujoco), **Apache License 2.0**. Installed from PyPI, not carried here.

## The Crazyflie firmware's controller structure and gains

The cascaded PID in `src/drone_study/pid.py` is the study's own implementation of the structure and gains of Bitcraze's
`crazyflie-firmware` position and attitude controllers (`position_controller_pid.c` and the attitude PID, at commit
`e828ed2`; https://github.com/bitcraze/crazyflie-firmware, GPL-3.0). No firmware code is copied; the gains and the limits are
taken as numbers and cited in the module.

## Python packages

Installed from PyPI by `uv sync` and not carried here: NumPy (BSD-3), SciPy (BSD-3), Gymnasium (MIT), pandas (BSD-3),
PyArrow (Apache-2.0), Poisson Prove (`poisson-prove`, Poisson Labs), and, in the `gate` group, PyTorch (BSD-3) and
Stable-Baselines3 (MIT). Each carries its own licence in its distribution.
