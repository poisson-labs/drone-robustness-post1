"""Your own controller, as a Prove target:

    uv run python examples/where-does-my-controller-break/where_does_my_controller_break.py --flyer my_controller:rollout --quick --jobs 2

A controller is a class with `reset()` and `act(view)`. `view.reading` is the noisy, delayed state every flyer gets at
100 Hz (position, velocity, rotation, body rate), and `view.reference` is the path's next five points, 0.1 s apart. `act`
returns a collective thrust in newtons and three body rates in rad/s; the study's rate loop, mixer and motors do the rest,
the same for every flyer. See `drone_study.stack.Controller` and `ControllerView`.

This one is a stand-in for yours: the study's geometric controller with softer gains than the study picked. Replace it.
"""
from drone_study.controllers import GeometricController, GeometricGains
from drone_study.rollout import rollout_for

rollout = rollout_for(GeometricController(GeometricGains(kp=6.0, kv=3.5, ki=2.0)))
