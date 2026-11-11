"""Invariant 8 (spec sections 6, 17.9, 12): the tapes are arrays complete before step 1; changing the policy leaves all
three unchanged, to the bit; changing one stream's use leaves the others unchanged.
"""
import math
import re

import numpy as np
import pytest

import spec_numbers as S
from drone_study.condition import Condition
from drone_study.sensing import SensingConfig
from drone_study.stack import ControllerView, ScriptedController, StackEnv
from drone_study.streams import build_tapes

COND = Condition(case="fig8-faster", az=45.0, el=0.0, speed=8.0, phase=0.0, turb=0.3)


def arrays(tapes):
    s = tapes.sensor
    return {"start.position": tapes.start.position, "start.velocity": tapes.start.velocity, "start.tilt": tapes.start.tilt,
            "turbulence": tapes.turbulence, "sensor.gyro_white": s.gyro_white, "sensor.pos_offset": s.pos_offset,
            "sensor.pos_ar": s.pos_ar, "sensor.vel_ar": s.vel_ar, "sensor.att_ar": s.att_ar, "sensor.rate_white": s.rate_white}


def group(name):
    return name.split(".")[0]


def assert_same(a, b, groups):
    for k, v in arrays(a).items():
        if group(k) in groups:
            assert np.array_equal(v, arrays(b)[k]), k
            assert v.tobytes() == arrays(b)[k].tobytes(), k


def assert_differ(a, b, groups):
    for g in groups:
        assert any(not np.array_equal(v, arrays(b)[k]) for k, v in arrays(a).items() if group(k) == g), g


def hover(view):
    return [S.WEIGHT, 0.0, 0.0, 0.0]


def busy(view):
    return [0.8 * S.WEIGHT + 0.3 * S.WEIGHT * math.sin(11 * view.t), 3 * math.sin(5 * view.t), 2 * math.cos(3 * view.t), 1.0]


def test_inv8_the_tapes_are_arrays_complete_before_the_first_step():
    env = StackEnv(COND, 7)                                           # not reset, not stepped
    a = arrays(env.tapes)
    assert a["turbulence"].shape == (S.N_SUBSTEPS, 3)
    assert a["sensor.gyro_white"].shape == (S.N_SUBSTEPS, 3)
    for k in ("sensor.pos_ar", "sensor.vel_ar", "sensor.att_ar", "sensor.rate_white"):
        assert a[k].shape == (S.N_STEPS, 3)
    for k, v in a.items():
        assert v.dtype == np.float64 and np.all(np.isfinite(v)), k
        assert np.any(v != 0.0), k                                    # a tape with something in it
    d = env.tapes.digests()
    assert set(d) == {"start", "turbulence", "sensor"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", x) for x in d.values())


def test_inv8_changing_the_policy_leaves_all_three_tapes_unchanged_to_the_bit():
    fresh = build_tapes(COND, 7)
    again = build_tapes(COND, 7)
    assert_same(fresh, again, {"start", "turbulence", "sensor"})
    assert fresh.digests() == again.digests() and all(fresh.digests().values())
    seen = {}
    for name, fn in (("hover", hover), ("busy", busy)):
        env = StackEnv(COND, 7)
        ctrl = ScriptedController(fn)
        reading = env.reset()
        for _ in range(300):
            view = ControllerView(env.time, reading, np.zeros((5, 3)), None)
            result = env.step(ctrl.act(view))
            reading = result.reading
            if result.done:
                break
        seen[name] = env.tapes
        assert_same(env.tapes, fresh, {"start", "turbulence", "sensor"})
        assert env.tapes.digests() == fresh.digests()
    assert_same(seen["hover"], seen["busy"], {"start", "turbulence", "sensor"})
    other = build_tapes(COND, 8)                                      # the control: another seed changes every stream
    assert_differ(fresh, other, {"start", "turbulence", "sensor"})
    assert set(a != b for a, b in zip(fresh.digests().values(), other.digests().values())) == {True}


def test_inv8_changing_the_turbulence_leaves_the_start_and_sensor_streams_unchanged():
    a = build_tapes(COND, 9)
    b = build_tapes(Condition(case="fig8-faster", az=45.0, el=0.0, speed=8.0, phase=0.0, turb=1.5), 9)
    assert_same(a, b, {"start", "sensor"})
    assert_differ(a, b, {"turbulence"})
    assert a.digests()["start"] == b.digests()["start"] and a.digests()["sensor"] == b.digests()["sensor"]
    assert a.digests()["turbulence"] != b.digests()["turbulence"]


def test_inv8_changing_how_the_sensor_stream_is_used_leaves_the_others_unchanged():
    base = StackEnv(COND, 9).tapes
    for sensing in (SensingConfig(gyro_scale=0.0), SensingConfig(outer_scale=0.0), SensingConfig(gyro_scale=9.0, outer_scale=9.0)):
        assert_same(StackEnv(COND, 9, sensing=sensing).tapes, base, {"start", "turbulence", "sensor"})
    delayed = Condition(case="fig8-faster", az=45.0, el=0.0, speed=8.0, phase=0.0, turb=0.3, delay=60)
    assert_same(StackEnv(delayed, 9).tapes, base, {"start", "turbulence", "sensor"})
    assert np.any(base.turbulence != 0) and np.any(base.sensor.pos_ar != 0)                  # not three empty tapes agreeing


def test_inv8_calm_air_means_a_zero_turbulence_tape_and_the_other_streams_still_run():
    calm = build_tapes(Condition.calm("circle-fast"), 4)
    assert np.all(calm.turbulence == 0.0)                              # section 2: "No wind of any kind"
    assert np.any(calm.sensor.pos_ar != 0) and np.any(calm.sensor.gyro_white != 0) and np.any(calm.start.position != 0)
    windy = build_tapes(Condition(case="circle-fast", turb=0.3), 4)
    assert_same(calm, windy, {"start", "sensor"})


def test_inv8_the_start_stream_has_the_spread_section_17_9_gives():
    starts = [build_tapes(COND, s).start for s in range(2000)]
    for attr, sigma in (("position", S.START_POS_SIGMA), ("velocity", S.START_VEL_SIGMA), ("tilt", S.START_TILT_SIGMA)):
        x = np.stack([getattr(s, attr) for s in starts])
        se = sigma / math.sqrt(2 * x.size)
        assert abs(x.std() - sigma) <= S.SIGMA_K * se, (attr, x.std(), sigma)
        assert abs(x.mean()) <= S.SIGMA_K * sigma / math.sqrt(x.size)


def _heads(t):
    """The first numbers each stream draws, labelled by stream."""
    return {"start": np.r_[t.start.position, t.start.velocity, t.start.tilt],
            "turbulence": np.r_[t.turbulence[0], t.turbulence[1], t.turbulence[2500]],
            "sensor": np.r_[t.sensor.gyro_white[0], t.sensor.gyro_white[1], t.sensor.pos_offset, t.sensor.pos_ar[0],
                            t.sensor.vel_ar[0], t.sensor.att_ar[0], t.sensor.rate_white[0]]}


def test_inv8_the_streams_are_independent_of_one_another():
    heads = [_heads(build_tapes(COND, s)) for s in range(400)]
    cols = {k: np.stack([h[k] for h in heads]) for k in heads[0]}
    worst = 0.0
    for a, b in (("start", "turbulence"), ("start", "sensor"), ("turbulence", "sensor")):
        A, B = cols[a], cols[b]
        assert A.std(0).min() > 0 and B.std(0).min() > 0               # each number really varies from seed to seed
        corr = np.abs((A - A.mean(0)).T @ (B - B.mean(0)) / (len(A) * A.std(0)[:, None] * B.std(0)[None, :] + 1e-300))
        worst = max(worst, float(corr.max()))
    assert worst <= S.SIGMA_K / math.sqrt(400) * 1.3, worst              # no leading number of one stream tracks another's
