"""Every number the boundary tests take from the frozen spec, each with the section it comes from.

The spec is "2026-11-04 Drone Robustness Study - Spec" (frozen 2026-11-05, read with its log's addenda). A number that
is not here is derived in a test by a formula the test writes out, or is a ruling listed in OWED
(docs/boundary-invariant-table.md). Nothing here is imported from drone_study.
"""
# ---- section 3: Zone B, the anchor slice, the extension ----------------------------------------------------------
CASE = "fig8-faster"
HORIZON_AZ = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
OFF_HORIZON_AZ = (0.0, 90.0, 180.0, 270.0)
OFF_HORIZON_EL = (35.0, -35.0)                        # "plus 0, 90, 180, 270 at +-35"
HORIZON = tuple((az, 0.0) for az in HORIZON_AZ)
OFF_HORIZON = tuple((az, el) for el in OFF_HORIZON_EL for az in OFF_HORIZON_AZ)
DIRECTIONS = HORIZON + OFF_HORIZON                    # 16
SPEEDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0)
EXTENSION_SPEED = 16.0
SPEEDS_EXTENDED = SPEEDS + (EXTENSION_SPEED,)
PHASES = (0.0, 0.125)
ZONE_B_REPLICATES = 24                                # section 7: 16 x 10 x 2 x 24
ANCHOR_REPLICATES = 16                                # section 3: "16 episodes each"
N_CELLS = 16 * 10 * 2                                 # 320 (section 20.1, N1)
N_ANCHOR_CELLS = 16
EXTENSION_AT = 12.0                                   # the first reading is on the 12 m/s rows
EXTENSION_UNDER = 0.8                                 # "under 80%"
ANCHOR = {                                            # section 3: (azimuth, phase) -> speed
    (0.0, 0.0): 4.0, (45.0, 0.0): 6.0, (90.0, 0.0): 8.0, (135.0, 0.0): 10.0,
    (180.0, 0.0): 4.0, (225.0, 0.0): 6.0, (270.0, 0.0): 8.0, (315.0, 0.0): 10.0,
    (0.0, 0.125): 8.0, (45.0, 0.125): 10.0, (90.0, 0.125): 4.0, (135.0, 0.125): 6.0,
    (180.0, 0.125): 8.0, (225.0, 0.125): 10.0, (270.0, 0.125): 4.0, (315.0, 0.125): 6.0,
}

# ---- section 10 item 6: the salts; first 16 hex of sha256("drone-robustness-2026-11-04:<zone>") ----------------------
SALT_PREFIX = "drone-robustness-2026-11-04:"
SALTS = {"search": "c50ed4d8d7752332", "A": "3f3713da7fa65bea", "B": "9c3d4a1e317d59fa", "C": "df4a448fc1610db4",
         "D": "81e8521fd8bc0f85", "S1": "ea518d7c9fa57d5f", "S2": "fd2cd7470f8f5756", "S3": "ecb71cd5dabd1958",
         "anchor": "644f7723497d4cae", "bar": "76e7b463820f72ad"}
ZONE_B_SALT = SALTS["B"]
ANCHOR_SALT = SALTS["anchor"]

# ---- section 20.1: the flyers; section 14 and 20.1: the episode counts -------------------------------------------------
PI0_SEEDS = tuple(range(1, 17))
PI0 = tuple(f"pi0-seed-{s}" for s in PI0_SEEDS)
CONTROLLERS = ("pid", "geometric", "wind_only", "oracle")
FLYERS = PI0 + CONTROLLERS
B0 = 5_000_000                                        # 4B, ruling 4B-2: the value 5A's synthetic rows and N9 fixtures were built at. The flyers
                                                      # and the real N9 read B0 from 4D-B's pick file, never this pin (4DR-5, the 2026-11-08 addendum)
YAW_WEIGHT_LADDER = (0.004, 0.008, 0.016, 0.032, 0.064)   # the yaw addenda (4D-A): the base ladder and its extension
CRASH_PENALTIES = {0.004: 20.0, 0.008: 20.0, 0.016: 20.0, 0.032: 22.0, 0.064: 26.0}   # 4DR-3, per weight
RUNGS = (5_000_000, 10_000_000, 20_000_000, 40_000_000)
LINEAGES_FILE = "data/4d-lineages.json"               # 4D-B's bars file with the yaw readings and the selection (4DR-5)
PICK_FILE = "data/yaw-weight.json"                    # 4D-B's pick: weight, B0, crash penalty
COUNTS = {"zone_b_pi0": 122_880, "zone_b_pid_geometric": 15_360, "zone_b_wind_only": 7_680, "zone_b_oracle": 7_680,
          "anchor": 5_120, "extension": 15_360}       # section 14, 12 row 5B, 20.1
COUNTS["zone_b"] = 153_600                            # 122,880 + 15,360 + 7,680 + 7,680

# ---- section 20.1, 20.2, 20.4: what post 1 may print -----------------------------------------------------------------
PROVE_WORDS = ("safer", "worse", "trade", "no significant change", "too small to tell")
N_NUMBERS = tuple(f"N{i}" for i in range(1, 11))
CRASH_CLASSES = ("ground", "tilt", "radius")
S6_NAMES = ("tilt-75", "tilt-90", "radius-0.3", "radius-0.75", "radius-1.0", "ground-and-tilt")

# ---- section 20.3 point 9, ruling 5A-4: Prove for post 1 -------------------------------------------------------------
PROVE_VERSION = "0.1.2"
