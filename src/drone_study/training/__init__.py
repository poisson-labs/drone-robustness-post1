"""Training: the reward, the samplers, the training environment, the feasibility rule and table, reset logging,
budgets, the PPO settings, the calm-air bar, the gain grid and a trained network as a flyer (spec sections 2, 5, 17.2;
session 4B's row in section 12).

Session 4A shipped this subpackage as stubs with the frozen training tests; session 4B built it against them, without
changing a test (docs/training-invariant-table.md section 5 is the interface; docs/nominal-4b.md the build notes). The
trainer itself is scripts/train_run.py; the flyers Prove sweeps are in `targets`.
"""
