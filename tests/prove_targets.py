"""Two test policies, each exposed as a Prove --target (section 13: `--target` names the policy)."""
from drone_study.rollout import rollout_for
from drone_study.stack import ScriptedController

WEIGHT = 0.027 * 9.81

policy_a = rollout_for(ScriptedController(lambda view: [WEIGHT, 0.0, 0.0, 0.0]))
policy_a_again = rollout_for(ScriptedController(lambda view: [WEIGHT, 0.0, 0.0, 0.0]))
policy_b = rollout_for(ScriptedController(lambda view: [WEIGHT * 1.05, 0.0, 0.1, 0.05]))
