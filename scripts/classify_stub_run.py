"""Read a pytest JUnit report and say, for each test, whether it passed, failed on an assertion, or died on something else.
Usage: classify_stub_run.py REPORT.xml   (exit 1 if any test passed or died on a non-assertion error)"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

root = ET.parse(sys.argv[1]).getroot()
passed, other, assertion, errors = [], [], [], []
kinds = defaultdict(Counter)
for case in root.iter("testcase"):
    name = f'{case.get("classname", "").split(".")[-1]}::{case.get("name")}'
    f, e = case.find("failure"), case.find("error")
    node = f if f is not None else e
    if node is None:
        passed.append(name); continue
    text = (node.get("message") or "") + (node.text or "")
    first = (node.get("message") or "").split(":")[0].strip()
    is_assert = "AssertionError" in (node.text or "")[:4000] or first.startswith("assert") or "Failed: DID NOT RAISE" in (node.get("message") or "")
    (assertion if is_assert and e is None else other).append((name, (node.get("message") or "")[:160]))
    kinds[name.split("::")[0]]["assert" if is_assert and e is None else "other"] += 1
print(f"tests: {len(passed) + len(assertion) + len(other)}  passed: {len(passed)}  failed on an assertion: {len(assertion)}  other: {len(other)}")
for k in sorted(kinds):
    print(f"  {k}: assertion {kinds[k]['assert']}, other {kinds[k]['other']}")
if passed:
    print("\nPASSED on the stub:")
    for n in passed: print("  ", n)
if other:
    print("\nFAILED for a reason other than an assertion:")
    for n, m in other[:80]: print("  ", n, "|", m)
sys.exit(1 if passed or other else 0)
