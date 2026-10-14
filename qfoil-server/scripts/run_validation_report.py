"""Run the full test suite and print a per-file pass/fail/skip summary."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

RESULT_RE = re.compile(r"^(tests/\S+\.py)::.* (PASSED|FAILED|SKIPPED|ERROR)")
# Module-level skips (e.g. pytest.importorskip at import time) collect zero
# items and never produce a per-test line above; they only show up here.
SUMMARY_SKIP_RE = re.compile(r"^SKIPPED \[\d+\] (tests/\S+\.py):\d+:")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "-rs"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start

    by_file: dict[str, Counter] = defaultdict(Counter)
    for line in result.stdout.splitlines():
        match = RESULT_RE.match(line)
        if match:
            by_file[match.group(1)][match.group(2)] += 1

    for line in result.stdout.splitlines():
        match = SUMMARY_SKIP_RE.match(line)
        if match and match.group(1) not in by_file:
            by_file[match.group(1)]["SKIPPED"] += 1

    totals: Counter = Counter()
    header = f"{'file':<45} {'passed':>7} {'failed':>7} {'skipped':>8} {'error':>6}"
    print(header)
    for file, counts in sorted(by_file.items()):
        print(
            f"{file:<45} {counts['PASSED']:>7} {counts['FAILED']:>7} "
            f"{counts['SKIPPED']:>8} {counts['ERROR']:>6}"
        )
        totals.update(counts)

    print("-" * 80)
    print(
        f"{'TOTAL':<45} {totals['PASSED']:>7} {totals['FAILED']:>7} "
        f"{totals['SKIPPED']:>8} {totals['ERROR']:>6}"
    )
    print(f"elapsed: {elapsed:.2f}s")

    if result.returncode != 0 and not totals["FAILED"] and not totals["ERROR"]:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
