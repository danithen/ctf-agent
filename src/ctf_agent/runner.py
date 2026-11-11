"""Runner: owns the benchmark answer, stages contestant files, verifies flags.

The expected flag comes from exactly one place: the official ``nyuctf``
package's ``CTFChallenge.flag``. Nothing else in this codebase reads the
challenge flag, and nothing in the agent's view can reach the benchmark data
(the sandbox cannot see outside the staged workspace). Verification is
reachable only through the agent's ``submit_flag`` tool, so a run can pass
only by submitting the key for the benchmark itself to check.
"""

from __future__ import annotations

import hmac
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from nyuctf.challenge import CTFChallenge
from nyuctf.dataset import CTFDataset


@dataclass
class Target:
    challenge: CTFChallenge
    expected_flag: str     # official benchmark flag (CTFChallenge.flag)

    @property
    def challenge_id(self) -> str:
        return self.challenge.name

    def check(self, candidate: str) -> bool:
        """The verification gate: constant-time compare with the official flag."""
        return hmac.compare_digest(candidate.strip().encode(), self.expected_flag.encode())


def load_target(dataset_json: str | Path, challenge_id: str) -> Target:
    ds = CTFDataset(dataset_json=str(dataset_json))
    chal = CTFChallenge(ds.get(challenge_id), ds.basedir)
    expected = chal.flag      # official package's flag — the only flag source
    if not isinstance(expected, str):
        raise TypeError("Non-string flag format needs the benchmark's own checker")
    return Target(chal, expected)


def stage_workspace(target: Target, workspace: Path) -> Path:
    """Fresh workspace with only the contestant files and description.

    Deliberately excludes challenge.json and every other metadata file — the
    flag must not be observable from the workspace, not even by path.
    """
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    src_root = target.challenge.challenge_dir.resolve()
    for filename in target.challenge.files:
        src = (src_root / filename).resolve()
        src.relative_to(src_root)  # reject anything that escapes the challenge dir
        dst = workspace / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (workspace / "description.txt").write_text(target.challenge.description)
    return workspace


def new_run_id(challenge_id: str) -> str:
    return f"{challenge_id}-{time.strftime('%Y%m%dT%H%M%S')}"
