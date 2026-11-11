"""Runner: owns the benchmark answer, stages contestant files, verifies flags.

The answer and challenge metadata never leave this module. The agent only ever
sees the staged workspace, the description, and the verified/not-verified result.
"""

from __future__ import annotations

import hmac
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from nyuctf.challenge import CTFChallenge
from nyuctf.dataset import CTFDataset


@dataclass
class Target:
    challenge: CTFChallenge
    expected_flag: str

    @property
    def challenge_id(self) -> str:
        return self.challenge.name

    def check(self, candidate: str) -> bool:
        return hmac.compare_digest(candidate.strip().encode(), self.expected_flag.encode())


def load_target(dataset_json: str | Path, challenge_id: str) -> Target:
    ds = CTFDataset(dataset_json=str(dataset_json))
    chal = CTFChallenge(ds.get(challenge_id), ds.basedir)
    meta = json.loads((chal.challenge_dir / "challenge.json").read_text())
    expected = meta.get("flag")
    if not isinstance(expected, str):
        raise TypeError("Non-string flag format needs the benchmark's own checker")
    return Target(chal, expected)


def stage_workspace(target: Target, workspace: Path) -> Path:
    """Fresh workspace with only the contestant files and description."""
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
