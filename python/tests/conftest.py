import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def rust_verifier():
    subprocess.run(["cargo", "build", "--bin", "xir-verify"], cwd=ROOT, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    if not target.is_absolute():
        target = ROOT / target
    binary = target / "debug" / "xir-verify"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XIR_VERIFY_BIN", str(binary))
        yield binary
