"""A clean checkout provides the editor and loader, not installed research content."""

import subprocess
from pathlib import Path


def test_checkout_does_not_track_component_or_experiment_configs():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "src/usr/config/agent",
            "src/usr/config/env",
            "src/usr/config/model",
            "src/usr/config/experiment",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    tracked = {Path(path.decode()) for path in result.stdout.split(b"\0") if path}
    assert tracked == {
        Path("src/usr/config/agent/.gitkeep"),
        Path("src/usr/config/env/.gitkeep"),
        Path("src/usr/config/model/.gitkeep"),
        Path("src/usr/config/experiment/.gitkeep"),
    }
