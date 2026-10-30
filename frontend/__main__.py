import os
import subprocess
import sys
from pathlib import Path


def _relaunch_through_macos_app_bundle() -> bool:
    """Open the local .app bundle so macOS gives the GUI a real ThetaIDE identity."""
    if (
        sys.platform != "darwin"
        or os.environ.get("THETAIDE_APP_BUNDLE") == "1"
        or os.environ.get("THETAIDE_DIRECT") == "1"
        or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        or any(arg in {"-h", "--help"} for arg in sys.argv[1:])
    ):
        return False

    repo_root = Path(__file__).resolve().parent.parent
    bundle_builder = repo_root / "scripts" / "create_macos_app.py"
    bundle_path = repo_root / ".thetaide" / "ThetaIDE.app"
    launcher = bundle_path / "Contents" / "MacOS" / "ThetaIDE"
    try:
        bundle_is_current = "export THETAIDE_APP_BUNDLE=1" in launcher.read_text(encoding="utf-8")
    except OSError:
        bundle_is_current = False
    if not bundle_is_current:
        subprocess.run([sys.executable, str(bundle_builder)], cwd=repo_root, check=True)

    command = ["/usr/bin/open", "-n", str(bundle_path)]
    if sys.argv[1:]:
        command.extend(["--args", *sys.argv[1:]])
    subprocess.Popen(command, cwd=repo_root)
    return True


if _relaunch_through_macos_app_bundle():
    raise SystemExit(0)

if __package__:
    from .app import main
else:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from frontend.app import main

raise SystemExit(main())
