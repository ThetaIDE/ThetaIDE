import os
import subprocess
import sys
import time
from pathlib import Path


def _relaunch_through_macos_app_bundle() -> bool:
    """From a source checkout, open its local .app bundle so macOS gives the GUI a real ThetaIDE identity."""
    repo_root = Path(__file__).resolve().parent.parent
    bundle_builder = repo_root / "scripts" / "create_macos_app.py"
    if (
        sys.platform != "darwin"
        or not bundle_builder.is_file()
        or os.environ.get("THETAIDE_APP_BUNDLE") == "1"
        or os.environ.get("THETAIDE_DIRECT") == "1"
        or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        or any(arg in {"-h", "--help"} for arg in sys.argv[1:])
    ):
        return False

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


def launch() -> int:
    """The ``theta-ide`` command and ``python -m frontend``."""
    os.environ["THETAIDE_STARTUP_EPOCH"] = str(time.perf_counter())
    if "--startup-diagnostics" in sys.argv:
        index = sys.argv.index("--startup-diagnostics")
        if index + 1 < len(sys.argv):
            from frontend.startup_diagnostics import begin
            begin(sys.argv[index + 1], profile="--startup-latency-only" not in sys.argv)
    from frontend.startup import mark
    mark("process_entry")
    if _relaunch_through_macos_app_bundle():
        return 0
    from frontend.app import main
    return main()


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(launch())
