"""A raising Qt slot is logged and reported; it never aborts the desktop."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = r'''
import sys
from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from frontend import error_log

state = Path(sys.argv[1])
app = QApplication([])
path = error_log.install(state)
reported = []
error_log.set_reporter(lambda message, where: reported.append((message, where)))

def broken():
    raise RuntimeError("slot failed on purpose")

QTimer.singleShot(0, broken)
QTimer.singleShot(50, app.quit)
app.exec()
assert reported and reported[0][0] == "RuntimeError: slot failed on purpose", reported
assert reported[0][1] == path
print("still running")
'''


def test_raising_slot_is_logged_reported_and_survived(tmp_path):
    result = subprocess.run([sys.executable, "-c", PROGRAM, str(tmp_path)], cwd=ROOT, capture_output=True,
                            text=True, timeout=120, env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin",
                                                         "HOME": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert "still running" in result.stdout
    log = (tmp_path / "logs" / "desktop-errors.log").read_text()
    assert "RuntimeError: slot failed on purpose" in log and "in broken" in log


def test_without_the_hook_pyqt_aborts(tmp_path):
    """Documents why the hook exists: PyQt's default turns a raising slot into an abort."""
    program = PROGRAM.replace("path = error_log.install(state)", "path = None")
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)], cwd=ROOT, capture_output=True,
                            text=True, timeout=120, env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin",
                                                         "HOME": str(tmp_path)})
    assert result.returncode != 0 and "still running" not in result.stdout
