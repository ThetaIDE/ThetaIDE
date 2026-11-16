"""Splash screen: shown while the main window is built, in the theme the window will open with."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.splash import PHASES, SplashScreen, splash_theme  # noqa: E402
from frontend.theme import BUILTINS, DEFAULT  # noqa: E402


def test_progress_only_moves_forward_and_is_clamped():
    splash = SplashScreen()
    splash.set_progress("Building the workspace…", 0.5)
    splash.set_progress("Starting plugins…", 0.3)
    assert splash.progress == 0.5 and splash.message == "Starting plugins…"
    splash.set_progress("Opening window…", 7)
    assert splash.progress == 1.0


def test_phase_follows_known_steps_and_never_moves_backwards():
    splash = SplashScreen()
    assert splash.phase == -1
    splash.set_progress("Starting plugins…", 0.56)
    assert splash.phase == PHASES.index("Starting plugins…")
    splash.set_progress("Loading settings…", 0.02)  # an earlier phase
    assert splash.phase == PHASES.index("Starting plugins…")
    splash.set_progress("Doing something unlisted…", 0.6)
    assert splash.message == "Doing something unlisted…" and splash.phase == PHASES.index("Starting plugins…")


def test_phase_circles_fill_as_phases_complete():
    theme = BUILTINS["Gruvbox Dark"]
    splash = SplashScreen(theme)
    splash.set_progress(PHASES[3], 0.08)
    splash.show()
    app.processEvents()
    image = splash.grab().toImage()
    scale = image.width() / splash.width()
    accent = QColor(theme["colors"]["accent"])

    def is_accent(point):
        pixel = image.pixelColor(int(point.x() * scale), int(point.y() * scale))
        return abs(pixel.red() - accent.red()) + abs(pixel.green() - accent.green()) + abs(pixel.blue() - accent.blue()) < 40

    centers = splash.phase_centers()
    assert len(centers) == len(PHASES)
    assert all(is_accent(centers[i]) for i in range(3))  # completed: filled
    assert is_accent(centers[3])  # under way: the centre dot
    assert not any(is_accent(centers[i]) for i in range(4, len(PHASES)))  # still to come: hollow
    splash.close()


def test_splash_paints_the_painting_faded_into_the_theme():
    theme = BUILTINS["Nord"]
    splash = SplashScreen(theme)
    assert not splash.background.isNull()  # images/splash_landscape.jpg is bundled
    splash.set_progress("Loading settings…", 0.4)
    splash.show()
    app.processEvents()
    image = splash.grab().toImage()
    scale = image.width() / splash.width()

    def distance(x, y):
        a, b = image.pixelColor(int(x * scale), int(y * scale)), QColor(theme["colors"]["base"])
        return abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())

    assert distance(splash.width() / 2, 30) > 120  # the painting's sky shows at the top
    assert distance(20, splash.height() - 12) < 30  # the theme's base colour behind the text
    splash.close()


def test_splash_without_the_painting_falls_back_to_the_theme(monkeypatch):
    import frontend.splash as splash_module

    monkeypatch.setattr(splash_module, "BACKGROUND_PATH", Path("missing.jpg"))
    theme = BUILTINS["Nord"]
    splash = SplashScreen(theme)
    splash.show()
    app.processEvents()
    image = splash.grab().toImage()
    scale = image.width() / splash.width()
    assert image.pixelColor(int(40 * scale), int(40 * scale)).name() == theme["colors"]["base"]
    splash.close()


def test_splash_theme_follows_saved_settings(monkeypatch):
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp) / ".thetaide" / "runs"
        data_dir.mkdir(parents=True)
        import frontend.settings as settings

        monkeypatch.setattr(settings.SettingsManager, "theme", property(lambda self: "Dracula"))
        assert splash_theme(data_dir) == BUILTINS["Dracula"]

        custom = {"name": "Mine", "colors": dict(BUILTINS["Paper"]["colors"], accent="#123456")}
        (data_dir / ".appearance.json").write_text(json.dumps({"version": 1, "themes": [custom]}), encoding="utf-8")
        monkeypatch.setattr(settings.SettingsManager, "theme", property(lambda self: "Mine"))
        assert splash_theme(data_dir)["colors"]["accent"] == "#123456"

        monkeypatch.setattr(settings.SettingsManager, "theme", property(lambda self: "Missing"))
        assert splash_theme(data_dir) == DEFAULT


