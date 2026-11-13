"""Theme palettes, application-wide preview, and portable local preferences."""
import copy
import json
from pathlib import Path
import re
import tempfile

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication


ROLES = {
    "base": ("Window / editor background", "#1d2021"),
    "panel": ("Panel background", "#282828"),
    "surface": ("Cards / dock headers", "#32302f"),
    "raised": ("Buttons / plot grid", "#3c3836"),
    "border": ("Borders / selection", "#504945"),
    "disabled": ("Disabled text", "#665c54"),
    "text": ("Main text", "#ebdbb2"),
    "muted": ("Secondary text", "#a89984"),
    "comment": ("Code comments", "#928374"),
    "accent": ("Accent / logo", "#fabd2f"),
    "secondary": ("Secondary curve / code keys", "#83a598"),
    "primary": ("Primary action / reward curve", "#b8bb26"),
    "primary_hover": ("Primary action hover", "#c7c94b"),
    "focus": ("Focus / logo shading", "#d79921"),
    "number": ("Code numbers", "#d3869b"),
}


def preset(name, values):
    return {"name": name, "colors": dict(zip(ROLES, values.split()))}


BUILTINS = {
    "Gruvbox Dark": {"name": "Gruvbox Dark", "colors": {key: value for key, (_, value) in ROLES.items()}},
    "Gruvbox Light": preset("Gruvbox Light", "#fbf1c7 #f9f5d7 #f2e5bc #ebdbb2 #d5c4a1 #a89984 #3c3836 #665c54 #7c6f64 #9d6000 #076678 #79740e #8b8610 #af3a03 #8f3f71"),
    "Nord": preset("Nord", "#242933 #2e3440 #343c4b #3b4252 #4c566a #69768c #eceff4 #b4bfd1 #8996ad #88c0d0 #81a1c1 #a3be8c #b5cf9f #8fbcbb #b48ead"),
    "Dracula": preset("Dracula", "#21222c #282a36 #303341 #383b4b #44475a #6272a4 #f8f8f2 #bcc2dc #929dc4 #bd93f9 #8be9fd #50fa7b #80ff9f #ffb86c #ff79c6"),
    "Catppuccin": preset("Catppuccin", "#1e1e2e #181825 #313244 #45475a #585b70 #6c7086 #cdd6f4 #a6adc8 #7f849c #cba6f7 #89b4fa #a6e3a1 #94e2d5 #fab387 #f5c2e7"),
    "Catppuccin Macchiato": preset("Catppuccin Macchiato", "#24273a #1e2030 #363a4f #494d64 #5b6078 #6e738d #cad3f5 #a5adcb #8087a2 #eed49f #8aadf4 #a6da95 #8bd5ca #c6a0f6 #f5bde6"),
    "Catppuccin Latte": preset("Catppuccin Latte", "#eff1f5 #e6e9ef #ccd0da #bcc0cc #acb0be #9ca0b0 #4c4f69 #6c6f85 #8c8fa1 #8839ef #1e66f5 #40a02b #179299 #fe640b #ea76cb"),
    "Paper": preset("Paper", "#ffffff #f5f7fa #edf0f5 #e2e7ef #c1cad8 #8794a6 #202b3b #526176 #65748a #6246b5 #176b91 #287448 #328a57 #8851c5 #a03876"),
    # ── Greek / Theta Palettes ────────────────────────────────────────────────
    # Apollo — Sunlit marble + Attic pottery (Light / Warm)
    #   base     panel    surface  raised   border   disabled
    #   text     muted    comment  accent   secondary primary  primary_hover focus  number
    "Apollo": preset("Apollo",
        "#f5f0e6 #eae2d3 #dfd6c5 #d9c8a9 #d1c6b4 #9c9485 "
        "#292722 #817a6d #918a76 #b56d32 #496a8a #637447 #748754 #a34832 #765b7a"),
    # Athena — White marble + Aegean sky + oxidized bronze (Light / Cool)
    #   base     panel    surface  raised   border   disabled
    #   text     muted    comment  accent   secondary primary  primary_hover focus  number
    "Athena": preset("Athena",
        "#f1f3f0 #e4e9e7 #d8dfdc #cbdad9 #c8d0cd #939f9e "
        "#252b2c #737d7d #7b8580 #397b7d #496a8c #55745f #658770 #a96e3f #6d617f"),
    # Dionysus — Blackened Fig, dark plum, warm ivory, pomegranate & wine (Dark / Warm)
    #   base     panel    surface  raised   border   disabled
    #   text     muted    comment  accent   secondary primary  primary_hover focus  number
    "Dionysus": preset("Dionysus",
        "#211820 #2d2029 #382833 #4a3040 #42303a #5e5257 "
        "#e5d8c8 #88787d #71656b #c5a05a #687a9b #71875a #839d69 #c45a5a #a66a91"),
    # Ares — Charcoal, blackened iron, bone, vermilion blood & fire bronze (Dark / Warm-Iron)
    #   base     panel    surface  raised   border   disabled
    #   text     muted    comment  accent   secondary primary  primary_hover focus  number
    "Ares": preset("Ares",
        "#171514 #211c1a #2b2522 #38302c #472522 #5a524c "
        "#e4ddd2 #877d73 #6f6861 #c09a55 #596b7a #697354 #7b8663 #d04a3e #765866"),
    # Poseidon — Abyss, deep water, submerged stone, sea foam & kelp (Dark / Cool)
    #   base     panel    surface  raised   border   disabled
    #   text     muted    comment  accent   secondary primary  primary_hover focus  number
    "Poseidon": preset("Poseidon",
        "#080f14 #0d171d #122129 #18343c #1c2b31 #3d4f55 "
        "#d3dedc #66777c #526369 #3f9698 #527ba3 #527d69 #63947e #b77d4e #716b8c"),
}
DEFAULT = BUILTINS["Gruvbox Dark"]
_LEGACY_ROLES = {value: key for key, (_, value) in ROLES.items()}


def current_theme():
    app = QApplication.instance()
    return getattr(app, "theta_theme", DEFAULT)


def theme_color(value):
    """Resolve semantic roles or legacy Gruvbox swatches for painted widgets."""
    colors = current_theme()["colors"]
    return colors.get(_LEGACY_ROLES.get(value, value), value)


def validate_theme(value):
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        raise ValueError("A theme needs a name and a colors object.")
    name = value["name"].strip()
    if not name or len(name) > 60 or any(ord(c) < 32 for c in name):
        raise ValueError("Theme names must contain 1–60 printable characters.")
    colors = value.get("colors")
    if not isinstance(colors, dict) or set(colors) != set(ROLES):
        raise ValueError("Theme colors must include exactly the supported color roles.")
    if any(not isinstance(c, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colors.values()):
        raise ValueError("Use six-digit hex colors, for example #fabd2f.")
    return {"name": name, "colors": {key: colors[key].lower() for key in ROLES}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


_BASE_STYLE = """
* { color: #ebdbb2; font-family: .AppleSystemUIFont, "Segoe UI", system-ui, sans-serif; font-size: 12px; }
QMainWindow, QDialog { background: #1d2021; }
QWidget { background: #282828; }
QMenuBar, QMenu, QToolBar, QStatusBar { background: #1d2021; }
QMenuBar::item { background: transparent; padding: 7px 12px; }
QMenuBar::item:selected, QMenu::item:selected { background: #504945; }
QToolBar { border: 0; spacing: 12px; padding: 8px; }
QToolBar QLabel, QToolBar QLabel#badge, QToolBar QWidget#toolbarSpacer { background: transparent; }
QToolBar::separator { background: #504945; width: 1px; margin: 5px; }
QDockWidget { font-weight: 600; }
QDockWidget::title { background: #32302f; padding: 9px; border-bottom: 1px solid #504945; }
QTabWidget::pane { border: 1px solid #3c3836; }
QTabBar::tab { background: #1d2021; color: #a89984; padding: 11px 18px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { background: #282828; color: #fabd2f; border-bottom: 2px solid #fabd2f; }
QTabBar::tab:hover { color: #ebdbb2; background: #32302f; }
QLabel#brand { color: #fabd2f; font-size: 24px; font-weight: 700; padding-right: 12px; }
QLabel#muted { color: #a89984; }
QLabel#heading { font-size: 23px; font-weight: 600; }
QLabel#cardTitle { font-size: 16px; font-weight: 600; }
QLabel#eyebrow { color: #83a598; font-size: 10px; font-weight: 700; }
QLabel#badge { background: #3c3836; color: #fabd2f; border-radius: 4px; padding: 5px 9px; }
QLabel#value { color: #b8bb26; font-size: 25px; font-weight: 600; }
QFrame#card, QFrame[configBox="true"] { background: #32302f; border: 1px solid #504945; border-radius: 5px; }
QFrame#card QLabel, QFrame[configBox="true"] QLabel { background: transparent; }
QWidget[configBoxBody="true"] { background: transparent; }
QWidget[parameterBoolean="true"] { background: transparent; }
QPushButton, QToolButton { background: #3c3836; border: 1px solid #504945; border-radius: 4px; padding: 7px 12px; }
QPushButton:hover, QToolButton:hover { background: #504945; border-color: #a89984; }
QPushButton:checked { background: #504945; color: #fabd2f; border-color: #fabd2f; font-weight: 600; }
QPushButton#primary { background: #b8bb26; color: #1d2021; border-color: #b8bb26; font-weight: 700; }
QPushButton#primary:hover { background: #c7c94b; }
QPushButton:disabled, QPushButton#primary:disabled { color: #665c54; background: #32302f; border-color: #3c3836; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #1d2021; border: 1px solid #504945; border-radius: 3px; padding: 6px; min-height: 18px; selection-background-color: #665c54; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #d79921; }
QComboBox { combobox-popup: 0; padding-right: 30px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: top right; width: 24px; background: #3c3836; border: 0; border-left: 1px solid #504945; border-top-right-radius: 3px; border-bottom-right-radius: 3px; }
QComboBox::drop-down:hover { background: #504945; }
QComboBox::drop-down:on { background: #665c54; }
QComboBox::down-arrow { image: url("@ARROW_DOWN@"); width: 10px; height: 10px; }
QComboBox::down-arrow:disabled { image: url("@ARROW_DOWN_OFF@"); }
QComboBox QAbstractItemView { background: #32302f; border: 1px solid #504945; padding: 0; outline: 0; selection-background-color: #504945; selection-color: #fabd2f; }
QComboBox QAbstractItemView::item { min-height: 26px; padding: 0 8px; }
QSpinBox, QDoubleSpinBox { padding-right: 6px; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { width: 0; height: 0; border: 0; }
QPlainTextEdit, QTextEdit { background: #1d2021; border: 0; padding: 10px; selection-background-color: #504945; font-family: Menlo, Monaco, 'Cascadia Code', 'Consolas', monospace; font-size: 12px; }
QTextEdit[parameterText="true"] { background: #1d2021; border: 1px solid #504945; border-radius: 3px; padding: 6px; min-height: 18px; selection-background-color: #665c54; font-family: ".AppleSystemUIFont", "Segoe UI", system-ui, sans-serif; }
QTextEdit[parameterText="true"]:focus { border-color: #d79921; }
QTreeWidget, QTableWidget { background: #282828; alternate-background-color: #32302f; border: 0; outline: 0; }
QTreeWidget::item { padding: 6px 2px; }
QTreeWidget::item:selected, QTableWidget::item:selected { background: #504945; color: #fabd2f; }
QHeaderView::section { background: #32302f; color: #a89984; border: 0; border-bottom: 1px solid #504945; padding: 8px; }
QTableWidget::item { padding: 7px; }
QProgressBar { background: #3c3836; border: 0; height: 5px; border-radius: 2px; }
QProgressBar::chunk { background: #b8bb26; border-radius: 2px; }
QSplitter::handle { background: #1d2021; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #282828; width: 9px; }
QScrollBar::handle:vertical { background: #504945; min-height: 25px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #282828; height: 9px; }
QScrollBar::handle:horizontal { background: #504945; min-width: 25px; border-radius: 4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QCheckBox, QRadioButton { background: transparent; spacing: 8px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #504945; border-radius: 3px; background: #1d2021; }
QCheckBox::indicator:hover { border-color: #a89984; }
QCheckBox::indicator:checked { background: #d79921; border-color: #d79921; image: url("@CHECK@"); }
QCheckBox::indicator:disabled { background: #32302f; border-color: #3c3836; }
QCheckBox:disabled { color: #665c54; }
QSlider { background: transparent; }
QSlider::groove:horizontal { height: 4px; background: #3c3836; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #d79921; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #ebdbb2; }
QSlider::handle:horizontal:hover { background: #fabd2f; }
QStatusBar { color: #a89984; border-top: 1px solid #504945; }
QToolTip { background: #3c3836; color: #ebdbb2; border: 1px solid #665c54; padding: 5px; }
QStatusBar QLabel { background: transparent; }
QToolButton#footerReload { background: transparent; border: 0; padding: 0 6px; max-height: 1em; }
QLabel#swatch { border: 1px solid #504945; border-radius: 4px; min-width: 28px; min-height: 24px; }
QLabel#themeError { color: #d79921; }
QWidget#sideTabs { background: #282828; border-right: 1px solid #504945; min-width: 70px; max-width: 70px; }
QWidget#sideTabsEdge { background: #282828; border-right: 1px solid #504945; }
QToolButton#sideTab { background: transparent; border: 0; border-radius: 6px; padding: 7px 0 5px 0; color: #a89984; font-size: 10px; }
QToolButton#sideTab:hover { background: #32302f; color: #ebdbb2; }
QToolButton#sideTab:checked { background: #3c3836; color: #fabd2f; }
QToolButton#sideTabLogo { background: transparent; border: 0; border-radius: 6px; padding: 6px 0; margin-bottom: 2px; }
QToolButton#sideTabLogo:hover { background: #32302f; }
QToolButton#sideTabLogo:checked { background: #3c3836; border: 1px solid #504945; }
QFrame#launcherCard { background: #282828; border: 1px solid #504945; border-radius: 12px; }
QLineEdit#launcherInput { background: transparent; border: 0; border-bottom: 1px solid #504945; border-radius: 0; padding: 14px 18px; font-size: 15px; color: #ebdbb2; }
QListWidget#launcherList { background: transparent; border: 0; outline: 0; padding: 3px 0; }
QLabel#launcherHint { color: #928374; font-size: 10px; padding: 4px 0 0 0; }
QToolButton#sideTabBin { background: transparent; border: 0; border-radius: 5px; padding: 3px 2px 3px 3px; color: #a89984; font-size: 10px; font-weight: 600; text-align: left; }
QToolButton#sideTabBin:hover { background: #32302f; color: #ebdbb2; }
QToolButton#sideTabBin:checked { color: #fabd2f; }
QLineEdit#sideTabBinEditor { font-size: 10px; padding: 1px 3px; border: 1px solid #fabd2f; border-radius: 4px; background: #1d2021; color: #ebdbb2; }
QFrame#monitorProblemBar { background: #1d2021; border: 1px solid #3c3836; border-radius: 6px; }
QPushButton#monitorViewButton { padding: 4px 10px; }
QPushButton#monitorMethodTitle { background: transparent; border: 0; padding: 0; font-size: 23px; font-weight: 600; text-align: left; }
QPushButton#monitorMethodTitle:hover, QPushButton#monitorMethodTitle:pressed, QPushButton#monitorMethodTitle:open { background: #504945; }
QMenu#monitorTitleMenu { background: #282828; border: 1px solid #504945; }
QMenu#monitorTitleMenu::item { font-size: 23px; font-weight: 600; padding: 5px 14px 5px 30px; }
QMenu#monitorTitleMenu::item:selected { background: #504945; }
QPushButton#monitorMethodTitle::menu-indicator { image: none; width: 0; }
QFrame#resultsChartCard { border: 1px solid #3c3836; border-radius: 5px; }
QFrame#resultsChartCard[flat="true"] { border: 0; }
QWidget#resultsCardCorner, QLabel#resultsCardValue { background: transparent; }
QLabel#resultsCardValue { font-weight: 600; }
QToolButton#monitorPlotHide { background: transparent; border: 0; border-radius: 3px; color: #928374; font-size: 15px; font-weight: 500; padding: 0 5px; }
QToolButton#monitorPlotHide:hover { background: #32302f; color: #ebdbb2; }
QToolButton#resultsSection { background: transparent; border: 0; border-bottom: 1px solid #3c3836; border-radius: 0; color: #83a598; font-size: 10px; font-weight: 700; padding: 6px 2px; text-align: left; }
QToolButton#resultsSection:hover { color: #ebdbb2; }
QWidget#resultsKpiStrip { background: transparent; }
QFrame#resultsKpi { background: #32302f; border: 1px solid #3c3836; border-radius: 6px; }
QFrame#resultsKpi:hover { border-color: #504945; }
QFrame#resultsKpi[primary="true"] { border: 1px solid #fabd2f; }
QFrame#resultsKpi QLabel { background: transparent; }
QLabel#resultsKpiName { font-weight: 600; }
QLabel#resultsKpiCaption { color: #928374; font-size: 10px; }
QLabel#resultsKpiValue { font-size: 15px; font-weight: 600; }
QPushButton#resultsChip { background: #32302f; border: 1px solid #3c3836; border-radius: 4px; padding: 5px 10px; }
QPushButton#resultsChip:checked { background: #3c3836; border-color: #fabd2f; color: #fabd2f; }
QFrame#resultsSparkline { background: #32302f; border: 1px solid #3c3836; border-radius: 5px; }
QFrame#resultsSparkline:hover { border-color: #504945; }
QFrame#resultsSparkline[focused="true"] { border: 1px solid #fabd2f; }
QListWidget#resultsPlotsList { background: #282828; border: 0; outline: 0; }
QListWidget#resultsPlotsList::item { padding: 3px 2px; }
QLabel#configOk { color: #b8bb26; }
QLabel#configError { color: #d79921; }
"""


_CHEVRON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><polyline points="{points}" fill="none" '
            'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
_CHEVRON_POINTS = {"UP": "2 6.5 5 3.5 8 6.5", "DOWN": "2 3.5 5 6.5 8 3.5", "CHECK": "2 5.2 4.2 7.4 8 2.8"}


def _chevron(direction, color):
    """Write a chevron in this color for spin-box arrows; Qt stylesheets can only load them from files."""
    folder = Path(tempfile.gettempdir()) / "thetaide_style"
    path = folder / f"chevron_{direction.lower()}_{color.lstrip('#')}.svg"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(_CHEVRON.format(points=_CHEVRON_POINTS[direction], color=color), encoding="utf-8")
    return path.as_posix()


def stylesheet(theme, zoom: float = 1.0):
    colors = theme["colors"]
    style = re.sub(r"#[0-9a-fA-F]{6}", lambda m: colors[_LEGACY_ROLES[m[0]]], _BASE_STYLE)
    for direction in _CHEVRON_POINTS:
        style = style.replace(f"@ARROW_{direction}@", _chevron(direction, colors["text"]))
        style = style.replace(f"@ARROW_{direction}_OFF@", _chevron(direction, colors["disabled"]))
    style = style.replace("@CHECK@", _chevron("CHECK", colors["base"]))
    if abs(zoom - 1.0) > 1e-4:
        def _scale_font(m):
            size = int(m.group(1))
            scaled = max(6, round(size * zoom))
            return f"font-size: {scaled}px"
        style = re.sub(r"font-size:\s*(\d+)px", _scale_font, style)
    return style


STYLE = stylesheet(DEFAULT)  # Backward-compatible default for tests and previews.


class ThemeManager(QObject):
    changed = pyqtSignal()
    committed = pyqtSignal(str)  # a theme the user chose to keep (not a builder preview)

    def __init__(self, path, parent=None, initial_theme: str | None = None, initial_zoom: float = 1.0):
        super().__init__(parent)
        self.path = Path(path)
        self.custom = {}
        self.active = copy.deepcopy(DEFAULT)
        self.error = None
        self.zoom = round(float(initial_zoom), 2)
        # Widget that set_zoom() styled directly. A widget's own stylesheet overrides the
        # application's, so apply() must restyle it too or theme changes never reach it.
        self._style_target = None
        # Migrate existing .appearance.json on first load (for backward compat)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("themes"), list):
                    raise ValueError("Unsupported appearance settings format.")
                custom = {}
                for item in data["themes"]:
                    theme = validate_theme(item)
                    if theme["name"].casefold() in {n.casefold() for n in BUILTINS}:
                        raise ValueError("A custom theme cannot replace a built-in theme.")
                    custom[theme["name"]] = theme
                self.custom = custom
                # Honour initial_theme (from settings.toml) over .appearance.json
                selected = initial_theme if initial_theme and initial_theme in {**BUILTINS, **custom} \
                    else data.get("selected")
                if not isinstance(selected, str) or selected not in {**BUILTINS, **custom}:
                    raise ValueError("Selected theme was not found.")
                self.active = copy.deepcopy(self.themes()[selected])
            except (OSError, ValueError, TypeError) as exc:
                self.error = f"Appearance settings could not be loaded; using Gruvbox Dark. {exc}"
        elif initial_theme:
            # No .appearance.json yet — apply whatever settings.toml declares
            self.select_by_name(initial_theme, _emit=False)
        self.apply(self.active, zoom=self.zoom)

    def themes(self):
        return {**BUILTINS, **self.custom}

    def apply(self, theme, zoom: float | None = None):
        theme = validate_theme(theme)
        if zoom is not None:
            self.zoom = round(float(zoom), 2)
        app = QApplication.instance()
        app.theta_theme = copy.deepcopy(theme)
        self.active = copy.deepcopy(theme)
        colors = theme["colors"]
        palette = QPalette()
        for role, key in ((QPalette.ColorRole.Window, "panel"), (QPalette.ColorRole.Base, "base"),
                          (QPalette.ColorRole.AlternateBase, "surface"), (QPalette.ColorRole.Text, "text"),
                          (QPalette.ColorRole.WindowText, "text"), (QPalette.ColorRole.Button, "raised"),
                          (QPalette.ColorRole.ButtonText, "text"), (QPalette.ColorRole.Highlight, "border"),
                          (QPalette.ColorRole.HighlightedText, "text"), (QPalette.ColorRole.PlaceholderText, "muted"),
                          (QPalette.ColorRole.ToolTipBase, "raised"), (QPalette.ColorRole.ToolTipText, "text"),
                          (QPalette.ColorRole.Link, "secondary")):
            palette.setColor(role, QColor(colors[key]))
        for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
            palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(colors["disabled"]))
        app.setPalette(palette)
        sheet = stylesheet(theme, zoom=self.zoom)
        # QApplication.setStyleSheet repolishes each widget once per ancestor (O(widgets x depth):
        # ~15k restyles and ~2s for the main window), while QWidget.setStyleSheet restyles each
        # descendant about once. So once there is a style target, theme it alone: its dialogs,
        # menus, popups and tooltips are parented to it and inherit its sheet. The app-level sheet
        # only covers windows created without a parent, before the target exists.
        if not self._restyle_target(sheet):
            app.setStyleSheet(sheet)
        self.changed.emit()
        for widget in app.allWidgets():
            widget.update()

    def set_zoom(self, zoom: float, target_widget=None):
        """Update zoom factor and reapply current stylesheet without triggering full theme reload."""
        self.zoom = round(float(zoom), 2)
        ss = stylesheet(self.active, zoom=self.zoom)
        target = target_widget or self.parent()
        if target is not None and hasattr(target, "setStyleSheet"):
            self._style_target = target
            target.setStyleSheet(ss)
        else:
            app = QApplication.instance()
            if app:
                app.setStyleSheet(ss)

    def _restyle_target(self, sheet):
        """Give the zoom target the new stylesheet; forget it if Qt has already deleted it.

        Returns True if a live target was restyled."""
        target = self._style_target
        if target is None:
            return False
        try:
            target.setStyleSheet(sheet)
        except RuntimeError:  # wrapped C++ widget was deleted
            self._style_target = None
            return False
        return True

    def select_by_name(self, name: str, _emit: bool = True) -> bool:
        """Switch the active theme to *name* without writing to .appearance.json.

        Called by the :class:`SettingsManager` when a hot-reload or GUI
        selection changes ``appearance.theme`` in ``settings.toml``.

        Returns ``True`` if the named theme was found and applied.
        """
        theme = self.themes().get(name)
        if theme is None:
            return False
        self.apply(theme)
        if not _emit:
            # apply() already emits; this param exists only for __init__ usage
            pass
        return True

    def commit(self, theme):
        theme = validate_theme(theme)
        custom = copy.deepcopy(self.custom)
        if theme["name"] in BUILTINS:
            if theme != BUILTINS[theme["name"]]:
                raise ValueError("Give your edited theme a new name to preserve the built-in preset.")
        else:
            if theme["name"].casefold() in {n.casefold() for n in BUILTINS}:
                raise ValueError("Choose a name different from the built-in themes.")
            custom[theme["name"]] = theme
        # Write first: a failed save must not change the committed preferences.
        write_json(self.path, {"version": 1, "selected": theme["name"], "themes": list(custom.values())})
        self.custom = custom
        self.apply(theme)
        self.committed.emit(theme["name"])
