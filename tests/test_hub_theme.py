"""Community Hub cards take their text colors from the active theme, so they stay readable on light themes."""
import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.hub.dialog import HubComponentCard
    from frontend.hub.models import AuthorInfo, HubComponent
    from frontend.theme import BUILTINS
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

DESCRIPTION = "Link experiments to an Obsidian vault."


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHubCardFollowsTheme(unittest.TestCase):
    def setUp(self):
        self.previous = getattr(app, "theta_theme", None)

    def tearDown(self):
        if self.previous is None:
            if hasattr(app, "theta_theme"):
                del app.theta_theme
        else:
            app.theta_theme = self.previous

    def card(self, theme_name):
        app.theta_theme = BUILTINS[theme_name]
        component = HubComponent(id="obsidian", name="Obsidian Notes", kind="plugin", version="1.0",
                                 description=DESCRIPTION, author=AuthorInfo(name="Team"), tags=["notes"])
        card = HubComponentCard(component, None)
        self.addCleanup(card.deleteLater)
        return card

    def description_style(self, card):
        return next(lbl for lbl in card.findChildren(QLabel) if lbl.text() == DESCRIPTION).styleSheet()

    def test_description_uses_the_light_theme_text_color(self):
        style = self.description_style(self.card("Gruvbox Light"))
        self.assertIn(BUILTINS["Gruvbox Light"]["colors"]["text"], style)
        self.assertNotIn("#ebdbb2", style)  # Gruvbox Dark's cream, unreadable on a light card

    def test_dark_theme_keeps_its_colors(self):
        style = self.description_style(self.card("Gruvbox Dark"))
        self.assertIn(BUILTINS["Gruvbox Dark"]["colors"]["text"], style)

    def test_install_button_uses_the_theme_primary(self):
        card = self.card("Gruvbox Light")
        self.assertIn(BUILTINS["Gruvbox Light"]["colors"]["primary"], card.btn_action.styleSheet())

    def test_standalone_package_cards_do_not_require_a_client(self):
        for kind in ("suite", "experiment"):
            component = HubComponent(id="preview", name="Preview", kind=kind, version="1.0.0")
            card = HubComponentCard(component, None)
            self.addCleanup(card.deleteLater)
            card.update_action_state()
            self.assertEqual(card.btn_action.text(), "Install")
            self.assertEqual(card.child_rows, {})


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHubRestylesOnThemeSwitch(unittest.TestCase):
    """An open hub follows a theme switch: ThemeManager.apply sets the app palette, and the cards
    and the view restyle on the PaletteChange that follows."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from frontend.theme import ThemeManager

        self.previous = getattr(app, "theta_theme", None)
        self.previous_palette = app.palette()
        self.manager = ThemeManager(Path(tempfile.mkdtemp()) / "appearance.json")
        self.manager.apply(BUILTINS["Gruvbox Dark"])

    def tearDown(self):
        app.setPalette(self.previous_palette)
        if self.previous is None:
            if hasattr(app, "theta_theme"):
                del app.theta_theme
        else:
            app.theta_theme = self.previous

    def make_card(self, installed=False):
        component = HubComponent(id="cql", name="CQL", kind="method", version="1.0", description=DESCRIPTION,
                                 author=AuthorInfo(name="Team"), tags=["rl"])
        component.is_installed = installed
        component.installed_version = "1.0" if installed else None
        card = HubComponentCard(component, None)
        card.show()
        self.addCleanup(card.deleteLater)
        app.processEvents()
        return card

    def switch_to(self, name):
        self.manager.apply(BUILTINS[name])
        app.processEvents()

    def test_open_card_restyles_on_theme_switch(self):
        card = self.make_card()
        light = BUILTINS["Gruvbox Light"]["colors"]
        self.switch_to("Gruvbox Light")
        self.assertIn(light["text"], card.desc_label.styleSheet())
        self.assertIn(light["primary"], card.btn_action.styleSheet())

    def test_badge_and_button_text_stays_readable_in_every_theme(self):
        from frontend.hub.dialog import _contrast, _text_on, theme_color

        for name in BUILTINS:
            self.switch_to(name)
            for role in ("number", "primary", "secondary", "accent", "focus", "disabled"):
                background = theme_color(role)
                self.assertGreaterEqual(_contrast(_text_on(background), background), 4.5, (name, role))

    def test_theme_switch_during_an_install_keeps_the_button_state(self):
        card = self.make_card()
        card.btn_action.setEnabled(False)
        card.btn_action.setText("Installing…")
        self.switch_to("Nord")
        self.assertEqual(card.btn_action.text(), "Installing…")
        self.assertFalse(card.btn_action.isEnabled())
        self.assertIn(BUILTINS["Nord"]["colors"]["primary"], card.btn_action.styleSheet())

    def test_uninstall_stays_red_in_every_theme(self):
        card = self.make_card(installed=True)
        for name in ("Gruvbox Light", "Nord", "Paper"):
            self.switch_to(name)
            self.assertIn("#cc241d", card.btn_action.styleSheet())


if __name__ == "__main__":
    unittest.main()
