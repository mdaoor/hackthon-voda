"""Static smoke checks for the build-free Vodafone companion UI."""
from html.parser import HTMLParser
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


class UIParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.roles = []
        self.aria_labels = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])
        if values.get("role"):
            self.roles.append(values["role"])
        if values.get("aria-label"):
            self.aria_labels.append(values["aria-label"])
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])


def parsed_ui():
    parser = UIParser()
    parser.feed((WEB / "index.html").read_text(encoding="utf-8"))
    return parser


def test_branded_shell_and_local_asset_are_present():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "Together, your day made simpler." in html
    assert html.count("/static/assets/vodafone-logo.svg") == 2
    logo = WEB / "assets" / "vodafone-logo.svg"
    assert logo.is_file()
    assert ET.parse(logo).getroot().tag.endswith("svg")
    assert "Vodafone Group Plc" in logo.read_text(encoding="utf-8")


def test_required_behavior_hooks_are_unique():
    ui = parsed_ui()
    required = {
        "entry", "entry-form", "user-id", "entry-error", "samples", "sample-list",
        "app", "who-id", "who-meta", "btn-new", "btn-switch", "messages", "typing",
        "composer", "input", "btn-mic", "mic-ring", "voice-language", "speak-replies",
        "voice-status", "basket-count", "tab-home", "tab-shop", "shop-form", "shop-q",
        "shop-cat", "shop-max", "shop-results", "tab-rewards", "tab-basket",
        "quick-prompts", "toast-region", "panel-loading", "mobile-nav", "mobile-basket-count",
    }
    assert required <= set(ui.ids)
    assert len(ui.ids) == len(set(ui.ids)), "HTML ids must be unique"
    assert ui.scripts == ["/static/app.js"]


def test_accessibility_and_responsive_controls_exist():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    ui = parsed_ui()
    assert ui.roles.count("tab") == 4
    assert ui.roles.count("tabpanel") == 4
    assert ui.roles.count("status") >= 2
    assert "Mobile navigation" in ui.aria_labels
    assert 'aria-live="polite"' in html
    assert '@media (max-width: 767px)' in css
    assert '@media (prefers-reduced-motion: reduce)' in css
    assert ':focus-visible' in css
    assert 'overflow-x: auto' in css


def test_javascript_selectors_and_recovery_features_are_wired():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    static_js_ids = set(re.findall(r'\$\("#([A-Za-z][\w-]*)"\)', js))
    assert static_js_ids <= html_ids
    assert "alert(" not in js
    assert "function toast(" in js
    assert 'data-mobile' in js
    assert 'data-prompt' in js
