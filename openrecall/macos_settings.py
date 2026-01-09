import os
import sys
import threading
import plistlib
import subprocess

import objc
from AppKit import (
    NSAlert,
    NSAlertFirstButtonReturn,
    NSAlertSecondButtonReturn,
    NSButton,
    NSStackView,
    NSTextField,
    NSPopUpButton,
    NSLayoutConstraint,
    NSLayoutAttributeWidth,
    NSLayoutRelationGreaterThanOrEqual,
    NSApplication,
    NSScrollView,
    NSView,
    NSWorkspace,
)
from PyObjCTools import AppHelper

from openrecall.settings import (
    EXCLUDED_DOMAIN_DEFAULTS,
    SENSITIVE_DEFAULTS,
    load_settings,
    save_settings,
)
from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import create_db

LAUNCH_AGENT_ID = "com.openrecall.app"
LAUNCH_AGENT_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCH_AGENT_ID}.plist")


def _format_size(bytes_size: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_size)
    for suffix in suffixes:
        if size < 1024:
            return f"{size:.1f} {suffix}"
        size /= 1024
    return f"{size:.1f} PB"


def _folder_size(path: str) -> int:
    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _delete_all_data():
    for root, dirs, files in os.walk(appdata_folder, topdown=False):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except OSError:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except OSError:
                pass
    os.makedirs(appdata_folder, exist_ok=True)
    os.makedirs(screenshots_path, exist_ok=True)
    create_db()


def _label(text: str, bold: bool = False, size: float = 13.0) -> NSTextField:
    lbl = NSTextField.alloc().init()
    lbl.setStringValue_(text)
    lbl.setEditable_(False)
    lbl.setBezeled_(False)
    lbl.setDrawsBackground_(False)
    font = lbl.font()
    if bold:
        font = font.fontDescriptor().fontDescriptorWithSymbolicTraits_(0x0002)
        font = objc.lookUpClass("NSFont").fontWithDescriptor_size_(font, size)
    else:
        font = font.fontWithSize_(size)
    lbl.setFont_(font)
    return lbl


def _checkbox(title: str, state: bool) -> NSButton:
    btn = NSButton.alloc().init()
    btn.setButtonType_(1)  # NSSwitch/checkbox
    btn.setTitle_(title)
    btn.setState_(1 if state else 0)
    return btn


def _retention_popup(current: str) -> NSPopUpButton:
    popup = NSPopUpButton.alloc().init()
    options = ["1w", "1m", "3m", "6m", "1y", "Forever"]
    popup.addItemsWithTitles_(options)
    if current in options:
        popup.selectItemWithTitle_(current)
    return popup


def _add_width_constraint(view, min_width: float):
    constraint = NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
        view,
        NSLayoutAttributeWidth,
        NSLayoutRelationGreaterThanOrEqual,
        None,
        NSLayoutAttributeWidth,
        1.0,
        min_width,
    )
    view.addConstraint_(constraint)


def _set_login_item(enabled: bool):
    os.makedirs(os.path.dirname(LAUNCH_AGENT_PATH), exist_ok=True)
    if enabled:
        plist = {
            "Label": LAUNCH_AGENT_ID,
            "ProgramArguments": [sys.executable, "-m", "openrecall.app"],
            "RunAtLoad": True,
            "KeepAlive": False,
            "StandardOutPath": os.path.expanduser("~/Library/Logs/openrecall.log"),
            "StandardErrorPath": os.path.expanduser("~/Library/Logs/openrecall.log"),
            "WorkingDirectory": os.path.expanduser("~"),
        }
        with open(LAUNCH_AGENT_PATH, "wb") as f:
            plistlib.dump(plist, f)
    else:
        try:
            os.remove(LAUNCH_AGENT_PATH)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _get_running_apps() -> list[str]:
    # Prefer GUI apps via NSWorkspace (activationPolicyRegular)
    try:
        workspace = NSWorkspace.sharedWorkspace()
        apps = workspace.runningApplications()
        gui_apps = {
            app.localizedName()
            for app in apps
            if app.localizedName() and app.activationPolicy() == 0
        }
        if gui_apps:
            return sorted(gui_apps)
    except Exception:
        pass
    # Fallback: process list
    try:
        output = subprocess.check_output(["ps", "-axo", "comm"], text=True)
        apps = sorted({line.strip().split("/")[-1] for line in output.splitlines() if line.strip()})
        return apps[:50]
    except Exception:
        return []


def show_settings_panel():
    settings = load_settings()
    size_bytes = _folder_size(appdata_folder)
    size_str = _format_size(size_bytes)

    alert = NSAlert.alloc().init()
    alert.setMessageText_("Settings")
    alert.setInformativeText_("Configure capture and retention in one place.")

    stack = NSStackView.alloc().init()
    stack.setOrientation_(1)  # vertical
    stack.setSpacing_(8)
    stack.setTranslatesAutoresizingMaskIntoConstraints_(True)
    stack.setFrameSize_((440, 320))

    # Disk usage
    stack.addArrangedSubview_(_label(f"Current Disk Space Used: {size_str}", bold=True))
    stack.addArrangedSubview_(
        _label(
            "Retention affects how long recordings are kept. Shorter retention saves space.",
            bold=False,
            size=12.0,
        )
    )

    # Retention
    stack.addArrangedSubview_(_label("Retention Period:", bold=True))
    retention_popup = _retention_popup(settings.retention)
    _add_width_constraint(retention_popup, 240)
    stack.addArrangedSubview_(retention_popup)

    # Startup toggle
    startup_cb = _checkbox("Open at Startup", settings.startup_enabled)
    stack.addArrangedSubview_(startup_cb)

    # Incognito toggle
    incognito_cb = _checkbox(
        "Do not record Incognito/Private windows (Chrome/Firefox/Safari)",
        settings.incognito_block,
    )
    stack.addArrangedSubview_(incognito_cb)

    # Sensitive window title exclusion
    stack.addArrangedSubview_(
        _label("Exclude sensitive activities (window titles contain):", bold=True)
    )
    sensitive_field = NSTextField.alloc().init()
    sensitive_field.setPlaceholderString_("Examples: bank|stripe|checkout|paypal|2fa|code|otp")
    current_patterns = settings.sensitive_patterns or SENSITIVE_DEFAULTS
    sensitive_field.setStringValue_("|".join(current_patterns))
    _add_width_constraint(sensitive_field, 400)
    stack.addArrangedSubview_(sensitive_field)

    # Domain exclusion (e.g., sensitive sites)
    stack.addArrangedSubview_(_label("Exclude domains (matches window title):", bold=True))
    domains_field = NSTextField.alloc().init()
    domains_field.setPlaceholderString_("Examples: paypal.com|bankofamerica.com|stripe.com")
    current_domains = settings.excluded_domains or EXCLUDED_DOMAIN_DEFAULTS
    domains_field.setStringValue_("|".join(current_domains))
    _add_width_constraint(domains_field, 400)
    stack.addArrangedSubview_(domains_field)

    # High-risk OCR triggers
    stack.addArrangedSubview_(_label("High-risk OCR triggers (mask image when detected):", bold=True))
    ocr_field = NSTextField.alloc().init()
    ocr_field.setPlaceholderString_("Examples: cvv|iban|seed phrase|routing number|2fa code")
    current_ocr = settings.high_risk_ocr_triggers or []
    ocr_field.setStringValue_("|".join(current_ocr))
    _add_width_constraint(ocr_field, 400)
    stack.addArrangedSubview_(ocr_field)

    # Whitelist apps list with checkboxes
    stack.addArrangedSubview_(_label("Whitelist Apps: Select apps you want recorded.", bold=True))
    running = sorted(set(_get_running_apps()))
    checked = set(settings.whitelist or [])
    checkbox_views = []
    if running:
        checkbox_stack = NSStackView.alloc().init()
        checkbox_stack.setOrientation_(1)
        checkbox_stack.setSpacing_(8)
        checkbox_stack.setAlignment_(1)  # align leading
        checkbox_stack.setTranslatesAutoresizingMaskIntoConstraints_(True)
        checkbox_stack.setEdgeInsets_((8, 12, 8, 12))  # top, left, bottom, right
        checkbox_stack.setFrameSize_((380, 28 * max(3, len(running))))
        for app_name in running:
            cb = _checkbox(app_name, app_name in checked)
            cb.sizeToFit()
            checkbox_stack.addArrangedSubview_(cb)
            checkbox_views.append((app_name, cb))

        scroll = NSScrollView.alloc().init()
        scroll.setHasVerticalScroller_(True)
        scroll.setDocumentView_(checkbox_stack)
        scroll.setFrameSize_((420, 240))
        scroll.setAutohidesScrollers_(True)
        stack.addArrangedSubview_(scroll)
    else:
        stack.addArrangedSubview_(_label("No running apps detected.", bold=False, size=12.0))

    # Delete note
    stack.addArrangedSubview_(_label("Delete all data will remove screenshots and the database.", bold=False, size=12.0))

    stack.setAutoresizingMask_(18)  # flexible width/height
    alert.setAccessoryView_(stack)

    # Bring alert to front
    app = NSApplication.sharedApplication()
    if app is not None:
        app.activateIgnoringOtherApps_(True)
    alert.window().makeKeyAndOrderFront_(None)

    alert.addButtonWithTitle_("Save")
    alert.addButtonWithTitle_("Delete All Data")
    alert.addButtonWithTitle_("Cancel")

    response = alert.runModal()

    if response == NSAlertSecondButtonReturn:
        # Delete
        confirm = NSAlert.alloc().init()
        confirm.setMessageText_("Delete all data?")
        confirm.setInformativeText_("This will remove all recordings and database. This cannot be undone.")
        confirm.addButtonWithTitle_("Delete")
        confirm.addButtonWithTitle_("Cancel")
        res = confirm.runModal()
        if res == NSAlertFirstButtonReturn:
            _delete_all_data()
        return

    if response != NSAlertFirstButtonReturn:
        return

    # Save settings
    settings.startup_enabled = bool(startup_cb.state())
    settings.retention = retention_popup.titleOfSelectedItem()
    settings.incognito_block = bool(incognito_cb.state())
    if checkbox_views:
        chosen = [name for name, cb in checkbox_views if bool(cb.state())]
        settings.whitelist = chosen
    else:
        settings.whitelist = []

    # Sensitive patterns (split on | or newline/comma)
    raw_patterns = sensitive_field.stringValue() or ""
    parts = []
    for chunk in raw_patterns.replace(",", "|").split("|"):
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    settings.sensitive_patterns = parts or SENSITIVE_DEFAULTS.copy()

    # Excluded domains (split on | or newline/comma)
    raw_domains = domains_field.stringValue() or ""
    domains_parts = []
    for chunk in raw_domains.replace(",", "|").split("|"):
        cleaned = chunk.strip()
        if cleaned:
            domains_parts.append(cleaned)
    settings.excluded_domains = domains_parts or EXCLUDED_DOMAIN_DEFAULTS.copy()

    # High-risk OCR triggers (split on | or newline/comma)
    raw_ocr = ocr_field.stringValue() or ""
    ocr_parts = []
    for chunk in raw_ocr.replace(",", "|").split("|"):
        cleaned = chunk.strip()
        if cleaned:
            ocr_parts.append(cleaned)
    settings.high_risk_ocr_triggers = ocr_parts or HIGH_RISK_OCR_DEFAULTS.copy()

    # Apply side effects
    try:
        _set_login_item(settings.startup_enabled)
    except Exception:
        pass

    save_settings(settings)


def present_settings_panel():
    """Ensure the settings panel is shown on the main thread."""
    if threading.current_thread() is threading.main_thread():
        show_settings_panel()
    else:
        # Schedule on main thread using PyObjC helper
        AppHelper.callAfter(show_settings_panel)
