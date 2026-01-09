import logging
import os
import threading
import subprocess
import platform
import webbrowser
from typing import Optional

from importlib.metadata import PackageNotFoundError, version as pkg_version
import pystray
from PIL import Image

from openrecall.config import appdata_folder
from openrecall.settings import load_settings, save_settings
from openrecall.database import create_db
from openrecall.config import screenshots_path

logger = logging.getLogger(__name__)

present_settings_panel = None
if platform.system().lower() == "darwin":
    try:
        from openrecall.macos_settings import present_settings_panel  # type: ignore
    except Exception as exc:
        logger.warning("Failed to load macOS settings panel: %s", exc)

ICON_FILENAME = "lookback-icon.png"
ICON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images", ICON_FILENAME))


def _load_icon() -> Optional[Image.Image]:
    try:
        return Image.open(ICON_PATH)
    except Exception as exc:
        logger.warning("Tray icon could not be loaded from %s: %s", ICON_PATH, exc)
        return None


def _open_app(_icon=None, _item=None):
    webbrowser.open_new_tab("http://127.0.0.1:8082")


def _open_homepage(_icon=None, _item=None):
    webbrowser.open_new_tab("https://github.com/vinnyglennon/openrecall")


def _quit(icon: pystray.Icon, _item=None):
    icon.stop()


_tray_icon: Optional[pystray.Icon] = None
_menu_lock = threading.Lock()


def _get_version() -> str:
    try:
        return pkg_version("OpenRecall")
    except PackageNotFoundError:
        return "unknown"


def _pause_label() -> str:
    from openrecall import state

    if state.is_paused():
        return "🟢 Resume Capture"
    return "Pause Capture"


def _toggle_pause(icon=None, item=None):
    from openrecall import state

    if state.is_paused():
        state.resume_capture()
        logger.info("Capture resumed.")
    else:
        state.pause_capture()
        logger.info("Capture paused.")
    _refresh_menu()


def _refresh_menu():
    global _tray_icon
    if _tray_icon is None:
        return
    with _menu_lock:
        _tray_icon.menu = _build_menu()
        try:
            _tray_icon.update_menu()
        except Exception:
            logger.debug("Tray menu update may not be supported; menu reassigned.")


def _build_menu() -> pystray.Menu:
    items = [
        pystray.MenuItem("Search", _open_app),
        pystray.MenuItem(_pause_label(), _toggle_pause),
        pystray.MenuItem("Settings", _open_settings),
        pystray.MenuItem("Project Homepage", _open_homepage),
        pystray.MenuItem(f"Version: {_get_version()}", None, enabled=False),
        pystray.MenuItem("Quit", _quit),
    ]
    return pystray.Menu(*items)


def create_system_tray_icon() -> Optional[pystray.Icon]:
    image = _load_icon()
    if image is None:
        return None

    menu = _build_menu()
    icon = pystray.Icon("OpenRecall", image, "OpenRecall", menu)
    icon.visible = True
    return icon


def stop_tray_icon():
    global _tray_icon
    if _tray_icon:
        try:
            _tray_icon.stop()
        except Exception:
            pass


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


def _get_running_apps() -> list[str]:
    try:
        output = subprocess.check_output(["ps", "-axo", "comm"], text=True)
        # dedupe and clean empty lines
        apps = sorted({line.strip().split("/")[-1] for line in output.splitlines() if line.strip()})
        return apps
    except Exception as exc:
        logger.warning("Failed to list running apps: %s", exc)
        return []


def _osascript(script: str) -> str:
    try:
        out = subprocess.check_output(["osascript", "-e", script], text=True)
        return out.strip()
    except subprocess.CalledProcessError as exc:
        logger.warning("osascript failed: %s", exc)
        return ""
    except FileNotFoundError:
        logger.warning("osascript not available on this system.")
        return ""


def _escape_item(item: str) -> str:
    # Escape double quotes for AppleScript
    return item.replace('"', '\\"')


def _macos_settings_flow():

    settings = load_settings()

    size_bytes = _folder_size(appdata_folder)
    size_str = _format_size(size_bytes)
    running_apps = _get_running_apps()

    info = (
        f"Current Disk Space Used: {size_str}\\n"
        "Retention affects how long recordings are kept. Shorter retention saves space."
    )
    _osascript(
        f'display dialog "{info}" buttons {{"OK"}} default button 1 with title "OpenRecall Settings"'
    )

    # Startup
    startup_default = "Enable" if settings.startup_enabled else "Disable"
    startup_choice = _osascript(
        f'choose from list {{"Enable", "Disable"}} with prompt "Open at Startup?" default items {{"{startup_default}"}} with title "OpenRecall Settings"'
    )

    # Retention
    retention_default = settings.retention
    retention_choice = _osascript(
        f'choose from list {{"1w", "1m", "3m", "6m", "1y", "Forever"}} with prompt "Retention period — how long to keep recordings" default items {{"{retention_default}"}} with title "OpenRecall Settings"'
    )

    # Incognito toggle
    incognito_default = "Do not record incognito" if settings.incognito_block else "Record all"
    incognito_choice = _osascript(
        f'choose from list {{"Do not record incognito", "Record all"}} with prompt "Private browsing: skip incognito/private windows?" default items {{"{incognito_default}"}} with title "OpenRecall Settings"'
    )

    # Whitelist / running apps
    # Exclude apps (stored in settings.whitelist for now)
    exclude_selection = None
    if running_apps:
        limited_apps = running_apps[:30]
        apps_list = "{" + ", ".join(f'"{_escape_item(app)}"' for app in limited_apps) + "}"
        default_items = (
            "{" + ", ".join(f'"{_escape_item(app)}"' for app in settings.whitelist or []) + "}"
            if settings.whitelist
            else "{}"
        )
        exclude_selection = _osascript(
            f'choose from list {apps_list} with prompt "Exclude apps from recording (select to exclude)" default items {default_items} multiple selections allowed with title "OpenRecall Settings"'
        )
    else:
        logger.info("No running apps detected for whitelist selection.")

    # Delete data confirmation
    delete_choice = _osascript(
        'display dialog "Delete all data from application folder?" buttons {"Cancel", "Delete"} default button "Cancel" cancel button "Cancel" with icon stop with title "OpenRecall Settings"'
    )
    deleted = False
    if delete_choice and "button returned:Delete" in delete_choice:
        try:
            # Delete files but leave the top-level folder so app can recreate structure
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
            # Recreate empty database/tables after deletion
            create_db()
            _osascript('display dialog "All data deleted." buttons {"OK"} default button "OK" with title "OpenRecall Settings"')
            deleted = True
        except Exception as exc:
            logger.error("Failed to delete data: %s", exc)

    # Persist choices (except deletion which is immediate)
    settings.startup_enabled = "Enable" in startup_choice
    settings.retention = (retention_choice or settings.retention).strip("{}")
    settings.incognito_block = "Do not record incognito" in incognito_choice
    if running_apps and exclude_selection:
        chosen = [w.strip() for w in exclude_selection.split(",") if w.strip()]
        settings.whitelist = chosen

    if not deleted:
        save_settings(settings)


def _open_settings(icon=None, item=None):
    if platform.system().lower() == "darwin" and present_settings_panel:
        try:
            present_settings_panel()
        except Exception as exc:
            logger.exception("Failed to open macOS settings panel: %s", exc)
        return
    elif platform.system().lower() == "darwin" and not present_settings_panel:
        logger.warning("macOS settings panel not available (PyObjC not installed or import failed).")

    # Fallback for other platforms: simple message
    logger.warning("Settings UI not implemented for this platform yet.")


def start_tray_icon_async():
    global _tray_icon
    _tray_icon = create_system_tray_icon()
    if _tray_icon is None:
        logger.warning("Tray icon not started (icon missing).")
        return

    try:
        # On some platforms run_detached is safer than manually threading run()
        _tray_icon.run_detached()
        logger.info("Tray icon started.")
    except Exception:
        logger.exception("Tray icon failed to start with run_detached; falling back to thread.")
        threading.Thread(target=_tray_icon.run, daemon=True).start()


def start_tray_icon_blocking():
    """Run the tray icon in the main thread (needed on some macOS setups)."""
    global _tray_icon
    _tray_icon = create_system_tray_icon()
    if _tray_icon is None:
        logger.warning("Tray icon not started (icon missing).")
        return
    try:
        _tray_icon.run()
    except Exception:
        logger.exception("Tray icon failed to run; retrying detached.")
        try:
            _tray_icon.run_detached()
            logger.info("Tray icon started (detached fallback).")
        except Exception:
            logger.exception("Tray icon detached fallback failed.")
