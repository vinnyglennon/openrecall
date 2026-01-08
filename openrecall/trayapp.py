import logging
import os
import threading
import webbrowser
from typing import Optional

import pystray
from PIL import Image

logger = logging.getLogger(__name__)

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


def create_system_tray_icon() -> Optional[pystray.Icon]:
    image = _load_icon()
    if image is None:
        return None

    menu = pystray.Menu(
        pystray.MenuItem("Open OpenRecall", _open_app),
        pystray.MenuItem("Project Homepage", _open_homepage),
        pystray.MenuItem("Quit", _quit),
    )

    icon = pystray.Icon("OpenRecall", image, "OpenRecall", menu)
    icon.visible = True
    return icon


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
    _tray_icon.run()
