import threading

# Simple pause control shared between tray and capture thread
_pause_event = threading.Event()


def pause_capture():
    _pause_event.set()


def resume_capture():
    _pause_event.clear()


def is_paused() -> bool:
    return _pause_event.is_set()
