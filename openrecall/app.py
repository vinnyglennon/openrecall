import os
import sys
from threading import Thread, Event
from pathlib import Path
from datetime import datetime, timedelta
import logging
import numpy as np
from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for
from jinja2 import BaseLoader
from dotenv import load_dotenv
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Trim optional heavy deps that crash in frozen builds (torchvision)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# Load environment variables from .env if present
load_dotenv()

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    auto_enabling_integrations=False,
    integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
    traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0")),
    profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0")),
    send_default_pii=False,
)

# Enable standard logging early
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import (
    create_db,
    get_all_entries,
    get_timestamps,
    get_entries_by_time_range,
)
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import record_screenshots_thread
from openrecall.utils import human_readable_time, timestamp_to_human_readable
from openrecall.trayapp import start_tray_icon_blocking, stop_tray_icon

app = Flask(__name__)
images_path = Path(__file__).resolve().parent.parent / "images"

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable

base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRecall</title>
  <link rel="icon" type="image/png" href="/favicon.ico">
  <!-- Bootstrap CSS -->
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css">
  <style>
    .slider-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }
    .slider {
      width: 80%;
    }
    .slider-value {
      margin-top: 10px;
      font-size: 1.2em;
    }
    .image-container {
      margin-top: 20px;
      text-align: center;
    }
    .image-container img {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-light bg-light">
  <div class="container">
    <form class="form-inline my-2 my-lg-0 w-100 d-flex justify-content-between align-items-center" action="/search" method="get">
        <div class="form-group mb-0 flex-grow-1 mr-3">
            <input type="text" class="form-control w-100" name="q" placeholder="Search" value="{{ request.args.get('q', '') }}">
        </div>
        <div class="form-group mb-0 mx-sm-2">
            <label for="start_time" class="mr-2 mb-0">Start</label>
            <input type="datetime-local" class="form-control" name="start_time" value="{{ request.args.get('start_time', '') }}">
        </div>
        <div class="form-group mb-0 mx-sm-2">
            <label for="end_time" class="mr-2 mb-0">End</label>
            <input type="datetime-local" class="form-control" name="end_time" value="{{ request.args.get('end_time', '') }}">
        </div>
        <div class="form-group mb-0 mx-sm-2">
            <select name="app" class="form-control">
                <option value="" {% if not request.args.get('app') %}selected{% endif %}>All apps</option>
                {% for app_name in apps_seen %}
                    <option value="{{ app_name }}" {% if request.args.get('app') == app_name %}selected{% endif %}>{{ app_name }}</option>
                {% endfor %}
            </select>
        </div>
        <button class="btn btn-outline-secondary my-2 my-sm-0" type="submit">
            <i class="bi bi-search"></i>
        </button>
    </form>
  </div>
</nav>
{% block content %}

{% endblock %}

  <!-- Bootstrap and jQuery JS -->
  <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""


class StringLoader(BaseLoader):
    def get_source(self, environment, template):
        if template == "base_template":
            return base_template, None, lambda: True
        return None, None, None


app.jinja_env.loader = StringLoader()


@app.route("/")
def timeline():
    # connect to db
    timestamps = get_timestamps()
    apps_seen = sorted({e.app for e in get_all_entries() if e.app})
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
{% if timestamps|length > 0 %}
  <div class="container">
    <div class="slider-container">
      <input type="range" class="slider custom-range" id="discreteSlider" min="0" max="{{timestamps|length - 1}}" step="1" value="{{timestamps|length - 1}}">
      <div class="slider-value" id="sliderValue">{{timestamps[0] | timestamp_to_human_readable }}</div>
    </div>
    <div class="image-container">
      <img id="timestampImage" src="/static/{{timestamps[0]}}.webp" alt="Image for timestamp">
    </div>
  </div>
  <script>
    const timestamps = {{ timestamps|tojson }};
    const slider = document.getElementById('discreteSlider');
    const sliderValue = document.getElementById('sliderValue');
    const timestampImage = document.getElementById('timestampImage');

    slider.addEventListener('input', function() {
      const reversedIndex = timestamps.length - 1 - slider.value;
      const timestamp = timestamps[reversedIndex];
      sliderValue.textContent = new Date(timestamp * 1000).toLocaleString();  // Convert to human-readable format
      timestampImage.src = `/static/${timestamp}.webp`;
    });

    // Initialize the slider with a default value
    slider.value = timestamps.length - 1;
    sliderValue.textContent = new Date(timestamps[0] * 1000).toLocaleString();  // Convert to human-readable format
    timestampImage.src = `/static/${timestamps[0]}.webp`;
  </script>
{% else %}
  <div class="container">
      <div class="alert alert-info" role="alert">
          Nothing recorded yet, wait a few seconds.
      </div>
  </div>
{% endif %}
{% endblock %}
""",
        timestamps=timestamps,
        apps_seen=apps_seen,
    )


def _day_bounds(date_str: str) -> tuple[int, int]:
    """Return start/end unix timestamps (inclusive) for a given YYYY-MM-DD."""
    day = datetime.strptime(date_str, "%Y-%m-%d")
    start = int(day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    end = int((day + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()) - 1
    return start, end


def _summarize_daily_usage(date_str: str) -> tuple[list[dict], int]:
    """Aggregate per-app usage for a day, approximating duration from capture cadence."""
    start_ts, end_ts = _day_bounds(date_str)
    entries = get_entries_by_time_range(start_ts, end_ts)
    if not entries:
        return [], 0

    # Sort ascending by timestamp to measure deltas
    entries = sorted(entries, key=lambda e: e.timestamp)
    per_app_seconds: dict[str, int] = {}
    total_active = 0
    default_interval = 3  # seconds between captures
    max_interval = 5 * 60  # cap long gaps to avoid overcounting

    for idx, entry in enumerate(entries):
        next_ts = entries[idx + 1].timestamp if idx + 1 < len(entries) else None
        delta = default_interval
        if next_ts:
            delta = max(default_interval, min(next_ts - entry.timestamp, max_interval))
        per_app_seconds[entry.app] = per_app_seconds.get(entry.app, 0) + delta
        total_active += delta

    usage = [
        {"app": app, "seconds": secs, "minutes": int(round(secs / 60))}
        for app, secs in per_app_seconds.items()
    ]
    usage.sort(key=lambda x: x["seconds"], reverse=True)
    return usage, total_active


def _hourly_activity(date_str: str) -> list[int]:
    """Return per-hour active seconds approximation."""
    start_ts, end_ts = _day_bounds(date_str)
    entries = get_entries_by_time_range(start_ts, end_ts)
    if not entries:
        return [0] * 24

    entries = sorted(entries, key=lambda e: e.timestamp)
    default_interval = 3
    max_interval = 5 * 60
    hourly = [0] * 24

    for idx, entry in enumerate(entries):
        next_ts = entries[idx + 1].timestamp if idx + 1 < len(entries) else None
        delta = default_interval
        if next_ts:
            delta = max(default_interval, min(next_ts - entry.timestamp, max_interval))
        hour = datetime.fromtimestamp(entry.timestamp).hour
        hourly[hour] += delta
    return hourly


@app.route("/daily")
def daily():
    date_str = request.args.get("date")
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    try:
        usage, total_active = _summarize_daily_usage(date_str)
        hourly = _hourly_activity(date_str)
    except Exception:
        usage, total_active, hourly = [], 0, [0] * 24

    prev_day = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
<style>
  body { background: #e8eaed; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .daily-wrap { max-width: 960px; margin: 32px auto; padding-bottom: 24px; }
  .card { border-radius: 16px; box-shadow: none; border: 1px solid #dfe3e8; background: #f9fafc; }
  .section-header { font-weight: 600; color: #3f4450; }
  .pill { background: #eef0f4; border: 1px solid #e3e6ec; border-radius: 12px; padding: 6px 12px; display: inline-flex; gap: 8px; align-items: center; font-size: 14px; color: #4b5563; }
  .active-hours { display: grid; grid-template-columns: repeat(24, 1fr); gap: 4px; align-items: end; height: 80px; margin-top: 8px; }
  .hour-bar { background: linear-gradient(180deg, #a855f7 0%, #7c3aed 100%); border-radius: 6px 6px 4px 4px; transition: opacity 0.2s ease; }
  .hour-bar.zero { opacity: 0.15; background: #d9dce3; }
  .hour-labels { display: grid; grid-template-columns: repeat(8, 1fr); font-size: 12px; color: #6b7280; margin-top: 8px; }
  .app-list { list-style: none; padding: 0; margin: 0; }
  .app-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e5e7eb; }
  .app-row:last-child { border-bottom: none; }
  .app-info { display: flex; align-items: center; gap: 12px; }
  .app-avatar { width: 34px; height: 34px; border-radius: 9px; background: linear-gradient(145deg, #f4f5f8 0%, #e4e7ec 100%); display: grid; place-items: center; font-weight: 700; color: #4f46e5; }
  .app-name { font-weight: 600; color: #2f3340; }
  .app-time { color: #6b7280; font-weight: 600; }
  .empty { color: #6b7280; }
</style>

<div class="daily-wrap">
  <div class="d-flex justify-content-between align-items-center mb-3">
    <h4 class="mb-0">Daily Recall</h4>
    <div class="d-flex align-items-center gap-2">
      <a class="btn btn-light btn-sm" href="{{ url_for('daily') }}?date={{ prev_day }}">◀</a>
      <input type="date" class="form-control form-control-sm" id="datePicker" value="{{ date_str }}" style="width: 180px;">
      <a class="btn btn-light btn-sm" href="{{ url_for('daily') }}?date={{ next_day }}">▶</a>
    </div>
  </div>

  <div class="card p-3 mb-3">
    <div class="d-flex justify-content-between align-items-center mb-2">
      <div class="section-header">Active Hours</div>
      <div class="pill">
        <span class="bi bi-clock-history"></span>
        <span>{{ (total_active // 3600) }}h {{ ((total_active % 3600) // 60) }}m</span>
      </div>
    </div>
    <div class="active-hours">
      {% set max_val = hourly|max if hourly else 0 %}
      {% for val in hourly %}
        {% set height = 8 if max_val == 0 else (max(8, (val / max_val) * 76)) %}
        <div class="hour-bar {% if val == 0 %}zero{% endif %}" style="height: {{ height }}px;"></div>
      {% endfor %}
    </div>
    <div class="hour-labels">
      <div>3 AM</div>
      <div>6 AM</div>
      <div>9 AM</div>
      <div>12 PM</div>
      <div>3 PM</div>
      <div>6 PM</div>
      <div>9 PM</div>
      <div></div>
    </div>
  </div>

  <div class="card p-3">
    <div class="d-flex justify-content-between align-items-center mb-2">
      <div class="section-header">Apps</div>
    </div>
    {% if usage %}
      <ul class="app-list">
      {% for item in usage %}
        <li class="app-row">
          <div class="app-info">
            <div class="app-avatar">{{ (item.app or "U")[:1] }}</div>
            <div class="app-name">{{ item.app or "Unknown" }}</div>
          </div>
          <div class="app-time">{{ item.minutes }}m</div>
        </li>
      {% endfor %}
      </ul>
    {% else %}
      <div class="empty py-3">No activity recorded for this day.</div>
    {% endif %}
  </div>
</div>

<script>
  const picker = document.getElementById('datePicker');
  picker.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val) {
      window.location.href = `${location.pathname}?date=${val}`;
    }
  });
</script>
{% endblock %}
""",
        usage=usage,
        total_active=total_active,
        hourly=hourly,
        date_str=date_str,
        prev_day=prev_day,
        next_day=next_day,
    )

@app.route("/search")
def search():
    q = request.args.get("q")
    start_time_str = request.args.get("start_time")
    end_time_str = request.args.get("end_time")
    app_filter = request.args.get("app")

    if not q and not start_time_str and not end_time_str and not app_filter:
        return redirect(url_for("timeline"))

    entries = get_all_entries()

    if start_time_str:
        start_time = int(
            datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M").timestamp()
        )
        if end_time_str:
            end_time = int(
                datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M").timestamp()
            )
        else:
            end_time = int(datetime.now().timestamp())
        entries = get_entries_by_time_range(start_time, end_time)

    apps_seen = sorted({e.app for e in entries if e.app})

    # If no query provided, just render the page with filter options
    if not q:
        return render_template_string(
            """
{% extends "base_template" %}
{% block content %}
    <div class="container">
        <div class="alert alert-info">Use the search bar above to search and filter by application.</div>
    </div>
{% endblock %}
""",
            entries=[],
            apps_seen=apps_seen,
            popular_icons={},
            default_icon="bi-app-indicator",
        )

    if app_filter:
        entries = [e for e in entries if e.app == app_filter]

    query_embedding = get_embedding(q)
    query_dim = query_embedding.shape[0]

    # If the embedding is empty/zero (e.g., model unavailable), return no results
    if not np.any(query_embedding):
        sorted_entries = []
        return render_template_string(
            """
{% extends "base_template" %}
{% block content %}
    <div class="container">
        <div class="alert alert-warning" role="alert">
            No results found for your search. Try adjusting the query, app filter, or time range.
        </div>
    </div>
{% endblock %}
""",
            entries=[],
            apps_seen=apps_seen,
            popular_icons={},
            default_icon="bi-app-indicator",
        )

    # Similarity threshold to decide if a result is meaningful
    similarity_threshold = 0.35

    filtered = []
    similarities = []
    for entry in entries:
        emb = np.frombuffer(entry.embedding, dtype=np.float32)
        if emb.shape[0] != query_dim:
            continue
        if not np.any(emb):
            continue
        sim = cosine_similarity(query_embedding, emb)
        if not np.isfinite(sim):
            continue
        # Drop very low-similarity matches to avoid noisy results on nonsense queries
        if sim < similarity_threshold:
            continue
        filtered.append(entry)
        similarities.append(sim)

    # If nothing meets the threshold or best score is too low, treat as no results
    if not filtered or max(similarities, default=0.0) < similarity_threshold:
        sorted_entries = []
    else:
        indices = np.argsort(similarities)[::-1]
        sorted_entries = [filtered[i] for i in indices]

    popular_icons = {
        "Google Chrome": "bi-google",
        "Chrome": "bi-google",
        "Safari": "bi-compass",
        "Firefox": "bi-fire",
        "Edge": "bi-microsoft",
        "Visual Studio Code": "bi-code-slash",
        "Code": "bi-code-slash",
        "Terminal": "bi-terminal",
        "iTerm2": "bi-terminal",
        "Slack": "bi-chat-dots",
        "Notion": "bi-journal-richtext",
        "Word": "bi-file-earmark-text",
        "Excel": "bi-table",
        "PowerPoint": "bi-easel",
    }
    default_icon = "bi-app-indicator"

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
    <div class="container">
        {% if entries %}
            <div class="row">
                {% for entry in entries %}
                    <div class="col-md-3 mb-4">
                        <div class="card shadow-sm h-100 result-card" data-modal-id="modal-{{ loop.index0 }}">
                            <a href="#" data-toggle="modal" data-target="#modal-{{ loop.index0 }}" data-modal-id="modal-{{ loop.index0 }}">
                                <img src="/static/{{ entry.timestamp }}.webp" alt="Image" class="card-img-top">
                            </a>
                            <div class="card-body py-2">
                                <div class="d-flex align-items-center">
                                    <i class="bi {% if entry.app in popular_icons %}{{ popular_icons[entry.app] }}{% else %}{{ default_icon }}{% endif %} mr-2"></i>
                                    <span class="small text-muted">{{ entry.app or 'Unknown app' }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal fade" id="modal-{{ loop.index0 }}" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
                        <div class="modal-dialog modal-xl" role="document" style="max-width: none; width: 100vw; height: 100vh; padding: 20px;">
                            <div class="modal-content" style="height: calc(100vh - 40px); width: calc(100vw - 40px); padding: 0; position: relative;">
                                <button type="button" class="close position-absolute" data-dismiss="modal" aria-label="Close" style="right: 16px; top: 8px; z-index: 10;">
                                    <span aria-hidden="true">&times;</span>
                                </button>
                                <button type="button" class="btn btn-light position-absolute prev-btn" data-nav="prev" style="left: 10px; top: 50%; transform: translateY(-50%); z-index: 10;">
                                    <i class="bi bi-chevron-left"></i>
                                </button>
                                <button type="button" class="btn btn-light position-absolute next-btn" data-nav="next" style="right: 10px; top: 50%; transform: translateY(-50%); z-index: 10;">
                                    <i class="bi bi-chevron-right"></i>
                                </button>
                                <div class="modal-body" style="padding: 0;">
                                    <img src="/static/{{ entry.timestamp }}.webp" alt="Image" style="width: 100%; height: 100%; object-fit: contain; margin: 0 auto;">
                                </div>
                            </div>
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="alert alert-warning" role="alert">
                No results found for your search. Try adjusting the query, app filter, or time range.
            </div>
        {% endif %}
    </div>
    <script>
      (function() {
        const modalIds = Array.from(document.querySelectorAll('.result-card')).map(el => el.dataset.modalId);
        if (!modalIds.length) return;

        let currentIndex = 0;

        function showModalByIndex(idx) {
          const clamped = ((idx % modalIds.length) + modalIds.length) % modalIds.length;
          currentIndex = clamped;
          const id = modalIds[clamped];
          $('.modal').modal('hide');
          $('#' + id).modal('show');
        }

        document.querySelectorAll('.result-card a[data-modal-id]').forEach((link, idx) => {
          link.addEventListener('click', () => {
            currentIndex = idx;
          });
        });

        document.querySelectorAll('.modal .prev-btn').forEach(btn => {
          btn.addEventListener('click', () => showModalByIndex(currentIndex - 1));
        });

        document.querySelectorAll('.modal .next-btn').forEach(btn => {
          btn.addEventListener('click', () => showModalByIndex(currentIndex + 1));
        });

        document.addEventListener('keydown', (e) => {
          const anyOpen = document.querySelector('.modal.show');
          if (!anyOpen) return;
          if (e.key === 'ArrowLeft') {
            e.preventDefault();
            showModalByIndex(currentIndex - 1);
          } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            showModalByIndex(currentIndex + 1);
          }
        });
      })();
    </script>
{% endblock %}
""",
        entries=sorted_entries,
        apps_seen=apps_seen,
        popular_icons=popular_icons,
        default_icon=default_icon,
    )


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(images_path, "favicon.png")


def ensure_single_instance():
    """Prevent multiple instances of the app from running simultaneously."""
    pid_file = Path(appdata_folder) / "openrecall.pid"
    current_pid = os.getpid()

    if pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
        except Exception:
            existing_pid = None

        if existing_pid and existing_pid != current_pid:
            proc_alive = False
            try:
                import psutil

                if psutil.pid_exists(existing_pid):
                    p = psutil.Process(existing_pid)
                    # A stale pid file from a previous crash should not block reuse
                    proc_alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except Exception:
                # If psutil is unavailable or errors, be conservative and assume running
                proc_alive = True

            if proc_alive:
                msg = f"OpenRecall already running with PID {existing_pid}. Exiting."
                logging.error(msg)
                print(msg)
                sys.exit(1)

        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        pid_file.write_text(str(current_pid))
    except Exception as exc:
        logging.warning("Could not write PID file %s: %s", pid_file, exc)


if __name__ == "__main__":
    ensure_single_instance()
    # Apply dock visibility preference on macOS before showing UI
    try:
        from openrecall.settings import load_settings
        settings = load_settings()
        # Lazy import helper to avoid hard dependency when not on macOS
        try:
            from openrecall.trayapp import _set_dock_visibility  # type: ignore
            _set_dock_visibility(settings.show_in_dock)
        except Exception:
            pass
    except Exception:
        pass
    create_db()

    print(f"Appdata folder: {appdata_folder}")

    stop_evt = Event()

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread, args=(stop_evt,), daemon=True)
    t.start()

    # Run Flask in a background thread so we can keep the tray icon on the main thread
    web_thread = Thread(target=app.run, kwargs={"port": 8082, "use_reloader": False}, daemon=True)
    web_thread.start()

    try:
        # Start tray icon in the main thread (more reliable on macOS)
        start_tray_icon_blocking()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        stop_evt.set()
        # Clean up PID file if it still points to this process
        try:
            pid_path = Path(appdata_folder) / "openrecall.pid"
            if pid_path.exists():
                recorded = pid_path.read_text().strip()
                if recorded == str(os.getpid()):
                    pid_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            t.join(timeout=5)
        except Exception:
            pass
        try:
            web_thread.join(timeout=2)
        except Exception:
            pass
        try:
            stop_tray_icon()
        except Exception:
            pass
