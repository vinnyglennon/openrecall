from threading import Thread
from pathlib import Path
from datetime import datetime
import numpy as np
from flask import Flask, render_template_string, request, send_from_directory
from jinja2 import BaseLoader

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import (
    create_db,
    get_all_entries,
    get_timestamps,
    get_entries_by_time_range,
)
from openrecall.nlp import cosine_similarity, get_embedding
from openrecall.screenshot import record_screenshots_thread
from openrecall.trayapp import start_tray_icon_async, start_tray_icon_blocking
from openrecall.utils import human_readable_time, timestamp_to_human_readable

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


@app.route("/search")
def search():
    q = request.args.get("q")
    start_time_str = request.args.get("start_time")
    end_time_str = request.args.get("end_time")
    app_filter = request.args.get("app")

    entries = get_all_entries()

    if start_time_str:
        start_time = int(
            datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M").timestamp()
        )
        if end_time_str:
            end_time = int(datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M").timestamp())
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
                        <div class="card shadow-sm h-100">
                            <a href="#" data-toggle="modal" data-target="#modal-{{ loop.index0 }}">
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
                            <div class="modal-content" style="height: calc(100vh - 40px); width: calc(100vw - 40px); padding: 0;">
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


if __name__ == "__main__":
    create_db()

    print(f"Appdata folder: {appdata_folder}")

    # Start the thread to record screenshots
    t = Thread(target=record_screenshots_thread, daemon=True)
    t.start()

    # Run Flask in a background thread so we can keep the tray icon on the main thread
    web_thread = Thread(target=app.run, kwargs={"port": 8082}, daemon=True)
    web_thread.start()

    # Start tray icon in the main thread (more reliable on macOS)
    start_tray_icon_blocking()
