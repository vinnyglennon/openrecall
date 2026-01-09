import os
import time
import threading
from typing import List

import mss
import numpy as np
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False
    raise ImportError(
        "OpenCV (cv2) is required for fast SSIM. Install opencv-python or opencv-python-headless."
    )

from openrecall import state
from openrecall.config import args, screenshots_path
from openrecall.database import insert_entry, delete_entries_older_than
from openrecall.settings import (
    EXCLUDED_DOMAIN_DEFAULTS,
    HIGH_RISK_OCR_DEFAULTS,
    SENSITIVE_DEFAULTS,
    load_settings,
)
from openrecall.utils import (
    get_active_app_name,
    get_active_window_title,
    is_user_active,
)

RETENTION_SECONDS = {
    "1 week": 7 * 24 * 3600,
    "1 month": 30 * 24 * 3600,
    "3 months": 90 * 24 * 3600,
    "6 months": 180 * 24 * 3600,
    "1 year": 365 * 24 * 3600,
    "Forever": None,
}


def _retention_cutoff(retention: str) -> int | None:
    """Return cutoff timestamp for retention label, or None to keep forever."""
    seconds = RETENTION_SECONDS.get(retention, None)
    if seconds is None:
        return None
    return int(time.time()) - seconds


def _cleanup_old_entries(cutoff_timestamp: int) -> None:
    """Delete db entries and screenshot files older than cutoff."""
    if cutoff_timestamp is None:
        return
    delete_entries_older_than(cutoff_timestamp)
    # Remove old screenshot files
    try:
        for fname in os.listdir(screenshots_path):
            if not fname.endswith(".webp"):
                continue
            try:
                ts = int(os.path.splitext(fname)[0])
            except ValueError:
                continue
            if ts < cutoff_timestamp:
                try:
                    os.remove(os.path.join(screenshots_path, fname))
                except OSError:
                    pass
    except FileNotFoundError:
        pass


def mean_structured_similarity_index(
    img1: np.ndarray, img2: np.ndarray, L: int = 255
) -> float:
    """Calculates the Mean Structural Similarity Index (MSSIM) between two images.

    Args:
        img1: The first image as a NumPy array (RGB).
        img2: The second image as a NumPy array (RGB).
        L: The dynamic range of the pixel values (default is 255).

    Returns:
        The MSSIM value between the two images (float between -1 and 1).
    """
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2

    def rgb2gray(img: np.ndarray) -> np.ndarray:
        """Converts an RGB image to grayscale."""
        return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]

    img1_gray: np.ndarray = rgb2gray(img1)
    img2_gray: np.ndarray = rgb2gray(img2)
    mu1: float = np.mean(img1_gray)
    mu2: float = np.mean(img2_gray)
    sigma1_sq = np.var(img1_gray)
    sigma2_sq = np.var(img2_gray)
    sigma12 = np.mean((img1_gray - mu1) * (img2_gray - mu2))
    ssim_index = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_index


def _fast_ssim_cv(img1: np.ndarray, img2: np.ndarray) -> float:
    """Fast SSIM using OpenCV (grayscale)."""
    # Ensure uint8 and grayscale
    if img1.dtype != np.uint8:
        img1 = img1.astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = img2.astype(np.uint8)
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    C1 = 6.5025
    C2 = 58.5225

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(cv2.mean(ssim_map)[0])


def is_similar(
    img1: np.ndarray, img2: np.ndarray, similarity_threshold: float = 0.9
) -> bool:
    """Checks if two images are similar based on MSSIM.

    Args:
        img1: The first image as a NumPy array.
        img2: The second image as a NumPy array.
        similarity_threshold: The threshold above which images are considered similar.

    Returns:
        True if the images are similar, False otherwise.
    """
    # Compress images to reduce size and improve performance
    compress_img1: np.ndarray = resize_image(img1)
    compress_img2: np.ndarray = resize_image(img2)

    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV not available; fast SSIM required.")

    similarity: float = _fast_ssim_cv(compress_img1, compress_img2)
    return similarity >= similarity_threshold


def take_screenshots() -> List[np.ndarray]:
    """Takes screenshots of all connected monitors or just the primary one.

    Depending on the `args.primary_monitor_only` flag, captures either
    all monitors or only the primary monitor (index 1 in mss.monitors).

    Returns:
        A list of screenshots, where each screenshot is a NumPy array (RGB).
    """
    screenshots: List[np.ndarray] = []
    with mss.mss() as sct:
        # sct.monitors[0] is the combined view of all monitors
        # sct.monitors[1] is the primary monitor
        # sct.monitors[2:] are other monitors
        monitor_indices = range(1, len(sct.monitors))  # Skip the 'all monitors' entry

        if args.primary_monitor_only:
            monitor_indices = [1]  # Only index 1 corresponds to the primary monitor

        for i in monitor_indices:
            # Ensure the index is valid before attempting to grab
            if i < len(sct.monitors):
                monitor_info = sct.monitors[i]
                # Grab the screen
                sct_img = sct.grab(monitor_info)
                # Convert to numpy array and change BGRA to RGB
                screenshot = np.array(sct_img)[:, :, [2, 1, 0]]
                screenshots.append(screenshot)
            else:
                # Handle case where primary_monitor_only is True but only one monitor exists (all monitors view)
                # This case might need specific handling depending on desired behavior.
                # For now, we just skip if the index is out of bounds.
                print(f"Warning: Monitor index {i} out of bounds. Skipping.")

    return screenshots


def _ensure_pil():
    """Lazy-load minimal PIL plugins (WebP only) to speed startup."""
    global _PIL_Image
    if _PIL_Image is not None:
        return _PIL_Image
    # Limit plugin loading before importing PIL
    os.environ.setdefault("PILLOW_DISABLE_PLUGIN_LOADING", "1")
    from PIL import Image
    try:
        from PIL import WebPImagePlugin  # noqa: F401
    except Exception:
        pass
    _PIL_Image = Image
    return _PIL_Image


def resize_image(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resizes an image to fit within a maximum dimension while maintaining aspect ratio.

    Args:
        image: The input image as a NumPy array (RGB).
        max_dim: The maximum dimension for resizing.

    Returns:
        The resized image as a NumPy array (RGB).
    """
    Image = _ensure_pil()
    pil_image = Image.fromarray(image)
    pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return np.array(pil_image)


_lazy_get_embedding = None
_lazy_extract_text = None
_PIL_Image = None
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2 default


def _ensure_nlp_ocr():
    """Lazy import heavy NLP/OCR only when first needed."""
    global _lazy_get_embedding, _lazy_extract_text
    if _lazy_get_embedding is None:
        from openrecall.nlp import get_embedding

        _lazy_get_embedding = get_embedding
    if _lazy_extract_text is None:
        from openrecall.ocr import extract_text_from_image

        _lazy_extract_text = extract_text_from_image


def _is_sensitive(title: str, patterns: list[str]) -> bool:
    """Return True if title contains any sensitive pattern (case-insensitive)."""
    title_cf = (title or "").casefold()
    for pat in patterns:
        if pat and pat.casefold() in title_cf:
            return True
    return False


def _is_excluded_domain(title: str, domains: list[str]) -> bool:
    """Return True if title contains any excluded domain hint (case-insensitive)."""
    title_cf = (title or "").casefold()
    for dom in domains:
        if dom and dom.casefold() in title_cf:
            return True
    return False


def _contains_high_risk_ocr(text: str, triggers: list[str]) -> bool:
    """Return True if OCR text contains any high-risk trigger (case-insensitive)."""
    if not text:
        return False
    text_cf = text.casefold()
    for trig in triggers:
        if trig and trig.casefold() in text_cf:
            return True
    return False


def _make_restricted_image(shape: tuple[int, ...]) -> np.ndarray:
    """Return a placeholder image with 'RESTRICTED' text."""
    Image = _ensure_pil()
    from PIL import ImageDraw, ImageFont

    height, width = shape[0], shape[1]
    img = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Helvetica", size=max(24, width // 20))
    except Exception:
        font = ImageFont.load_default()
    text = "RESTRICTED"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pos = ((width - text_w) // 2, (height - text_h) // 2)
    draw.text(pos, text, fill=(220, 50, 50), font=font)
    return np.array(img)


def record_screenshots_thread(stop_event: threading.Event | None = None) -> None:
    """
    Continuously records screenshots, processes them, and stores relevant data.

    Checks for user activity and image similarity before processing and saving
    screenshots, associated OCR text, embeddings, and active application info.
    Runs in an infinite loop, intended to be executed in a separate thread.
    """
    # TODO: Move this environment variable setting to the application's entry point.
    # HACK: Prevents a warning/error from the huggingface/tokenizers library
    # when used in environments where multiprocessing fork safety is a concern.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    last_screenshots: List[np.ndarray] = take_screenshots()

    settings = load_settings()
    last_settings_refresh = time.time()
    last_retention_cleanup = 0.0

    stop_evt = stop_event or threading.Event()

    while not stop_evt.is_set():
        # Refresh settings periodically (every 60s)
        now = time.time()
        if now - last_settings_refresh > 60:
            settings = load_settings()
            last_settings_refresh = now

        # Apply retention cleanup every 30 minutes
        if now - last_retention_cleanup > 1800:
            cutoff = _retention_cutoff(settings.retention)
            if cutoff:
                _cleanup_old_entries(cutoff)
            last_retention_cleanup = now

        if state.is_paused():
            time.sleep(3)
            continue

        if not is_user_active():
            time.sleep(3)  # Wait longer if user is inactive
            continue

        current_screenshots: List[np.ndarray] = take_screenshots()

        # Ensure we have a last_screenshot for each current_screenshot
        # This handles cases where monitor setup might change (though unlikely mid-run)
        if len(last_screenshots) != len(current_screenshots):
            # If monitor count changes, reset last_screenshots and continue
            last_screenshots = current_screenshots
            time.sleep(3)
            continue

        for i, current_screenshot in enumerate(current_screenshots):
            last_screenshot = last_screenshots[i]

            if not is_similar(current_screenshot, last_screenshot):
                last_screenshots[i] = current_screenshot  # Update the last screenshot for this monitor
                sanitized_image = current_screenshot
                timestamp = int(time.time())
                filename = (
                    f"{timestamp}.webp"  # Align filename with search page expectations
                )
                filepath = os.path.join(screenshots_path, filename)
                # Apply privacy/whitelist rules before processing OCR
                active_app_name: str = get_active_app_name() or "Unknown App"
                active_window_title: str = get_active_window_title() or "Unknown Title"

                # Whitelist behavior: if a whitelist is set, only process those apps
                if settings.whitelist:
                    if active_app_name not in settings.whitelist:
                        continue
                else:
                    # No whitelist set: do not record until user selects apps
                    continue

                # Skip incognito/private windows by window title hint
                if (
                    settings.incognito_block
                    and "incognito" in active_window_title.lower()
                ):
                    continue

                # Skip sensitive window titles
                patterns = settings.sensitive_patterns or SENSITIVE_DEFAULTS
                if _is_sensitive(active_window_title, patterns):
                    continue

                domains = settings.excluded_domains or []
                if _is_excluded_domain(active_window_title, domains):
                    continue

                _ensure_nlp_ocr()

                text: str = _lazy_extract_text(current_screenshot)

                triggers = settings.high_risk_ocr_triggers or HIGH_RISK_OCR_DEFAULTS
                high_risk = _contains_high_risk_ocr(text, triggers)
                if high_risk:
                    sanitized_image = _make_restricted_image(current_screenshot.shape)
                    text = "RESTRICTED"
                    embedding: np.ndarray = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                else:
                    # Only proceed if OCR actually extracts text
                    if text.strip():
                        embedding: np.ndarray = _lazy_get_embedding(text)
                    else:
                        continue

                Image = _ensure_pil()
                image = Image.fromarray(sanitized_image)
                image.save(
                    filepath,
                    format="webp",
                    lossless=True,
                )

                insert_entry(
                    text, timestamp, embedding, active_app_name, active_window_title
                )

        time.sleep(3)  # Wait before taking the next screenshot
    return None


def resize_image(image: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """
    Resizes an image to fit within a maximum dimension while maintaining aspect ratio.

    Args:
        image: The input image as a NumPy array (RGB).
        max_dim: The maximum dimension for resizing.

    Returns:
        The resized image as a NumPy array (RGB).
    """
    Image = _ensure_pil()
    pil_image = Image.fromarray(image)
    pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return np.array(pil_image)
