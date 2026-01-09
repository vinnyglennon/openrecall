import sys
from setuptools import find_packages, setup
sys.setrecursionlimit(5000)
with open("version.txt", "r", encoding="utf-8") as vf:
    version = vf.read().strip()
APP = ["openrecall/app.py"]
OPTIONS = {
    "argv_emulation": True,
    "plist": {
        "LSUIElement": True
    },  # menu-bar app; remove if you want Dock icon/appear in Cmd+Tab like a regular app
     "includes": [
      "sentence_transformers",
    ],
    "excludes": [
        "pypdfium2", "pypdfium2_raw",  # avoid bundling libpdfium.dylib
        "IPython", "Cython", "PyQt6", "PySide6", "hypothesis", "numba",
        "docutils", "pyinstaller", "watchdog", "docutils", "pyinstaller", "watchdog", "pytest", "mypy", "ipywidgets",
        "decord", "deepspeed", "detectron2", "einops", "faiss", "flash_attn",
        "jax", "jaxlib", "onnxruntime", "tensorflow", "torchaudio", "timm",
        "sentencepiece", "pandas", "matplotlib", "ray", "optimum", "peft",
        "wandb", "gradio",
    ],  # trim heavy libs if not needed in the app bundle
}
setup(
    name="OpenRecall",
    version=version,
    app=APP,
    options={"py2app": OPTIONS},
    packages=find_packages(),
)
