from setuptools import find_packages, setup

with open("version.txt", "r", encoding="utf-8") as vf:
    version = vf.read().strip()
APP = ["openrecall/app.py"]
OPTIONS = {
    "argv_emulation": True,
    "plist": {
        "LSUIElement": True
    },  # menu-bar app; remove if you want Dock icon/appear in Cmd+Tab like a regular app
    "excludes": [
        "torch",
        "torchvision",
        "sentence_transformers",
        "doctr",
    ],  # trim heavy libs if not needed in the app bundle
}
setup(
    name="OpenRecall",
    version=version,
    app=APP,
    options={"py2app": OPTIONS},
    packages=find_packages(),
)
