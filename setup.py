import io
import platform

from setuptools import find_packages, setup

# Read the README.md file
with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "Flask==3.1.2",
    "numpy==2.2.6",
    "opencv-python-headless==4.12.0.88",
    "mss==10.1.0",
    "sentence-transformers==5.2.0",
    "torch==2.9.1",
    "torchvision==0.24.1",
    "shapely==2.1.2",
    "h5py==3.15.1",
    "rapidfuzz==3.14.3",
    "Pillow==12.1.0",
    "pystray==0.19.5",
]

# Define OS-specific dependencies
extras_require = {
    "windows": ["pywin32", "psutil"],
    "macos": ["pyobjc==10.3"],
    "linux": [],
    "python-doctr": [
        "python-doctr"
    ],
    "dev": [
        "pytest==9.0.2",
    ],
}

# Determine the current OS
current_os = platform.system().lower()
if current_os.startswith("win"):
    current_os = "windows"
elif current_os == "darwin":
    current_os = "macos"
elif current_os == "linux":
    current_os = "linux"
else:
    current_os = None

# Include the OS-specific dependencies if the current OS is recognized
if current_os and current_os in extras_require:
    install_requires.extend(extras_require[current_os])

install_requires.extend(extras_require.get("python-doctr", []))

setup(
    name="OpenRecall",
    version="0.8.1",
    packages=find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require=extras_require,
)
