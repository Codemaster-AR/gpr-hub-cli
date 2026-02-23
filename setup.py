from setuptools import setup, find_packages

setup(
    name="gpr-hub",
    version="4.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "gpr_hub": ["*.mp3"],
    },
    install_requires=[
        "matplotlib",
        "numpy",
        "google-genai",
        "colorama",
        "cinetext",
        "keyboard",
        "KeyboardGate",
        "pygame",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "gpr-hub=gpr_hub.main:run",
        ],
    },
    author="Anay Rustogi",
    description="A GPR Image Reader and AI Analyzer",
    url="https://github.com/Codemaster-AR/GPR-Hub-Python",
)
