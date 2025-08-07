#!/usr/bin/env python3
"""
Setup script for the legally validated voice anonymization evaluation framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="legal-eval",
    version="1.0.0",
    author="Nijta SAS",
    author_email="brij@nijta.com",
    description="Legally validated evaluation framework for voice anonymization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brijmohan/voice-anonymization-legal-eval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "legal-eval=legal_eval.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "legal_eval": ["*.yaml", "*.json"],
    },
    keywords=[
        "voice anonymization",
        "privacy",
        "legal compliance",
        "speaker recognition",
        "evaluation metrics",
        "data protection",
        "GDPR",
    ],
    project_urls={
        "Bug Reports": "https://github.com/brijmohan/voice-anonymization-legal-eval/issues",
        "Source": "https://github.com/brijmohan/voice-anonymization-legal-eval",
        "Documentation": "https://github.com/brijmohan/voice-anonymization-legal-eval#readme",
    },
)
