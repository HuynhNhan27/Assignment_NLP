"""
Setup configuration for the project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="text-to-kg",
    version="0.1.0",
    author="NLP Team",
    description="Educational Text to Knowledge Graph Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "spacy>=3.7.0",
        "nltk>=3.8.0",
        "sentence-transformers>=2.3.0",
        "networkx>=3.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.23.0",
        "scikit-learn>=1.3.0",
        "flask>=3.0.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
)
