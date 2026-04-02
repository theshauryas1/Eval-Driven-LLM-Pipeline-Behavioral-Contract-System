from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="llmtest",
    version="0.1.0",
    description="Local Behavioral Contract Testing CLI",
    packages=find_packages(),
    install_requires=[
        "click",
        "pydantic",
        "langchain",
        "langchain-groq",
        "pyyaml"
    ],
    entry_points={
        "console_scripts": [
            "llmtest=cli.cli:cli",
        ],
    },
)
