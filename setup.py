from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="causal-dags-classifier",
    version="1.0.0",
    author="Pedro AM",
    author_email="pedroam.dev@gmail.com",
    description="Clasificador de Aprendizaje Causal usando DAGs (Directed Acyclic Graphs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pedroam-dev/DAGs",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "causal-demo=causal_classifier:main",
            "marketing-demo=marketing_example:main",
        ],
    },
    keywords=[
        "causal inference",
        "machine learning", 
        "directed acyclic graphs",
        "DAG",
        "causal classification",
        "graphical models",
        "causality",
        "d-separation",
    ],
    project_urls={
        "Bug Reports": "https://github.com/pedroam-dev/DAGs/issues",
        "Source": "https://github.com/pedroam-dev/DAGs",
        "Documentation": "https://github.com/pedroam-dev/DAGs/README.md",
    },
)