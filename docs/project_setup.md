# Project Setup Guide

This explains how to turn a fresh clone of this template into a real project.

## 1. Rename the Package

Rename:
src/project_name/
to:
src/<your_project_name>/

Then update pyproject.toml:

[project]
name = "<your_project_name>"

Your import name becomes:
import your_project_name

## 2. Update environment.yml BEFORE creating the environment

Open environment.yml and:
- rename the environment if you want
- remove dependencies you don't need
- add project-specific dependencies (PyTorch, TensorFlow, Lightning, etc.)

## 3. Create and activate the environment

After editing environment.yml:
conda env create -f environment.yml
conda activate <env-name>

Install the project in editable mode:
pip install -e .

## 4. Adjust the README

Replace the template description with:
- project name
- what the project does
- how to install dependencies
- how to run scripts or notebooks

## 5. Start Working

Common entry points:
- notebooks/ – exploration
- src/<your_project_name>/ – project code
- scripts/ – CLI entry points (train, eval, preprocess)
- configs/ – experiment configs
- tests/ – pytest tests
