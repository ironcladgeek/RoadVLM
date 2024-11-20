#!/bin/bash

# Get real path of the parent directory of this script
SCRIPTDIR=$(dirname $(realpath $0))
BASEDIR=$(dirname $SCRIPTDIR)
alias uv=$HOME/.local/bin/uv

echo "Setting up environment for project..."
echo "Base directory: $BASEDIR"

# Install 'uv' if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing 'uv'..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $BASEDIR/.venv/bin/activate

# Install project dependencies
echo "Installing project dependencies..."
uv sync

# Install 'ollama' if not already installed and on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! command -v ollama &> /dev/null; then
        echo "Installing 'ollama'..."
        curl -LsSf https://ollama.sh/install.sh | sh
    fi
fi
