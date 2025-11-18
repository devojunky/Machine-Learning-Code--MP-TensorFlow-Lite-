#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
ENV_NAME="xgbenv"
# Find an installed Python 3.11.x version from pyenv, otherwise fallback to a default
PYTHON_VERSION=$(pyenv versions --bare | grep "^3\.11\." | head -n 1)
if [ -z "$PYTHON_VERSION" ]; then
    echo "[WARNING] No Python 3.11 version found in pyenv. Defaulting to 3.11.2."
    echo "Please install a version with 'pyenv install 3.11.2' if this fails."
    PYTHON_VERSION="3.11.2"
fi

PY_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1,2)
REQUIREMENTS_FILE="requirements_pi.txt"

# --- System Library Paths ---
# These are the libraries installed via 'apt' that we need to link.
SYSTEM_SITE_PACKAGES="/usr/lib/python3/dist-packages"
PACKAGES_TO_LINK=(
    "picamera2"
    "yaml" # PyYAML, a dependency for picamera2
    "simplejpeg"
    "v4l2"
    "drm"
    # Add other system packages here if needed
)

echo "--- Starting Environment Setup for '$ENV_NAME' ---"
echo "Using Python version: $PYTHON_VERSION"

# 1. Check for pyenv
if ! command -v pyenv &> /dev/null; then
    echo "[ERROR] pyenv command not found. Please install pyenv first."
    exit 1
fi

# 2. Cleanup existing environment
if pyenv virtualenvs --bare | grep -q "^$ENV_NAME$"; then
    echo "Existing '$ENV_NAME' environment found. Removing it for a clean setup."
    pyenv uninstall -f "$ENV_NAME"
fi

# 3. Create the pyenv virtual environment
echo "Creating new pyenv virtual environment: '$ENV_NAME'"
pyenv virtualenv "$PYTHON_VERSION" "$ENV_NAME"

# 4. Activate the environment (for the current script's context)
export PYENV_VERSION="$ENV_NAME"

# 5. Create symbolic links to system packages
VENV_SITE_PACKAGES=$(pyenv which python | sed 's|/bin/python$|/lib/python'"$PY_MAJOR_MINOR"'/site-packages|')

echo "Linking system libraries to '$VENV_SITE_PACKAGES'..."

for pkg in "${PACKAGES_TO_LINK[@]}"; do
    if [ -e "$SYSTEM_SITE_PACKAGES/$pkg" ]; then
        echo "  - Linking $pkg"
        ln -s "$SYSTEM_SITE_PACKAGES/$pkg" "$VENV_SITE_PACKAGES/"
    else
        echo "  - [WARNING] System package '$pkg' not found in '$SYSTEM_SITE_PACKAGES'. Skipping link."
    fi
done

# 6. Install Python requirements from the file
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing requirements from '$REQUIREMENTS_FILE'..."
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "[WARNING] '$REQUIREMENTS_FILE' not found. Skipping pip install."
fi

echo "--- Setup Complete! ---"
echo "To activate your new environment, run:"
echo "pyenv activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "pyenv deactivate"
