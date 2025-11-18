#!/bin/bash
# setup_python_deps.sh - Install Python dependencies for HW3
# Works with Python 3.6+ and handles permission issues

set -e

echo "=================================================="
echo "Installing Python dependencies for HW3"
echo "=================================================="
echo ""

# Detect Python version
py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.6")
echo "Detected Python version: $py_version"
echo ""

# Install based on Python version
if [[ "$py_version" == "3.6" ]]; then
    echo "Installing Python 3.6 compatible packages..."
    echo "Note: Using pre-compiled wheels to avoid compilation issues"
    
    # Install core dependencies first
    pip3 install --user --only-binary=:all: 'numpy<1.20' 2>/dev/null || pip3 install --user 'numpy<1.20'
    pip3 install --user --only-binary=:all: 'pandas<1.2' 2>/dev/null || pip3 install --user 'pandas<1.2'
    
    # Install matplotlib without Pillow dependency (use minimal backend)
    pip3 install --user --only-binary=:all: 'matplotlib<3.4' 2>/dev/null || pip3 install --user 'matplotlib<3.4'
    
    # Install fpdf2 with compatible dependencies
    pip3 install --user 'fpdf2<2.6.0'
    
    echo ""
    echo "If you see Pillow errors, that's OK - matplotlib will work without it for our use case"
else
    echo "Installing latest packages for Python $py_version..."
    pip3 install --user -r requirements.txt
fi

echo ""
echo "=================================================="
echo "Installation complete!"
echo "=================================================="
echo ""
echo "You can now run: ./run_all.sh"
