#!/usr/bin/env bash
set -e

# Activate virtual environment (assumes .venv in project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_ROOT="$SCRIPT_DIR/../.venv"
source "$VENV_ROOT/bin/activate"

PYTHON_SITE=$(python -c "import site; print(site.getsitepackages()[0])")

echo "============================================="
echo "PyKDL + kdl_parser Installation Script"
echo "============================================="

# ── 0. Always check and patch Joint.None compatibility ────────────────────
# Newer PyKDL builds do not expose Joint.None; fall back to Fixed.
patch_kdl_parser() {
    PATCHED_FILE="$PYTHON_SITE/kdl_parser/urdf.py"
    if [ -f "$PATCHED_FILE" ] && grep -q "getattr(kdl.Joint, 'None')" "$PATCHED_FILE" 2>/dev/null; then
        sed -i "s/getattr(kdl.Joint, 'None')/getattr(kdl.Joint, 'None', kdl.Joint.Fixed)/g" \
            "$PATCHED_FILE"
        echo "[PATCH] Fixed Joint.None compatibility in $PATCHED_FILE"
    fi
}

# Run patch first (in case package is already installed)
patch_kdl_parser

echo ""

# Check if PyKDL is available
if python -c "import PyKDL" 2>/dev/null; then
    echo "[OK] PyKDL is already importable."
else
    echo "[BUILD] PyKDL not found, building from source..."

    # Clean previous attempts (local to project)
    rm -rf "$SCRIPT_DIR/orocos_kinematics_dynamics"

    git clone https://github.com/orocos/orocos_kinematics_dynamics.git \
        "$SCRIPT_DIR/orocos_kinematics_dynamics"

    cd "$SCRIPT_DIR/orocos_kinematics_dynamics/orocos_kdl"
    mkdir -p build && cd build
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$VENV_ROOT" \
      -DCMAKE_PREFIX_PATH="$VENV_ROOT"
    make -j$(nproc)
    make install

    cd "$SCRIPT_DIR/orocos_kinematics_dynamics/python_orocos_kdl"
    if [ ! -d "pybind11" ]; then
        wget -q https://github.com/pybind/pybind11/archive/refs/tags/v2.13.0.zip -O pybind11.zip
        unzip -q pybind11-2.13.0 pybind11
    fi
    mkdir -p build && cd build
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python) \
      -DCMAKE_INSTALL_PREFIX="$VENV_ROOT" \
      -DCMAKE_PREFIX_PATH="$VENV_ROOT"
    make -j$(nproc)
    make install

    # Ensure PyKDL .so is importable; copy to site-packages if needed
    echo "Verifying PyKDL installation..."
    if ! python -c "import PyKDL" 2>/dev/null; then
        SHARED=$(find "$SCRIPT_DIR/orocos_kinematics_dynamics/python_orocos_kdl" \
                     -name 'PyKDL*.so' | head -n 1)
        if [ -n "$SHARED" ]; then
            cp "$SHARED" "$PYTHON_SITE/"
            echo "Copied $SHARED -> $PYTHON_SITE/"
        else
            echo "ERROR: Could not find built PyKDL .so" >&2; exit 1
        fi
    fi

    # Re-run patch after building PyKDL (in case kdl_parser was installed before)
    patch_kdl_parser
fi

# ---------------------------------------------------------------------------
# Check if kdl_parser is already available
# ---------------------------------------------------------------------------
if python -c "import kdl_parser.urdf" 2>/dev/null; then
    echo "[OK] kdl_parser is already importable."
else
    echo "[INSTALL] Installing kdl_parser..."

    # Clean previous attempts (local to project)
    rm -rf "$SCRIPT_DIR/kdl_parser"

    cd "$SCRIPT_DIR"
    timeout 120 git clone https://github.com/jvytee/kdl_parser.git "$SCRIPT_DIR/kdl_parser" \
        || { echo "ERROR: Failed to clone kdl_parser (timeout or network issue)" >&2; exit 1; }

    cd "$SCRIPT_DIR/kdl_parser"
    uv pip install . 2>&1 || { echo "ERROR: Failed to install kdl_parser via pip" >&2; exit 1; }

    # Verify installation
    if ! python -c "import kdl_parser.urdf" 2>/dev/null; then
        echo "WARNING: kdl_parser installed but not importable!"
        echo "Available modules in site-packages:"
        ls -la "$PYTHON_SITE/" | grep -i kdl || echo "No kdl-related packages found"
    fi

    # Apply patch after installation
    patch_kdl_parser
fi

# ---------------------------------------------------------------------------
# Add venv lib directory to activate script (for liborocos-kdl runtime)
# ---------------------------------------------------------------------------
ACTIVATE="$VENV_ROOT/bin/activate"
if ! grep -q "LD_LIBRARY_PATH.*VIRTUAL_ENV" "$ACTIVATE"; then
    cat >> "$ACTIVATE" << 'HOOK'

# Auto-inject venv lib path for PyKDL (liborocos-kdl) shared library
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
HOOK
    echo "[UPDATE] Updated $ACTIVATE with LD_LIBRARY_PATH."
fi

echo ""
echo "============================================="
echo "✅ PyKDL + kdl_parser installation complete."
echo "============================================="
echo ""
echo "Verification:"
python -c "import PyKDL; print('  PyKDL: OK')" 2>/dev/null || echo "  PyKDL: FAILED"
python -c "import kdl_parser.urdf; print('  kdl_parser.urdf: OK')" 2>/dev/null || echo "  kdl_parser.urdf: FAILED"

# Test the fix
if python -c "from kdl_parser.urdf import treeFromFile; print('  Joint.None patch: OK')" 2>/dev/null; then
    echo ""
fi