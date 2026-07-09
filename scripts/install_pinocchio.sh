#!/usr/bin/env bash
# Restore Pinocchio with CasADi bindings.
#
# The pip-installed `pin` wheel (from PyPI) provides the Python module but
# does NOT include the casadi submodule — that must be compiled from source.
#
# This script:
#   1. Installs pure-Python deps (pin, eigenpy) via uv pip.
#   2. Creates Eigen3::Eigen cmake shim in $VIRTUAL_ENV/lib/cmake/.
#   3. Clones pinocchio from GitHub and builds ONLY the C++ library with CASADI.
#      Uses **system Boost** (compatible version) instead of cmeel.prefix's partial Boost 1.90.0.
#   4. Installs the built libraries into $VIRTUAL_ENV, overwriting the pip wheel's.
#      (The pip wheel's Python bindings are kept — they don't need rebuilding.)
#   5. Generates env_vars.sh with LD_LIBRARY_PATH needed for runtime.
#
# Usage (inside .venv):
#   bash scripts/install_pinocchio.sh [--rebuild]
#
# After installation, run:
#   source ./env_vars.sh   # Sets LD_LIBRARY_PATH for pinocchio/casadi
#   export LD_LIBRARY_PATH=...  # Or set manually before running Python
#
# Architecture: both aarch64 and amd64 are supported.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REBUILD=false
if [[ "${1:-}" == "--rebuild" ]]; then REBUILD=true; fi

# ── 0. Detect architecture ────────────────────────────────────────────────
ARCH="$(uname -m)"
echo "Architecture: $ARCH (aarch64/amd64 both supported)"

# ── 0a. Find system Boost and CasADi library paths for runtime ────────────
# These will be used to generate env_vars.sh
SYS_BOOST_LIB_DIR=""
for _d in /usr/lib/{aarch64-linux-gnu,x86_64-linux-gnu,arm-linux-gnueabihf} /usr/lib64; do
    [ -f "$_d/libboost_system.so" ] && { SYS_BOOST_LIB_DIR="$_d"; break; }
done

if [ -z "$SYS_BOOST_LIB_DIR" ]; then
    echo "==> WARNING: system libboost_system.so not found. Will attempt to use pip boost."
fi

CASADI_DIR="$(python3 -c "import casadi; import os; print(os.path.dirname(casadi.__file__))")"
CMEEL_PREFIX="$VIRTUAL_ENV/lib/python3.12/site-packages/cmeel.prefix/lib"

# ── 1. Install pure-Python deps via uv ─────────────────────────────────────
echo "==> Installing pure-Python dependencies ..."
uv pip install "pin>=4.0.0"

# Remove the pip-installed eigenpy — we'll build it from source below to match
# system Boost version (pip wheel is built with Boost 1.90.0, but system has 1.83).
uv pip uninstall --yes eigenpy 2>/dev/null || true

# ── 2. System dependencies ────────────────────────────────────────────────
EIGEN3_FOUND=false
if [ -d /usr/include/eigen3 ]; then
    EIGEN3_FOUND=true
elif command -v apt-get &>/dev/null && sudo -n apt-get install -y --dry-run libeigen3-dev &>/dev/null; then
    echo "==> Installing libeigen3-dev ..."
    sudo apt-get update -qq
    sudo apt-get install -y libeigen3-dev
    EIGEN3_FOUND=true
fi

if [[ "$EIGEN3_FOUND" == true ]]; then
    EIGEN3_INCLUDE="/usr/include/eigen3"
else
    EIGEN3_INCLUDE=""
    echo "==> Eigen3 headers not found via apt; will use bundled (casadi) or download at build time."
fi

# ── 2a. Fix cmake shims in $VIRTUAL_ENV/lib/cmake/ ────────────────────────
# We place Eigen3::Eigen target here (NOT in cmeel.prefix) so cmake finds it
# via CMAKE_INSTALL_PREFIX search path without needing cmeel.prefix on the
# CMAKE_PREFIX_PATH (which would bring broken Boost 1.90.0 configs).

CASADI_DIR_SHIM="$(python3 -c "import casadi; import os; print(os.path.dirname(casadi.__file__))")"

if [ -d /usr/include/eigen3 ]; then
    EIGEN_DIR="/usr/include/eigen3"
elif [ -d "${CASADI_DIR_SHIM}/include/eigen3" ]; then
    EIGEN_DIR="${CASADI_DIR_SHIM}/include/eigen3"
else
    EIGEN_DIR="${VIRTUAL_ENV}/include"
fi

# Eigen3::Eigen imported target — placed in $VIRTUAL_ENV/lib/cmake/ so cmake
# finds it automatically when CMAKE_INSTALL_PREFIX is set to $VIRTUAL_ENV.
EIGEN_CONFIG_DIR="$VIRTUAL_ENV/lib/cmake/Eigen3"
mkdir -p "$EIGEN_CONFIG_DIR"
cat > "$EIGEN_CONFIG_DIR/Eigen3Config.cmake" << EOF
set(Eigen3_INCLUDE_DIRS "${EIGEN_DIR}")
set(Eigen3_VERSION "3.4.0")
set(Eigen3_VERSION_MAJOR 3)
set(Eigen3_VERSION_MINOR 4)
set(Eigen3_VERSION_PATCH 0)

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_DIR}")
endif()
EOF

# Version file (needed by some find_package callers)
cat > "$EIGEN_CONFIG_DIR/Eigen3ConfigVersion.cmake" << 'EOF'
set(PACKAGE_VERSION "3.4.0")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
EOF

echo "   Eigen3 shim created at $EIGEN_CONFIG_DIR"

# ── 3. Build pinocchio from source with CASADI support ─────────────────────
PIN_DIR="$PROJECT_ROOT/pinocchio"

if [[ "$REBUILD" == true ]] || [ ! -d "$PIN_DIR/.git" ]; then
    echo "==> Building pinocchio from source ..."

    if [ -d "$PIN_DIR" ]; then rm -rf "$PIN_DIR"; fi

    # high version cannot cmake example-robot-data
    git clone https://github.com/stack-of-tasks/pinocchio.git "$PIN_DIR"
    cd "$PIN_DIR" && git reset --hard bb5658416724a36d5e8d2fb6c65614f39796f7f1 && cd -

    # Initialize cmake submodule (needed for jrl-cmakemodules)
    echo "   Initializing git submodules..."
    cd "$PIN_DIR" && git submodule update --init cmake && cd -
else
    echo "==> Pinocchio source already cloned — skip.  Use --rebuild to force."
fi

if [[ "$REBUILD" == true ]] || [ ! -d "$PIN_DIR/build" ]; then
    mkdir -p "$PIN_DIR/build"
fi

cd "$PIN_DIR/build"

CASADI_DIR="$(python3 -c "import casadi; import os; print(os.path.dirname(casadi.__file__))")"
CASADE_DIR_FOR_CMAKE="${CASADI_DIR}/cmake"

# Build Eigen3 include path from available sources
if [ -d /usr/include/eigen3 ]; then
    EIGEN3_INCLUDE="/usr/include/eigen3"
elif [ -d "$CASADI_DIR/include/eigen3" ]; then
    EIGEN3_INCLUDE="$CASADI_DIR/include/eigen3"
else
        TMP_EIGEN="$(pwd)/_deps/eigen3"
        mkdir -p "$TMP_EIGEN"
        echo "   Downloading Eigen3 3.4.0 headers ..."
        curl -fsSL --retry 3 \
            "https://github.com/eigenteam/eigen-git-mirror/archive/refs/tags/3.4.0.tar.gz" \
            | tar -xz --strip-components=1 -C "$TMP_EIGEN"
        EIGEN3_INCLUDE="$TMP_EIGEN"
    fi

    # NOTE: cmeel.prefix is excluded from CMAKE_PREFIX_PATH to avoid its
    # partial Boost 1.90.0 configs (missing system component). Eigen3::Eigen
    # is available via $VIRTUAL_ENV/lib/cmake/Eigen3/ (created in step 2a).
    export CMAKE_PREFIX_PATH="${CASADE_DIR_FOR_CMAKE}:${CMAKE_PREFIX_PATH:-}"

    # ABI match: casadi/cmeel boost was built with _GLIBCXX_USE_CXX11_ABI=0.
    # GCC 10+ changed parameter passing for class types between C++14/C++17, so
    # the ABI flag must be shared or it causes Boost.Python link errors.
    export CMAKE_CXX_FLAGS="-I${VIRTUAL_ENV}/include -L${VIRTUAL_ENV}/lib -D_GLIBCXX_USE_CXX11_ABI=0 ${CMAKE_CXX_FLAGS:-}"

    # ── Boost: force system Boost (not cmeel.prefix partial one) ──────────────
    # On aarch64, system boost libs are in /usr/lib/<triplet>/, not in
    # cmeel.prefix.  FindBoost prefers CMAKE_PREFIX_PATH entries and will
    # find the wrong version there — we override everything explicitly.
    SYS_BOOST_LIB_DIR=""
    for _d in /usr/lib/{aarch64-linux-gnu,x86_64-linux-gnu,arm-linux-gnueabihf} /usr/lib64; do
        [ -f "$_d/libboost_system.so" ] && { SYS_BOOST_LIB_DIR="$_d"; break; }
    done
    if [ -z "$SYS_BOOST_LIB_DIR" ]; then
        echo "  ERROR: cannot find system libboost_system.so — aborting." >&2
        exit 1
    fi

    # Wipe stale CMakeCache that may hold cmeel.prefix Boost paths
    rm -f CMakeCache.txt

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
        -DPYTHON_EXECUTABLE="$(which python3)" \
        -DBUILD_WITH_CASADI_SUPPORT=ON \
        -DCASADI_INCLUDE_DIR="$CASADI_DIR/include" \
        -DCASADI_LIBRARY="$CASADI_DIR/libcasadi.so" \
        -DEigen3_INCLUDE_DIR="$EIGEN3_INCLUDE" \
        -DPINOCCHIO_INSTALL_MODEL=OFF \
        -DBUILD_WITH_URDF_SUPPORT=OFF \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_PYTHON_INTERFACE=ON \
        -DBoost_NO_BOOST_CMAKE=ON \
        -DBoost_INCLUDE_DIR="/usr/include" \
        -DBoost_LIBRARY_DIR="$SYS_BOOST_LIB_DIR"

    # Fix pinocchio_casadi target: add missing casadi include directory.
    # Pinocchio's CMakeLists.txt doesn't properly propagate casadi's include dirs to the target.
    # We patch the generated flags.make to add -I<path-to-casadi/include>.
    sed -i 's|-isystem '"$CASADI_DIR"'/include/eigen3|-isystem '"$CASADI_DIR"'/include/eigen3 -isystem '"$CASADI_DIR"'/include|g' \
        "$PIN_DIR/build/src/CMakeFiles/pinocchio_casadi.dir/flags.make"

    # Fix link.txt: CasADi's cmake config uses `-lcasadi` without full path, but the library
    # is in pip package at <casadi_dir>/libcasadi.so. We need to use full path for linking.
    sed -i 's|-lcasadi /usr/lib/|'"$CASADI_DIR"'/libcasadi.so /usr/lib/|g' \
        "$PIN_DIR/build/src/CMakeFiles/pinocchio_casadi.dir/link.txt"

    # Also fix Python bindings for casadi (pinocchio_pywrap_casadi) if building with python interface
    if [ -f "$PIN_DIR/build/bindings/python/CMakeFiles/pinocchio_pywrap_casadi.dir/flags.make" ]; then
        sed -i 's|-isystem '"$CASADI_DIR"'/include/eigen3|-isystem '"$CASADI_DIR"'/include/eigen3 -isystem '"$CASADI_DIR"'/include|g' \
            "$PIN_DIR/build/bindings/python/CMakeFiles/pinocchio_pywrap_casadi.dir/flags.make"
    fi

    # Also fix link.txt for pinocchio_pywrap_casadi (same casadi library path issue)
    if [ -f "$PIN_DIR/build/bindings/python/CMakeFiles/pinocchio_pywrap_casadi.dir/link.txt" ]; then
        sed -i 's|-lcasadi /usr/lib/|'"$CASADI_DIR"'/libcasadi.so /usr/lib/|g' \
            "$PIN_DIR/build/bindings/python/CMakeFiles/pinocchio_pywrap_casadi.dir/link.txt"
    fi

    make -j"$(nproc)"
    make install
    cd "$PROJECT_ROOT"
    echo "   C++ library + Python bindings built and installed OK."

# ── 4. Install eigenpy from source (needed for some visualizers) ────────────
EIGENPY_DIR="$PROJECT_ROOT/eigenpy"

if [[ "$REBUILD" == true ]] || [ ! -d "$EIGENPY_DIR/.git" ]; then
    echo "==> Building eigenpy from source ..."

    if [ -d "$EIGENPY_DIR" ]; then rm -rf "$EIGENPY_DIR"; fi

    git clone --depth 1 https://github.com/stack-of-tasks/eigenpy.git "$EIGENPY_DIR"
    cd "$EIGENPY_DIR"
    mkdir -p build && cd build

    CASADI_DIR="$(python3 -c "import casadi; import os; print(os.path.dirname(casadi.__file__))")"
    CASADE_DIR_FOR_CMAKE="${CASADI_DIR}/cmake"

    if [ -d /usr/include/eigen3 ]; then
        EIGEN3_INCLUDE="/usr/include/eigen3"
    elif [ -d "$CASADI_DIR/include/eigen3" ]; then
        EIGEN3_INCLUDE="$CASADI_DIR/include/eigen3"
    fi

    export CMAKE_PREFIX_PATH="${CASADE_DIR_FOR_CMAKE}:${EIGENPY_DIR}"

    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
        -DPYTHON_EXECUTABLE="$(which python3)" \
        -DEIGENPY_USE_EIGEN_FOR_RANDOM=ON \
        -DBUILD_TESTING=OFF ..

    make -j"$(nproc)"
    make install
    cd "$PROJECT_ROOT"
    echo "   eigenpy built OK."
else
    echo "==> Eigenpy source already cloned — skip.  Use --rebuild to force."
fi

# ── 5. Generate env_vars.sh for runtime library path ──────────────────────
echo ""
echo "==> Generating env_vars.sh ..."

# Build the LD_LIBRARY_PATH that includes:
#   - CasADi's lib (contains casadi.so)
#   - System Boost (compatible version, e.g., 1.83.0)
#   - CMEEL prefix (for any remaining dependencies from pin wheel)
ENV_SCRIPT="$PROJECT_ROOT/env_vars.sh"

cat > "$ENV_SCRIPT" << EOF
#!/usr/bin/env bash
# Auto-generated by install_pinocchio.sh
# Source this file to set LD_LIBRARY_PATH for Pinocchio + CasADi runtime

export CASADI_LIB_PATH="${CASADI_DIR}"
export BOOST_LIB_PATH="${SYS_BOOST_LIB_DIR:-/usr/lib/\$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo lib)}"
export CMEEL_PREFIX_PATH="${CMEEL_PREFIX}"

# Combine all library paths (system boost takes precedence over cmeel)
if [ -n "${SYS_BOOST_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="\${BOOST_LIB_PATH}:\${CASADI_LIB_PATH}:\${CMEEL_PREFIX_PATH}:\${LD_LIBRARY_PATH:-}"
else
    # Fallback: use only CasADi and cmeel
    export LD_LIBRARY_PATH="\${CASADI_LIB_PATH}:\${CMEEL_PREFIX_PATH}:\${LD_LIBRARY_PATH:-}"
fi

echo "LD_LIBRARY_PATH set to:"
echo "  \${LD_LIBRARY_PATH}"
EOF

chmod +x "$ENV_SCRIPT"
echo "   Created: $ENV_SCRIPT"
echo ""
echo "   To activate before running Python scripts:"
echo "     source ./env_vars.sh"
echo ""

# Also create a small auto-setup script that can be used with LD_PRELOAD approach
cat > "$PROJECT_ROOT/setup_pinocchio_env.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Auto-configure LD_LIBRARY_PATH for Pinocchio + CasADi runtime.
Import this module before importing pinocchio to ensure correct shared libraries are found.

Usage:
    from setup_pinocchio_env import setup_pinocchio_libs
    setup_pinocchio_libs()
    import pinocchio
"""

import os
import sys


def setup_pinocchio_libs():
    """Set LD_LIBRARY_PATH for Pinocchio runtime dependencies."""
    lib_paths = []

    # CasADi library path (from installed package)
    try:
        import casadi
        casadi_dir = os.path.dirname(casadi.__file__)
        casadi_lib = os.path.join(casadi_dir, "lib")
        if os.path.isdir(casadi_lib):
            lib_paths.append(casadi_lib)
    except ImportError:
        pass

    # System Boost (prefer system over cmeel)
    import platform
    machine = platform.machine()
    sys_boost_dirs = []
    for d in [f"/usr/lib/{m}" for m in ["aarch64-linux-gnu", "x86_64-linux-gnu", "arm-linux-gnueabihf"]] + ["/usr/lib64", "/usr/lib"]:
        if os.path.exists(os.path.join(d, "libboost_system.so")):
            sys_boost_dirs.append(d)
    if sys_boost_dirs:
        lib_paths.insert(0, sys_boost_dirs[0])  # Put system boost first

    # CMEEL prefix (for remaining dependencies from pin wheel)
    cmeel_prefix = os.path.join(sys.prefix, "lib", "python3.12", "site-packages", "cmeel.prefix", "lib")
    if os.path.isdir(cmeel_prefix):
        lib_paths.append(cmeel_prefix)

    # Update LD_LIBRARY_PATH
    if lib_paths:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        new_path = ":".join(lib_paths + [current] if current else lib_paths)
        os.environ["LD_LIBRARY_PATH"] = new_path
        print(f"[setup_pinocchio_env] LD_LIBRARY_PATH updated with {len(lib_paths)} paths")


# Auto-run on import if PINOCCHIO_AUTO_SETUP is set
if os.environ.get("PINOCCHIO_AUTO_SETUP", "").lower() in ("1", "true", "yes"):
    setup_pinocchio_libs()
PYEOF

chmod +x "$PROJECT_ROOT/setup_pinocchio_env.py"
echo "   Created: setup_pinocchio_env.py (auto-configure helper)"

# ── 6. Verify ─────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation ..."

# Test with env_vars.sh sourced first
if [ -n "$SYS_BOOST_LIB_DIR" ]; then
    echo "   Testing with LD_LIBRARY_PATH..."
    export LD_LIBRARY_PATH="${SYS_BOOST_LIB_DIR}:${CASADI_DIR}/lib:${CMEEL_PREFIX}:${LD_LIBRARY_PATH:-}"
fi

python3 -c "import pinocchio; print('  pinocchio', getattr(pinocchio, '__version__', 'OK'))"
python3 -c "from pinocchio import casadi as cpin; print('  pinocchio.casadi  OK')" 2>&1 || echo "  WARNING: casadi submodule not found — check build log above."

echo ""
echo "==> Installation complete!"
echo ""
