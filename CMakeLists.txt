# Root CMake — drives the Python wheel build via scikit-build-core.
#
# scikit-build-core invokes this file when `pip install .` / `python -m
# build` is run. We turn on SWIG and direct the SWIG outputs into the
# wheel's `iq_saver/` package directory. The standalone C++ install of
# libiqsaver (pkg-config, headers under /usr/local) is driven directly
# from the libiqsaver/ subdirectory CMake; see libiqsaver/CMakeLists.txt.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# Tell libiqsaver/CMakeLists.txt to: (1) build SWIG bindings, (2) skip
# the system-wide install logic, (3) install SWIG outputs into the
# wheel's iq_saver/ package directory.
set(LIBIQSAVER_ENABLE_SWIG   ON      CACHE BOOL   "" FORCE)
set(LIBIQSAVER_BUILD_TESTS   OFF     CACHE BOOL   "" FORCE)
set(LIBIQSAVER_PY_INSTALL_DIR "iq_saver" CACHE STRING "" FORCE)

project(iqsaver_pywheel LANGUAGES CXX)

add_subdirectory(libiqsaver)
