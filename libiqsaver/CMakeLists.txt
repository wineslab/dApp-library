# libiqsaver - SigMF-compliant IQ-sample writer
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# ----------------------------------------------------------------------------
# VERSION
# ----------------------------------------------------------------------------
# Honor a VERSION file at either libiqsaver/VERSION or the parent directory
# (the spear-dApp repository root).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LIBIQSAVER_VERSION)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" LIBIQSAVER_VERSION)
else()
    message(FATAL_ERROR "VERSION file not found")
endif()
string(STRIP "${LIBIQSAVER_VERSION}" LIBIQSAVER_VERSION)
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${LIBIQSAVER_VERSION}")
set(LIBIQSAVER_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(LIBIQSAVER_VERSION_MINOR "${CMAKE_MATCH_2}")
set(LIBIQSAVER_VERSION_PATCH "${CMAKE_MATCH_3}")

project(libiqsaver
    VERSION ${LIBIQSAVER_VERSION_MAJOR}.${LIBIQSAVER_VERSION_MINOR}.${LIBIQSAVER_VERSION_PATCH}
    DESCRIPTION "SigMF-compliant IQ-sample writer for SPEAR dApp"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(LIBIQSAVER_BUILD_TESTS  "Build C++ unit tests"                  OFF)
option(LIBIQSAVER_ENABLE_SWIG  "Build SWIG Python bindings"            OFF)
# When the wheel build drives this CMake, it sets LIBIQSAVER_PY_INSTALL_DIR
# to direct the SWIG outputs into the wheel's iq_saver package directory.
set(LIBIQSAVER_PY_INSTALL_DIR "" CACHE STRING
    "If non-empty, install SWIG Python module here (used by scikit-build-core)")

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
# SigMF .sigmf-meta JSON is emitted directly via nlohmann/json. We do not
# pull in the C++ libsigmf — its modern releases lock the schema to a
# specific (older) flatbuffers ABI that does not coexist cleanly with
# common system flatbuffers packages. The Python `sigmf` library used in
# the test suite validates the JSON we produce, so spec compliance is
# covered.
#
# Prefer a system-installed nlohmann/json (e.g. the nlohmann-json3-dev apt
# package — faster and offline-friendly), but fall back to fetching the
# header-only library at configure time when it is absent. This keeps the
# wheel build (scikit-build-core) and `pip install` from source working on
# machines without the system package — including the PyPI publish CI and
# end users — with no extra prerequisite.
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found on the system; fetching v3.11.3 via FetchContent")
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ----------------------------------------------------------------------------
# Generated version header
# ----------------------------------------------------------------------------
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/iqsaver/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/iqsaver/version.hpp
    @ONLY
)

# ----------------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------------
# In wheel mode (LIBIQSAVER_PY_INSTALL_DIR set), build iqsaver as a static
# library so the SWIG extension links it in directly — no second .so to
# ship inside the wheel, no RPATH dance. In standalone mode, build a
# shared library so external C++ consumers can link against -liqsaver.
if(NOT "${LIBIQSAVER_PY_INSTALL_DIR}" STREQUAL "")
    set(LIBIQSAVER_LIB_TYPE STATIC)
else()
    set(LIBIQSAVER_LIB_TYPE SHARED)
endif()

add_library(iqsaver ${LIBIQSAVER_LIB_TYPE}
    src/iq_saver_writer.cpp
)
add_library(iqsaver::iqsaver ALIAS iqsaver)

set_target_properties(iqsaver PROPERTIES
    OUTPUT_NAME iqsaver
    VERSION     ${PROJECT_VERSION}
    SOVERSION   ${PROJECT_VERSION_MAJOR}
)

target_include_directories(iqsaver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# nlohmann/json is used only in the implementation (.cpp), not in any public
# header, so it is a PRIVATE dependency. Keeping it PRIVATE also means it is
# not part of the exported interface — so the standalone install(EXPORT) below
# works even when nlohmann/json was provided via FetchContent rather than an
# installed package.
target_link_libraries(iqsaver
    PRIVATE
        nlohmann_json::nlohmann_json
)

target_compile_features(iqsaver PUBLIC cxx_std_17)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
if(LIBIQSAVER_BUILD_TESTS)
    enable_testing()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_iqsaver_cpp.cpp")
        add_executable(test_iqsaver_cpp tests/test_iqsaver_cpp.cpp)
        target_link_libraries(test_iqsaver_cpp PRIVATE iqsaver::iqsaver)
        add_test(NAME test_iqsaver_cpp COMMAND test_iqsaver_cpp)
    endif()
endif()

# ----------------------------------------------------------------------------
# SWIG Python bindings
# ----------------------------------------------------------------------------
if(LIBIQSAVER_ENABLE_SWIG)
    find_package(SWIG 4.0 REQUIRED COMPONENTS python)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
    include(UseSWIG)

    set(SWIG_IFACE "${CMAKE_CURRENT_SOURCE_DIR}/swig/iqsaver.i")
    set(SWIG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/swig")
    file(MAKE_DIRECTORY "${SWIG_OUTPUT_DIR}")

    set_property(SOURCE "${SWIG_IFACE}" PROPERTY CPLUSPLUS ON)
    set_property(SOURCE "${SWIG_IFACE}" PROPERTY SWIG_MODULE_NAME iqsaver_native)
    set_property(SOURCE "${SWIG_IFACE}" PROPERTY INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/swig"
        "${Python3_NumPy_INCLUDE_DIRS}"
    )
    set_property(SOURCE "${SWIG_IFACE}" PROPERTY SWIG_FLAGS
        "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/swig"
    )

    swig_add_library(iqsaver_native
        TYPE      SHARED
        LANGUAGE  python
        SOURCES   "${SWIG_IFACE}"
                  "${CMAKE_CURRENT_SOURCE_DIR}/swig/iqsaver_swig.cpp"
        OUTPUT_DIR "${SWIG_OUTPUT_DIR}"
    )

    set_target_properties(iqsaver_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${SWIG_OUTPUT_DIR}"
    )

    target_include_directories(iqsaver_native PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/swig
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )

    target_link_libraries(iqsaver_native PRIVATE
        iqsaver::iqsaver
        ${Python3_LIBRARIES}
    )

    # When the wheel build drives this CMake, install the SWIG .so + .py shim
    # into the wheel package directory (iq_saver/).
    if(NOT "${LIBIQSAVER_PY_INSTALL_DIR}" STREQUAL "")
        install(TARGETS iqsaver_native
            LIBRARY DESTINATION ${LIBIQSAVER_PY_INSTALL_DIR}
        )
        install(FILES "${SWIG_OUTPUT_DIR}/iqsaver_native.py"
            DESTINATION ${LIBIQSAVER_PY_INSTALL_DIR}
        )
    endif()
endif()

# ----------------------------------------------------------------------------
# Standalone install (C++ users); skipped when driven by scikit-build-core
# pointing at a Python wheel directory.
# ----------------------------------------------------------------------------
if("${LIBIQSAVER_PY_INSTALL_DIR}" STREQUAL "")
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install library
    install(TARGETS iqsaver
        EXPORT  libiqsaverTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Install public headers
    install(DIRECTORY include/iqsaver
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp"
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/iqsaver/version.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iqsaver
    )

    # pkg-config (.pc) file
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libiqsaver.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/libiqsaver.pc
        @ONLY
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libiqsaver.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    )

    # Export targets
    install(EXPORT libiqsaverTargets
        FILE      libiqsaverTargets.cmake
        NAMESPACE iqsaver::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libiqsaver
    )

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libiqsaverConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/libiqsaverConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libiqsaver
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/libiqsaverConfigVersion.cmake
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/libiqsaverConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/libiqsaverConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libiqsaver
    )
endif()

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "libiqsaver Configuration")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build Tests:    ${LIBIQSAVER_BUILD_TESTS}")
message(STATUS "  Enable SWIG:    ${LIBIQSAVER_ENABLE_SWIG}")
if(NOT "${LIBIQSAVER_PY_INSTALL_DIR}" STREQUAL "")
    message(STATUS "  Wheel mode:     installing SWIG to ${LIBIQSAVER_PY_INSTALL_DIR}")
else()
    message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
endif()
message(STATUS "")
