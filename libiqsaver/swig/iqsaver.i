/* SWIG interface for libiqsaver — Python bindings used by the
 * spear-dApp `iq_saver` package.
 *
 * The Python facade in src/iq_saver/iq_saver.py wraps these bindings to
 * preserve the original Python `IQSaver` API (numpy ndarrays, **kwargs,
 * context manager). On the SWIG side we expose only POD-friendly types:
 * binary buffers cross via the buffer protocol; optional values use
 * sentinel-bearing numbers; metadata crosses as JSON strings.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

%module iqsaver_native

%{
#include "iqsaver_swig.hpp"
%}

%include "std_string.i"
%include "stdint.i"
%include "exception.i"
%include "pybuffer.i"

%exception {
    try {
        $action
    } catch (const std::exception& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
}

/* save_samples_buf takes a Python object implementing the buffer
 * protocol (bytes, bytearray, memoryview, numpy ndarray). SWIG's
 * %pybuffer_binary fills (data, n_bytes) from the buffer.
 */
%pybuffer_binary(const char* data, std::size_t n_bytes);

%include "iqsaver_swig.hpp"
