#pragma once

// Umbrella header for libiqsaver.
//
// libiqsaver is a SigMF-compliant IQ-sample writer with sample-count and
// time-based file rotation, deferred annotations, and a configurable SigMF
// extension namespace (default: "spear:"). The library is the C++ backend
// for the Python `iq_saver` package used by the SPEAR spectrum dApp; it is
// also usable directly as a standalone C++ library installed via CMake +
// pkg-config (libiqsaver.pc).
//
// Public API: see iqsaver/iq_saver_writer.hpp.

#include "iqsaver/annotation.hpp"
#include "iqsaver/iq_saver_writer.hpp"
#include "iqsaver/types.hpp"
#include "iqsaver/version.hpp"
