#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace iqsaver {

struct IQSaverConfig {
    std::string base_path;
    double      center_freq      = 3.6192e9;
    double      bandwidth        = 0.0;
    double      sample_rate      = 0.0;
    int         annotation_flush_interval = 200;
    std::string author           = "SPEAR dApp";
    std::string description      = "5G NR Spectrum Sharing IQ Captures";
    std::string hw_info;
    std::string dtype            = "ci16_le";
    std::string filename;
    std::optional<uint64_t> max_samples_per_file;
    std::optional<double>   rotation_interval;
    std::string extension_namespace = "dapp";
    // JSON-encoded object of extra global metadata fields. Keys whose names
    // do not already start with `core:` or `<extension_namespace>:` are
    // prefixed with `<extension_namespace>:` before being written.
    // The key `sampling_threshold` is reserved: it is moved to the first
    // capture segment instead of being placed in the global section.
    std::string extra_metadata_json = "{}";
};

}  // namespace iqsaver
