#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "iqsaver/types.hpp"

namespace iqsaver {

class IQSaverWriter {
public:
    explicit IQSaverWriter(const IQSaverConfig& cfg);
    ~IQSaverWriter();

    IQSaverWriter(const IQSaverWriter&)            = delete;
    IQSaverWriter& operator=(const IQSaverWriter&) = delete;
    IQSaverWriter(IQSaverWriter&&)                 = default;
    IQSaverWriter& operator=(IQSaverWriter&&)      = default;

    // Save a raw IQ buffer to disk. n_bytes is the buffer's byte length;
    // num_samples is the IQ-sample count the buffer represents (used for
    // annotation indexing). Returns the IQ-sample index of the first
    // sample in this write (the value of the running iq_sample_count
    // BEFORE the write was applied), matching the Python contract.
    uint64_t save_samples(const void* data,
                          std::size_t  n_bytes,
                          uint64_t     num_samples,
                          std::optional<double> timestamp = std::nullopt);

    // Add an annotation. If start_sample is std::nullopt, the current
    // iq_sample_count is used. custom_fields_json must be a JSON object;
    // its keys are prefixed with the configured extension namespace and
    // placed in the SigMF annotation. Returns true on success, false if
    // the writer has not been initialized (no save_samples call yet).
    bool add_annotation(std::optional<uint64_t> start_sample,
                        const std::string&      label,
                        const std::string&      comment,
                        std::optional<double>   timestamp,
                        const std::string&      custom_fields_json);

    // Flush pending annotations into the in-memory SigMF object and write
    // the .sigmf-meta file to disk.
    void finalize_annotations();

    // Add a new capture segment marking a sampling-threshold change.
    // No-op if the writer is not initialized or sampling_threshold has
    // no value.
    void update_sample_rate(double new_sample_rate,
                            std::optional<int> sampling_threshold = std::nullopt);

    // Shortcut for add_annotation(..., label="waveform_description", ...).
    bool add_waveform_description(std::optional<double> timestamp,
                                  const std::string&    fields_json);

    // Returns the current session snapshot encoded as a JSON object string:
    // { "total_samples", "total_files", "pending_annotations",
    //   "duration_seconds", "file_paths", "file_size_bytes" }.
    std::string get_recording_info() const;

    // Close any open file, flush pending annotations, and write the final
    // .sigmf-meta. Safe to call multiple times.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace iqsaver
