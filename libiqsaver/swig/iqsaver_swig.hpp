/* SWIG-friendly view of the libiqsaver public API.
 *
 * Mirrors iqsaver::IQSaverWriter using only constructs SWIG 4.1 parses
 * cleanly: no std::optional in signatures, no nlohmann::json, no
 * variant. Optional values cross the boundary as sentinel-bearing
 * integers (negative => absent) so the Python facade can present a
 * proper Python API.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IQSAVER_SWIG_HPP
#define IQSAVER_SWIG_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace iqsaver {

// SWIG-visible config; default values mirror IQSaverConfig.
struct IQSaverConfigSwig {
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
    // Negative => unset.
    long long   max_samples_per_file = -1;
    double      rotation_interval    = -1.0;
    std::string extension_namespace  = "spear";
    std::string extra_metadata_json  = "{}";
};

// SWIG-visible writer wrapper. Pimpl owns the real IQSaverWriter.
class IQSaverWriterSwig {
public:
    explicit IQSaverWriterSwig(const IQSaverConfigSwig& cfg);
    ~IQSaverWriterSwig();

    IQSaverWriterSwig(const IQSaverWriterSwig&)            = delete;
    IQSaverWriterSwig& operator=(const IQSaverWriterSwig&) = delete;
    IQSaverWriterSwig(IQSaverWriterSwig&&)                 = delete;
    IQSaverWriterSwig& operator=(IQSaverWriterSwig&&)      = delete;
    // Buffer must be contiguous; n_bytes is the buffer length in bytes,
    // num_samples is the corresponding IQ-sample count (n_bytes /
    // sample_size). Pass timestamp = -1.0 to use the current wall clock.
    unsigned long long save_samples_buf(const char* data,
                                        std::size_t n_bytes,
                                        unsigned long long num_samples,
                                        double timestamp);

    // start_sample < 0 => use current iq_sample_count. timestamp < 0
    // => no timestamp. custom_fields_json must be a JSON object string.
    bool add_annotation(long long          start_sample,
                        const std::string& label,
                        const std::string& comment,
                        double             timestamp,
                        const std::string& custom_fields_json);

    void finalize_annotations();

    // sampling_threshold < 0 => unset (no new capture).
    void update_sample_rate(double new_sample_rate,
                            long long sampling_threshold);

    bool add_waveform_description(double             timestamp,
                                  const std::string& fields_json);

    std::string get_recording_info() const;
    void        close();

private:
    struct Pimpl;
    Pimpl* pimpl_;
};

}  // namespace iqsaver

#endif  // IQSAVER_SWIG_HPP
