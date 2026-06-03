/* SWIG shim implementation. Owns the real IQSaverWriter via Pimpl and
 * converts between SWIG-friendly sentinel-bearing types and the
 * std::optional-using core API.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "iqsaver_swig.hpp"

#include <memory>
#include <optional>

#include "iqsaver/iq_saver_writer.hpp"
#include "iqsaver/types.hpp"

namespace iqsaver {

namespace {

IQSaverConfig from_swig(const IQSaverConfigSwig& s) {
    IQSaverConfig c;
    c.base_path                 = s.base_path;
    c.center_freq               = s.center_freq;
    c.bandwidth                 = s.bandwidth;
    c.sample_rate               = s.sample_rate;
    c.annotation_flush_interval = s.annotation_flush_interval;
    c.author                    = s.author;
    c.description               = s.description;
    c.hw_info                   = s.hw_info;
    c.dtype                     = s.dtype;
    c.filename                  = s.filename;
    if (s.max_samples_per_file >= 0) {
        c.max_samples_per_file =
            static_cast<uint64_t>(s.max_samples_per_file);
    }
    if (s.rotation_interval > 0.0) {
        c.rotation_interval = s.rotation_interval;
    }
    c.extension_namespace = s.extension_namespace;
    c.extra_metadata_json = s.extra_metadata_json;
    return c;
}

}  // namespace

struct IQSaverWriterSwig::Pimpl {
    std::unique_ptr<IQSaverWriter> writer;
};

IQSaverWriterSwig::IQSaverWriterSwig(const IQSaverConfigSwig& cfg)
    : pimpl_(new Pimpl{
          std::make_unique<IQSaverWriter>(from_swig(cfg))}) {}

IQSaverWriterSwig::~IQSaverWriterSwig() {
    delete pimpl_;
}

unsigned long long IQSaverWriterSwig::save_samples_buf(
    const char* data, std::size_t n_bytes,
    unsigned long long num_samples, double timestamp) {
    std::optional<double> ts;
    if (timestamp >= 0.0) ts = timestamp;
    return pimpl_->writer->save_samples(data, n_bytes,
                                        static_cast<uint64_t>(num_samples),
                                        ts);
}

bool IQSaverWriterSwig::add_annotation(long long start_sample,
                                       const std::string& label,
                                       const std::string& comment,
                                       double timestamp,
                                       const std::string& custom_fields_json) {
    std::optional<uint64_t> start;
    if (start_sample >= 0) start = static_cast<uint64_t>(start_sample);
    std::optional<double> ts;
    if (timestamp >= 0.0) ts = timestamp;
    return pimpl_->writer->add_annotation(start, label, comment, ts,
                                          custom_fields_json);
}

void IQSaverWriterSwig::finalize_annotations() {
    pimpl_->writer->finalize_annotations();
}

void IQSaverWriterSwig::update_sample_rate(double new_sample_rate,
                                           long long sampling_threshold) {
    std::optional<int> st;
    if (sampling_threshold >= 0) st = static_cast<int>(sampling_threshold);
    pimpl_->writer->update_sample_rate(new_sample_rate, st);
}

bool IQSaverWriterSwig::add_waveform_description(double timestamp,
                                                 const std::string& fields_json) {
    std::optional<double> ts;
    if (timestamp >= 0.0) ts = timestamp;
    return pimpl_->writer->add_waveform_description(ts, fields_json);
}

std::string IQSaverWriterSwig::get_recording_info() const {
    return pimpl_->writer->get_recording_info();
}

void IQSaverWriterSwig::close() {
    pimpl_->writer->close();
}

}  // namespace iqsaver
