#include "iqsaver/iq_saver_writer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// SigMF (.sigmf-meta) JSON is emitted directly via nlohmann/json. The
// schema we produce (core:* required fields + optional <ns>:* extension
// keys) is fully validated by the Python `sigmf` library used in the
// test suite, which is the acceptance gate.

namespace iqsaver {

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace {

constexpr const char* kSigMFVersion = "1.0.0";

// Fallback version stamped in the SigMF `core:extensions` entry for the
// configured extension namespace when the caller supplies no `schema_version`
// in extra metadata. Callers (e.g. the spectrum dApp) that co-version their
// namespace pass an explicit `schema_version`, which takes precedence.
constexpr const char* kDefaultExtensionVersion = "1.0.0";

double now_seconds() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               system_clock::now().time_since_epoch())
        .count();
}

std::string iso8601_utc(double unix_seconds) {
    using namespace std::chrono;
    auto whole   = static_cast<std::time_t>(unix_seconds);
    auto frac    = unix_seconds - static_cast<double>(whole);
    auto us      = static_cast<long long>(frac * 1'000'000.0 + 0.5);
    if (us >= 1'000'000) {
        whole += 1;
        us -= 1'000'000;
    }
    std::tm tm_buf{};
    gmtime_r(&whole, &tm_buf);
    char date_part[32];
    std::strftime(date_part, sizeof(date_part), "%Y-%m-%dT%H:%M:%S", &tm_buf);

    std::ostringstream oss;
    oss << date_part << '.' << std::setw(6) << std::setfill('0') << us
        << "+00:00";
    return oss.str();
}

bool has_namespace_prefix(const std::string& key, const std::string& ns) {
    if (key.rfind("core:", 0) == 0) return true;
    if (key.rfind(ns + ":", 0) == 0) return true;
    return false;
}

std::string with_namespace(const std::string& key, const std::string& ns) {
    if (has_namespace_prefix(key, ns)) return key;
    return ns + ":" + key;
}

[[maybe_unused]] uint32_t sample_size_bytes(const std::string& dtype_str) {
    // Mirrors sigmf::get_sample_size(): bit width from the digit substring,
    // doubled for complex types (leading 'c').
    uint32_t size = 0;
    if (dtype_str.find("64") != std::string::npos) size = 8;
    else if (dtype_str.find("32") != std::string::npos) size = 4;
    else if (dtype_str.find("16") != std::string::npos) size = 2;
    else if (dtype_str.find("8")  != std::string::npos) size = 1;
    else throw std::invalid_argument("Invalid datatype string: " + dtype_str);
    if (!dtype_str.empty() && dtype_str.front() == 'c') size *= 2;
    return size;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct IQSaverWriter::Impl {
    IQSaverConfig config;

    // Parsed extra metadata (default {})
    json extra_metadata;
    // Pending annotations buffered until flush
    std::vector<json> annotation_buffer;
    // Annotations already flushed into the record JSON
    json record_global;
    json record_captures;
    json record_annotations;
    bool record_active = false;

    std::ofstream                data_stream;
    std::vector<char>            data_buffer;  // 1 MiB rdbuf storage

    std::string filename_base;   // current file's base name (no extension)
    fs::path    base_path;
    std::string data_path;

    uint64_t frame_count       = 0;
    uint64_t iq_sample_count   = 0;
    bool     initialized       = false;

    double   session_start_time = 0.0;
    std::optional<double> file_start_time;
    std::optional<int64_t> session_timestamp_ms;
    uint64_t file_index        = 0;
    std::vector<std::string> all_files;  // base names of fully closed files

    explicit Impl(const IQSaverConfig& cfg) : config(cfg) {
        if (config.max_samples_per_file.has_value() &&
            *config.max_samples_per_file == 0) {
            throw std::invalid_argument("max_samples_per_file must be positive.");
        }
        if (config.rotation_interval.has_value() &&
            *config.rotation_interval <= 0.0) {
            throw std::invalid_argument("rotation_interval must be positive.");
        }
        if (config.extension_namespace.empty()) {
            throw std::invalid_argument("extension_namespace must not be empty.");
        }
        if (config.sample_rate <= 0.0) {
            config.sample_rate = config.bandwidth;
        }

        base_path = config.base_path.empty() ? fs::current_path()
                                             : fs::path(config.base_path);
        fs::create_directories(base_path);

        if (config.extra_metadata_json.empty()) {
            extra_metadata = json::object();
        } else {
            extra_metadata = json::parse(config.extra_metadata_json);
            if (!extra_metadata.is_object()) {
                throw std::invalid_argument(
                    "extra_metadata_json must be a JSON object.");
            }
        }

        session_start_time = now_seconds();
    }

    void initialize_file(double timestamp) {
        if (initialized) return;

        if (!config.filename.empty()) {
            char idx_buf[16];
            std::snprintf(idx_buf, sizeof(idx_buf), "_%04llu",
                          static_cast<unsigned long long>(file_index));
            filename_base = config.filename + idx_buf;
        } else {
            if (file_index == 0) {
                session_timestamp_ms =
                    static_cast<int64_t>(timestamp * 1000.0);
            }
            char buf[64];
            std::snprintf(buf, sizeof(buf), "spectrum_iq_%lld_%04llu",
                          static_cast<long long>(*session_timestamp_ms),
                          static_cast<unsigned long long>(file_index));
            filename_base = buf;
        }

        data_path = (base_path / (filename_base + ".sigmf-data")).string();
        data_buffer.assign(1 << 20, 0);  // 1 MiB rdbuf
        data_stream.rdbuf()->pubsetbuf(data_buffer.data(),
                                       static_cast<std::streamsize>(data_buffer.size()));
        data_stream.open(data_path,
                         std::ios::out | std::ios::binary | std::ios::trunc);
        if (!data_stream.is_open()) {
            throw std::runtime_error("Failed to open data file: " + data_path);
        }

        // Build the in-memory SigMF skeleton.
        record_global = json::object();
        record_global["core:datatype"]    = config.dtype;
        record_global["core:sample_rate"] = config.sample_rate;
        record_global["core:version"]     = kSigMFVersion;
        record_global["core:author"]      = config.author;
        record_global["core:description"] = config.description;
        if (!config.hw_info.empty()) {
            record_global["core:hw"] = config.hw_info;
        }
        // Declare the extension namespace we stamp so SigMF readers know which
        // extension the `<ns>:*` fields belong to. Version is co-versioned with
        // the namespace: use the caller's `schema_version` when provided, else
        // fall back to kDefaultExtensionVersion. `optional=false` — the `<ns>:*`
        // geometry/domain fields are required to interpret the recording.
        record_global["core:extensions"] = json::array(
            {json{{"name", config.extension_namespace},
                  {"version",
                   extra_metadata.value(
                       "schema_version",
                       std::string(kDefaultExtensionVersion))},
                  {"optional", false}}});
        for (auto it = extra_metadata.begin(); it != extra_metadata.end();
             ++it) {
            if (it.key() == "sampling_threshold") continue;
            record_global[with_namespace(it.key(),
                                         config.extension_namespace)] =
                it.value();
        }

        record_captures    = json::array();
        record_annotations = json::array();

        // First capture segment.
        json first_cap = json::object();
        first_cap["core:sample_start"] = 0;
        first_cap["core:frequency"]    = config.center_freq;
        first_cap["core:datetime"]     = iso8601_utc(timestamp);
        if (config.bandwidth > 0.0) {
            first_cap["core:bandwidth"] = config.bandwidth;
        }
        if (extra_metadata.contains("sampling_threshold")) {
            first_cap[config.extension_namespace + ":sampling_threshold"] =
                extra_metadata.at("sampling_threshold");
        }
        record_captures.push_back(std::move(first_cap));

        record_active   = true;
        file_start_time = timestamp;
        initialized     = true;

        // Crash-safety: write metadata immediately.
        write_metadata_file();
    }

    void rotate_file(double timestamp) {
        (void)timestamp;
        flush_annotations();
        if (data_stream.is_open()) {
            data_stream.close();
        }
        if (initialized) {
            write_metadata_file();
            all_files.push_back(filename_base);
        }
        record_active      = false;
        record_global      = json::object();
        record_captures    = json::array();
        record_annotations = json::array();
        filename_base.clear();
        data_path.clear();
        frame_count        = 0;
        iq_sample_count    = 0;
        initialized        = false;
        annotation_buffer.clear();
        file_start_time.reset();
        file_index += 1;
    }

    void finalize_file() {
        if (data_stream.is_open()) {
            data_stream.close();
        }
        if (initialized) {
            flush_annotations();
            write_metadata_file();
        }
    }

    // Drain annotation_buffer into the in-memory record.
    void flush_annotations() {
        if (!record_active || annotation_buffer.empty()) {
            return;
        }
        for (auto& ann : annotation_buffer) {
            json out = json::object();
            out["core:sample_start"] = ann.value("sample_start", 0ULL);
            if (ann.contains("label") && ann.at("label").is_string()) {
                out["core:label"] = ann.at("label").get<std::string>();
            }
            if (ann.contains("comment") && ann.at("comment").is_string()) {
                out["core:comment"] = ann.at("comment").get<std::string>();
            }
            const std::string ts_key =
                config.extension_namespace + ":timestamp";
            if (ann.contains(ts_key)) {
                out["core:datetime"] =
                    iso8601_utc(ann.at(ts_key).get<double>());
            }
            for (auto it = ann.begin(); it != ann.end(); ++it) {
                const std::string& k = it.key();
                if (k.rfind(config.extension_namespace + ":", 0) == 0 &&
                    k != ts_key) {
                    out[k] = it.value();
                }
            }
            record_annotations.push_back(std::move(out));
        }
        annotation_buffer.clear();
    }

    json compose_metadata_json() const {
        json j = json::object();
        j["global"]      = record_global;
        j["captures"]    = record_captures;
        j["annotations"] = record_annotations;
        return j;
    }

    void write_metadata_file() {
        if (filename_base.empty()) return;
        const fs::path meta_path =
            base_path / (filename_base + ".sigmf-meta");
        json j = compose_metadata_json();
        std::ofstream out(meta_path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            throw std::runtime_error(
                "Failed to open metadata file: " + meta_path.string());
        }
        out << j.dump(4);
    }

    uint64_t save_samples(const void* data, std::size_t n_bytes,
                          uint64_t num_samples,
                          std::optional<double> timestamp) {
        double ts = timestamp.value_or(now_seconds());

        // Rotation check. max_samples_per_file is a count of true IQ *samples*,
        // so it is compared against iq_sample_count — NOT frame_count, which
        // counts save_samples() calls. The check runs before the write so each
        // call's samples stay whole within one segment; a segment therefore
        // reaches *at least* max_samples_per_file and may exceed it by up to one
        // call's worth of samples. It is a rotation threshold, not a hard cap —
        // splitting a single call across files would fragment an indication.
        if (config.max_samples_per_file.has_value() &&
            iq_sample_count >= *config.max_samples_per_file) {
            rotate_file(ts);
        } else if (config.rotation_interval.has_value() &&
                   file_start_time.has_value() &&
                   (ts - *file_start_time) >= *config.rotation_interval) {
            rotate_file(ts);
        }

        if (!initialized) {
            initialize_file(ts);
        }

        if (n_bytes > 0 && data != nullptr) {
            data_stream.write(static_cast<const char*>(data),
                              static_cast<std::streamsize>(n_bytes));
            if (!data_stream.good()) {
                throw std::runtime_error("Failed to write IQ data to: " +
                                         data_path);
            }
        }

        const uint64_t sample_index = iq_sample_count;
        frame_count += 1;
        iq_sample_count += num_samples;
        return sample_index;
    }

    bool add_annotation(std::optional<uint64_t> start_sample,
                        const std::string& label,
                        const std::string& comment,
                        std::optional<double> timestamp,
                        const std::string& custom_fields_json) {
        if (!initialized) return false;

        json ann = json::object();
        ann["sample_start"] =
            start_sample.value_or(iq_sample_count);
        ann["label"] = label;
        if (!comment.empty()) {
            ann["comment"] = comment;
        }
        if (timestamp.has_value()) {
            ann[config.extension_namespace + ":timestamp"] = *timestamp;
        }

        if (!custom_fields_json.empty()) {
            json fields = json::parse(custom_fields_json);
            if (!fields.is_object()) {
                throw std::invalid_argument(
                    "custom_fields_json must be a JSON object.");
            }
            for (auto it = fields.begin(); it != fields.end(); ++it) {
                ann[with_namespace(it.key(), config.extension_namespace)] =
                    it.value();
            }
        }

        annotation_buffer.push_back(std::move(ann));
        if (static_cast<int>(annotation_buffer.size()) >=
            config.annotation_flush_interval) {
            finalize_annotations();
        }
        return true;
    }

    void finalize_annotations() {
        flush_annotations();
        if (initialized) {
            write_metadata_file();
        }
    }

    void update_sample_rate(double new_sample_rate,
                            std::optional<int> sampling_threshold) {
        if (!record_active || !sampling_threshold.has_value()) {
            return;
        }
        json cap = json::object();
        cap["core:sample_start"] = iq_sample_count;
        cap["core:frequency"]    = config.center_freq;
        cap["core:datetime"]     = iso8601_utc(now_seconds());
        if (config.bandwidth > 0.0) {
            cap["core:bandwidth"] = config.bandwidth;
        }
        cap[config.extension_namespace + ":sampling_threshold"] =
            *sampling_threshold;
        cap[config.extension_namespace + ":effective_sample_rate"] =
            new_sample_rate;
        record_captures.push_back(std::move(cap));

        extra_metadata["sampling_threshold"] = *sampling_threshold;
    }

    json recording_info_json() const {
        std::vector<std::string> file_names = all_files;
        if (!filename_base.empty()) {
            file_names.push_back(filename_base);
        }
        std::vector<std::string> file_paths;
        json sizes = json::object();
        for (const auto& name : file_names) {
            fs::path p = base_path / (name + ".sigmf-data");
            file_paths.push_back(p.string());
            std::uintmax_t sz = 0;
            std::error_code ec;
            if (fs::exists(p, ec)) {
                sz = fs::file_size(p, ec);
                if (ec) sz = 0;
            }
            sizes[p.string()] = sz;
        }
        json info;
        info["total_samples"]       = iq_sample_count;
        info["total_files"]         = file_names.size();
        info["pending_annotations"] = annotation_buffer.size();
        info["duration_seconds"]    = now_seconds() - session_start_time;
        info["file_paths"]          = file_paths;
        info["file_size_bytes"]     = sizes;
        return info;
    }

    void close() {
        finalize_annotations();
        finalize_file();

        if (!filename_base.empty() &&
            (all_files.empty() || all_files.back() != filename_base)) {
            all_files.push_back(filename_base);
        }

        // Make close() idempotent and prevent double-counting/overwrites if the
        // writer instance is (unexpectedly) reused after closing.
        record_active = false;
        initialized   = false;
        file_start_time.reset();
        filename_base.clear();
        data_path.clear();
    }
};

// ---------------------------------------------------------------------------
// IQSaverWriter
// ---------------------------------------------------------------------------

IQSaverWriter::IQSaverWriter(const IQSaverConfig& cfg)
    : impl_(std::make_unique<Impl>(cfg)) {}

IQSaverWriter::~IQSaverWriter() {
    try {
        if (impl_) impl_->close();
    } catch (...) {
        // Destructors must not throw.
    }
}

uint64_t IQSaverWriter::save_samples(const void* data, std::size_t n_bytes,
                                     uint64_t num_samples,
                                     std::optional<double> timestamp) {
    return impl_->save_samples(data, n_bytes, num_samples, timestamp);
}

bool IQSaverWriter::add_annotation(std::optional<uint64_t> start_sample,
                                   const std::string& label,
                                   const std::string& comment,
                                   std::optional<double> timestamp,
                                   const std::string& custom_fields_json) {
    return impl_->add_annotation(start_sample, label, comment, timestamp,
                                 custom_fields_json);
}

void IQSaverWriter::finalize_annotations() {
    impl_->finalize_annotations();
}

void IQSaverWriter::update_sample_rate(double new_sample_rate,
                                       std::optional<int> sampling_threshold) {
    impl_->update_sample_rate(new_sample_rate, sampling_threshold);
}

bool IQSaverWriter::add_waveform_description(std::optional<double> timestamp,
                                             const std::string& fields_json) {
    return add_annotation(std::nullopt, "waveform_description", "",
                          timestamp, fields_json);
}

std::string IQSaverWriter::get_recording_info() const {
    return impl_->recording_info_json().dump();
}

void IQSaverWriter::close() {
    impl_->close();
}

}  // namespace iqsaver
