#pragma once

// Reserved for internal use. The public API does not expose Annotation;
// callers add annotations via IQSaverWriter::add_annotation with a JSON
// string of custom fields. This header is kept so consumers including
// "iqsaver/iqsaver.hpp" do not get a missing-file error.
