import sigmf
from pathlib import Path

# Find the most recent SigMF file in /tmp
tmp_path = Path("/tmp")
sigmf_files = list(tmp_path.glob("spectrum_iq_*.sigmf-meta"))

if not sigmf_files:
    print("No SigMF files found in /tmp")
    exit(1)

# Sort by modification time and get the most recent
latest_file = max(sigmf_files, key=lambda p: p.stat().st_mtime)
base_filename = str(latest_file).replace(".sigmf-meta", "")

print(f"Reading latest file: {base_filename}")
print()

handle = sigmf.fromfile(base_filename)

print("=== Timeseries Data ===")
samples = handle.read_samples()
print(f"Type: {type(samples)}")
print(f"Shape: {samples.shape if hasattr(samples, 'shape') else 'N/A'}")
print(f"Dtype: {samples.dtype if hasattr(samples, 'dtype') else 'N/A'}")
print(f"First 5 samples: {samples[:5] if len(samples) > 0 else 'Empty'}")
print()

print("=== Global Info ===")
global_info = handle.get_global_info()
for key, value in global_info.items():
    print(f"{key}: {value}")
print()

print("=== Captures ===")
captures = handle.get_captures()
print(f"Number of captures: {len(captures)}")
for i, capture in enumerate(captures):
    print(f"\nCapture {i}:")
    for key, value in capture.items():
        print(f"  {key}: {value}")
print()

print("=== Annotations ===")
annotations = handle.get_annotations()
print(f"Number of annotations: {len(annotations)}")
for i, annotation in enumerate(annotations):
    print(f"\nAnnotation {i}:")
    for key, value in annotation.items():
        print(f"  {key}: {value}")
