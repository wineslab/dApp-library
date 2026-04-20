#!/usr/bin/env python3
"""
Test suite for IQSaver class - SigMF-compliant IQ recording
"""
import tempfile
import time
import numpy as np
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from iq_saver.iq_saver import IQSaver
import sigmf


def test_basic_recording():
    """Test basic IQ recording functionality"""
    print("=" * 80)
    print("Test 1: Basic IQ Recording")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            sample_rate=38.16e6,
            rotation_interval=10.0,
            hw_info='Test RU - USRP B210',
            num_prbs=106,
            fft_size=2048
        )
        
        # Generate and save samples
        samples = (np.random.randn(1536) + 1j * np.random.randn(1536)) * 1000
        ts = time.time()
        idx = saver.save_samples(samples.astype(np.complex64), timestamp=ts)
        
        print(f"✓ Saved {len(samples)} samples at index {idx}")
        
        saver.close()
        
        # Verify files were created
        data_files = list(Path(tmpdir).glob('*.sigmf-data'))
        meta_files = list(Path(tmpdir).glob('*.sigmf-meta'))
        
        assert len(data_files) == 1, f"Expected 1 data file, got {len(data_files)}"
        assert len(meta_files) == 1, f"Expected 1 meta file, got {len(meta_files)}"
        
        print(f"✓ Created {len(data_files)} data file and {len(meta_files)} metadata file")
        print("✓ Test 1 PASSED\n")


def test_annotations():
    """Test annotation functionality"""
    print("=" * 80)
    print("Test 2: Annotations")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            num_prbs=106
        )
        
        # Save samples
        samples = (np.random.randn(1536) + 1j * np.random.randn(1536)) * 1000
        ts = time.time()
        saver.save_samples(samples.astype(np.complex64), timestamp=ts)
        
        # Add PRB control annotation
        saver.add_annotation(
            timestamp=ts,
            label='prb_control',
            comment='Blacklisted PRBs due to interference',
            prb_blacklist=[76, 77, 78, 79, 80],
            noise_threshold=53,
            control_action='blacklist'
        )
        
        # Add waveform description
        saver.add_waveform_description(
            timestamp=ts,
            waveform_type='5G_NR_UPLINK_PUSCH',
            modulation='QPSK',
            protocol='PUSCH',
            mcs=5
        )
        
        info = saver.get_recording_info()
        print(f"✓ Added {info['pending_annotations']} annotations")
        
        saver.close()
        
        # Verify annotations in metadata
        meta_files = list(Path(tmpdir).glob('*.sigmf-meta'))
        with open(meta_files[0], 'r') as f:
            metadata = json.load(f)
        
        annotations = metadata.get('annotations', [])
        assert len(annotations) == 2, f"Expected 2 annotations, got {len(annotations)}"
        
        # Check PRB control annotation
        prb_ann = next(a for a in annotations if a.get('core:label') == 'prb_control')
        assert prb_ann.get('spear:prb_blacklist') == [76, 77, 78, 79, 80]
        assert prb_ann.get('spear:noise_threshold') == 53
        
        print(f"✓ Verified {len(annotations)} annotations in metadata")
        print("✓ Test 2 PASSED\n")


def test_file_rotation():
    """Test time-based file rotation"""
    print("=" * 80)
    print("Test 3: File Rotation")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            rotation_interval=0.5,  # Rotate every 500ms
            num_prbs=106
        )
        
        # Save samples over time to trigger rotation
        for i in range(3):
            samples = (np.random.randn(1024) + 1j * np.random.randn(1024)) * 1000
            saver.save_samples(samples.astype(np.complex64))
            time.sleep(0.6)  # Wait to trigger rotation
        
        saver.close()
        
        # Verify multiple files were created
        data_files = list(Path(tmpdir).glob('*.sigmf-data'))
        meta_files = list(Path(tmpdir).glob('*.sigmf-meta'))
        
        assert len(data_files) >= 2, f"Expected at least 2 data files, got {len(data_files)}"
        assert len(meta_files) >= 2, f"Expected at least 2 meta files, got {len(meta_files)}"
        
        print(f"✓ Created {len(data_files)} files through rotation")
        print("✓ Test 3 PASSED\n")


def test_sigmf_compliance():
    """Test SigMF compliance by loading with sigmf library"""
    print("=" * 80)
    print("Test 4: SigMF Compliance Verification")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a recording
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            sample_rate=38.16e6,
            rotation_interval=10.0,
            hw_info='Test RU - USRP B210',
            num_prbs=106,
            fft_size=2048,
            subcarrier_spacing_khz=30
        )
        
        # Save samples
        samples = (np.random.randn(1536) + 1j * np.random.randn(1536)) * 1000
        ts = time.time()
        saver.save_samples(samples.astype(np.complex64), timestamp=ts)
        
        # Add annotations
        saver.add_annotation(
            timestamp=ts,
            label='prb_control',
            comment='Blacklisted PRBs due to interference',
            prb_blacklist=[76, 77, 78, 79, 80],
            noise_threshold=53
        )
        
        saver.close()
        
        # Load and verify using sigmf library
        files = list(Path(tmpdir).glob('*.sigmf-meta'))
        assert len(files) > 0, "No metadata files found"
        
        base = str(files[0]).replace('.sigmf-meta', '')
        recording = sigmf.fromfile(base)
        
        print(f"✓ Successfully loaded SigMF recording from: {Path(base).name}")
        
        # Verify global metadata
        global_info = recording.get_global_info()
        assert global_info.get('core:sample_rate') == 38.16e6, f"Sample rate mismatch: {global_info.get('core:sample_rate')}"
        # Note: default dtype is cf32_le, but we're using ci16_le in IQSaver
        # assert global_info.get('core:datatype') == 'cf32_le'
        assert global_info.get('spear:num_prbs') == 106, f"num_prbs mismatch: {global_info.get('spear:num_prbs')}"
        assert global_info.get('spear:fft_size') == 2048, f"fft_size mismatch: {global_info.get('spear:fft_size')}"
        
        print(f"\nGlobal Metadata:")
        print(f"  Sample Rate: {global_info.get('core:sample_rate')} Hz")
        print(f"  Data Type: {global_info.get('core:datatype')}")
        print(f"  Author: {global_info.get('core:author')}")
        print(f"  Hardware: {global_info.get('core:hw')}")
        
        # Verify captures
        captures = recording.get_captures()
        assert len(captures) > 0, "No captures found"
        assert captures[0].get('core:frequency') == 3.6192e9
        assert captures[0].get('core:bandwidth') == 38.16e6
        
        print(f"\nCaptures: {len(captures)}")
        print(f"  Center Frequency: {captures[0].get('core:frequency')} Hz")
        print(f"  Bandwidth: {captures[0].get('core:bandwidth')} Hz")
        
        # Verify annotations
        annotations = recording.get_annotations()
        assert len(annotations) > 0, "No annotations found"
        
        ann = annotations[0]
        assert ann.get('core:label') == 'prb_control'
        assert ann.get('spear:prb_blacklist') == [76, 77, 78, 79, 80]
        
        print(f"\nAnnotations: {len(annotations)}")
        print(f"  Label: {ann.get('core:label')}")
        print(f"  Comment: {ann.get('core:comment')}")
        print(f"  Custom - PRB Blacklist: {ann.get('spear:prb_blacklist')}")
        
        print(f"\n✓ All SigMF metadata is correctly formatted and compliant!")
        print("✓ Test 4 PASSED\n")


def test_context_manager():
    """Test context manager functionality"""
    print("=" * 80)
    print("Test 5: Context Manager")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            num_prbs=106
        ) as saver:
            samples = (np.random.randn(2048) + 1j * np.random.randn(2048)) * 1000
            idx = saver.save_samples(samples.astype(np.complex64))
            
            saver.add_annotation(
                start_sample=idx,
                label='test_signal',
                comment='Test using context manager'
            )
            
            print(f"✓ Saved samples and annotation using context manager")
        
        # Verify files exist after context manager exit
        data_files = list(Path(tmpdir).glob('*.sigmf-data'))
        meta_files = list(Path(tmpdir).glob('*.sigmf-meta'))
        
        assert len(data_files) == 1
        assert len(meta_files) == 1
        
        print(f"✓ Context manager properly finalized recording")
        print("✓ Test 5 PASSED\n")


def test_deferred_annotations():
    """Test adding annotations to past recordings"""
    print("=" * 80)
    print("Test 6: Deferred Annotations")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            rotation_interval=0.5,
            num_prbs=106
        )
        
        # Save samples and record timestamps
        timestamps = []
        for i in range(3):
            samples = (np.random.randn(1024) + 1j * np.random.randn(1024)) * 1000
            ts = time.time()
            timestamps.append(ts)
            saver.save_samples(samples.astype(np.complex64), timestamp=ts)
            time.sleep(0.6)  # Trigger rotation
        
        # Add annotations to past timestamps (different files)
        for i, ts in enumerate(timestamps):
            saver.add_annotation(
                timestamp=ts,
                label=f'deferred_annotation_{i}',
                comment=f'Added after the fact to capture {i}'
            )
        
        saver.close()
        
        # Verify annotations were distributed to correct files
        meta_files = sorted(Path(tmpdir).glob('*.sigmf-meta'))
        
        total_annotations = 0
        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            annotations = metadata.get('annotations', [])
            total_annotations += len(annotations)
        
        assert total_annotations == 3, f"Expected 3 annotations, got {total_annotations}"
        
        print(f"✓ Successfully added deferred annotations to {len(meta_files)} files")
        print(f"✓ Total annotations: {total_annotations}")
        print("✓ Test 6 PASSED\n")


def test_spectrum_dapp_integration_pattern():
    """Test integration pattern for SpectrumSharingDApp"""
    print("=" * 80)
    print("Test 7: SpectrumSharingDApp Integration Pattern")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Configuration matching SpectrumSharingDApp
        num_prbs = 106
        num_subcarrier_spacing = 30  # kHz
        center_freq = 3.6192e9  # Hz
        bandwidth = num_prbs * 12 * num_subcarrier_spacing * 1e3  # Hz
        
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=center_freq,
            bandwidth=bandwidth,
            sample_rate=bandwidth,
            rotation_interval=10.0,
            annotation_flush_interval=200,
            hw_info=f"FFT:2048, PRBs:{num_prbs}",
            description="5G NR Spectrum Sharing - RAN Function 1",
            fft_size=2048,
            num_prbs=num_prbs,
            subcarrier_spacing_khz=num_subcarrier_spacing,
            sampling_threshold=5
        )
        
        # Simulate IQ data reception (as in get_iqs_from_ran callback)
        fft_size = 2048
        for i in range(5):
            # Simulate received IQ samples
            iq_arr = np.random.randint(-1000, 1000, size=fft_size*2, dtype=np.int16)
            
            # Convert to complex (as done in spectrum_dapp.py)
            iq_comp = iq_arr[::2] + iq_arr[1::2] * 1j
            
            timestamp = time.time()
            sample_idx = saver.save_samples(iq_comp.astype(np.complex64), timestamp=timestamp)
            
            # Simulate PRB blacklisting (as in control logic)
            if i == 2:
                prb_blk_list = np.array([76, 77, 78, 79, 80], dtype=np.uint16)
                saver.add_annotation(
                    timestamp=timestamp,
                    label="prb_control",
                    comment=f"Blacklisted {prb_blk_list.size} PRBs",
                    prb_blacklist=prb_blk_list.tolist(),
                    noise_threshold=53,
                    control_action="blacklist"
                )
                print(f"  Capture {i+1}: Added PRB blacklist annotation")
            
            time.sleep(0.1)
        
        info = saver.get_recording_info()
        saver.close()
        
        print(f"\n✓ Recorded {info['total_samples']} samples")
        print(f"✓ Created {info['total_files']} file(s)")
        print(f"✓ Duration: {info['duration_seconds']:.2f}s")
        
        # Verify the recording can be loaded
        meta_files = list(Path(tmpdir).glob('*.sigmf-meta'))
        for meta_file in meta_files:
            base = str(meta_file).replace('.sigmf-meta', '')
            recording = sigmf.fromfile(base)
            assert recording.get_global_info().get('spear:num_prbs') == num_prbs
        
        print(f"✓ All recordings are valid SigMF files")
        print("✓ Test 7 PASSED\n")


def test_metadata_accumulation_across_flushes():
    """Verify that repeated tofile calls accumulate all annotations, never losing earlier ones."""
    print("=" * 80)
    print("Test 8: Metadata accumulation across flushes")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        saver = IQSaver(
            base_path=tmpdir,
            center_freq=3.6192e9,
            bandwidth=38.16e6,
            annotation_flush_interval=200,  # manual flush only
            num_prbs=106
        )

        samples = (np.random.randn(1024) + 1j * np.random.randn(1024)) * 1000
        saver.save_samples(samples.astype(np.complex64))

        # First batch of annotations, flushed to disk
        saver.add_annotation(label='batch_1_a', comment='first flush, annotation A')
        saver.add_annotation(label='batch_1_b', comment='first flush, annotation B')
        saver.finalize_annotations()

        meta_file = list(Path(tmpdir).glob('*.sigmf-meta'))[0]
        with open(meta_file) as f:
            meta = json.load(f)
        assert len(meta.get('annotations', [])) == 2, "Expected 2 annotations after first flush"
        print("✓ First flush: 2 annotations on disk")

        # Second batch, flushed on top of the first
        saver.add_annotation(label='batch_2_a', comment='second flush, annotation A')
        saver.finalize_annotations()

        with open(meta_file) as f:
            meta = json.load(f)
        assert len(meta.get('annotations', [])) == 3, \
            f"Expected 3 annotations after second flush, got {len(meta.get('annotations', []))}"
        print("✓ Second flush: all 3 annotations preserved on disk")

        # Third batch flushed via close()
        saver.add_annotation(label='batch_3_a', comment='third flush via close')
        saver.close()

        with open(meta_file) as f:
            meta = json.load(f)
        assert len(meta.get('annotations', [])) == 4, \
            f"Expected 4 annotations after close, got {len(meta.get('annotations', []))}"
        labels = [a.get('core:label') for a in meta['annotations']]
        assert 'batch_1_a' in labels and 'batch_1_b' in labels
        assert 'batch_2_a' in labels
        assert 'batch_3_a' in labels
        print("✓ close(): all 4 annotations preserved, no data lost across flushes")
        print("✓ Test 8 PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IQSaver Test Suite")
    print("=" * 80 + "\n")
    
    tests = [
        test_basic_recording,
        test_annotations,
        test_file_rotation,
        test_sigmf_compliance,
        test_context_manager,
        test_deferred_annotations,
        test_spectrum_dapp_integration_pattern,
        test_metadata_accumulation_across_flushes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test FAILED: {test.__name__}")
            print(f"  Error: {e}\n")
    
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All tests PASSED!")
    else:
        print(f"⚠️  {failed} test(s) FAILED")
        sys.exit(1)
