#!/usr/bin/env python3
"""
Test script for the source separation system.
Creates synthetic test data and runs the pipeline.
"""
import numpy as np
import torch
import torchaudio
from pathlib import Path
import tempfile
import shutil

def create_synthetic_audio():
    """Create synthetic test audio files."""
    sr = 16000
    
    # Create test directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Generate 10-second mixture (sine waves at different frequencies)
    duration_mix = 10.0
    t_mix = np.linspace(0, duration_mix, int(sr * duration_mix))
    
    # Component 1: 440 Hz sine (like a musical note)
    component1 = 0.5 * np.sin(2 * np.pi * 440 * t_mix)
    
    # Component 2: 880 Hz sine (octave higher)
    component2 = 0.3 * np.sin(2 * np.pi * 880 * t_mix)
    
    # Add some segments where component1 is stronger (simulating target presence)
    # Segments: 2-3s, 5-6.5s, 8-9s
    target_segments = [(2.0, 3.0), (5.0, 6.5), (8.0, 9.0)]
    
    mix_audio = component1 + component2
    for start, end in target_segments:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        # Boost component1 in these segments
        mix_audio[start_idx:end_idx] += 0.4 * np.sin(2 * np.pi * 440 * t_mix[start_idx:end_idx])
    
    # Add some noise
    mix_audio += 0.05 * np.random.randn(len(mix_audio))
    
    # Generate 2-second query (pure 440 Hz sine)
    duration_query = 2.0
    t_query = np.linspace(0, duration_query, int(sr * duration_query))
    query_audio = 0.6 * np.sin(2 * np.pi * 440 * t_query)
    
    # Add slight variations to make it more realistic
    query_audio += 0.02 * np.random.randn(len(query_audio))
    
    # Normalize
    mix_audio = mix_audio / np.max(np.abs(mix_audio)) * 0.8
    query_audio = query_audio / np.max(np.abs(query_audio)) * 0.8
    
    # Save as WAV files
    mix_path = test_dir / "mix.wav"
    query_path = test_dir / "query.wav"
    
    torchaudio.save(str(mix_path), torch.from_numpy(mix_audio).unsqueeze(0), sr)
    torchaudio.save(str(query_path), torch.from_numpy(query_audio).unsqueeze(0), sr)
    
    return str(mix_path), str(query_path), target_segments


def run_basic_test():
    """Run basic functionality test."""
    print("Creating synthetic test data...")
    mix_path, query_path, expected_segments = create_synthetic_audio()
    
    print(f"Mix audio: {mix_path}")
    print(f"Query audio: {query_path}")
    print(f"Expected segments: {expected_segments}")
    
    # Test individual components
    print("\nTesting individual components...")
    
    # Test I/O preprocessing
    try:
        from src.io_preprocessing import load_wav_mono16k, enforce_length_policy
        
        mix_wav = load_wav_mono16k(mix_path)
        query_wav = load_wav_mono16k(query_path)
        
        print(f"✓ Audio loading successful")
        print(f"  Mix shape: {mix_wav.shape}, duration: {len(mix_wav)/16000:.2f}s")
        print(f"  Query shape: {query_wav.shape}, duration: {len(query_wav)/16000:.2f}s")
        
        # Test length policies
        mix_wav, _ = enforce_length_policy(mix_wav, 10.0, policy="mix")
        query_wav, _ = enforce_length_policy(query_wav, 3.0, policy="query")
        
        print(f"✓ Length policy enforcement successful")
        
    except Exception as e:
        print(f"✗ I/O preprocessing failed: {e}")
        return False
    
    # Test mel processing
    try:
        from src.mel_processing import compute_mel, create_mel_filter_bank
        
        X_complex, Mix_mel = compute_mel(mix_wav, 16000, 512, 256, 64)
        Q_complex, Q_mel = compute_mel(query_wav, 16000, 512, 256, 64)
        mel_fb = create_mel_filter_bank(16000, 512, 64)
        
        print(f"✓ Mel processing successful")
        print(f"  Mix STFT shape: {X_complex.shape}")
        print(f"  Mix mel shape: {Mix_mel.shape}")
        print(f"  Query mel shape: {Q_mel.shape}")
        print(f"  Mel filter bank shape: {mel_fb.shape}")
        
    except Exception as e:
        print(f"✗ Mel processing failed: {e}")
        return False
    
    # Test mask generation
    try:
        from src.mask_generation import make_initial_mask
        from src.segment_detection import interpolate_to_stft_frames
        
        # Create dummy segments and weights for testing
        dummy_segments = [(2.0, 3.0), (5.0, 6.5)]
        times = np.linspace(0, 10, 20)  # 20 analysis windows
        w_raw = np.zeros(20)
        w_raw[4:6] = 1.0  # Active in middle
        w_raw[10:13] = 1.0  # Active later
        
        T_stft = X_complex.shape[0]
        w_stft = interpolate_to_stft_frames(w_raw, times, T_stft, 10.0)
        
        M0 = make_initial_mask(Mix_mel, Mix_mel * 0.5, mel_fb, w_stft)
        
        print(f"✓ Mask generation successful")
        print(f"  Initial mask shape: {M0.shape}")
        print(f"  Mask range: [{M0.min():.3f}, {M0.max():.3f}]")
        print(f"  Mask mean: {M0.mean():.3f}")
        
    except Exception as e:
        print(f"✗ Mask generation failed: {e}")
        return False
    
    print("\n✓ All basic component tests passed!")
    return True


def test_query_trimming():
    """Test query trimming functionality."""
    print("\nTesting query trimming functionality...")
    
    # Create a 5-second query audio (longer than 3s limit)
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    long_query = 0.6 * np.sin(2 * np.pi * 440 * t)
    long_query += 0.02 * np.random.randn(len(long_query))
    long_query = long_query / np.max(np.abs(long_query)) * 0.8
    
    # Create test directory and save long query
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    long_query_path = test_dir / "long_query.wav"
    torchaudio.save(str(long_query_path), torch.from_numpy(long_query).unsqueeze(0), sr)
    
    try:
        from src.io_preprocessing import load_wav_mono16k, enforce_length_policy
        
        # Load and test trimming
        query_wav = load_wav_mono16k(long_query_path)
        print(f"  Original query duration: {len(query_wav)/sr:.2f}s")
        
        # Test trimming with save functionality
        trimmed_wav, trimmed_path = enforce_length_policy(
            query_wav, 3.0, policy="query", 
            save_trimmed=True,
            original_path=str(long_query_path),
            output_dir="test_outputs"
        )
        
        print(f"  Trimmed query duration: {len(trimmed_wav)/sr:.2f}s")
        if trimmed_path:
            print(f"  Trimmed file saved to: {trimmed_path}")
            
            # Verify the saved file
            if Path(trimmed_path).exists():
                saved_wav = load_wav_mono16k(trimmed_path)
                print(f"  Verified saved file duration: {len(saved_wav)/sr:.2f}s")
                print("✓ Query trimming test successful!")
                return True
            else:
                print("✗ Trimmed file was not saved!")
                return False
        else:
            print("✗ No trimmed path returned!")
            return False
            
    except Exception as e:
        print(f"✗ Query trimming test failed: {e}")
        return False


def run_integration_test():
    """Run full pipeline integration test."""
    print("\nRunning integration test...")
    
    # Create test data
    mix_path, query_path, expected_segments = create_synthetic_audio()
    
    # Run main pipeline with minimal parameters for faster testing
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "main.py",
        "--mix_wav", mix_path,
        "--query_wav", query_path,
        "--opt_steps", "5",  # Reduced for faster testing
        "--out_dir", "test_outputs",
        "--cls_percentile", "90",  # Adjusted for tight preset
        "--save_trimmed_query",  # Test the trimming feature
        "--save_renamed_query",  # Test query renaming
        "--save_visualizations"  # Test visualizations
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Integration test successful!")
            print("Output:")
            print(result.stdout)
            
            # Check outputs
            output_dir = Path("test_outputs")
            if output_dir.exists():
                files = list(output_dir.glob("*"))
                print(f"Generated {len(files)} output files:")
                for f in files:
                    print(f"  {f.name}")
            
            return True
        else:
            print("✗ Integration test failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Integration test timed out!")
        return False
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        return False


def cleanup():
    """Clean up test files."""
    print("\nCleaning up test files...")
    
    dirs_to_remove = ["test_data", "test_outputs"]
    for dir_name in dirs_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed {dir_name}/")


if __name__ == "__main__":
    print("=" * 50)
    print("Source Separation System Test")
    print("=" * 50)
    
    try:
        # Run tests
        basic_success = run_basic_test()
        trimming_success = test_query_trimming()
        
        if basic_success and trimming_success:
            integration_success = run_integration_test()
        else:
            integration_success = False
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"Basic component tests: {'PASS' if basic_success else 'FAIL'}")
        print(f"Query trimming test: {'PASS' if trimming_success else 'FAIL'}")
        print(f"Integration test: {'PASS' if integration_success else 'FAIL'}")
        
        if basic_success and trimming_success and integration_success:
            print("✓ All tests passed! System is ready to use.")
        else:
            print("✗ Some tests failed. Please check the implementation.")
        
    finally:
        cleanup()
