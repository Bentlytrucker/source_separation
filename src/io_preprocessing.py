"""
I/O preprocessing functions for audio loading and normalization.
"""
import numpy as np
import torchaudio
import torch
from pathlib import Path
from typing import Union, Tuple, Optional


def load_wav_mono16k(path: Union[str, Path]) -> np.ndarray:
    """
    Load audio file and convert to mono 16kHz float32 numpy array.
    
    Args:
        path: Path to audio file
        
    Returns:
        np.ndarray: Audio data as float32 array, shape (samples,)
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    try:
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(str(path))
        
        # Convert to mono by averaging channels if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=16000,
                resampling_method='linear'
            )
            waveform = resampler(waveform)
        
        # Convert to numpy float32 and squeeze to 1D
        wav_array = waveform.squeeze().numpy().astype(np.float32)
        
        return wav_array
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {path}: {str(e)}")


def enforce_length_policy(wav: np.ndarray, target_length_sec: float, 
                         sr: int = 16000, policy: str = "mix", 
                         save_trimmed: bool = False, 
                         original_path: str = None,
                         output_dir: str = None) -> Tuple[np.ndarray, Optional[str]]:
    """
    Enforce length policy for input audio.
    
    Args:
        wav: Input audio array
        target_length_sec: Target length in seconds
        sr: Sample rate (default: 16000)
        policy: "mix" for mix_wav (pad/trim to exact length), 
               "query" for query_wav (error if <1s, trim if >3s)
        save_trimmed: Whether to save trimmed query audio to file
        original_path: Path to original audio file (for generating trimmed filename)
        output_dir: Directory to save trimmed file
               
    Returns:
        Tuple of (length-adjusted audio array, path to trimmed file if saved)
        
    Raises:
        ValueError: If query audio is too short
    """
    target_samples = int(target_length_sec * sr)
    current_samples = len(wav)
    trimmed_path = None
    
    if policy == "mix":
        # Mix policy: exactly 10 seconds
        if current_samples < target_samples:
            # Pad with zeros at the end
            pad_samples = target_samples - current_samples
            wav = np.pad(wav, (0, pad_samples), mode='constant', constant_values=0)
        elif current_samples > target_samples:
            # Take first 10 seconds
            wav = wav[:target_samples]
            
    elif policy == "query":
        # Query policy: 1-3 seconds range
        min_samples = int(1.0 * sr)  # 1 second minimum
        max_samples = int(3.0 * sr)  # 3 seconds maximum
        
        if current_samples < min_samples:
            raise ValueError(f"Query audio too short: {current_samples/sr:.2f}s < 1.0s minimum")
        elif current_samples > max_samples:
            # Trim to 3 seconds
            original_duration = current_samples / sr
            wav = wav[:max_samples]
            
            print(f"  Query audio trimmed: {original_duration:.2f}s → 3.00s")
            
            # Save trimmed query if requested
            if save_trimmed and original_path and output_dir:
                trimmed_path = save_trimmed_query(wav, original_path, output_dir, sr)
                print(f"  Trimmed query saved to: {trimmed_path}")
            
    return wav, trimmed_path


def save_renamed_query(wav: np.ndarray, original_path: str, output_dir: str, 
                      class_label: str, sr: int = 16000, suffix: str = "") -> str:
    """
    Save query audio with class label as filename.
    
    Args:
        wav: Query audio array
        original_path: Path to original query file
        output_dir: Output directory
        class_label: Classified class label
        sr: Sample rate
        suffix: Optional suffix for filename (e.g., "_trimmed_3s")
        
    Returns:
        Path to saved renamed file
    """
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename from class label
    safe_name = create_safe_slug(class_label)
    filename = f"{safe_name}{suffix}.wav"
    renamed_path = out_path / filename
    
    # Convert to torch tensor and save
    if wav.ndim == 1:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # Add channel dimension
    else:
        wav_tensor = torch.from_numpy(wav)
    
    # Normalize to prevent clipping
    max_val = torch.abs(wav_tensor).max()
    if max_val > 0:
        wav_tensor = wav_tensor / max_val * 0.95
    
    # Save using torchaudio
    torchaudio.save(str(renamed_path), wav_tensor, sr)
    
    return str(renamed_path)


def save_trimmed_query(wav: np.ndarray, original_path: str, output_dir: str, sr: int = 16000) -> str:
    """
    Save trimmed query audio to output directory.
    
    Args:
        wav: Trimmed audio array
        original_path: Path to original query file
        output_dir: Output directory
        sr: Sample rate
        
    Returns:
        Path to saved trimmed file
    """
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename for trimmed query
    original_name = Path(original_path).stem
    trimmed_filename = f"{original_name}_trimmed_3s.wav"
    trimmed_path = out_path / trimmed_filename
    
    # Convert to torch tensor and save
    if wav.ndim == 1:
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)  # Add channel dimension
    else:
        wav_tensor = torch.from_numpy(wav)
    
    # Normalize to prevent clipping
    max_val = torch.abs(wav_tensor).max()
    if max_val > 0:
        wav_tensor = wav_tensor / max_val * 0.95
    
    # Save using torchaudio
    torchaudio.save(str(trimmed_path), wav_tensor, sr)
    
    return str(trimmed_path)


def create_safe_slug(text: str) -> str:
    """
    Convert class label to filesystem-safe slug.
    
    Args:
        text: Original class label (e.g., "Glass breaking")
        
    Returns:
        str: Safe slug (e.g., "glass_breaking")
    """
    import re
    
    # Convert to lowercase
    slug = text.lower()
    
    # Replace non-alphanumeric characters with underscores
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    
    # Collapse multiple consecutive underscores
    slug = re.sub(r'_+', '_', slug)
    
    return slug
