"""
Audio reconstruction and output functions.
"""
import numpy as np
import torch
import torchaudio
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


def reconstruct_and_save(X_complex: np.ndarray, mask: np.ndarray, 
                        out_dir: str, sr: int = 16000, 
                        nfft: int = 512, hop: int = 256) -> Dict[str, str]:
    """
    Reconstruct audio using iSTFT and save output files.
    
    Args:
        X_complex: Complex STFT of mixture [T, F]
        mask: Optimized mask [T, F]
        out_dir: Output directory path
        sr: Sample rate
        nfft: STFT window size
        hop: STFT hop size
        
    Returns:
        Dictionary of output file paths
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Apply mask to get separated complex spectrogram
    Y_complex = X_complex * mask
    
    # Compute residual
    R_complex = X_complex - Y_complex
    
    # Reconstruct audio using iSTFT
    target_audio = istft_reconstruction(Y_complex, sr, nfft, hop)
    residual_audio = istft_reconstruction(R_complex, sr, nfft, hop)
    
    # Save audio files
    target_path = out_path / "target.wav"
    residual_path = out_path / "residual.wav"
    
    save_audio(target_audio, target_path, sr)
    save_audio(residual_audio, residual_path, sr)
    
    return {
        "target": str(target_path),
        "residual": str(residual_path)
    }


def istft_reconstruction(complex_spec: np.ndarray, sr: int = 16000, 
                        nfft: int = 512, hop: int = 256) -> np.ndarray:
    """
    Reconstruct audio from complex STFT using inverse STFT.
    
    Args:
        complex_spec: Complex STFT [T, F]
        sr: Sample rate
        nfft: STFT window size
        hop: STFT hop size
        
    Returns:
        Reconstructed audio array
    """
    # Convert to torch tensor and transpose to [F, T]
    complex_tensor = torch.from_numpy(complex_spec).transpose(0, 1)
    
    # Create iSTFT transform
    istft_transform = torchaudio.transforms.InverseSpectrogram(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window_fn=torch.hann_window
    )
    
    # Reconstruct audio
    audio_tensor = istft_transform(complex_tensor)
    
    # Convert to numpy
    audio_array = audio_tensor.numpy().astype(np.float32)
    
    return audio_array


def save_audio(audio: np.ndarray, path: Path, sr: int = 16000) -> None:
    """
    Save audio array to file.
    
    Args:
        audio: Audio array
        path: Output file path
        sr: Sample rate
    """
    # Ensure audio is in correct shape for torchaudio
    if audio.ndim == 1:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Add channel dimension
    else:
        audio_tensor = torch.from_numpy(audio)
    
    # Normalize to prevent clipping
    max_val = torch.abs(audio_tensor).max()
    if max_val > 0:
        audio_tensor = audio_tensor / max_val * 0.95
    
    # Save using torchaudio
    torchaudio.save(str(path), audio_tensor, sr)


def dump_event_json(metadata: Dict[str, Any], out_dir: str) -> str:
    """
    Save event metadata to JSON file.
    
    Args:
        metadata: Event metadata dictionary
        out_dir: Output directory path
        
    Returns:
        Path to saved JSON file
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    json_path = out_path / "event.json"
    
    # Ensure all values are JSON serializable
    serializable_metadata = make_json_serializable(metadata)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
    
    return str(json_path)


def make_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON serializable format.
    
    Args:
        obj: Input object
        
    Returns:
        JSON serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def create_event_metadata(
    present: bool,
    segments: List[Tuple[float, float]],
    query_label: str,
    params: Dict[str, Any],
    processing_time: float,
    mask_stats: Dict[str, Any] = None,
    audio_paths: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive event metadata.
    
    Args:
        present: Whether target sound was detected
        segments: List of detected time segments
        query_label: Exact class label from AST
        params: Processing parameters
        processing_time: Total processing time in seconds
        mask_stats: Optional mask statistics
        audio_paths: Optional output audio paths
        
    Returns:
        Event metadata dictionary
    """
    metadata = {
        "present": present,
        "segments": segments,
        "query_label": query_label,
        "processing_time_seconds": processing_time,
        "parameters": params
    }
    
    if mask_stats:
        metadata["mask_statistics"] = mask_stats
    
    if audio_paths:
        metadata["output_files"] = audio_paths
    
    # Add derived statistics
    if segments:
        total_duration = sum(end - start for start, end in segments)
        metadata["total_detected_duration"] = total_duration
        metadata["num_segments"] = len(segments)
        metadata["average_segment_duration"] = total_duration / len(segments)
    else:
        metadata["total_detected_duration"] = 0.0
        metadata["num_segments"] = 0
        metadata["average_segment_duration"] = 0.0
    
    return metadata


def compute_separation_metrics(target_audio: np.ndarray, residual_audio: np.ndarray, 
                              mix_audio: np.ndarray) -> Dict[str, float]:
    """
    Compute basic separation quality metrics.
    
    Args:
        target_audio: Separated target audio
        residual_audio: Residual audio
        mix_audio: Original mixture audio
        
    Returns:
        Dictionary of metrics
    """
    # Energy ratios
    target_energy = np.sum(target_audio ** 2)
    residual_energy = np.sum(residual_audio ** 2)
    mix_energy = np.sum(mix_audio ** 2)
    
    # Signal-to-residual ratio (higher is better)
    if residual_energy > 0:
        srr_db = 10 * np.log10(target_energy / residual_energy)
    else:
        srr_db = float('inf')
    
    # Energy preservation ratio (should be close to 1.0)
    if mix_energy > 0:
        energy_ratio = (target_energy + residual_energy) / mix_energy
    else:
        energy_ratio = 0.0
    
    # Target energy ratio (fraction of energy in target)
    if mix_energy > 0:
        target_ratio = target_energy / mix_energy
    else:
        target_ratio = 0.0
    
    return {
        "signal_to_residual_ratio_db": float(srr_db),
        "energy_preservation_ratio": float(energy_ratio),
        "target_energy_ratio": float(target_ratio),
        "target_rms": float(np.sqrt(np.mean(target_audio ** 2))),
        "residual_rms": float(np.sqrt(np.mean(residual_audio ** 2)))
    }


def create_visualization_data(mask: np.ndarray, segments: List[Tuple[float, float]], 
                             times: np.ndarray, w_stft: np.ndarray) -> Dict[str, Any]:
    """
    Create data for visualization (optional).
    
    Args:
        mask: Final optimized mask [T, F]
        segments: Detected segments
        times: Time positions
        w_stft: Temporal weights
        
    Returns:
        Visualization data dictionary
    """
    return {
        "mask_shape": mask.shape,
        "mask_mean_over_time": np.mean(mask, axis=1).tolist(),
        "mask_mean_over_frequency": np.mean(mask, axis=0).tolist(),
        "temporal_weights": w_stft.tolist(),
        "time_positions": times.tolist(),
        "segments": segments
    }
