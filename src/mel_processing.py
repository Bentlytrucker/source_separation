"""
Mel spectrogram processing and "stitching" approach for reference generation.
"""
import numpy as np
import torch
import torchaudio
from typing import List, Tuple
from scipy.ndimage import gaussian_filter1d


def compute_mel(wav: np.ndarray, sr: int = 16000, nfft: int = 512, 
               hop: int = 256, n_mels: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STFT and mel spectrogram from audio.
    
    Args:
        wav: Input audio array
        sr: Sample rate
        nfft: FFT window size
        hop: Hop size
        n_mels: Number of mel bands
        
    Returns:
        Tuple of (complex_spectrogram, mel_spectrogram)
        complex_spectrogram: Complex STFT [T, F]
        mel_spectrogram: Mel spectrogram [T, M]
    """
    # Convert to torch tensor
    wav_tensor = torch.from_numpy(wav).float()
    
    # Compute STFT
    stft_transform = torchaudio.transforms.Spectrogram(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window_fn=torch.hann_window,
        power=None  # Return complex values
    )
    
    complex_spec = stft_transform(wav_tensor)  # Shape: [F, T]
    complex_spec = complex_spec.transpose(0, 1)  # Shape: [T, F]
    
    # Compute magnitude spectrogram
    mag_spec = torch.abs(complex_spec)
    
    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window_fn=torch.hann_window,
        n_mels=n_mels,
        f_min=0.0,
        f_max=sr//2
    )
    
    mel_spec = mel_transform(wav_tensor)  # Shape: [M, T]
    mel_spec = mel_spec.transpose(0, 1)  # Shape: [T, M]
    
    # Convert to numpy
    complex_spec_np = complex_spec.numpy()
    mel_spec_np = mel_spec.numpy()
    
    return complex_spec_np, mel_spec_np


def stretch_or_tile_query_mel(Q_mel: np.ndarray, L: int) -> np.ndarray:
    """
    Stretch or tile query mel to match segment length using the "stitching" approach.
    
    Args:
        Q_mel: Query mel spectrogram [Tq, M]
        L: Target length in frames
        
    Returns:
        Length-matched query mel [L, M]
    """
    Tq, M = Q_mel.shape
    
    if L <= 0:
        return np.zeros((L, M))
    
    if L <= int(1.5 * Tq):
        # Case 1: Linear interpolation (stretch)
        # Create time grids
        original_times = np.arange(Tq)
        target_times = np.linspace(0, Tq - 1, L)
        
        # Interpolate each mel band
        Qm_seg = np.zeros((L, M))
        for m in range(M):
            Qm_seg[:, m] = np.interp(target_times, original_times, Q_mel[:, m])
            
    else:
        # Case 2: Tiling with partial copy
        k = L // Tq  # Number of full repetitions
        r = L % Tq   # Remaining frames
        
        # Create tiled array
        Qm_seg = np.zeros((L, M))
        
        # Fill with full repetitions
        for i in range(k):
            start_idx = i * Tq
            end_idx = (i + 1) * Tq
            Qm_seg[start_idx:end_idx, :] = Q_mel
            
            # Apply crossfade at boundaries (except first)
            if i > 0:
                crossfade_frames = min(2, Tq // 4)  # 2-frame crossfade or 1/4 of query length
                fade_start = start_idx
                fade_end = start_idx + crossfade_frames
                
                # Linear crossfade
                for f in range(crossfade_frames):
                    if fade_start + f < L:
                        alpha = f / crossfade_frames
                        # Blend with previous content
                        prev_val = Qm_seg[fade_start + f, :]
                        curr_val = Q_mel[f, :]
                        Qm_seg[fade_start + f, :] = (1 - alpha) * prev_val + alpha * curr_val
        
        # Fill remaining frames
        if r > 0:
            start_idx = k * Tq
            Qm_seg[start_idx:start_idx + r, :] = Q_mel[:r, :]
            
            # Apply crossfade if there was a previous segment
            if k > 0:
                crossfade_frames = min(2, r)
                for f in range(crossfade_frames):
                    alpha = f / crossfade_frames
                    prev_val = Qm_seg[start_idx + f, :]
                    curr_val = Q_mel[f, :]
                    Qm_seg[start_idx + f, :] = (1 - alpha) * prev_val + alpha * curr_val
    
    return Qm_seg


def assemble_R_mel_timeline(Q_mel: np.ndarray, segments: List[Tuple[float, float]], 
                           Tmix: int, M: int, sr: int = 16000, 
                           hop: int = 256) -> np.ndarray:
    """
    Assemble reference mel timeline using the "stitching" approach.
    
    Args:
        Q_mel: Query mel spectrogram [Tq, M]
        segments: List of (start_time, end_time) segments in seconds
        Tmix: Total number of STFT frames in mixture
        M: Number of mel bands
        sr: Sample rate
        hop: STFT hop size
        
    Returns:
        Reference mel timeline [Tmix, M]
    """
    R_mel_timeline = np.zeros((Tmix, M))
    
    # Convert time segments to frame indices
    for start_time, end_time in segments:
        # Convert to frame indices
        tau0 = int(start_time * sr / hop)
        tau1 = int(end_time * sr / hop)
        
        # Ensure bounds
        tau0 = max(0, tau0)
        tau1 = min(Tmix, tau1)
        
        if tau1 > tau0:
            L = tau1 - tau0
            
            # Generate length-matched query mel for this segment
            Qm_seg = stretch_or_tile_query_mel(Q_mel, L)
            
            # Place in timeline
            R_mel_timeline[tau0:tau1, :] = Qm_seg
    
    return R_mel_timeline


def create_mel_filter_bank(sr: int = 16000, nfft: int = 512, n_mels: int = 64) -> np.ndarray:
    """
    Create mel filter bank matrix for mel-to-linear conversion.
    
    Args:
        sr: Sample rate
        nfft: FFT size
        n_mels: Number of mel bands
        
    Returns:
        Mel filter bank [n_mels, n_freqs]
    """
    # Use torchaudio's mel scale
    mel_transform = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sr,
        f_min=0.0,
        f_max=sr//2,
        n_stft=nfft//2 + 1
    )
    
    # Create dummy input to extract filter bank
    dummy_spec = torch.eye(nfft//2 + 1).unsqueeze(0)  # [1, F, F]
    mel_fb = mel_transform(dummy_spec).squeeze(0)  # [M, F]
    
    return mel_fb.numpy()


def apply_crossfade(signal: np.ndarray, fade_length: int = 2) -> np.ndarray:
    """
    Apply crossfade at segment boundaries to reduce artifacts.
    
    Args:
        signal: Input signal [T, ...]
        fade_length: Length of crossfade in frames
        
    Returns:
        Crossfaded signal
    """
    if fade_length <= 0 or len(signal) <= fade_length:
        return signal
    
    # Apply gentle smoothing at boundaries
    smoothed = signal.copy()
    
    # Smooth beginning
    for i in range(min(fade_length, len(signal))):
        alpha = i / fade_length
        smoothed[i] = alpha * smoothed[i]
    
    # Smooth end
    for i in range(min(fade_length, len(signal))):
        idx = len(signal) - 1 - i
        alpha = i / fade_length
        smoothed[idx] = alpha * smoothed[idx]
    
    return smoothed


def compute_mel_to_linear_weights(mel_fb: np.ndarray) -> np.ndarray:
    """
    Compute normalization weights for mel-to-linear conversion.
    
    Args:
        mel_fb: Mel filter bank [M, F]
        
    Returns:
        Normalization weights [F]
    """
    # Sum of mel filter bank weights for each frequency bin
    weights = np.sum(mel_fb, axis=0)
    
    # Avoid division by zero
    weights = np.maximum(weights, 1e-8)
    
    return weights
