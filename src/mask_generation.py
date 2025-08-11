"""
Initial mask generation from mel spectrograms using the "stitching" approach.
"""
import numpy as np
from typing import Tuple
from .mel_processing import create_mel_filter_bank, compute_mel_to_linear_weights


def make_initial_mask(Mix_mel: np.ndarray, R_mel_timeline: np.ndarray, 
                     mel_fb: np.ndarray, w_stft: np.ndarray, 
                     gamma_profile: float = 0.3, mask_floor: float = 0.0,
                     epsilon: float = 1e-6, soft_masking: bool = True) -> np.ndarray:
    """
    Generate initial time-frequency mask from mel spectrograms.
    
    This implements the "stitching" approach where:
    1. Compute band-wise ratio mask from reference and mixture mel spectrograms
    2. Convert mel-domain mask to linear frequency domain
    3. Apply temporal weighting based on segment detection
    
    Args:
        Mix_mel: Mixture mel spectrogram [T, M]
        R_mel_timeline: Reference mel timeline [T, M] (from stitching)
        mel_fb: Mel filter bank [M, F]
        w_stft: Temporal weights [T] (from segment detection)
        gamma_profile: Query frequency profile influence (loose-lite: 0.2)
        mask_floor: Minimum mask value to prevent complete cutoff (loose-lite: 0.05)
        epsilon: Small constant for numerical stability
        
    Returns:
        Initial mask [T, F]
    """
    T, M = Mix_mel.shape
    M_fb, F = mel_fb.shape
    
    assert M == M_fb, f"Mel dimensions mismatch: {M} vs {M_fb}"
    assert len(w_stft) == T, f"Temporal weight length mismatch: {len(w_stft)} vs {T}"
    
    # Step 1: Compute band-wise ratio mask in mel domain with gamma_profile
    # Tight preset: stronger query frequency profile influence for better separation
    
    if gamma_profile > 0:
        # Blend reference with mixture, but give more weight to query profile
        blended_ref = gamma_profile * R_mel_timeline + (1 - gamma_profile) * Mix_mel
        B_mel = np.minimum(1.0, blended_ref / (Mix_mel + epsilon))
        
        # Apply gentle selectivity: softer boost for more natural separation
        confidence_boost = np.where(B_mel > 0.6, B_mel * 1.1, B_mel * 0.7)
        B_mel = np.minimum(1.0, confidence_boost)
    else:
        # Original approach
        B_mel = np.minimum(1.0, R_mel_timeline / (Mix_mel + epsilon))
    
    # Step 2: Convert mel-domain mask to linear frequency domain
    # M0[t, f] = (Σ_m B[t, m] * mel_fb[m, f]) / (Σ_m mel_fb[m, f])
    
    # Compute normalization weights for each frequency bin
    mel_weights = compute_mel_to_linear_weights(mel_fb)  # Shape: [F]
    
    # Apply mel-to-linear transformation
    M0 = np.zeros((T, F))
    for t in range(T):
        # Weighted sum over mel bands for each frequency bin
        numerator = np.dot(B_mel[t, :], mel_fb)  # [M] @ [M, F] -> [F]
        M0[t, :] = numerator / mel_weights
    
    # Step 3: Apply temporal weighting
    # M0 = w_stft[:, None] ⊙ M0
    M0 = w_stft[:, None] * M0
    
    # Step 4: Apply mask floor to prevent complete cutoff (tight preset)
    if mask_floor > 0:
        M0 = np.maximum(M0, mask_floor)
    
    # Step 5: Apply selective enhancement based on masking mode
    if soft_masking:
        M0 = apply_soft_selective_enhancement(M0, w_stft)
    else:
        M0 = apply_selective_enhancement(M0, w_stft)
    
    # Ensure mask is in [0, 1] range
    M0 = np.clip(M0, 0.0, 1.0)
    
    return M0


def compute_band_mask_with_smoothing(Mix_mel: np.ndarray, R_mel_timeline: np.ndarray, 
                                   epsilon: float = 1e-6, 
                                   smooth_sigma: float = 0.5) -> np.ndarray:
    """
    Compute band-wise mask with optional smoothing.
    
    Args:
        Mix_mel: Mixture mel spectrogram [T, M]
        R_mel_timeline: Reference mel timeline [T, M]
        epsilon: Numerical stability constant
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)
        
    Returns:
        Band mask [T, M]
    """
    from scipy.ndimage import gaussian_filter
    
    # Compute basic ratio mask
    B_mel = np.minimum(1.0, R_mel_timeline / (Mix_mel + epsilon))
    
    # Apply smoothing if requested
    if smooth_sigma > 0:
        B_mel = gaussian_filter(B_mel, sigma=smooth_sigma)
        B_mel = np.clip(B_mel, 0.0, 1.0)
    
    return B_mel


def enhance_mask_with_frequency_continuity(mask: np.ndarray, 
                                          freq_smooth_sigma: float = 1.0) -> np.ndarray:
    """
    Enhance mask with frequency continuity constraint.
    
    Args:
        mask: Input mask [T, F]
        freq_smooth_sigma: Smoothing sigma along frequency axis
        
    Returns:
        Enhanced mask [T, F]
    """
    from scipy.ndimage import gaussian_filter1d
    
    enhanced_mask = mask.copy()
    
    # Apply smoothing along frequency axis for each time frame
    for t in range(mask.shape[0]):
        enhanced_mask[t, :] = gaussian_filter1d(
            mask[t, :], sigma=freq_smooth_sigma, mode='reflect'
        )
    
    return np.clip(enhanced_mask, 0.0, 1.0)


def apply_spectral_gate(mask: np.ndarray, Mix_spec: np.ndarray, 
                       gate_threshold: float = 0.01) -> np.ndarray:
    """
    Apply spectral gating to suppress low-energy regions.
    
    Args:
        mask: Input mask [T, F]
        Mix_spec: Mixture magnitude spectrogram [T, F]
        gate_threshold: Relative threshold for gating
        
    Returns:
        Gated mask [T, F]
    """
    # Compute relative energy threshold
    max_energy = np.max(Mix_spec)
    energy_threshold = gate_threshold * max_energy
    
    # Create gate mask
    gate_mask = Mix_spec > energy_threshold
    
    # Apply gate
    gated_mask = mask * gate_mask.astype(np.float32)
    
    return gated_mask


def apply_selective_enhancement(mask: np.ndarray, w_stft: np.ndarray) -> np.ndarray:
    """
    Apply selective enhancement to boost high-confidence regions and suppress others.
    
    Args:
        mask: Initial mask [T, F]
        w_stft: Temporal weights [T]
        
    Returns:
        Enhanced mask [T, F]
    """
    enhanced_mask = mask.copy()
    T, F = mask.shape
    
    for t in range(T):
        temporal_weight = w_stft[t]
        
        if temporal_weight > 0.7:
            # High confidence temporal region: gentle boost
            enhanced_mask[t, :] = np.where(
                mask[t, :] > 0.4,
                np.minimum(mask[t, :] * 1.2, 0.9),   # Softer boost
                np.maximum(mask[t, :] * 0.5, 0.05)   # Less suppression
            )
        elif temporal_weight > 0.3:
            # Medium confidence: very gentle enhancement
            enhanced_mask[t, :] = np.where(
                mask[t, :] > 0.5,
                np.minimum(mask[t, :] * 1.05, 0.75), # Very gentle boost
                np.maximum(mask[t, :] * 0.7, 0.08)   # Much less suppression
            )
        else:
            # Low confidence: moderate suppression (not extreme)
            enhanced_mask[t, :] = mask[t, :] * 0.3
    
    return enhanced_mask


def apply_soft_selective_enhancement(mask: np.ndarray, w_stft: np.ndarray) -> np.ndarray:
    """
    Apply very gentle selective enhancement for natural separation.
    
    Args:
        mask: Initial mask [T, F]
        w_stft: Temporal weights [T]
        
    Returns:
        Softly enhanced mask [T, F]
    """
    enhanced_mask = mask.copy()
    T, F = mask.shape
    
    for t in range(T):
        temporal_weight = w_stft[t]
        
        if temporal_weight > 0.8:
            # Very high confidence: minimal boost
            enhanced_mask[t, :] = np.where(
                mask[t, :] > 0.5,
                np.minimum(mask[t, :] * 1.1, 0.85),  # Very gentle boost
                np.maximum(mask[t, :] * 0.6, 0.08)   # Minimal suppression
            )
        elif temporal_weight > 0.5:
            # High confidence: very gentle enhancement
            enhanced_mask[t, :] = np.where(
                mask[t, :] > 0.6,
                np.minimum(mask[t, :] * 1.05, 0.8),  # Minimal boost
                np.maximum(mask[t, :] * 0.8, 0.1)    # Very gentle suppression
            )
        elif temporal_weight > 0.2:
            # Medium confidence: preserve most content
            enhanced_mask[t, :] = np.where(
                mask[t, :] > 0.7,
                np.minimum(mask[t, :] * 1.02, 0.75), # Almost no boost
                np.maximum(mask[t, :] * 0.9, 0.12)   # Very minimal suppression
            )
        else:
            # Low confidence: preserve some content
            enhanced_mask[t, :] = mask[t, :] * 0.5
    
    return enhanced_mask


def validate_mask_properties(mask: np.ndarray) -> dict:
    """
    Validate mask properties and return statistics.
    
    Args:
        mask: Input mask [T, F]
        
    Returns:
        Dictionary of mask statistics
    """
    stats = {
        'shape': mask.shape,
        'min': float(np.min(mask)),
        'max': float(np.max(mask)),
        'mean': float(np.mean(mask)),
        'std': float(np.std(mask)),
        'sparsity': float(np.mean(mask < 0.1)),  # Fraction of near-zero values
        'activity': float(np.mean(mask > 0.5)),  # Fraction of active values
        'in_range': bool(np.all((mask >= 0) & (mask <= 1)))
    }
    
    return stats
