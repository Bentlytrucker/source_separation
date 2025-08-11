"""
Embedding-aware mask generation using AST embeddings for semantic similarity.
This module creates softer, more natural masks based on semantic content rather than
just query audio matching.
"""
import numpy as np
from typing import Tuple, Dict, List
from scipy.ndimage import gaussian_filter
from .ast_analysis import cosine_similarity


def interpolate_to_stft_frames(
    weights: np.ndarray, 
    times: np.ndarray, 
    T_stft: int, 
    total_duration: float
) -> np.ndarray:
    """
    Interpolate window-based weights to STFT frame resolution.
    
    Args:
        weights: Window weights [T_win]
        times: Time positions for windows [T_win]
        T_stft: Number of STFT frames
        total_duration: Total duration in seconds
        
    Returns:
        Interpolated weights [T_stft]
    """
    # Create STFT time grid
    stft_times = np.linspace(0, total_duration, T_stft)
    
    # Interpolate weights to STFT resolution
    interpolated_weights = np.interp(stft_times, times, weights)
    
    return interpolated_weights


def create_embedding_aware_mask(
    Mix_mel: np.ndarray,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    query_class: str,
    w_stft: np.ndarray,
    mel_fb: np.ndarray,
    times: np.ndarray,
    total_duration: float = 10.0,
    mask_floor: float = 0.05,
    similarity_smoothing: float = 1.0,
    class_aware_blending: bool = True
) -> np.ndarray:
    """
    Create embedding-aware mask using semantic similarity.
    
    Args:
        Mix_mel: Mixture mel spectrogram [T, M]
        embeddings: Window embeddings [T_win, D]
        query_embedding: Query embedding vector [D]
        query_class: Query class name (e.g., "Speech")
        w_stft: Temporal weights [T]
        mel_fb: Mel filter bank [M, F]
        times: Time positions for embedding windows [T_win]
        total_duration: Total audio duration in seconds
        mask_floor: Minimum mask value
        similarity_smoothing: Gaussian smoothing for similarity weights
        class_aware_blending: Whether to use class-specific blending
        
    Returns:
        Embedding-aware mask [T, F]
    """
    T, M = Mix_mel.shape
    M_fb, F = mel_fb.shape
    
    # Step 1: Compute semantic similarity weights
    similarity_weights_raw = compute_semantic_similarity_weights(
        embeddings, query_embedding, query_class, similarity_smoothing
    )
    
    # Step 1.5: Interpolate similarity weights to STFT frame resolution
    similarity_weights = interpolate_to_stft_frames(
        similarity_weights_raw, times, T, total_duration
    )
    
    # Step 2: Create class-aware frequency profile
    freq_profile = create_class_aware_frequency_profile(
        query_class, M, F, mel_fb, class_aware_blending
    )
    
    # Step 3: Generate embedding-aware mel mask
    mel_mask = create_semantic_mel_mask(
        Mix_mel, similarity_weights, freq_profile, mask_floor
    )
    
    # Step 4: Convert to linear frequency domain
    linear_mask = convert_mel_to_linear_mask(mel_mask, mel_fb)
    
    # Step 5: Apply temporal weighting with semantic awareness
    final_mask = apply_semantic_temporal_weighting(
        linear_mask, w_stft, similarity_weights
    )
    
    # Safety check: replace any NaN or inf values
    final_mask = np.nan_to_num(final_mask, nan=mask_floor, posinf=1.0, neginf=0.0)
    final_mask = np.clip(final_mask, 0.0, 1.0)
    
    # Final validation
    if np.any(np.isnan(final_mask)) or np.any(np.isinf(final_mask)):
        print("Warning: NaN/inf values detected in embedding mask, using fallback")
        final_mask = np.full_like(final_mask, mask_floor)
    
    return final_mask


def compute_semantic_similarity_weights(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    query_class: str,
    smoothing: float = 1.0
) -> np.ndarray:
    """
    Compute semantic similarity weights with class-specific adjustments.
    
    Args:
        embeddings: Window embeddings [T, D]
        query_embedding: Query embedding [D]
        query_class: Query class name
        smoothing: Gaussian smoothing sigma
        
    Returns:
        Semantic similarity weights [T]
    """
    T = len(embeddings)
    similarities = np.zeros(T)
    
    # Compute cosine similarities
    for t in range(T):
        similarities[t] = cosine_similarity(embeddings[t], query_embedding)
    
    # Ensure similarities are in valid range before adjustments
    similarities = np.clip(similarities, -1.0, 1.0)
    
    # Apply class-specific adjustments
    adjusted_similarities = apply_class_specific_adjustments(
        similarities, query_class
    )
    
    # Apply smoothing for continuity
    if smoothing > 0:
        adjusted_similarities = gaussian_filter(adjusted_similarities, sigma=smoothing)
    
    # Normalize to [0, 1]
    sim_min, sim_max = adjusted_similarities.min(), adjusted_similarities.max()
    if sim_max > sim_min:
        normalized_similarities = (adjusted_similarities - sim_min) / (sim_max - sim_min)
    else:
        normalized_similarities = np.ones_like(adjusted_similarities) * 0.5
    
    return normalized_similarities


def apply_class_specific_adjustments(
    similarities: np.ndarray,
    query_class: str
) -> np.ndarray:
    """
    Apply class-specific adjustments to similarity scores.
    
    Args:
        similarities: Raw similarity scores [T] (should be in [-1, 1])
        query_class: Query class name
        
    Returns:
        Adjusted similarities [T] in [0, 1]
    """
    # First normalize to [0, 1] range to avoid NaN in power operations
    normalized = (similarities + 1.0) / 2.0  # [-1,1] -> [0,1]
    normalized = np.clip(normalized, 0.0, 1.0)
    
    adjusted = normalized.copy()
    
    # Class-specific adjustments for more natural separation
    if "speech" in query_class.lower():
        # Speech: More gradual transitions, preserve more content
        adjusted = np.power(adjusted, 0.7)  # Softer thresholding (now safe)
        adjusted = np.where(adjusted > 0.3, adjusted * 1.2, adjusted * 0.8)
        
    elif "music" in query_class.lower():
        # Music: Preserve harmonic content
        adjusted = np.power(adjusted, 0.8)
        adjusted = np.where(adjusted > 0.4, adjusted * 1.1, adjusted * 0.9)
        
    elif "noise" in query_class.lower() or "sound" in query_class.lower():
        # Environmental sounds: More selective
        adjusted = np.power(adjusted, 1.2)
        adjusted = np.where(adjusted > 0.5, adjusted * 1.3, adjusted * 0.6)
        
    else:
        # Default: Balanced approach
        adjusted = np.power(adjusted, 0.9)
        adjusted = np.where(adjusted > 0.4, adjusted * 1.1, adjusted * 0.8)
    
    return np.clip(adjusted, 0.0, 1.0)


def create_class_aware_frequency_profile(
    query_class: str,
    n_mels: int,
    n_freq: int,
    mel_fb: np.ndarray,
    use_blending: bool = True
) -> np.ndarray:
    """
    Create class-aware frequency profile for different sound types.
    
    Args:
        query_class: Query class name
        n_mels: Number of mel bands
        n_freq: Number of frequency bins
        mel_fb: Mel filter bank [M, F]
        use_blending: Whether to use frequency blending
        
    Returns:
        Frequency profile [M]
    """
    profile = np.ones(n_mels)
    
    if not use_blending:
        return profile
    
    # Class-specific frequency profiles
    if "speech" in query_class.lower():
        # Speech: Emphasize mid frequencies (formants)
        mel_centers = np.linspace(0, 1, n_mels)
        speech_weights = np.exp(-((mel_centers - 0.4) ** 2) / (2 * 0.2 ** 2))
        profile = 0.3 + 0.7 * speech_weights
        
    elif "music" in query_class.lower():
        # Music: Broader frequency range with harmonic emphasis
        mel_centers = np.linspace(0, 1, n_mels)
        music_weights = np.exp(-((mel_centers - 0.5) ** 2) / (2 * 0.3 ** 2))
        profile = 0.4 + 0.6 * music_weights
        
    elif "noise" in query_class.lower():
        # Noise: More uniform across frequencies
        profile = np.ones(n_mels) * 0.8
        
    else:
        # Default: Slight emphasis on mid frequencies
        mel_centers = np.linspace(0, 1, n_mels)
        default_weights = np.exp(-((mel_centers - 0.5) ** 2) / (2 * 0.25 ** 2))
        profile = 0.5 + 0.5 * default_weights
    
    return profile


def create_semantic_mel_mask(
    Mix_mel: np.ndarray,
    similarity_weights: np.ndarray,
    freq_profile: np.ndarray,
    mask_floor: float
) -> np.ndarray:
    """
    Create semantic-aware mel mask using similarity weights and frequency profile.
    
    Args:
        Mix_mel: Mixture mel spectrogram [T, M]
        similarity_weights: Semantic similarity weights [T]
        freq_profile: Frequency profile [M]
        mask_floor: Minimum mask value
        
    Returns:
        Semantic mel mask [T, M]
    """
    T, M = Mix_mel.shape
    
    # Create base mask using similarity weights
    base_mask = similarity_weights[:, None] * freq_profile[None, :]
    
    # Apply adaptive thresholding based on mixture energy
    energy_threshold = np.percentile(Mix_mel, 20)  # 20th percentile
    energy_mask = Mix_mel > energy_threshold
    
    # Combine similarity and energy information
    semantic_mask = base_mask * energy_mask.astype(np.float32)
    
    # Apply mask floor
    semantic_mask = np.maximum(semantic_mask, mask_floor)
    
    # Smooth the mask for natural transitions
    semantic_mask = gaussian_filter(semantic_mask, sigma=0.5)
    
    return semantic_mask


def convert_mel_to_linear_mask(
    mel_mask: np.ndarray,
    mel_fb: np.ndarray
) -> np.ndarray:
    """
    Convert mel-domain mask to linear frequency domain.
    
    Args:
        mel_mask: Mel mask [T, M]
        mel_fb: Mel filter bank [M, F]
        
    Returns:
        Linear frequency mask [T, F]
    """
    T, M = mel_mask.shape
    M_fb, F = mel_fb.shape
    
    # Compute mel filter bank weights for normalization
    mel_weights = np.sum(mel_fb, axis=0)  # [F]
    mel_weights = np.maximum(mel_weights, 1e-6)  # Avoid division by zero
    
    # Convert mel to linear frequency
    linear_mask = np.zeros((T, F))
    for t in range(T):
        # Weighted sum over mel bands
        numerator = np.dot(mel_mask[t, :], mel_fb)  # [M] @ [M, F] -> [F]
        linear_mask[t, :] = numerator / mel_weights
    
    return linear_mask


def apply_semantic_temporal_weighting(
    linear_mask: np.ndarray,
    w_stft: np.ndarray,
    similarity_weights: np.ndarray
) -> np.ndarray:
    """
    Apply semantic-aware temporal weighting.
    
    Args:
        linear_mask: Linear frequency mask [T, F]
        w_stft: Temporal weights [T]
        similarity_weights: Semantic similarity weights [T]
        
    Returns:
        Final mask [T, F]
    """
    T, F = linear_mask.shape
    
    # Combine traditional temporal weights with semantic similarity
    semantic_temporal_weights = 0.6 * w_stft + 0.4 * similarity_weights
    
    # Apply semantic-aware weighting
    final_mask = linear_mask * semantic_temporal_weights[:, None]
    
    # Apply additional semantic smoothing
    for f in range(F):
        final_mask[:, f] = gaussian_filter(final_mask[:, f], sigma=1.0)
    
    return final_mask


def create_adaptive_mask_blending(
    traditional_mask: np.ndarray,
    semantic_mask: np.ndarray,
    query_class: str,
    blending_weight: float = 0.7
) -> np.ndarray:
    """
    Create adaptive blending between traditional and semantic masks.
    
    Args:
        traditional_mask: Traditional stitching mask [T, F]
        semantic_mask: Semantic similarity mask [T, F]
        query_class: Query class name
        blending_weight: Weight for semantic mask (0-1)
        
    Returns:
        Blended mask [T, F]
    """
    # Class-specific blending weights
    if "speech" in query_class.lower():
        # Speech: More semantic, less traditional
        semantic_weight = 0.8
    elif "music" in query_class.lower():
        # Music: Balanced approach
        semantic_weight = 0.6
    else:
        # Default: Use provided weight
        semantic_weight = blending_weight
    
    # Blend masks
    blended_mask = (1 - semantic_weight) * traditional_mask + semantic_weight * semantic_mask
    
    return blended_mask


def create_spectrogram_feature_mask(
    Mix_mel: np.ndarray,
    ast_features: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
    query_class: str,
    w_stft: np.ndarray,
    mel_fb: np.ndarray,
    mask_floor: float = 0.05
) -> np.ndarray:
    """
    Create mask using AST's internal spectrogram features.
    
    Args:
        Mix_mel: Mixture mel spectrogram [T, M]
        ast_features: AST feature maps from mixture
        query_embedding: Query embedding vector [D]
        query_class: Query class name
        w_stft: Temporal weights [T]
        mel_fb: Mel filter bank [M, F]
        mask_floor: Minimum mask value
        
    Returns:
        Feature-aware mask [T, F]
    """
    T, M = Mix_mel.shape
    M_fb, F = mel_fb.shape
    
    # Step 1: Convert patch features to time-frequency domain
    patch_embeddings = ast_features['patch_embeddings']  # [num_patches, hidden_dim]
    
    # AST uses 16x16 patches on 128x1024 spectrogram
    # Calculate patch grid dimensions
    patch_height, patch_width = 16, 16
    mel_height, mel_time = 128, 1024  # AST's internal resolution
    
    patches_h = mel_height // patch_height  # 8 patches in frequency
    patches_w = mel_time // patch_width     # 64 patches in time
    
    # Step 2: Compute patch similarities with query
    patch_similarities = compute_patch_similarities(
        patch_embeddings, query_embedding, query_class
    )
    
    # Step 3: Reshape to 2D patch grid
    if len(patch_similarities) == patches_h * patches_w:
        patch_grid = patch_similarities.reshape(patches_h, patches_w)  # [8, 64]
    else:
        # Fallback: use average similarity
        avg_sim = np.mean(patch_similarities)
        patch_grid = np.full((patches_h, patches_w), avg_sim)
    
    # Step 4: Upsample patch grid to our mel resolution
    feature_mask_mel = upsample_patch_grid_to_mel(
        patch_grid, T, M, patches_h, patches_w
    )
    
    # Step 5: Convert to linear frequency domain
    linear_mask = convert_mel_to_linear_mask(feature_mask_mel, mel_fb)
    
    # Step 6: Apply temporal weighting
    final_mask = linear_mask * w_stft[:, None]
    
    # Step 7: Apply mask floor and normalize
    final_mask = np.maximum(final_mask, mask_floor)
    final_mask = np.clip(final_mask, 0.0, 1.0)
    
    # Safety check: replace any NaN or inf values
    final_mask = np.nan_to_num(final_mask, nan=mask_floor, posinf=1.0, neginf=0.0)
    
    # Final validation
    if np.any(np.isnan(final_mask)) or np.any(np.isinf(final_mask)):
        print("Warning: NaN/inf values detected in mask, using fallback")
        final_mask = np.full_like(final_mask, mask_floor)
    
    return final_mask


def compute_patch_similarities(
    patch_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    query_class: str
) -> np.ndarray:
    """
    Compute similarity between each patch and query embedding.
    
    Args:
        patch_embeddings: Patch embeddings [num_patches, hidden_dim]
        query_embedding: Query embedding [hidden_dim]
        query_class: Query class name
        
    Returns:
        Patch similarities [num_patches]
    """
    similarities = np.zeros(len(patch_embeddings))
    
    for i, patch_emb in enumerate(patch_embeddings):
        similarities[i] = cosine_similarity(patch_emb, query_embedding)
    
    # Ensure similarities are in valid range
    similarities = np.clip(similarities, -1.0, 1.0)
    
    # Apply class-specific adjustments
    adjusted_similarities = apply_class_specific_adjustments(
        similarities, query_class
    )
    
    return adjusted_similarities


def upsample_patch_grid_to_mel(
    patch_grid: np.ndarray,
    target_T: int,
    target_M: int,
    patches_h: int,
    patches_w: int
) -> np.ndarray:
    """
    Upsample patch grid to target mel spectrogram resolution.
    
    Args:
        patch_grid: Patch similarity grid [patches_h, patches_w]
        target_T: Target time frames
        target_M: Target mel bands
        patches_h: Number of patches in frequency
        patches_w: Number of patches in time
        
    Returns:
        Upsampled mask [target_T, target_M]
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_t = target_T / patches_w  # Time axis
    zoom_f = target_M / patches_h  # Frequency axis
    
    # Upsample using bilinear interpolation
    upsampled = zoom(patch_grid, (zoom_f, zoom_t), order=1)
    
    # Ensure exact target size
    if upsampled.shape != (target_M, target_T):
        # Resize to exact target
        upsampled_resized = np.zeros((target_M, target_T))
        min_f = min(upsampled.shape[0], target_M)
        min_t = min(upsampled.shape[1], target_T)
        upsampled_resized[:min_f, :min_t] = upsampled[:min_f, :min_t]
        upsampled = upsampled_resized
    
    # Transpose to [T, M] format
    return upsampled.T
