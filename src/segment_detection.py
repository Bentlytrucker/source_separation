"""
Temporal segment detection based on AST predictions and embeddings.
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from .ast_analysis import cosine_similarity


def build_segments(
    times: np.ndarray,
    probs: np.ndarray, 
    embeddings: np.ndarray,
    query_class_idx: int,
    query_embedding: np.ndarray,
    cls_threshold: Optional[float] = None,
    cls_percentile: Optional[int] = None,
    use_cosine_gate: bool = True,
    cosine_threshold: float = 0.35,
    alpha: float = 0.5,
    beta: float = 0.5,
    min_dur_ms: int = 500,
    merge_gap_ms: int = 250,
    segment_padding_ms: int = 0,
    time_mask_sharpening: float = 1.0,
    hysteresis_enter: Optional[float] = None,
    hysteresis_exit: Optional[float] = None
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Build temporal segments based on class probabilities and embedding similarity.
    
    Args:
        times: Time positions for each window
        probs: Class probabilities [T, C]
        embeddings: Window embeddings [T, D] 
        query_class_idx: Target class index
        query_embedding: Query embedding vector
        cls_threshold: Absolute threshold for class probability
        cls_percentile: Percentile threshold for class probability
        use_cosine_gate: Whether to use cosine similarity gating
        cosine_threshold: Threshold for cosine similarity
        alpha: Weight for class probability component
        beta: Weight for cosine similarity component
        min_dur_ms: Minimum segment duration in milliseconds
        merge_gap_ms: Maximum gap to merge segments in milliseconds
        segment_padding_ms: Padding around segments in milliseconds (loose-lite: 120)
        time_mask_sharpening: Power for time mask sharpening (loose-lite: 0.9)
        hysteresis_enter: Enter threshold for hysteresis (if None, uses main threshold)
        hysteresis_exit: Exit threshold for hysteresis (if None, uses main threshold * 0.8)
        
    Returns:
        Tuple of (segments_list, w_stft_interpolated)
    """
    T = len(times)
    
    # Extract class timeline
    p_cls = probs[:, query_class_idx]  # Shape: [T]
    
    # Normalize class probabilities to [0, 1]
    p_min, p_max = p_cls.min(), p_cls.max()
    if p_max > p_min:
        p_norm = (p_cls - p_min) / (p_max - p_min)
    else:
        p_norm = np.ones_like(p_cls) * 0.5
    
    # Compute cosine similarities if enabled
    cos_sim = np.zeros(T)
    if use_cosine_gate:
        for t in range(T):
            cos_sim[t] = cosine_similarity(embeddings[t], query_embedding)
    
    # Normalize cosine similarities to [0, 1]
    if use_cosine_gate:
        cos_min, cos_max = cos_sim.min(), cos_sim.max()
        if cos_max > cos_min:
            cos_norm = (cos_sim - cos_min) / (cos_max - cos_min)
        else:
            cos_norm = np.ones_like(cos_sim) * 0.5
    else:
        cos_norm = np.zeros(T)
    
    # Combine signals
    w_raw = alpha * p_norm + beta * cos_norm
    
    # Determine threshold
    if cls_threshold is not None:
        threshold = cls_threshold
    elif cls_percentile is not None:
        threshold = np.percentile(w_raw, cls_percentile)
    else:
        threshold = 0.9
    
    # Set hysteresis thresholds
    if hysteresis_enter is None:
        hysteresis_enter = threshold
    if hysteresis_exit is None:
        hysteresis_exit = threshold * 0.8
    
    # Apply hysteresis thresholding
    binary_mask = apply_hysteresis(w_raw, hysteresis_enter, hysteresis_exit)
    
    # Convert to segments
    segments = binary_mask_to_segments(times, binary_mask)
    
    # Apply minimum duration filter
    min_dur_sec = min_dur_ms / 1000.0
    segments = filter_min_duration(segments, min_dur_sec)
    
    # Merge nearby segments
    merge_gap_sec = merge_gap_ms / 1000.0
    segments = merge_segments(segments, merge_gap_sec)
    
    # Apply segment padding (loose-lite feature)
    if segment_padding_ms > 0:
        padding_sec = segment_padding_ms / 1000.0
        segments = apply_segment_padding(segments, padding_sec, max_duration=times[-1])
    
    # Apply time mask sharpening (loose-lite feature)
    if time_mask_sharpening != 1.0:
        w_raw = apply_time_mask_sharpening(w_raw, time_mask_sharpening)
    
    return segments, w_raw


def apply_hysteresis(signal: np.ndarray, enter_thresh: float, exit_thresh: float) -> np.ndarray:
    """
    Apply hysteresis thresholding to signal.
    
    Args:
        signal: Input signal
        enter_thresh: Threshold to enter active state
        exit_thresh: Threshold to exit active state
        
    Returns:
        Binary mask
    """
    binary_mask = np.zeros_like(signal, dtype=bool)
    active = False
    
    for i, val in enumerate(signal):
        if not active and val >= enter_thresh:
            active = True
        elif active and val <= exit_thresh:
            active = False
        
        binary_mask[i] = active
    
    return binary_mask


def binary_mask_to_segments(times: np.ndarray, binary_mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert binary mask to list of time segments.
    
    Args:
        times: Time positions
        binary_mask: Binary activity mask
        
    Returns:
        List of (start_time, end_time) tuples
    """
    segments = []
    
    if len(binary_mask) == 0:
        return segments
    
    # Find transitions
    transitions = np.diff(binary_mask.astype(int))
    starts = np.where(transitions == 1)[0] + 1  # Rising edges
    ends = np.where(transitions == -1)[0] + 1   # Falling edges
    
    # Handle edge cases
    if binary_mask[0]:  # Starts active
        starts = np.concatenate([[0], starts])
    if binary_mask[-1]:  # Ends active
        ends = np.concatenate([ends, [len(binary_mask) - 1]])
    
    # Create segments
    for start_idx, end_idx in zip(starts, ends):
        start_time = times[start_idx]
        end_time = times[min(end_idx, len(times) - 1)]
        segments.append((start_time, end_time))
    
    return segments


def filter_min_duration(segments: List[Tuple[float, float]], min_dur: float) -> List[Tuple[float, float]]:
    """
    Filter segments by minimum duration.
    
    Args:
        segments: List of (start, end) segments
        min_dur: Minimum duration in seconds
        
    Returns:
        Filtered segments
    """
    return [(start, end) for start, end in segments if (end - start) >= min_dur]


def merge_segments(segments: List[Tuple[float, float]], max_gap: float) -> List[Tuple[float, float]]:
    """
    Merge segments that are close together.
    
    Args:
        segments: List of (start, end) segments
        max_gap: Maximum gap to merge in seconds
        
    Returns:
        Merged segments
    """
    if not segments:
        return segments
    
    # Sort by start time
    segments = sorted(segments)
    merged = [segments[0]]
    
    for current_start, current_end in segments[1:]:
        last_start, last_end = merged[-1]
        
        # Check if gap is small enough to merge
        gap = current_start - last_end
        if gap <= max_gap:
            # Merge with previous segment
            merged[-1] = (last_start, current_end)
        else:
            # Add as new segment
            merged.append((current_start, current_end))
    
    return merged


def apply_segment_padding(segments: List[Tuple[float, float]], 
                         padding_sec: float, max_duration: float) -> List[Tuple[float, float]]:
    """
    Apply padding around segments and merge overlapping ones.
    
    Args:
        segments: List of (start, end) segments
        padding_sec: Padding in seconds
        max_duration: Maximum duration to clip to
        
    Returns:
        Padded and merged segments
    """
    if not segments or padding_sec <= 0:
        return segments
    
    # Apply padding
    padded_segments = []
    for start, end in segments:
        padded_start = max(0, start - padding_sec)
        padded_end = min(max_duration, end + padding_sec)
        padded_segments.append((padded_start, padded_end))
    
    # Merge overlapping segments
    if not padded_segments:
        return padded_segments
    
    # Sort by start time
    padded_segments = sorted(padded_segments)
    merged = [padded_segments[0]]
    
    for current_start, current_end in padded_segments[1:]:
        last_start, last_end = merged[-1]
        
        # Check for overlap
        if current_start <= last_end:
            # Merge overlapping segments
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # Add as new segment
            merged.append((current_start, current_end))
    
    return merged


def apply_time_mask_sharpening(w_raw: np.ndarray, power: float) -> np.ndarray:
    """
    Apply power-based sharpening to time mask.
    
    Args:
        w_raw: Raw temporal weights [T]
        power: Sharpening power (< 1.0 makes it softer, > 1.0 makes it sharper)
        
    Returns:
        Sharpened weights [T]
    """
    if power == 1.0:
        return w_raw
    
    # Apply power transformation
    # Normalize to [0, 1] first
    w_min, w_max = w_raw.min(), w_raw.max()
    if w_max > w_min:
        w_norm = (w_raw - w_min) / (w_max - w_min)
    else:
        return w_raw
    
    # Apply power
    w_sharp = np.power(w_norm, power)
    
    # Scale back to original range
    w_result = w_sharp * (w_max - w_min) + w_min
    
    return w_result


def interpolate_to_stft_frames(w_raw: np.ndarray, times: np.ndarray, 
                              T_stft: int, total_duration: float) -> np.ndarray:
    """
    Interpolate temporal weights to STFT frame resolution.
    
    Args:
        w_raw: Raw temporal weights [T_windows]
        times: Time positions for windows [T_windows]  
        T_stft: Number of STFT frames
        total_duration: Total audio duration in seconds
        
    Returns:
        Interpolated weights [T_stft]
    """
    # Create STFT time grid
    stft_times = np.linspace(0, total_duration, T_stft)
    
    # Interpolate weights to STFT resolution
    w_stft = np.interp(stft_times, times, w_raw)
    
    return w_stft
