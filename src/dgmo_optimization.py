"""
DGMO (Differentiable Gaussian Mixture Optimization) for mask refinement.
"""
import numpy as np
import torch
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


def dgmo_optimize(
    X_mag: np.ndarray,
    M0: np.ndarray, 
    R_mel_timeline: np.ndarray,
    mel_fb: np.ndarray,
    steps: int = 20,
    lr: float = 0.3,
    lambda_tv: float = 0.01,
    lambda_sat: float = 0.001,
    gaussian_kernel_size: int = 3,
    gaussian_sigma: float = 1.0,
    epsilon: float = 1e-6,
    max_time_limit: Optional[float] = None
) -> np.ndarray:
    """
    Optimize mask using DGMO with total variation and saturation regularization.
    
    Args:
        X_mag: Mixture magnitude spectrogram [T, F]
        M0: Initial mask [T, F]
        R_mel_timeline: Reference mel timeline [T, M]
        mel_fb: Mel filter bank [M, F]
        steps: Number of optimization steps
        lr: Learning rate
        lambda_tv: Total variation regularization weight
        lambda_sat: Saturation regularization weight
        gaussian_kernel_size: Size of Gaussian smoothing kernel
        gaussian_sigma: Standard deviation for Gaussian smoothing
        epsilon: Numerical stability constant
        max_time_limit: Maximum time limit in seconds (None = no limit)
        
    Returns:
        Optimized mask [T, F]
    """
    import time
    start_time = time.time() if max_time_limit else None
    
    # Convert to torch tensors
    device = torch.device("cpu")
    X_mag_torch = torch.from_numpy(X_mag).float().to(device)
    R_mel_torch = torch.from_numpy(R_mel_timeline).float().to(device)
    mel_fb_torch = torch.from_numpy(mel_fb).float().to(device)
    
    # Initialize mask as learnable parameter
    M = torch.from_numpy(M0.copy()).float().to(device)
    M.requires_grad_(True)
    
    # Optimizer
    optimizer = torch.optim.Adam([M], lr=lr)
    
    for step in range(steps):
        # Check time limit
        if max_time_limit and start_time:
            if time.time() - start_time > max_time_limit:
                print(f"DGMO optimization stopped early due to time limit at step {step}")
                break
        
        optimizer.zero_grad()
        
        # Ensure mask is in [0, 1]
        M_clamped = torch.clamp(M, 0.0, 1.0)
        
        # Apply mask to get separated magnitude spectrogram
        Y_mag = X_mag_torch * M_clamped
        
        # Convert to mel domain for comparison
        Y_mel = linear_to_mel(Y_mag, mel_fb_torch)
        
        # Reconstruction loss with frame-wise L2 normalization
        recon_loss = compute_normalized_mel_loss(Y_mel, R_mel_torch, epsilon)
        
        # Total variation regularization
        tv_loss = compute_total_variation_2d(M_clamped)
        
        # Saturation regularization (encourages values to be 0 or 1)
        sat_loss = torch.mean(M_clamped * (1 - M_clamped))
        
        # Combined loss
        total_loss = recon_loss + lambda_tv * tv_loss + lambda_sat * sat_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Apply Gaussian smoothing after each step
        with torch.no_grad():
            M_np = M.detach().cpu().numpy()
            M_smoothed = gaussian_filter(
                M_np, 
                sigma=gaussian_sigma,
                mode='reflect'
            )
            M.data = torch.from_numpy(M_smoothed).float().to(device)
            
            # Clamp again after smoothing
            M.data = torch.clamp(M.data, 0.0, 1.0)
        
        # Optional: print progress
        if step % 5 == 0 or step == steps - 1:
            print(f"Step {step:2d}: Loss={total_loss.item():.4f} "
                  f"(recon={recon_loss.item():.4f}, "
                  f"tv={tv_loss.item():.4f}, "
                  f"sat={sat_loss.item():.4f})")
    
    # Return final mask
    final_mask = torch.clamp(M, 0.0, 1.0).detach().cpu().numpy()
    return final_mask


def linear_to_mel(linear_spec: torch.Tensor, mel_fb: torch.Tensor) -> torch.Tensor:
    """
    Convert linear spectrogram to mel spectrogram.
    
    Args:
        linear_spec: Linear magnitude spectrogram [T, F]
        mel_fb: Mel filter bank [M, F]
        
    Returns:
        Mel spectrogram [T, M]
    """
    # linear_spec: [T, F], mel_fb: [M, F]
    # Result: [T, M]
    mel_spec = torch.matmul(linear_spec, mel_fb.t())
    return mel_spec


def compute_normalized_mel_loss(Y_mel: torch.Tensor, R_mel: torch.Tensor, 
                               epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute reconstruction loss with frame-wise L2 normalization.
    
    Args:
        Y_mel: Predicted mel spectrogram [T, M]
        R_mel: Reference mel spectrogram [T, M]
        epsilon: Numerical stability constant
        
    Returns:
        Normalized reconstruction loss
    """
    T, M = Y_mel.shape
    
    total_loss = 0.0
    active_frames = 0
    
    for t in range(T):
        # Get frame vectors
        y_frame = Y_mel[t, :]  # [M]
        r_frame = R_mel[t, :]  # [M]
        
        # Skip frames where reference is nearly zero
        r_energy = torch.sum(r_frame ** 2)
        if r_energy < epsilon:
            continue
        
        # L2 normalize both frames
        y_norm = y_frame / (torch.norm(y_frame) + epsilon)
        r_norm = r_frame / (torch.norm(r_frame) + epsilon)
        
        # Compute frame loss
        frame_loss = torch.mean((y_norm - r_norm) ** 2)
        total_loss += frame_loss
        active_frames += 1
    
    # Average over active frames
    if active_frames > 0:
        total_loss = total_loss / active_frames
    
    return total_loss


def compute_total_variation_2d(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D total variation for mask smoothness.
    
    Args:
        mask: Input mask [T, F]
        
    Returns:
        Total variation loss
    """
    # Time differences
    diff_time = torch.abs(mask[1:, :] - mask[:-1, :])
    tv_time = torch.mean(diff_time)
    
    # Frequency differences  
    diff_freq = torch.abs(mask[:, 1:] - mask[:, :-1])
    tv_freq = torch.mean(diff_freq)
    
    # Combined TV loss
    tv_loss = tv_time + tv_freq
    
    return tv_loss


def apply_gaussian_smoothing_2d(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply 2D Gaussian smoothing to mask.
    
    Args:
        mask: Input mask [T, F]
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed mask [T, F]
    """
    if sigma <= 0:
        return mask
    
    smoothed = gaussian_filter(mask, sigma=sigma, mode='reflect')
    return np.clip(smoothed, 0.0, 1.0)


def validate_optimization_inputs(X_mag: np.ndarray, M0: np.ndarray, 
                               R_mel_timeline: np.ndarray, mel_fb: np.ndarray) -> bool:
    """
    Validate inputs for DGMO optimization.
    
    Args:
        X_mag: Mixture magnitude spectrogram [T, F]
        M0: Initial mask [T, F]
        R_mel_timeline: Reference mel timeline [T, M]
        mel_fb: Mel filter bank [M, F]
        
    Returns:
        True if inputs are valid
        
    Raises:
        ValueError: If inputs are invalid
    """
    T_x, F_x = X_mag.shape
    T_m, F_m = M0.shape
    T_r, M_r = R_mel_timeline.shape
    M_fb, F_fb = mel_fb.shape
    
    # Check shapes
    if T_x != T_m or T_x != T_r:
        raise ValueError(f"Time dimension mismatch: X_mag={T_x}, M0={T_m}, R_mel={T_r}")
    
    if F_x != F_m or F_x != F_fb:
        raise ValueError(f"Frequency dimension mismatch: X_mag={F_x}, M0={F_m}, mel_fb={F_fb}")
    
    if M_r != M_fb:
        raise ValueError(f"Mel dimension mismatch: R_mel={M_r}, mel_fb={M_fb}")
    
    # Check value ranges
    if not np.all(M0 >= 0) or not np.all(M0 <= 1):
        raise ValueError("Initial mask M0 must be in range [0, 1]")
    
    if not np.all(X_mag >= 0):
        raise ValueError("Magnitude spectrogram X_mag must be non-negative")
    
    if not np.all(R_mel_timeline >= 0):
        raise ValueError("Reference mel R_mel_timeline must be non-negative")
    
    return True
