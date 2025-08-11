"""
Evaluation metrics for source separation performance.
"""
import numpy as np
import torch
import torchaudio
from typing import Dict, Tuple, Optional
from pathlib import Path


def compute_si_sdr(reference: np.ndarray, estimation: np.ndarray, 
                   epsilon: float = 1e-8) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    Args:
        reference: Reference (ground truth) signal
        estimation: Estimated signal
        epsilon: Small value to avoid numerical issues
        
    Returns:
        SI-SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # Zero-mean signals
    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)
    
    # Compute optimal scaling factor
    alpha = np.dot(estimation, reference) / (np.dot(reference, reference) + epsilon)
    
    # Scaled reference
    scaled_reference = alpha * reference
    
    # Error signal
    error = estimation - scaled_reference
    
    # SI-SDR computation
    signal_power = np.sum(scaled_reference ** 2)
    noise_power = np.sum(error ** 2)
    
    si_sdr = 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))
    
    return float(si_sdr)


def compute_sdr(reference: np.ndarray, estimation: np.ndarray,
                epsilon: float = 1e-8) -> float:
    """
    Compute Signal-to-Distortion Ratio (SDR).
    
    Args:
        reference: Reference signal
        estimation: Estimated signal
        epsilon: Small value to avoid numerical issues
        
    Returns:
        SDR in dB
    """
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # Compute SDR
    signal_power = np.sum(reference ** 2)
    error_power = np.sum((estimation - reference) ** 2)
    
    sdr = 10 * np.log10((signal_power + epsilon) / (error_power + epsilon))
    
    return float(sdr)


def compute_snr(signal: np.ndarray, noise: np.ndarray,
                epsilon: float = 1e-8) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR).
    
    Args:
        signal: Signal component
        noise: Noise component
        epsilon: Small value to avoid numerical issues
        
    Returns:
        SNR in dB
    """
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(noise ** 2)
    
    snr = 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))
    
    return float(snr)


def compute_spectral_metrics(reference: np.ndarray, estimation: np.ndarray,
                           sr: int = 16000, n_fft: int = 512, 
                           hop_length: int = 256) -> Dict[str, float]:
    """
    Compute spectral domain metrics.
    
    Args:
        reference: Reference signal
        estimation: Estimated signal
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        Dictionary of spectral metrics
    """
    # Compute spectrograms
    ref_spec = torch.stft(torch.from_numpy(reference).float(), 
                         n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True)
    est_spec = torch.stft(torch.from_numpy(estimation).float(), 
                         n_fft=n_fft, hop_length=hop_length, 
                         return_complex=True)
    
    # Ensure same shape
    min_time = min(ref_spec.shape[-1], est_spec.shape[-1])
    ref_spec = ref_spec[..., :min_time]
    est_spec = est_spec[..., :min_time]
    
    # Magnitude spectrograms
    ref_mag = torch.abs(ref_spec).numpy()
    est_mag = torch.abs(est_spec).numpy()
    
    # Spectral convergence
    numerator = np.sum((ref_mag - est_mag) ** 2)
    denominator = np.sum(ref_mag ** 2)
    spectral_convergence = np.sqrt(numerator / (denominator + 1e-8))
    
    # Log spectral distance
    ref_log = np.log(ref_mag + 1e-8)
    est_log = np.log(est_mag + 1e-8)
    log_spectral_distance = np.sqrt(np.mean((ref_log - est_log) ** 2))
    
    return {
        'spectral_convergence': float(spectral_convergence),
        'log_spectral_distance': float(log_spectral_distance)
    }


def compute_perceptual_metrics(reference: np.ndarray, estimation: np.ndarray,
                             sr: int = 16000) -> Dict[str, float]:
    """
    Compute perceptual quality metrics.
    
    Args:
        reference: Reference signal
        estimation: Estimated signal
        sr: Sample rate
        
    Returns:
        Dictionary of perceptual metrics
    """
    # Ensure same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]
    
    # RMS energy ratio
    ref_rms = np.sqrt(np.mean(reference ** 2))
    est_rms = np.sqrt(np.mean(estimation ** 2))
    energy_ratio = est_rms / (ref_rms + 1e-8)
    
    # Cross-correlation
    cross_corr = np.corrcoef(reference, estimation)[0, 1]
    if np.isnan(cross_corr):
        cross_corr = 0.0
    
    # Dynamic range
    ref_dynamic_range = 20 * np.log10(np.max(np.abs(reference)) / (ref_rms + 1e-8))
    est_dynamic_range = 20 * np.log10(np.max(np.abs(estimation)) / (est_rms + 1e-8))
    
    return {
        'energy_ratio': float(energy_ratio),
        'cross_correlation': float(cross_corr),
        'ref_dynamic_range_db': float(ref_dynamic_range),
        'est_dynamic_range_db': float(est_dynamic_range)
    }


def comprehensive_evaluation(reference_path: str, estimation_path: str,
                           mixture_path: Optional[str] = None,
                           sr: int = 16000) -> Dict[str, float]:
    """
    Perform comprehensive evaluation of separation results.
    
    Args:
        reference_path: Path to ground truth audio
        estimation_path: Path to separated audio
        mixture_path: Optional path to mixture audio for additional metrics
        sr: Sample rate
        
    Returns:
        Dictionary of all evaluation metrics
    """
    from .io_preprocessing import load_wav_mono16k
    
    # Load audio files
    reference = load_wav_mono16k(reference_path)
    estimation = load_wav_mono16k(estimation_path)
    
    print(f"  Reference: {len(reference)/sr:.2f}s")
    print(f"  Estimation: {len(estimation)/sr:.2f}s")
    
    # Core metrics
    metrics = {}
    
    try:
        metrics['si_sdr_db'] = compute_si_sdr(reference, estimation)
        print(f"  SI-SDR: {metrics['si_sdr_db']:.2f} dB")
    except Exception as e:
        print(f"  SI-SDR computation failed: {e}")
        metrics['si_sdr_db'] = float('nan')
    
    try:
        metrics['sdr_db'] = compute_sdr(reference, estimation)
        print(f"  SDR: {metrics['sdr_db']:.2f} dB")
    except Exception as e:
        print(f"  SDR computation failed: {e}")
        metrics['sdr_db'] = float('nan')
    
    # Spectral metrics
    try:
        spectral_metrics = compute_spectral_metrics(reference, estimation, sr)
        metrics.update(spectral_metrics)
        print(f"  Spectral Convergence: {metrics['spectral_convergence']:.4f}")
        print(f"  Log Spectral Distance: {metrics['log_spectral_distance']:.4f}")
    except Exception as e:
        print(f"  Spectral metrics computation failed: {e}")
        metrics.update({
            'spectral_convergence': float('nan'),
            'log_spectral_distance': float('nan')
        })
    
    # Perceptual metrics
    try:
        perceptual_metrics = compute_perceptual_metrics(reference, estimation, sr)
        metrics.update(perceptual_metrics)
        print(f"  Cross-correlation: {metrics['cross_correlation']:.3f}")
        print(f"  Energy ratio: {metrics['energy_ratio']:.3f}")
    except Exception as e:
        print(f"  Perceptual metrics computation failed: {e}")
        metrics.update({
            'energy_ratio': float('nan'),
            'cross_correlation': float('nan'),
            'ref_dynamic_range_db': float('nan'),
            'est_dynamic_range_db': float('nan')
        })
    
    # Additional metrics if mixture is provided
    if mixture_path:
        try:
            mixture = load_wav_mono16k(mixture_path)
            
            # Compute residual
            min_len = min(len(mixture), len(estimation))
            residual = mixture[:min_len] - estimation[:min_len]
            
            # SNR of separated vs residual
            metrics['separation_snr_db'] = compute_snr(estimation[:min_len], residual)
            print(f"  Separation SNR: {metrics['separation_snr_db']:.2f} dB")
            
        except Exception as e:
            print(f"  Mixture-based metrics computation failed: {e}")
            metrics['separation_snr_db'] = float('nan')
    
    return metrics


def create_evaluation_summary(metrics: Dict[str, float], 
                            method_name: str = "Unknown") -> str:
    """
    Create a formatted summary of evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        method_name: Name of the separation method
        
    Returns:
        Formatted summary string
    """
    summary = f"\n{'='*60}\n"
    summary += f"EVALUATION SUMMARY - {method_name}\n"
    summary += f"{'='*60}\n"
    
    # Core metrics
    summary += f"📊 Core Metrics:\n"
    summary += f"   SI-SDR: {metrics.get('si_sdr_db', 'N/A'):>8.2f} dB\n"
    summary += f"   SDR:    {metrics.get('sdr_db', 'N/A'):>8.2f} dB\n"
    
    if 'separation_snr_db' in metrics:
        summary += f"   Sep-SNR: {metrics['separation_snr_db']:>7.2f} dB\n"
    
    # Quality assessment
    si_sdr = metrics.get('si_sdr_db', float('nan'))
    if not np.isnan(si_sdr):
        if si_sdr > 10:
            quality = "Excellent 🟢"
        elif si_sdr > 5:
            quality = "Good 🟡"
        elif si_sdr > 0:
            quality = "Fair 🟠"
        else:
            quality = "Poor 🔴"
        summary += f"   Quality: {quality}\n"
    
    # Spectral metrics
    summary += f"\n🎵 Spectral Quality:\n"
    summary += f"   Spectral Conv: {metrics.get('spectral_convergence', 'N/A'):>6.4f}\n"
    summary += f"   Log Spec Dist: {metrics.get('log_spectral_distance', 'N/A'):>6.4f}\n"
    
    # Perceptual metrics
    summary += f"\n👂 Perceptual Quality:\n"
    summary += f"   Cross-corr:   {metrics.get('cross_correlation', 'N/A'):>7.3f}\n"
    summary += f"   Energy ratio: {metrics.get('energy_ratio', 'N/A'):>7.3f}\n"
    
    summary += f"\n{'='*60}\n"
    
    return summary
