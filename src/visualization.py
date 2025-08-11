"""
Visualization functions for mel spectrograms and masks.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def save_mel_spectrogram(mel_spec: np.ndarray, output_path: str, 
                        title: str = "Mel Spectrogram", 
                        sr: int = 16000, hop_length: int = 256,
                        figsize: Tuple[int, int] = (12, 6)) -> str:
    """
    Save mel spectrogram as image.
    
    Args:
        mel_spec: Mel spectrogram [T, M]
        output_path: Output file path
        title: Plot title
        sr: Sample rate
        hop_length: STFT hop length
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved image
    """
    plt.figure(figsize=figsize)
    
    # Convert to dB scale for better visualization
    mel_db = 20 * np.log10(np.maximum(mel_spec, 1e-8))
    
    # Create time axis
    time_frames = mel_spec.shape[0]
    time_axis = np.linspace(0, time_frames * hop_length / sr, time_frames)
    
    # Plot
    plt.imshow(mel_db.T, aspect='auto', origin='lower', 
               extent=[0, time_axis[-1], 0, mel_spec.shape[1]],
               cmap='viridis')
    
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Bands')
    plt.title(title)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def save_mask_visualization(mask: np.ndarray, output_path: str,
                           title: str = "Separation Mask",
                           sr: int = 16000, hop_length: int = 256, nfft: int = 512,
                           figsize: Tuple[int, int] = (12, 6)) -> str:
    """
    Save time-frequency mask as image.
    
    Args:
        mask: Time-frequency mask [T, F]
        output_path: Output file path
        title: Plot title
        sr: Sample rate
        hop_length: STFT hop length
        nfft: STFT window size
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved image
    """
    plt.figure(figsize=figsize)
    
    # Create axes
    time_frames = mask.shape[0]
    freq_bins = mask.shape[1]
    time_axis = np.linspace(0, time_frames * hop_length / sr, time_frames)
    freq_axis = np.linspace(0, sr / 2, freq_bins)
    
    # Plot mask
    plt.imshow(mask.T, aspect='auto', origin='lower',
               extent=[0, time_axis[-1], 0, freq_axis[-1] / 1000],  # kHz
               cmap='plasma', vmin=0, vmax=1)
    
    plt.colorbar(label='Mask Value')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (kHz)')
    plt.title(title)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def save_comparison_plot(mix_mel: np.ndarray, target_mel: np.ndarray, 
                        mask: np.ndarray, output_path: str,
                        sr: int = 16000, hop_length: int = 256, nfft: int = 512,
                        figsize: Tuple[int, int] = (18, 12)) -> str:
    """
    Save comparison plot with mixture, target, and mask.
    
    Args:
        mix_mel: Mixture mel spectrogram [T, M]
        target_mel: Target mel spectrogram [T, M]
        mask: Time-frequency mask [T, F]
        output_path: Output file path
        sr: Sample rate
        hop_length: STFT hop length
        nfft: STFT window size
        figsize: Figure size (width, height)
        
    Returns:
        Path to saved image
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Time axis for mel spectrograms
    time_frames_mel = mix_mel.shape[0]
    time_axis_mel = np.linspace(0, time_frames_mel * hop_length / sr, time_frames_mel)
    
    # Time axis for mask
    time_frames_mask = mask.shape[0]
    time_axis_mask = np.linspace(0, time_frames_mask * hop_length / sr, time_frames_mask)
    freq_axis = np.linspace(0, sr / 2, mask.shape[1])
    
    # Convert mel spectrograms to dB
    mix_mel_db = 20 * np.log10(np.maximum(mix_mel, 1e-8))
    target_mel_db = 20 * np.log10(np.maximum(target_mel, 1e-8))
    
    # Plot 1: Mixture Mel Spectrogram
    im1 = axes[0, 0].imshow(mix_mel_db.T, aspect='auto', origin='lower',
                           extent=[0, time_axis_mel[-1], 0, mix_mel.shape[1]],
                           cmap='viridis')
    axes[0, 0].set_title('Input Mixture (Mel)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Mel Bands')
    plt.colorbar(im1, ax=axes[0, 0], label='Magnitude (dB)')
    
    # Plot 2: Target Mel Spectrogram
    im2 = axes[0, 1].imshow(target_mel_db.T, aspect='auto', origin='lower',
                           extent=[0, time_axis_mel[-1], 0, target_mel.shape[1]],
                           cmap='viridis')
    axes[0, 1].set_title('Separated Target (Mel)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mel Bands')
    plt.colorbar(im2, ax=axes[0, 1], label='Magnitude (dB)')
    
    # Plot 3: Separation Mask
    im3 = axes[1, 0].imshow(mask.T, aspect='auto', origin='lower',
                           extent=[0, time_axis_mask[-1], 0, freq_axis[-1] / 1000],
                           cmap='plasma', vmin=0, vmax=1)
    axes[1, 0].set_title('Separation Mask')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (kHz)')
    plt.colorbar(im3, ax=axes[1, 0], label='Mask Value')
    
    # Plot 4: Mask Statistics Over Time
    mask_mean_time = np.mean(mask, axis=1)  # Average over frequency
    mask_max_time = np.max(mask, axis=1)    # Max over frequency
    
    axes[1, 1].plot(time_axis_mask, mask_mean_time, label='Mean Mask', linewidth=2)
    axes[1, 1].plot(time_axis_mask, mask_max_time, label='Max Mask', linewidth=2, alpha=0.7)
    axes[1, 1].set_title('Mask Activity Over Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Mask Value')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def save_segment_timeline(segments: list, total_duration: float, 
                         w_stft: np.ndarray, output_path: str,
                         figsize: Tuple[int, int] = (12, 4)) -> str:
    """
    Save segment detection timeline.
    
    Args:
        segments: List of (start, end) segments
        total_duration: Total audio duration
        w_stft: Temporal weights [T]
        output_path: Output file path
        figsize: Figure size
        
    Returns:
        Path to saved image
    """
    plt.figure(figsize=figsize)
    
    # Create time axis
    time_axis = np.linspace(0, total_duration, len(w_stft))
    
    # Plot temporal weights
    plt.plot(time_axis, w_stft, 'b-', linewidth=2, label='Detection Confidence')
    plt.fill_between(time_axis, 0, w_stft, alpha=0.3, color='blue')
    
    # Highlight detected segments
    for i, (start, end) in enumerate(segments):
        plt.axvspan(start, end, alpha=0.3, color='red', 
                   label='Detected Segment' if i == 0 else "")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Detection Weight')
    plt.title('Segment Detection Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, total_duration])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_visualization_summary(mix_mel: np.ndarray, target_mel: np.ndarray,
                               mask: np.ndarray, segments: list, w_stft: np.ndarray,
                               output_dir: str, class_label: str,
                               sr: int = 16000, hop_length: int = 256, 
                               nfft: int = 512) -> Dict[str, str]:
    """
    Create essential visualization files only.
    
    Args:
        mix_mel: Mixture mel spectrogram [T, M]
        target_mel: Target mel spectrogram [T, M]
        mask: Time-frequency mask [T, F]
        segments: List of detected segments
        w_stft: Temporal weights [T]
        output_dir: Output directory
        class_label: Query class label
        sr: Sample rate
        hop_length: STFT hop length
        nfft: STFT window size
        
    Returns:
        Dictionary of saved visualization paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename prefix
    from .io_preprocessing import create_safe_slug
    safe_name = create_safe_slug(class_label)
    
    viz_paths = {}
    
    try:
        # Only essential visualizations
        
        # 1. Comparison plot (shows everything in one image)
        viz_paths['comparison'] = save_comparison_plot(
            mix_mel, target_mel, mask,
            str(output_path / f"{safe_name}_comparison.png"),
            sr, hop_length, nfft
        )
        
        # 2. Segment timeline (shows detection quality)
        total_duration = len(w_stft) * hop_length / sr
        viz_paths['timeline'] = save_segment_timeline(
            segments, total_duration, w_stft,
            str(output_path / f"{safe_name}_timeline.png")
        )
        
        print(f"  Essential visualizations saved:")
        for name, path in viz_paths.items():
            print(f"    {name}: {Path(path).name}")
            
    except Exception as e:
        print(f"  Warning: Visualization failed: {e}")
        # Continue without visualizations
    
    return viz_paths
