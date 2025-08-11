#!/usr/bin/env python3
"""
Main CLI script for AST-based source separation with DGMO optimization.

This implements a complete pipeline:
1. I/O preprocessing with length policies
2. AST-based analysis for classification and embedding
3. Temporal segment detection with class and embedding gating
4. "Stitching" approach for reference mel generation
5. Initial mask generation from mel-to-linear mapping
6. DGMO mask optimization with regularization
7. iSTFT reconstruction and output saving
"""
import argparse
import time
import warnings
from pathlib import Path

import torch
import torchaudio
import numpy as np

# Set audio backend
torchaudio.set_audio_backend("soundfile")

# Import our modules
from src.io_preprocessing import (
    load_wav_mono16k, enforce_length_policy, create_safe_slug, save_renamed_query
)
from src.ast_analysis import ASTAnalyzer
from src.segment_detection import build_segments, interpolate_to_stft_frames
from src.mel_processing import (
    compute_mel, assemble_R_mel_timeline, create_mel_filter_bank
)
from src.mask_generation import make_initial_mask, validate_mask_properties
from src.dgmo_optimization import dgmo_optimize, validate_optimization_inputs
from src.reconstruction import (
    reconstruct_and_save, dump_event_json, create_event_metadata,
    compute_separation_metrics, create_visualization_data
)
from src.visualization import create_visualization_summary

from src.evaluation_metrics import comprehensive_evaluation, create_evaluation_summary
from src.embedding_aware_masking import create_embedding_aware_mask, create_adaptive_mask_blending, create_spectrogram_feature_mask


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AST-based source separation with DGMO optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--mix_wav", required=True, type=str,
                       help="Path to 10-second mixture WAV file")
    parser.add_argument("--query_wav", required=True, type=str,
                       help="Path to 1-3 second query WAV file")
    
    # AST analysis parameters
    parser.add_argument("--window_sec", type=float, default=1.0,
                       help="AST window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=0.5,
                       help="AST hop size in seconds")
    
    # Threshold parameters (tight preset for better separation)
    parser.add_argument("--cls_threshold", type=float, default=None,
                       help="Absolute classification threshold")
    parser.add_argument("--cls_percentile", type=int, default=92,
                       help="Percentile classification threshold (tight: 92)")
    parser.add_argument("--use_cosine_gate", type=bool, default=True,
                       help="Use cosine similarity gating")
    parser.add_argument("--cosine_threshold", type=float, default=0.45,
                       help="Cosine similarity threshold (tight: 0.45)")
    
    # Segment processing parameters (tight preset for better separation)
    parser.add_argument("--min_dur_ms", type=int, default=500,
                       help="Minimum segment duration in milliseconds (tight: 500)")
    parser.add_argument("--merge_gap_ms", type=int, default=200,
                       help="Maximum gap to merge segments in milliseconds (tight: 200)")
    parser.add_argument("--segment_padding_ms", type=int, default=50,
                       help="Padding around segments in milliseconds (tight: 50)")
    parser.add_argument("--time_mask_sharpening", type=float, default=1.2,
                       help="Time mask sharpening power (tight: 1.2)")
    
    # STFT parameters
    parser.add_argument("--nfft", type=int, default=512,
                       help="STFT window size")
    parser.add_argument("--hop", type=int, default=256,
                       help="STFT hop size")
    parser.add_argument("--n_mels", type=int, default=64,
                       help="Number of mel bands")
    
    # DGMO optimization parameters (balanced preset for natural separation)
    parser.add_argument("--opt_steps", type=int, default=18,
                       help="Number of DGMO optimization steps (balanced: 18)")
    parser.add_argument("--lr", type=float, default=0.28,
                       help="DGMO learning rate (balanced: 0.28)")
    parser.add_argument("--lambda_tv", type=float, default=0.008,
                       help="Total variation regularization weight (balanced: 0.008)")
    parser.add_argument("--lambda_sat", type=float, default=0.0010,
                       help="Saturation regularization weight (balanced: 0.0010)")
    
    # Mask generation parameters (balanced preset for natural separation)
    parser.add_argument("--gamma_profile", type=float, default=0.25,
                       help="Query frequency profile influence (balanced: 0.25)")
    parser.add_argument("--mask_floor", type=float, default=0.08,
                       help="Minimum mask value to prevent complete cutoff (balanced: 0.08)")
    
    # Other parameters
    parser.add_argument("--epsilon_mel", type=float, default=0.0,
                       help="Epsilon value for mel processing")
    parser.add_argument("--threads", type=int, default=2,
                       help="Number of CPU threads")
    parser.add_argument("--out_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--max_time_limit", type=float, default=10.0,
                       help="Maximum DGMO optimization time in seconds")
    parser.add_argument("--save_trimmed_query", action="store_true",
                       help="Save trimmed query file if query > 3 seconds")
    parser.add_argument("--save_renamed_query", action="store_true", default=True,
                       help="Save query file with classified class name")
    parser.add_argument("--save_visualizations", action="store_true", default=True,
                       help="Save mel spectrograms and mask visualizations")
    parser.add_argument("--gt_wav", type=str, default=None,
                       help="Path to ground truth target audio for evaluation")
    parser.add_argument("--soft_masking", action="store_true", default=True,
                       help="Use softer masking for more natural separation (default: True)")
    parser.add_argument("--use_embedding_masking", action="store_true", default=False,
                       help="Use embedding-aware semantic masking (experimental)")
    parser.add_argument("--semantic_blending", type=float, default=0.7,
                       help="Weight for semantic mask blending (0-1, default: 0.7)")
    parser.add_argument("--similarity_smoothing", type=float, default=1.0,
                       help="Gaussian smoothing for similarity weights (default: 1.0)")
    parser.add_argument("--use_spectrogram_features", action="store_true", default=False,
                       help="Use AST's internal spectrogram features for masking (experimental)")
    
    return parser.parse_args()


def main():
    """Main processing pipeline."""
    args = parse_arguments()
    start_time = time.time()
    
    # Set number of threads
    torch.set_num_threads(args.threads)
    
    print("=" * 60)
    print("AST-based Source Separation with DGMO")
    print("=" * 60)
    
    try:
        # Step 1: Load and preprocess audio
        print("Step 1: Loading and preprocessing audio...")
        
        mix_wav = load_wav_mono16k(args.mix_wav)
        query_wav = load_wav_mono16k(args.query_wav)
        
        # Enforce length policies
        mix_wav, _ = enforce_length_policy(mix_wav, 10.0, policy="mix")
        query_wav, trimmed_query_path = enforce_length_policy(
            query_wav, 3.0, policy="query", 
            save_trimmed=args.save_trimmed_query,
            original_path=args.query_wav,
            output_dir=args.out_dir
        )
        
        print(f"  Mix audio: {len(mix_wav)/16000:.2f}s")
        print(f"  Query audio: {len(query_wav)/16000:.2f}s")
        
        # Step 2: AST analysis
        print("Step 2: AST analysis...")
        
        analyzer = ASTAnalyzer()
        
        # Query analysis
        query_class_idx, query_label, query_embedding = analyzer.ast_predict_single(query_wav)
        print(f"  Query class: '{query_label}' (ID: {query_class_idx})")
        
        # Save query with class name
        renamed_query_path = None
        if args.save_renamed_query:
            suffix = "_trimmed_3s" if trimmed_query_path else ""
            renamed_query_path = save_renamed_query(
                query_wav, args.query_wav, args.out_dir, query_label, suffix=suffix
            )
            print(f"  Query saved as: {Path(renamed_query_path).name}")
        
        # Mixture analysis (sliding window)
        times, probs, embeddings = analyzer.ast_predict_windowed(
            mix_wav, args.window_sec, args.hop_sec
        )
        print(f"  Analyzed {len(times)} windows")
        
        # Step 3: Segment detection
        print("Step 3: Temporal segment detection...")
        
        segments, w_raw = build_segments(
            times, probs, embeddings, query_class_idx, query_embedding,
            cls_threshold=args.cls_threshold,
            cls_percentile=args.cls_percentile,
            use_cosine_gate=args.use_cosine_gate,
            cosine_threshold=args.cosine_threshold,
            min_dur_ms=args.min_dur_ms,
            merge_gap_ms=args.merge_gap_ms,
            segment_padding_ms=args.segment_padding_ms,
            time_mask_sharpening=args.time_mask_sharpening
        )
        
        print(f"  Detected {len(segments)} segments")
        for i, (start, end) in enumerate(segments):
            print(f"    Segment {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        
        # Check if any segments were found
        present = len(segments) > 0
        
        if not present:
            print("  No segments detected - saving metadata only")
            
            # Create minimal metadata
            metadata = create_event_metadata(
                present=False,
                segments=[],
                query_label=query_label,
                params=vars(args),
                processing_time=time.time() - start_time
            )
            
            # Save metadata
            json_path = dump_event_json(metadata, args.out_dir)
            print(f"  Saved metadata to: {json_path}")
            return
        
        # Step 4: Compute spectrograms
        print("Step 4: Computing spectrograms...")
        
        # Compute STFT and mel spectrograms
        X_complex, Mix_mel = compute_mel(mix_wav, 16000, args.nfft, args.hop, args.n_mels)
        Q_complex, Q_mel = compute_mel(query_wav, 16000, args.nfft, args.hop, args.n_mels)
        
        T_stft, F_stft = X_complex.shape
        print(f"  STFT shape: {T_stft} x {F_stft}")
        print(f"  Mel shape: {Mix_mel.shape[0]} x {Mix_mel.shape[1]}")
        
        # Create mel filter bank (needed for both approaches)
        mel_fb = create_mel_filter_bank(16000, args.nfft, args.n_mels)
        X_mag = np.abs(X_complex)
        
        # Traditional "Stitching" approach
        print("Step 5: Assembling reference mel timeline...")
        
        R_mel_timeline = assemble_R_mel_timeline(
            Q_mel, segments, T_stft, args.n_mels, 16000, args.hop
        )
        
        active_frames = np.sum(np.sum(R_mel_timeline, axis=1) > 0)
        print(f"  Reference mel timeline: {active_frames}/{T_stft} active frames")
        
        # Step 6: Generate initial mask
        print("Step 6: Generating initial mask...")
        
        # Interpolate temporal weights to STFT resolution
        w_stft = interpolate_to_stft_frames(w_raw, times, T_stft, 10.0)
        
        # Generate initial mask
        if args.use_spectrogram_features:
            print("  Using AST spectrogram feature masking...")
            
            # Extract AST's internal spectrogram features from mixture
            ast_features = analyzer.extract_spectrogram_features(mix_wav)
            
            # Create feature-aware mask
            feature_mask = create_spectrogram_feature_mask(
                Mix_mel, ast_features, query_embedding, query_label,
                w_stft, mel_fb, mask_floor=args.mask_floor
            )
            
            # Create traditional mask for blending
            traditional_mask = make_initial_mask(
                Mix_mel, R_mel_timeline, mel_fb, w_stft,
                gamma_profile=args.gamma_profile,
                mask_floor=args.mask_floor,
                soft_masking=args.soft_masking
            )
            
            # Blend traditional and feature masks
            M0 = create_adaptive_mask_blending(
                traditional_mask, feature_mask, query_label,
                blending_weight=args.semantic_blending
            )
            
        elif args.use_embedding_masking:
            print("  Using embedding-aware semantic masking...")
            
            # Create semantic mask using embeddings
            semantic_mask = create_embedding_aware_mask(
                Mix_mel, embeddings, query_embedding, query_label,
                w_stft, mel_fb, times, 10.0,
                mask_floor=args.mask_floor,
                similarity_smoothing=args.similarity_smoothing,
                class_aware_blending=True
            )
            
            # Create traditional mask for blending
            traditional_mask = make_initial_mask(
                Mix_mel, R_mel_timeline, mel_fb, w_stft,
                gamma_profile=args.gamma_profile,
                mask_floor=args.mask_floor,
                soft_masking=args.soft_masking
            )
            
            # Blend traditional and semantic masks
            M0 = create_adaptive_mask_blending(
                traditional_mask, semantic_mask, query_label,
                blending_weight=args.semantic_blending
            )
            
        else:
            # Traditional masking approach
            M0 = make_initial_mask(
                Mix_mel, R_mel_timeline, mel_fb, w_stft,
                gamma_profile=args.gamma_profile,
                mask_floor=args.mask_floor,
                soft_masking=args.soft_masking
            )
        
        # Validate mask
        mask_stats = validate_mask_properties(M0)
        print(f"  Initial mask stats: mean={mask_stats['mean']:.3f}, "
              f"activity={mask_stats['activity']:.3f}")
        
        # Step 7: DGMO optimization
        print("Step 7: DGMO mask optimization...")
        
        # Validate inputs
        validate_optimization_inputs(X_mag, M0, R_mel_timeline, mel_fb)
        
        # Optimize mask
        M_optimized = dgmo_optimize(
            X_mag, M0, R_mel_timeline, mel_fb,
            steps=args.opt_steps,
            lr=args.lr,
            lambda_tv=args.lambda_tv,
            lambda_sat=args.lambda_sat,
            max_time_limit=args.max_time_limit
        )
        
        # Final mask stats
        final_mask_stats = validate_mask_properties(M_optimized)
        print(f"  Final mask stats: mean={final_mask_stats['mean']:.3f}, "
              f"activity={final_mask_stats['activity']:.3f}")
        
        # Step 8: Reconstruction and output
        print("Step 8: Audio reconstruction...")
        
        # Reconstruct and save audio
        audio_paths = reconstruct_and_save(X_complex, M_optimized, args.out_dir)
        
        # Compute separation metrics
        target_audio = load_wav_mono16k(audio_paths["target"])
        residual_audio = load_wav_mono16k(audio_paths["residual"])
        metrics = compute_separation_metrics(target_audio, residual_audio, mix_wav)
        
        print(f"  Target saved to: {audio_paths['target']}")
        print(f"  Residual saved to: {audio_paths['residual']}")
        print(f"  Signal-to-residual ratio: {metrics['signal_to_residual_ratio_db']:.2f} dB")
        
        # Step 8.1: Ground Truth Evaluation (if provided)
        evaluation_metrics = {}
        if args.gt_wav:
            print("Step 8.1: Ground Truth Evaluation...")
            try:
                method_name = "Traditional Stitching"
                evaluation_metrics = comprehensive_evaluation(
                    args.gt_wav, audio_paths["target"], args.mix_wav
                )
                
                # Print evaluation summary
                eval_summary = create_evaluation_summary(evaluation_metrics, method_name)
                print(eval_summary)
                
            except Exception as e:
                print(f"  Evaluation failed: {str(e)}")
                evaluation_metrics = {}
        
        # Step 8.5: Create visualizations
        viz_paths = {}
        if args.save_visualizations:
            print("Step 8.5: Creating visualizations...")
            
            # Compute target mel spectrogram for visualization
            _, Target_mel = compute_mel(target_audio, 16000, args.nfft, args.hop, args.n_mels)
            
            # Create all visualizations
            viz_paths = create_visualization_summary(
                Mix_mel, Target_mel, M_optimized, segments, w_stft,
                args.out_dir, query_label, 16000, args.hop, args.nfft
            )
        
        # Step 9: Save comprehensive metadata
        print("Step 9: Saving metadata...")
        
        # Create visualization data
        viz_data = create_visualization_data(M_optimized, segments, times, w_stft)
        
        # Create comprehensive metadata
        metadata = create_event_metadata(
            present=present,
            segments=segments,
            query_label=query_label,
            params=vars(args),
            processing_time=time.time() - start_time,
            mask_stats=final_mask_stats,
            audio_paths=audio_paths
        )
        
        # Add additional information
        metadata["separation_metrics"] = metrics
        metadata["visualization_data"] = viz_data
        metadata["query_class_slug"] = create_safe_slug(query_label)
        
        # Add evaluation metrics if available
        if evaluation_metrics:
            metadata["evaluation_metrics"] = evaluation_metrics
            metadata["gt_wav_path"] = args.gt_wav
        
        # Add file paths
        if trimmed_query_path:
            metadata["trimmed_query_path"] = trimmed_query_path
        if renamed_query_path:
            metadata["renamed_query_path"] = renamed_query_path
        if viz_paths:
            metadata["visualization_paths"] = viz_paths
        
        # Save metadata
        json_path = dump_event_json(metadata, args.out_dir)
        print(f"  Metadata saved to: {json_path}")
        
        # Summary
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Query class: '{query_label}'")
        print(f"Segments detected: {len(segments)}")
        print(f"Total active duration: {metadata['total_detected_duration']:.2f}s")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
