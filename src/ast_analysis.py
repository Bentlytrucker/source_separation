"""
AST (Audio Spectrogram Transformer) based analysis functions.
"""
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from typing import Tuple, List, Optional, Dict
import warnings


class ASTAnalyzer:
    """AST-based audio analyzer for classification and embedding extraction."""
    
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        """
        Initialize AST analyzer.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = torch.device("cpu")  # CPU-only as specified
        
        try:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
            self.model = ASTForAudioClassification.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)
            
            # Set number of threads for CPU inference
            torch.set_num_threads(2)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load AST model {model_name}: {str(e)}")
    
    def ast_predict_single(self, query_wav: np.ndarray, sr: int = 16000) -> Tuple[int, str, np.ndarray]:
        """
        Perform single prediction on query audio.
        
        Args:
            query_wav: Query audio array
            sr: Sample rate
            
        Returns:
            Tuple of (top1_id, query_label, embedding_vector)
        """
        with torch.no_grad():
            # Prepare inputs
            inputs = self.feature_extractor(
                query_wav, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass with hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get top-1 prediction
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            top1_id = torch.argmax(probs, dim=-1).item()
            
            # Get exact class label from model config
            query_label = self.model.config.id2label[top1_id]
            
            # Extract penultimate layer global embedding
            # AST typically has 12 layers, so we take layer -2 (second to last)
            hidden_states = outputs.hidden_states
            if len(hidden_states) >= 2:
                # Take the [CLS] token (first token) from penultimate layer
                penultimate_layer = hidden_states[-2]  # Shape: [batch, seq_len, hidden_dim]
                embedding = penultimate_layer[:, 0, :].squeeze().cpu().numpy()  # [CLS] token
            else:
                # Fallback to last layer if not enough layers
                embedding = hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
            
            return top1_id, query_label, embedding
    
    def extract_spectrogram_features(self, mix_wav: np.ndarray, sr: int = 16000) -> Dict[str, np.ndarray]:
        """
        Extract AST's internal spectrogram feature maps from mixture audio.
        
        Args:
            mix_wav: Mixture audio array (10 seconds)
            sr: Sample rate
            
        Returns:
            Dictionary containing:
            - 'patch_embeddings': Patch-level embeddings [num_patches, hidden_dim]
            - 'attention_weights': Multi-head attention weights
            - 'layer_features': Features from different transformer layers
        """
        with torch.no_grad():
            # Prepare inputs
            inputs = self.feature_extractor(
                mix_wav, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass with all intermediate outputs
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            
            # Extract different levels of features
            hidden_states = outputs.hidden_states  # List of [batch, seq_len, hidden_dim]
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
            
            features = {}
            
            # Patch embeddings (after initial embedding layer)
            if len(hidden_states) > 0:
                # First layer contains patch embeddings
                patch_embeddings = hidden_states[0].squeeze().cpu().numpy()  # [seq_len, hidden_dim]
                features['patch_embeddings'] = patch_embeddings[1:, :]  # Remove [CLS] token
            
            # Multi-layer features (different semantic levels)
            layer_features = []
            for i, layer_output in enumerate(hidden_states):
                layer_feat = layer_output.squeeze().cpu().numpy()[1:, :]  # Remove [CLS] token
                layer_features.append(layer_feat)
            features['layer_features'] = layer_features
            
            # Attention weights (if available)
            if attentions is not None:
                attention_weights = []
                for attention in attentions:
                    att_weights = attention.squeeze().cpu().numpy()  # [num_heads, seq_len, seq_len]
                    attention_weights.append(att_weights)
                features['attention_weights'] = attention_weights
            
            return features

    def ast_predict_windowed(self, mix_wav: np.ndarray, window_sec: float = 1.0, 
                           hop_sec: float = 0.5, sr: int = 16000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform sliding window analysis on mixture audio.
        
        Args:
            mix_wav: Mixture audio array
            window_sec: Window size in seconds
            hop_sec: Hop size in seconds
            sr: Sample rate
            
        Returns:
            Tuple of (times, probs[T,C], embeddings[T,D])
        """
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)
        
        # Calculate number of windows
        num_windows = max(1, (len(mix_wav) - window_samples) // hop_samples + 1)
        
        times = []
        all_probs = []
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(num_windows):
                start_sample = i * hop_samples
                end_sample = min(start_sample + window_samples, len(mix_wav))
                
                # Extract window
                window = mix_wav[start_sample:end_sample]
                
                # Pad if window is too short
                if len(window) < window_samples:
                    window = np.pad(window, (0, window_samples - len(window)), 
                                  mode='constant', constant_values=0)
                
                # Calculate time position (center of window)
                time_pos = (start_sample + end_sample) / 2 / sr
                times.append(time_pos)
                
                try:
                    # Prepare inputs
                    inputs = self.feature_extractor(
                        window, 
                        sampling_rate=sr, 
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs, output_hidden_states=True)
                    
                    # Get probabilities
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                    all_probs.append(probs)
                    
                    # Get embedding
                    hidden_states = outputs.hidden_states
                    if len(hidden_states) >= 2:
                        penultimate_layer = hidden_states[-2]
                        embedding = penultimate_layer[:, 0, :].squeeze().cpu().numpy()
                    else:
                        embedding = hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
                    all_embeddings.append(embedding)
                    
                except Exception as e:
                    warnings.warn(f"Failed to process window {i}: {str(e)}")
                    # Use zero arrays as fallback
                    num_classes = len(self.model.config.id2label)
                    all_probs.append(np.zeros(num_classes))
                    all_embeddings.append(np.zeros(768))  # Typical AST hidden size
        
        times = np.array(times)
        probs = np.array(all_probs)  # Shape: [T, C]
        embeddings = np.array(all_embeddings)  # Shape: [T, D]
        
        return times, probs, embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: Input vectors
        
    Returns:
        Cosine similarity score
    """
    eps = 1e-8
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a) + eps
    norm_b = np.linalg.norm(b) + eps
    return dot_product / (norm_a * norm_b)
