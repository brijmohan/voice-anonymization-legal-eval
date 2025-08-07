"""
Speaker embedding extraction for voice anonymization evaluation.

This module provides speaker embedding extraction using ECAPA-TDNN architecture
and other speaker recognition models.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import librosa
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeakerEmbeddingExtractor:
    """
    Speaker embedding extractor using ECAPA-TDNN architecture.
    
    This class extracts speaker embeddings (x-vectors) from audio data using
    the ECAPA-TDNN model as described in the paper.
    """
    
    def __init__(self, 
                 model_type: str = "ecapa_tdnn",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the speaker embedding extractor.
        
        Args:
            model_type: Type of speaker embedding model
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        
        # Default configuration
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.frame_length = self.config.get("frame_length", 0.025)
        self.frame_shift = self.config.get("frame_shift", 0.010)
        self.embedding_dim = self.config.get("embedding_dim", 192)
        
        # Initialize model
        self.model = None
        self.is_initialized = False
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the speaker embedding model."""
        logger.info(f"Initializing {self.model_type} speaker embedding model...")
        
        if self.model_type == "ecapa_tdnn":
            self.model = self._initialize_ecapa_tdnn()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.is_initialized = True
        logger.info("Speaker embedding model initialized successfully.")
    
    def _initialize_ecapa_tdnn(self) -> nn.Module:
        """Initialize ECAPA-TDNN model."""
        # Simplified ECAPA-TDNN implementation
        # In practice, you would use a proper implementation from a library
        class SimplifiedECAPA(nn.Module):
            def __init__(self, input_dim=80, embedding_dim=192):
                super().__init__()
                self.input_dim = input_dim
                self.embedding_dim = embedding_dim
                
                # Simplified ECAPA-TDNN architecture
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, embedding_dim)
                )
                
                # Global average pooling layer
                self.pooling = nn.AdaptiveAvgPool1d(1)
                
            def forward(self, x):
                # x shape: (batch_size, time_steps, input_dim)
                batch_size, time_steps, _ = x.shape
                
                # Process through feature extractor
                features = []
                for t in range(time_steps):
                    feat = self.feature_extractor(x[:, t, :])
                    features.append(feat.unsqueeze(1))
                
                # Concatenate features
                features = torch.cat(features, dim=1)  # (batch_size, time_steps, embedding_dim)
                
                # Global average pooling
                features = features.transpose(1, 2)  # (batch_size, embedding_dim, time_steps)
                embeddings = self.pooling(features).squeeze(-1)  # (batch_size, embedding_dim)
                
                # L2 normalization
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings
        
        return SimplifiedECAPA(input_dim=80, embedding_dim=self.embedding_dim)
    
    def extract_embeddings(self, 
                          audio_data: np.ndarray, 
                          conversation_length: int = 1) -> np.ndarray:
        """
        Extract speaker embeddings from audio data.
        
        Args:
            audio_data: Audio data (can be single utterance or multiple utterances)
            conversation_length: Number of utterances to average over
            
        Returns:
            Speaker embeddings
        """
        if not self.is_initialized:
            raise RuntimeError("Model must be initialized before extracting embeddings")
        
        # Preprocess audio data
        features = self._extract_features(audio_data)
        
        # Extract embeddings
        if conversation_length == 1:
            # Single utterance
            embeddings = self._extract_single_embedding(features)
        else:
            # Multiple utterances - average embeddings
            embeddings = self._extract_averaged_embeddings(features, conversation_length)
        
        return embeddings
    
    def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract acoustic features from audio data.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Extracted features
        """
        # Resample if necessary
        if len(audio_data.shape) == 1:
            # Single audio file
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Extract MFCC features
            features = self._extract_mfcc(audio_data)
        else:
            # Multiple audio files
            features = []
            for audio in audio_data:
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                mfcc = self._extract_mfcc(audio)
                features.append(mfcc)
            features = np.array(features)
        
        return features
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Input audio data
            
        Returns:
            MFCC features
        """
        # Extract MFCC features using librosa
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=int(self.frame_shift * self.sample_rate),
            n_fft=int(self.frame_length * self.sample_rate)
        )
        
        # Add delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate features
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        # Transpose to (time_steps, feature_dim)
        features = features.T
        
        return features
    
    def _extract_single_embedding(self, features: np.ndarray) -> np.ndarray:
        """
        Extract embedding from single utterance features.
        
        Args:
            features: MFCC features
            
        Returns:
            Speaker embedding
        """
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(features_tensor)
        
        return embedding.numpy().squeeze()
    
    def _extract_averaged_embeddings(self, 
                                   features: np.ndarray, 
                                   conversation_length: int) -> np.ndarray:
        """
        Extract averaged embeddings from multiple utterances.
        
        Args:
            features: MFCC features for multiple utterances
            conversation_length: Number of utterances to average over
            
        Returns:
            Averaged speaker embedding
        """
        if len(features.shape) == 2:
            # Single utterance features
            return self._extract_single_embedding(features)
        
        # Multiple utterances
        embeddings = []
        
        # Process each utterance
        for i in range(min(conversation_length, features.shape[0])):
            utterance_features = features[i]
            embedding = self._extract_single_embedding(utterance_features)
            embeddings.append(embedding)
        
        # Average embeddings
        averaged_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        averaged_embedding = averaged_embedding / np.linalg.norm(averaged_embedding)
        
        return averaged_embedding
    
    def extract_embeddings_batch(self, 
                               audio_batch: List[np.ndarray], 
                               conversation_length: int = 1) -> np.ndarray:
        """
        Extract embeddings from a batch of audio data.
        
        Args:
            audio_batch: List of audio data
            conversation_length: Number of utterances to average over
            
        Returns:
            Batch of speaker embeddings
        """
        embeddings = []
        
        for audio in audio_batch:
            embedding = self.extract_embeddings(audio, conversation_length)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def load_pretrained_model(self, model_path: str) -> None:
        """
        Load a pretrained model from file.
        
        Args:
            model_path: Path to the pretrained model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading pretrained model from {model_path}")
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info("Pretrained model loaded successfully.")


class SidekitEmbeddingExtractor(SpeakerEmbeddingExtractor):
    """
    Speaker embedding extractor using Sidekit toolkit.
    
    This is an alternative implementation using the Sidekit library as mentioned
    in the paper for x-vector extraction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Sidekit embedding extractor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(model_type="sidekit", config=config)
    
    def _initialize_model(self) -> None:
        """Initialize Sidekit model."""
        try:
            import sidekit
            logger.info("Initializing Sidekit speaker embedding model...")
            
            # Initialize Sidekit extractor
            self.model = sidekit.FeaturesExtractor(
                audio_filename_structure="",
                feature_filename_structure="",
                sampling_frequency=self.sample_rate,
                lower_frequency=20,
                higher_frequency=7600,
                filter_bank="log",
                filter_bank_size=24,
                window_size=0.025,
                shift=0.010,
                ceps_number=20,
                vad="snr",
                snr=40,
                pre_emphasis=0.97,
                save_param=True,
                keep_all_features=True
            )
            
            self.is_initialized = True
            logger.info("Sidekit model initialized successfully.")
            
        except ImportError:
            raise ImportError("Sidekit is not installed. Please install it to use this extractor.")
    
    def extract_embeddings(self, 
                          audio_data: np.ndarray, 
                          conversation_length: int = 1) -> np.ndarray:
        """
        Extract embeddings using Sidekit.
        
        Args:
            audio_data: Audio data
            conversation_length: Number of utterances to average over
            
        Returns:
            Speaker embeddings
        """
        # This would implement actual Sidekit extraction
        # For now, return placeholder
        return np.random.randn(self.embedding_dim)


def create_embedding_extractor(extractor_type: str = "ecapa_tdnn", 
                             config: Optional[Dict[str, Any]] = None) -> SpeakerEmbeddingExtractor:
    """
    Factory function to create embedding extractor instances.
    
    Args:
        extractor_type: Type of extractor ('ecapa_tdnn', 'sidekit')
        config: Configuration dictionary
        
    Returns:
        Embedding extractor instance
    """
    if extractor_type == "ecapa_tdnn":
        return SpeakerEmbeddingExtractor(model_type="ecapa_tdnn", config=config)
    elif extractor_type == "sidekit":
        return SidekitEmbeddingExtractor(config=config)
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


def get_available_extractors() -> Dict[str, str]:
    """
    Get available embedding extractors.
    
    Returns:
        Dictionary mapping extractor names to descriptions
    """
    return {
        "ecapa_tdnn": "ECAPA-TDNN speaker embedding model",
        "sidekit": "Sidekit toolkit x-vector extractor"
    }
