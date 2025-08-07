"""
Main evaluator for the legally validated voice anonymization framework.

This module provides the main LegalEvaluator class that orchestrates the evaluation
of voice anonymization systems using the legally validated metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
from tqdm import tqdm

from .metrics import SinglingOutMetric, LinkabilityMetric, compute_chance_levels
from .attack_models import create_attacker, get_attack_scenarios
from .anonymization import BaseAnonymizer
from .speaker_embedding import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)


class LegalEvaluator:
    """
    Main evaluator for legally validated voice anonymization metrics.
    
    This class orchestrates the evaluation of voice anonymization systems using
    the Singling Out and Linkability metrics across different attack scenarios.
    """
    
    def __init__(self,
                 anonymization_system: str = "baseline_b1",
                 speaker_embedding_model: str = "ecapa_tdnn",
                 config_path: Optional[str] = None):
        """
        Initialize the legal evaluator.
        
        Args:
            anonymization_system: Type of anonymization system to evaluate
            speaker_embedding_model: Type of speaker embedding model to use
            config_path: Path to configuration file
        """
        self.anonymization_system = anonymization_system
        self.speaker_embedding_model = speaker_embedding_model
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.anonymizer = self._initialize_anonymizer()
        self.embedding_extractor = SpeakerEmbeddingExtractor(
            model_type=speaker_embedding_model,
            config=self.config.get("speaker_embedding", {})
        )
        
        # Initialize metrics
        self.singling_out_metric = SinglingOutMetric()
        self.linkability_metric = LinkabilityMetric()
        
        # Initialize attackers
        self.attackers = {}
        self._initialize_attackers()
        
        logger.info(f"LegalEvaluator initialized with {anonymization_system} anonymization system")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "evaluation": {
                "conversation_lengths": [1, 3, 30],
                "speaker_counts": [20, 100, 1000, 10000],
                "num_runs": 5,
                "num_folds": 10
            },
            "anonymization": {
                "system": "baseline_b1",
                "parameters": {
                    "candidate_count": 100,
                    "random_seed": 42
                }
            },
            "speaker_embedding": {
                "model": "ecapa_tdnn",
                "extractor_path": "models/ecapa_tdnn.pth",
                "sample_rate": 16000,
                "frame_length": 0.025,
                "frame_shift": 0.010
            },
            "attack_models": ["ignorant", "semi_informed", "informed"]
        }
    
    def _initialize_anonymizer(self) -> BaseAnonymizer:
        """Initialize the anonymization system."""
        # This would be implemented based on the specific anonymization system
        # For now, return a placeholder
        from .anonymization import BaselineB1Anonymizer
        return BaselineB1Anonymizer()
    
    def _initialize_attackers(self) -> None:
        """Initialize all attacker models."""
        attack_types = self.config.get("attack_models", ["ignorant", "semi_informed", "informed"])
        
        for attack_type in attack_types:
            self.attackers[attack_type] = create_attacker(
                attack_type=attack_type,
                embedding_model=self.speaker_embedding_model
            )
        
        logger.info(f"Initialized {len(self.attackers)} attacker models")
    
    def evaluate(self,
                 test_data_path: str,
                 enrollment_data_path: str,
                 training_data_path: Optional[str] = None,
                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete evaluation of the anonymization system.
        
        Args:
            test_data_path: Path to test data
            enrollment_data_path: Path to enrollment data
            training_data_path: Path to training data (for attacker training)
            output_path: Path to save results
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting legal evaluation...")
        
        # Load and preprocess data
        test_data = self._load_data(test_data_path)
        enrollment_data = self._load_data(enrollment_data_path)
        
        if training_data_path:
            training_data = self._load_data(training_data_path)
        else:
            training_data = None
        
        # Train attackers if training data is provided
        if training_data:
            self._train_attackers(training_data)
        
        # Run evaluation
        results = self._run_evaluation(test_data, enrollment_data)
        
        # Save results
        if output_path:
            self._save_results(results, output_path)
        
        logger.info("Legal evaluation completed.")
        return results
    
    def _load_data(self, data_path: str) -> Dict[str, np.ndarray]:
        """Load audio data from path."""
        # This would implement actual data loading
        # For now, return placeholder data
        logger.info(f"Loading data from {data_path}")
        return {"placeholder": np.random.randn(100, 16000)}  # Placeholder
    
    def _train_attackers(self, training_data: Dict[str, np.ndarray]) -> None:
        """Train all attacker models."""
        logger.info("Training attacker models...")
        
        for attack_type, attacker in self.attackers.items():
            logger.info(f"Training {attack_type} attacker...")
            attacker.train(training_data)
        
        logger.info("All attackers trained successfully.")
    
    def _run_evaluation(self,
                       test_data: Dict[str, np.ndarray],
                       enrollment_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        conversation_lengths = self.config["evaluation"]["conversation_lengths"]
        speaker_counts = self.config["evaluation"]["speaker_counts"]
        
        results = {
            "singling_out": {},
            "linkability": {},
            "eer": {},
            "chance_levels": compute_chance_levels(speaker_counts)
        }
        
        # Evaluate for each conversation length
        for L in conversation_lengths:
            logger.info(f"Evaluating with conversation length L={L}")
            
            # Anonymize test data
            anonymized_test_data = self._anonymize_data(test_data)
            
            # Extract embeddings
            test_embeddings = self._extract_embeddings(anonymized_test_data, L)
            enrollment_embeddings = self._extract_embeddings(enrollment_data, L)
            
            # Evaluate for each attack type
            for attack_type, attacker in self.attackers.items():
                logger.info(f"Evaluating {attack_type} attacker...")
                
                # Extract embeddings using attacker's model
                attacker_test_embeddings = attacker.extract_embeddings(
                    self._prepare_audio_for_attacker(anonymized_test_data, L)
                )
                attacker_enrollment_embeddings = attacker.extract_embeddings(
                    self._prepare_audio_for_attacker(enrollment_data, L)
                )
                
                # Compute Singling Out metric
                singling_out_scores = self._compute_singling_out(
                    attacker_test_embeddings, attacker_enrollment_embeddings, speaker_counts
                )
                
                # Compute Linkability metric
                linkability_scores = self._compute_linkability(
                    attacker_test_embeddings, attacker_enrollment_embeddings, speaker_counts
                )
                
                # Compute EER (simplified)
                eer_scores = self._compute_eer(
                    attacker_test_embeddings, attacker_enrollment_embeddings, speaker_counts
                )
                
                # Store results
                if attack_type not in results["singling_out"]:
                    results["singling_out"][attack_type] = {}
                if attack_type not in results["linkability"]:
                    results["linkability"][attack_type] = {}
                if attack_type not in results["eer"]:
                    results["eer"][attack_type] = {}
                
                results["singling_out"][attack_type][L] = singling_out_scores
                results["linkability"][attack_type][L] = linkability_scores
                results["eer"][attack_type][L] = eer_scores
        
        return results
    
    def _anonymize_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Anonymize audio data."""
        logger.info("Anonymizing data...")
        anonymized_data = {}
        
        for speaker_id, audio_data in tqdm(data.items(), desc="Anonymizing"):
            anonymized_audio = self.anonymizer.anonymize(audio_data)
            anonymized_data[speaker_id] = anonymized_audio
        
        return anonymized_data
    
    def _extract_embeddings(self, data: Dict[str, np.ndarray], conversation_length: int) -> np.ndarray:
        """Extract speaker embeddings from audio data."""
        logger.info(f"Extracting embeddings with conversation length {conversation_length}...")
        
        embeddings = []
        for speaker_id, audio_data in tqdm(data.items(), desc="Extracting embeddings"):
            # Extract embeddings for this speaker
            speaker_embeddings = self.embedding_extractor.extract_embeddings(
                audio_data, conversation_length
            )
            embeddings.append(speaker_embeddings)
        
        return np.array(embeddings)
    
    def _prepare_audio_for_attacker(self, data: Dict[str, np.ndarray], conversation_length: int) -> np.ndarray:
        """Prepare audio data for attacker processing."""
        # This would implement proper audio preparation
        # For now, return placeholder
        return np.random.randn(len(data), conversation_length, 16000)
    
    def _compute_singling_out(self,
                             test_embeddings: np.ndarray,
                             enrollment_embeddings: np.ndarray,
                             speaker_counts: List[int]) -> Dict[int, float]:
        """Compute Singling Out metric for different speaker counts."""
        scores = {}
        
        for N in speaker_counts:
            if N <= enrollment_embeddings.shape[0]:
                # Randomly select N enrollment speakers
                indices = np.random.choice(enrollment_embeddings.shape[0], size=N, replace=False)
                selected_enrollment = enrollment_embeddings[indices]
                
                # Create calibration embeddings (simplified)
                calibration_embeddings = test_embeddings  # Placeholder
                
                score = self.singling_out_metric.compute(
                    test_embeddings=test_embeddings,
                    enrollment_embeddings=selected_enrollment,
                    calibration_embeddings=calibration_embeddings
                )
                scores[N] = score
        
        return scores
    
    def _compute_linkability(self,
                            test_embeddings: np.ndarray,
                            enrollment_embeddings: np.ndarray,
                            speaker_counts: List[int]) -> Dict[int, float]:
        """Compute Linkability metric for different speaker counts."""
        return self.linkability_metric.compute_with_speaker_counts(
            test_embeddings, enrollment_embeddings, speaker_counts
        )
    
    def _compute_eer(self,
                    test_embeddings: np.ndarray,
                    enrollment_embeddings: np.ndarray,
                    speaker_counts: List[int]) -> Dict[int, float]:
        """Compute EER metric for different speaker counts."""
        # Simplified EER computation
        # In practice, implement proper EER calculation
        scores = {}
        
        for N in speaker_counts:
            if N <= enrollment_embeddings.shape[0]:
                # Placeholder EER computation
                eer = 0.1 + np.random.normal(0, 0.02)  # Placeholder
                scores[N] = 1 - eer  # Return 1-EER for consistency
        
        return scores
    
    def _save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        summary = {
            "anonymization_system": self.anonymization_system,
            "speaker_embedding_model": self.speaker_embedding_model,
            "attack_scenarios": list(self.attackers.keys()),
            "conversation_lengths": self.config["evaluation"]["conversation_lengths"],
            "speaker_counts": self.config["evaluation"]["speaker_counts"],
            "metrics": ["singling_out", "linkability", "eer"]
        }
        
        return summary


def run_evaluation(test_data: str,
                  enrollment_data: str,
                  training_data: Optional[str] = None,
                  anonymization_system: str = "baseline_b1",
                  config_path: Optional[str] = None,
                  output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run complete evaluation.
    
    Args:
        test_data: Path to test data
        enrollment_data: Path to enrollment data
        training_data: Path to training data (optional)
        anonymization_system: Type of anonymization system
        config_path: Path to configuration file
        output_path: Path to save results
        
    Returns:
        Evaluation results
    """
    evaluator = LegalEvaluator(
        anonymization_system=anonymization_system,
        config_path=config_path
    )
    
    return evaluator.evaluate(
        test_data_path=test_data,
        enrollment_data_path=enrollment_data,
        training_data_path=training_data,
        output_path=output_path
    )
