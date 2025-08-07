"""
Legally validated evaluation metrics for voice anonymization.

This module implements the Singling Out and Linkability metrics based on the
Article 29 Working Party's Opinion 05/2014 on Anonymization Techniques.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """Base class for legally validated privacy metrics."""
    
    def __init__(self, similarity_function: str = "cosine"):
        """
        Initialize the base metric.
        
        Args:
            similarity_function: Type of similarity function to use ('cosine', 'euclidean')
        """
        self.similarity_function = similarity_function
        
    def compute_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            x1: First embedding vector
            x2: Second embedding vector
            
        Returns:
            Similarity score
        """
        if self.similarity_function == "cosine":
            return self._cosine_similarity(x1, x2)
        elif self.similarity_function == "euclidean":
            return self._euclidean_similarity(x1, x2)
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
    
    def _cosine_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return dot_product / (norm_x1 * norm_x2)
    
    def _euclidean_similarity(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute negative euclidean distance as similarity."""
        return -np.linalg.norm(x1 - x2)
    
    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute the metric value."""
        pass


class SinglingOutMetric(BaseMetric):
    """
    Singling Out metric based on Predicate Singling Out (PSO) framework.
    
    This metric quantifies the risk that an attacker can isolate an individual
    from an anonymized dataset using the PSO framework.
    """
    
    def __init__(self, similarity_function: str = "cosine"):
        """
        Initialize the Singling Out metric.
        
        Args:
            similarity_function: Type of similarity function to use
        """
        super().__init__(similarity_function)
        self.baseline_probability = np.exp(-1)  # ≈ 37%
        
    def compute(self, 
                test_embeddings: np.ndarray,
                enrollment_embeddings: np.ndarray,
                calibration_embeddings: np.ndarray,
                num_runs: int = 5,
                num_folds: int = 10) -> float:
        """
        Compute the Singling Out metric.
        
        Args:
            test_embeddings: Test speaker embeddings [N_test, embedding_dim]
            enrollment_embeddings: Enrollment speaker embeddings [N_enroll, embedding_dim]
            calibration_embeddings: Calibration embeddings [M*N_test, embedding_dim]
            num_runs: Number of random runs for robustness
            num_folds: Number of cross-validation folds
            
        Returns:
            Singling Out probability π^sing
        """
        N_test = test_embeddings.shape[0]
        N_enroll = enrollment_embeddings.shape[0]
        
        isolation_probabilities = []
        
        for run in range(num_runs):
            # Randomly select enrollment speakers
            enroll_indices = np.random.choice(N_enroll, size=min(495, N_enroll), replace=False)
            selected_enrollment = enrollment_embeddings[enroll_indices]
            
            for fold in range(num_folds):
                # Split calibration embeddings into calibration and test subsets
                calib_indices = np.random.permutation(calibration_embeddings.shape[0])
                split_point = int(0.9 * len(calib_indices))
                
                calib_subset = calibration_embeddings[calib_indices[:split_point]]
                test_subset = test_embeddings  # Use original test embeddings
                
                # Compute isolation probability for this fold
                fold_prob = self._compute_isolation_probability(
                    test_subset, selected_enrollment, calib_subset
                )
                isolation_probabilities.append(fold_prob)
        
        # Return average isolation probability
        return np.mean(isolation_probabilities)
    
    def _compute_isolation_probability(self,
                                     test_embeddings: np.ndarray,
                                     enrollment_embeddings: np.ndarray,
                                     calibration_embeddings: np.ndarray) -> float:
        """
        Compute isolation probability for a single fold.
        
        Args:
            test_embeddings: Test embeddings
            enrollment_embeddings: Enrollment embeddings
            calibration_embeddings: Calibration embeddings
            
        Returns:
            Isolation probability
        """
        N_test = test_embeddings.shape[0]
        isolation_count = 0
        total_predicates = 0
        
        for enroll_embedding in enrollment_embeddings:
            # Compute optimal threshold using calibration data
            similarities = [self.compute_similarity(enroll_embedding, calib_emb) 
                          for calib_emb in calibration_embeddings]
            similarities.sort(reverse=True)
            
            # Threshold is average of 9th and 10th highest similarities
            if len(similarities) >= 10:
                threshold = (similarities[8] + similarities[9]) / 2
            else:
                # Fallback for smaller calibration sets
                k = len(similarities) - 1
                threshold = (similarities[k-1] + similarities[k]) / 2
            
            # Apply predicate to test embeddings
            predicate_values = []
            for test_emb in test_embeddings:
                similarity = self.compute_similarity(enroll_embedding, test_emb)
                predicate_values.append(1 if similarity > threshold else 0)
            
            # Check if exactly one test embedding satisfies the predicate
            if sum(predicate_values) == 1:
                isolation_count += 1
            total_predicates += 1
        
        return isolation_count / total_predicates if total_predicates > 0 else 0.0


class LinkabilityMetric(BaseMetric):
    """
    Linkability metric for voice anonymization evaluation.
    
    This metric measures the probability that an attacker can correctly match
    anonymized speech samples with the corresponding enrollment speaker.
    """
    
    def __init__(self, similarity_function: str = "cosine"):
        """
        Initialize the Linkability metric.
        
        Args:
            similarity_function: Type of similarity function to use
        """
        super().__init__(similarity_function)
        
    def compute(self,
                test_embeddings: np.ndarray,
                enrollment_embeddings: np.ndarray,
                speaker_mapping: Optional[Dict[int, int]] = None) -> float:
        """
        Compute the Linkability metric.
        
        Args:
            test_embeddings: Test speaker embeddings [N_test, embedding_dim]
            enrollment_embeddings: Enrollment speaker embeddings [N_enroll, embedding_dim]
            speaker_mapping: Mapping from test speaker indices to enrollment speaker indices
                           If None, assumes test and enrollment speakers are in same order
            
        Returns:
            Linkability probability π^link
        """
        N_test = test_embeddings.shape[0]
        N_enroll = enrollment_embeddings.shape[0]
        
        if speaker_mapping is None:
            # Assume test and enrollment speakers are in same order
            speaker_mapping = {i: i for i in range(min(N_test, N_enroll))}
        
        successful_linkages = 0
        total_attempts = 0
        
        for test_idx, test_emb in enumerate(test_embeddings):
            if test_idx not in speaker_mapping:
                continue
                
            enroll_idx = speaker_mapping[test_idx]
            if enroll_idx >= N_enroll:
                continue
                
            # Compute similarity with correct enrollment speaker
            correct_similarity = self.compute_similarity(test_emb, enrollment_embeddings[enroll_idx])
            
            # Compute similarities with all other enrollment speakers
            other_similarities = []
            for i, enroll_emb in enumerate(enrollment_embeddings):
                if i != enroll_idx:
                    other_similarities.append(self.compute_similarity(test_emb, enroll_emb))
            
            # Check if correct speaker has highest similarity
            if other_similarities:
                max_other_similarity = max(other_similarities)
                if correct_similarity > max_other_similarity:
                    successful_linkages += 1
                total_attempts += 1
        
        return successful_linkages / total_attempts if total_attempts > 0 else 0.0
    
    def compute_with_speaker_counts(self,
                                   test_embeddings: np.ndarray,
                                   enrollment_embeddings: np.ndarray,
                                   speaker_counts: List[int]) -> Dict[int, float]:
        """
        Compute Linkability metric for different enrollment speaker counts.
        
        Args:
            test_embeddings: Test speaker embeddings
            enrollment_embeddings: Enrollment speaker embeddings
            speaker_counts: List of enrollment speaker counts to test
            
        Returns:
            Dictionary mapping speaker count to linkability probability
        """
        results = {}
        
        for N in speaker_counts:
            if N <= enrollment_embeddings.shape[0]:
                # Randomly select N enrollment speakers
                indices = np.random.choice(enrollment_embeddings.shape[0], size=N, replace=False)
                selected_enrollment = enrollment_embeddings[indices]
                
                # Create mapping for selected speakers
                speaker_mapping = {i: i for i in range(min(test_embeddings.shape[0], N))}
                
                linkability = self.compute(test_embeddings, selected_enrollment, speaker_mapping)
                results[N] = linkability
        
        return results


def compute_chance_levels(speaker_counts: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Compute chance-level performance for different metrics.
    
    Args:
        speaker_counts: List of speaker counts
        
    Returns:
        Dictionary with chance levels for each metric
    """
    chance_levels = {
        "singling_out": {N: np.exp(-1) for N in speaker_counts},  # ≈ 37%
        "linkability": {N: 1.0/N for N in speaker_counts},
        "eer": {N: 0.5 for N in speaker_counts}  # 50% for 1-EER
    }
    
    return chance_levels
