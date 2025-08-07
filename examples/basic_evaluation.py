#!/usr/bin/env python3
"""
Basic example of using the legally validated voice anonymization evaluation framework.

This script demonstrates how to evaluate a voice anonymization system using
the Singling Out and Linkability metrics.
"""

import numpy as np
import logging
from pathlib import Path
import sys

# Add the parent directory to the path to import the legal_eval package
sys.path.append(str(Path(__file__).parent.parent))

from legal_eval import LegalEvaluator, run_evaluation
from legal_eval.metrics import SinglingOutMetric, LinkabilityMetric
from legal_eval.attack_models import create_attacker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(num_speakers=100, num_utterances=10, embedding_dim=192):
    """
    Create sample data for demonstration purposes.
    
    Args:
        num_speakers: Number of speakers
        num_utterances: Number of utterances per speaker
        embedding_dim: Dimension of speaker embeddings
        
    Returns:
        Dictionary containing sample data
    """
    logger.info(f"Creating sample data with {num_speakers} speakers...")
    
    # Create sample embeddings
    test_embeddings = np.random.randn(num_speakers, embedding_dim)
    enrollment_embeddings = np.random.randn(num_speakers, embedding_dim)
    calibration_embeddings = np.random.randn(num_speakers * 9, embedding_dim)  # 9 calibration embeddings per speaker
    
    # Normalize embeddings
    test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    enrollment_embeddings = enrollment_embeddings / np.linalg.norm(enrollment_embeddings, axis=1, keepdims=True)
    calibration_embeddings = calibration_embeddings / np.linalg.norm(calibration_embeddings, axis=1, keepdims=True)
    
    return {
        'test_embeddings': test_embeddings,
        'enrollment_embeddings': enrollment_embeddings,
        'calibration_embeddings': calibration_embeddings
    }


def demonstrate_metrics():
    """Demonstrate the Singling Out and Linkability metrics."""
    logger.info("Demonstrating legal evaluation metrics...")
    
    # Create sample data
    data = create_sample_data(num_speakers=50)
    
    # Initialize metrics
    singling_out_metric = SinglingOutMetric()
    linkability_metric = LinkabilityMetric()
    
    # Compute Singling Out metric
    logger.info("Computing Singling Out metric...")
    singling_out_score = singling_out_metric.compute(
        test_embeddings=data['test_embeddings'],
        enrollment_embeddings=data['enrollment_embeddings'],
        calibration_embeddings=data['calibration_embeddings'],
        num_runs=3,  # Reduced for demonstration
        num_folds=5   # Reduced for demonstration
    )
    
    # Compute Linkability metric
    logger.info("Computing Linkability metric...")
    linkability_score = linkability_metric.compute(
        test_embeddings=data['test_embeddings'],
        enrollment_embeddings=data['enrollment_embeddings']
    )
    
    # Print results
    print(f"\n=== Metric Results ===")
    print(f"Singling Out Risk: {singling_out_score:.3f} (baseline: 0.368)")
    print(f"Linkability Risk: {linkability_score:.3f} (baseline: 0.020)")
    
    # Interpret results
    print(f"\n=== Interpretation ===")
    if singling_out_score < 0.4:
        print("✓ Singling Out risk is acceptable (below 40%)")
    else:
        print("✗ Singling Out risk is too high (above 40%)")
    
    if linkability_score < 0.1:
        print("✓ Linkability risk is acceptable (below 10%)")
    else:
        print("✗ Linkability risk is too high (above 10%)")


def demonstrate_attack_models():
    """Demonstrate different attack models."""
    logger.info("Demonstrating attack models...")
    
    # Create sample training data
    training_data = {
        'speaker_1': np.random.randn(16000),  # 1 second of audio at 16kHz
        'speaker_2': np.random.randn(16000),
        'speaker_3': np.random.randn(16000)
    }
    
    # Create different attackers
    attackers = {
        'ignorant': create_attacker('ignorant'),
        'semi_informed': create_attacker('semi_informed'),
        'informed': create_attacker('informed')
    }
    
    print(f"\n=== Attack Models ===")
    for attack_type, attacker in attackers.items():
        print(f"{attack_type.capitalize()} Attacker: {attacker.get_attack_type()}")
        
        # Train attacker (simplified)
        try:
            attacker.train(training_data)
            print(f"  ✓ Training completed")
        except Exception as e:
            print(f"  ✗ Training failed: {e}")


def demonstrate_full_evaluation():
    """Demonstrate the complete evaluation pipeline."""
    logger.info("Demonstrating complete evaluation pipeline...")
    
    # Create sample data paths (in practice, these would be real data paths)
    test_data_path = "sample_data/test"
    enrollment_data_path = "sample_data/enrollment"
    training_data_path = "sample_data/training"
    
    # Create the evaluator
    evaluator = LegalEvaluator(
        anonymization_system="baseline_b1",
        speaker_embedding_model="ecapa_tdnn"
    )
    
    print(f"\n=== Evaluation Configuration ===")
    print(f"Anonymization System: {evaluator.anonymization_system}")
    print(f"Speaker Embedding Model: {evaluator.speaker_embedding_model}")
    print(f"Attack Models: {list(evaluator.attackers.keys())}")
    print(f"Conversation Lengths: {evaluator.config['evaluation']['conversation_lengths']}")
    print(f"Speaker Counts: {evaluator.config['evaluation']['speaker_counts']}")
    
    # Note: In practice, you would run the full evaluation here
    print(f"\n=== Full Evaluation ===")
    print("Note: This would run the complete evaluation pipeline with real data.")
    print("For demonstration purposes, we're showing the configuration only.")
    
    # Example of how to run the full evaluation:
    """
    results = evaluator.evaluate(
        test_data_path=test_data_path,
        enrollment_data_path=enrollment_data_path,
        training_data_path=training_data_path,
        output_path="results.json"
    )
    
    print("Evaluation completed!")
    print(f"Results saved to: results.json")
    """


def demonstrate_configuration():
    """Demonstrate configuration options."""
    logger.info("Demonstrating configuration options...")
    
    # Example configuration
    config = {
        "evaluation": {
            "conversation_lengths": [1, 3, 10],
            "speaker_counts": [20, 100, 500],
            "num_runs": 3,
            "num_folds": 5
        },
        "anonymization": {
            "system": "baseline_b1",
            "parameters": {
                "candidate_count": 50,
                "random_seed": 42
            }
        },
        "speaker_embedding": {
            "model": "ecapa_tdnn",
            "sample_rate": 16000,
            "frame_length": 0.025,
            "frame_shift": 0.010
        },
        "attack_models": ["ignorant", "informed"]
    }
    
    print(f"\n=== Configuration Example ===")
    print("Configuration can be customized for different evaluation scenarios:")
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("LEGALLY VALIDATED VOICE ANONYMIZATION EVALUATION FRAMEWORK")
    print("=" * 60)
    print("This example demonstrates the key features of the framework.\n")
    
    try:
        # Demonstrate individual components
        demonstrate_metrics()
        demonstrate_attack_models()
        demonstrate_configuration()
        demonstrate_full_evaluation()
        
        print(f"\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Prepare your voice data in the required format")
        print("2. Configure the evaluation parameters")
        print("3. Run the full evaluation using the LegalEvaluator class")
        print("4. Analyze the results to assess privacy protection")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
