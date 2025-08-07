#!/usr/bin/env python3
"""
Command-line interface for the legally validated voice anonymization evaluation framework.
"""

import click
import logging
import yaml
from pathlib import Path
from typing import Optional

from .evaluator import LegalEvaluator, run_evaluation
from .metrics import SinglingOutMetric, LinkabilityMetric
from .attack_models import create_attacker, get_attack_scenarios
from .anonymization import get_available_anonymizers


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def cli(verbose: bool, config: Optional[str]):
    """Legally validated voice anonymization evaluation framework."""
    # Set up logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@cli.command()
@click.option('--test-data', '-t', required=True, help='Path to test data')
@click.option('--enrollment-data', '-e', required=True, help='Path to enrollment data')
@click.option('--training-data', help='Path to training data (for attacker training)')
@click.option('--output', '-o', default='results.json', help='Output file path')
@click.option('--anonymization-system', '-a', default='baseline_b1', help='Anonymization system to evaluate')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def evaluate(test_data: str, enrollment_data: str, training_data: Optional[str], 
            output: str, anonymization_system: str, config: Optional[str]):
    """Run complete evaluation of voice anonymization system."""
    click.echo(f"Starting evaluation with {anonymization_system} anonymization system...")
    
    try:
        results = run_evaluation(
            test_data=test_data,
            enrollment_data=enrollment_data,
            training_data=training_data,
            anonymization_system=anonymization_system,
            config_path=config,
            output_path=output
        )
        
        click.echo(f"Evaluation completed successfully!")
        click.echo(f"Results saved to: {output}")
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--test-embeddings', required=True, help='Path to test embeddings file')
@click.option('--enrollment-embeddings', required=True, help='Path to enrollment embeddings file')
@click.option('--calibration-embeddings', required=True, help='Path to calibration embeddings file')
@click.option('--output', '-o', default='metrics_results.json', help='Output file path')
def compute_metrics(test_embeddings: str, enrollment_embeddings: str, 
                   calibration_embeddings: str, output: str):
    """Compute Singling Out and Linkability metrics from embeddings."""
    click.echo("Computing legal evaluation metrics...")
    
    try:
        import numpy as np
        
        # Load embeddings
        test_emb = np.load(test_embeddings)
        enrollment_emb = np.load(enrollment_embeddings)
        calibration_emb = np.load(calibration_embeddings)
        
        # Initialize metrics
        singling_out_metric = SinglingOutMetric()
        linkability_metric = LinkabilityMetric()
        
        # Compute metrics
        singling_out_score = singling_out_metric.compute(
            test_embeddings=test_emb,
            enrollment_embeddings=enrollment_emb,
            calibration_embeddings=calibration_emb
        )
        
        linkability_score = linkability_metric.compute(
            test_embeddings=test_emb,
            enrollment_embeddings=enrollment_emb
        )
        
        # Save results
        results = {
            'singling_out': singling_out_score,
            'linkability': linkability_score,
            'baseline_singling_out': 0.368,  # exp(-1)
            'baseline_linkability': 1.0 / test_emb.shape[0]
        }
        
        import json
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"Metrics computed successfully!")
        click.echo(f"Singling Out: {singling_out_score:.3f}")
        click.echo(f"Linkability: {linkability_score:.3f}")
        click.echo(f"Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Metric computation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
def list_attackers():
    """List available attack models."""
    click.echo("Available attack models:")
    scenarios = get_attack_scenarios()
    for attack_type, description in scenarios.items():
        click.echo(f"  {attack_type}: {description}")


@cli.command()
def list_anonymizers():
    """List available anonymization systems."""
    click.echo("Available anonymization systems:")
    anonymizers = get_available_anonymizers()
    for name, description in anonymizers.items():
        click.echo(f"  {name}: {description}")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def validate_config(config: str):
    """Validate configuration file."""
    click.echo(f"Validating configuration file: {config}")
    
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Basic validation
        required_sections = ['evaluation', 'anonymization', 'speaker_embedding']
        for section in required_sections:
            if section not in config_data:
                click.echo(f"Missing required section: {section}", err=True)
                raise click.Abort()
        
        click.echo("Configuration file is valid!")
        
        # Print configuration summary
        print_config_summary(config_data)
        
    except Exception as e:
        click.echo(f"Configuration validation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--output', '-o', default='config.yaml', help='Output configuration file path')
def create_config(output: str):
    """Create a sample configuration file."""
    sample_config = {
        'evaluation': {
            'conversation_lengths': [1, 3, 30],
            'speaker_counts': [20, 100, 1000, 10000],
            'num_runs': 5,
            'num_folds': 10
        },
        'anonymization': {
            'system': 'baseline_b1',
            'parameters': {
                'candidate_count': 100,
                'random_seed': 42
            }
        },
        'speaker_embedding': {
            'model': 'ecapa_tdnn',
            'sample_rate': 16000,
            'frame_length': 0.025,
            'frame_shift': 0.010
        },
        'attack_models': ['ignorant', 'semi_informed', 'informed']
    }
    
    try:
        with open(output, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        click.echo(f"Sample configuration file created: {output}")
        click.echo("Please customize the configuration for your specific needs.")
        
    except Exception as e:
        click.echo(f"Failed to create configuration file: {e}", err=True)
        raise click.Abort()


def print_summary(results: dict):
    """Print evaluation results summary."""
    click.echo("\n=== Evaluation Summary ===")
    
    if 'singling_out' in results:
        click.echo("Singling Out Results:")
        for attack_type, scores in results['singling_out'].items():
            for conv_length, score in scores.items():
                click.echo(f"  {attack_type} (L={conv_length}): {score:.3f}")
    
    if 'linkability' in results:
        click.echo("\nLinkability Results:")
        for attack_type, scores in results['linkability'].items():
            for conv_length, score in scores.items():
                click.echo(f"  {attack_type} (L={conv_length}): {score:.3f}")


def print_config_summary(config: dict):
    """Print configuration summary."""
    click.echo("\n=== Configuration Summary ===")
    
    if 'evaluation' in config:
        eval_config = config['evaluation']
        click.echo(f"Conversation lengths: {eval_config.get('conversation_lengths', 'N/A')}")
        click.echo(f"Speaker counts: {eval_config.get('speaker_counts', 'N/A')}")
        click.echo(f"Number of runs: {eval_config.get('num_runs', 'N/A')}")
    
    if 'anonymization' in config:
        anon_config = config['anonymization']
        click.echo(f"Anonymization system: {anon_config.get('system', 'N/A')}")
    
    if 'speaker_embedding' in config:
        emb_config = config['speaker_embedding']
        click.echo(f"Speaker embedding model: {emb_config.get('model', 'N/A')}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
