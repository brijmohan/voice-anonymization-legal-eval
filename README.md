# Voice Anonymization Legal Evaluation Framework

This repository implements the legally validated evaluation framework for voice anonymization systems, as described in the paper "Legally validated evaluation framework for voice anonymization" (Interspeech 2025). The framework introduces two key metrics that align with legal requirements for data anonymization:

## Overview

The framework addresses the critical gap between conventional evaluation metrics (like EER) and legally required assessment of re-identification risk in voice anonymization. It implements two legally grounded metrics:

1. **Singling Out**: Measures the probability that an attacker can isolate a single speaker from anonymized speech samples
2. **Linkability**: Measures the probability that anonymized speech samples can be correctly linked to the corresponding enrollment speaker

These metrics have been formally validated by the French Data Protection Authority (CNIL) and provide a legally compliant assessment of voice anonymization systems.

## Key Features

- **Legally Validated Metrics**: Based on Article 29 Working Party's Opinion 05/2014 on Anonymization Techniques
- **Multiple Attack Scenarios**: Supports Ignorant, Semi-Informed, and Informed attacker models
- **Robust Evaluation Framework**: Includes random sampling, threshold calibration, and statistical averaging
- **Comprehensive Documentation**: Detailed implementation guide and usage examples
- **Reproducible Results**: Complete evaluation pipeline with configurable parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/voice-anonymization-legal-eval.git
cd voice-anonymization-legal-eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from legal_eval import LegalEvaluator
from legal_eval.metrics import SinglingOutMetric, LinkabilityMetric

# Initialize evaluator
evaluator = LegalEvaluator(
    anonymization_system="baseline_b1",
    speaker_embedding_model="ecapa_tdnn"
)

# Compute Singling Out metric
singling_out = SinglingOutMetric()
singling_out_score = singling_out.compute(
    test_embeddings=test_xvectors,
    enrollment_embeddings=enrollment_xvectors,
    calibration_embeddings=calibration_xvectors
)

# Compute Linkability metric
linkability = LinkabilityMetric()
linkability_score = linkability.compute(
    test_embeddings=test_xvectors,
    enrollment_embeddings=enrollment_xvectors
)

print(f"Singling Out Risk: {singling_out_score:.3f}")
print(f"Linkability Risk: {linkability_score:.3f}")
```

## Metrics Description

### Singling Out Metric

The Singling Out metric implements the Predicate Singling Out (PSO) framework to quantify the risk that an attacker can isolate an individual from an anonymized dataset. It computes:

```
π^sing = Pr_{X,x^enroll}{∃i s.t. p(x^test_i) = 1 and p(x^test_j) = 0 ∀j≠i}
```

where the predicate function is defined as:
```
p(x^test) = 𝟙{s(x^test, x^enroll) > s^thresh}
```

### Linkability Metric

The Linkability metric measures the probability that an attacker can correctly match anonymized speech samples with the corresponding enrollment speaker:

```
π^link = Pr_{x_i^test}{s(x_i^test, x_i^enroll) > max_{j≠i} s(x_i^test, x_j^enroll)}
```

## Attack Models

The framework supports three attacker models with varying levels of knowledge:

1. **Ignorant Attacker**: Unaware that data is anonymized, trains on original data
2. **Semi-Informed Attacker**: Aware of anonymization but uses outdated system
3. **Informed Attacker**: Has full knowledge of the anonymization process (worst-case scenario)

## Usage Examples

### Basic Evaluation

```python
from legal_eval import run_evaluation

# Run complete evaluation
results = run_evaluation(
    test_data="path/to/test/data",
    enrollment_data="path/to/enrollment/data",
    anonymization_system="baseline_b1",
    conversation_lengths=[1, 3, 30],
    speaker_counts=[20, 100, 1000, 10000]
)

# Print results
for metric, scores in results.items():
    print(f"{metric}: {scores}")
```

### Custom Anonymization System

```python
from legal_eval import LegalEvaluator
from legal_eval.anonymization import BaseAnonymizer

class CustomAnonymizer(BaseAnonymizer):
    def anonymize(self, audio_data):
        # Implement your anonymization logic
        return anonymized_audio

evaluator = LegalEvaluator(
    anonymization_system=CustomAnonymizer(),
    speaker_embedding_model="ecapa_tdnn"
)
```

## Configuration

The framework can be configured through a YAML configuration file:

```yaml
# config.yaml
evaluation:
  conversation_lengths: [1, 3, 30]
  speaker_counts: [20, 100, 1000, 10000]
  num_runs: 5
  num_folds: 10

anonymization:
  system: "baseline_b1"
  parameters:
    candidate_count: 100
    random_seed: 42

speaker_embedding:
  model: "ecapa_tdnn"
  extractor_path: "models/ecapa_tdnn.pth"
  sample_rate: 16000
  frame_length: 0.025
  frame_shift: 0.010

attack_models:
  - "ignorant"
  - "semi_informed"
  - "informed"
```

## Dataset Requirements

The framework expects the following data structure:

```
data/
├── test/
│   ├── speaker_1/
│   │   ├── utterance_1.wav
│   │   ├── utterance_2.wav
│   │   └── ...
│   └── speaker_2/
├── enrollment/
│   ├── speaker_1/
│   └── speaker_2/
└── calibration/
    ├── speaker_1/
    └── speaker_2/
```

## Results Interpretation

- **Lower values indicate better privacy protection**
- **Singling Out**: Values close to 37% (baseline) indicate good protection
- **Linkability**: Values close to 1/N (chance level) indicate good protection
- **Compare across different attack models** to assess robustness
- **Consider conversation length effects** on privacy risk

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{nijta2025legally,
  title={Legally validated evaluation framework for voice anonymization},
  author={Vauquier, Nathalie and Srivastava, Brij Mohan Lal and Hosseini, Seyed Ahmad and Vincent, Emmanuel},
  booktitle={Interspeech},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by NIJTA SAS and builds upon the Voice Privacy Challenge framework.

## Contact

For questions and support, please open an issue on GitHub or contact the maintainers.
