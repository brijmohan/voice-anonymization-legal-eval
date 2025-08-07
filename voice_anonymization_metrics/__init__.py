"""Utilities to evaluate anonymized speech data.

This package provides implementations of the Singling Out and Linkability
metrics described in ``Legally validated evaluation framework for voice
anonymization``.  The functions operate on precomputed speaker embeddings.
"""

from .embeddings import average_embeddings
from .linkability import linkability_probability
from .singling_out import singling_out_probability

__all__ = [
    "average_embeddings",
    "linkability_probability",
    "singling_out_probability",
]
