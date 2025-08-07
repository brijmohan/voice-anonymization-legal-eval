"""
Voice Anonymization Legal Evaluation Framework

This package implements legally validated evaluation metrics for voice anonymization systems,
based on the Article 29 Working Party's Opinion 05/2014 on Anonymization Techniques.
"""

from .evaluator import LegalEvaluator
from .metrics import SinglingOutMetric, LinkabilityMetric
from .attack_models import IgnorantAttacker, SemiInformedAttacker, InformedAttacker
from .anonymization import BaseAnonymizer, BaselineB1Anonymizer

__version__ = "1.0.0"
__author__ = "Nijta SAS"
__email__ = "brij@nijta.com"

__all__ = [
    "LegalEvaluator",
    "SinglingOutMetric", 
    "LinkabilityMetric",
    "IgnorantAttacker",
    "SemiInformedAttacker", 
    "InformedAttacker",
    "BaseAnonymizer",
    "BaselineB1Anonymizer"
]
