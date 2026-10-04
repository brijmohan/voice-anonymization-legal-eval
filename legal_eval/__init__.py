"""Legally validated evaluation framework for voice anonymization.

Reference implementation of the Singling Out and Linkability metrics introduced
in:

    N. Vauquier, B. M. L. Srivastava, S. A. Hosseini and E. Vincent,
    "Legally validated evaluation framework for voice anonymization",
    Interspeech 2025.

The two metrics translate the "singling out" and "linkability" criteria of the
Article 29 Working Party's Opinion 05/2014 on Anonymization Techniques into
quantities that can be measured on speech data. The framework was formally
validated by the French Data Protection Authority (CNIL).

Everything in this package operates on *speaker embeddings* (x-vectors) or on
precomputed cosine score matrices. Anonymization and embedding extraction are
upstream steps performed with external toolkits; see ``docs/reproduction.md``.
"""

from legal_eval.__about__ import __version__
from legal_eval.embeddings import (
    average_embeddings,
    build_conversations,
    build_speaker_embeddings,
    build_test_embeddings,
    l2_normalize,
)
from legal_eval.io import (
    load_score_matrix,
    load_spk2utt,
    load_xvectors,
    read_results,
    save_score_matrix,
    write_results,
)
from legal_eval.metrics import (
    linkability,
    rocch_eer,
    singling_out,
)
from legal_eval.paper import PublishedCurve, load_paper_results
from legal_eval.scoring import cosine_score_matrix
from legal_eval.sweeps import (
    SweepResult,
    eer_sweep,
    linkability_sweep,
    singling_out_sweep,
)
from legal_eval.vpc import (
    benchmark_vpc_run,
    load_vpc_embeddings,
)

__all__ = [
    "__version__",
    "average_embeddings",
    "benchmark_vpc_run",
    "build_conversations",
    "build_speaker_embeddings",
    "build_test_embeddings",
    "cosine_score_matrix",
    "eer_sweep",
    "l2_normalize",
    "linkability",
    "linkability_sweep",
    "load_paper_results",
    "load_score_matrix",
    "load_vpc_embeddings",
    "load_spk2utt",
    "load_xvectors",
    "read_results",
    "rocch_eer",
    "save_score_matrix",
    "singling_out",
    "singling_out_sweep",
    "PublishedCurve",
    "SweepResult",
    "write_results",
]
