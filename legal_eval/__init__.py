"""Legally validated evaluation framework for voice anonymization.

Reference implementation of the Singling Out and Linkability metrics introduced
in:

    N. Vauquier, B. M. L. Srivastava, S. A. Hosseini and E. Vincent,
    "Legally validated evaluation framework for voice anonymization",
    Interspeech 2025.

The two metrics translate the "singling out" and "linkability" criteria of the
Article 29 Working Party's Opinion 05/2014 on Anonymization Techniques into
quantities that can be measured on speech data.

Opinion 05/2014 sets *three* criteria. The third, **inference**, is measured by
``legal_eval.inference`` as attribute inference: how much better than the
population prior an attacker predicts a sensitive attribute from the released
artifact. It is newer and less validated than the other two, and a null result
bounds one classifier on one set of labels rather than proving anonymity.

These metrics were the vehicle of a compliance demonstration that the CNIL's
Technological Expertise Department reviewed in August 2024, finding no obstacle
to the Voice Harbor solution demonstrating compliance with the three Opinion
05/2014 criteria (ref. MAI/SA241024). That is a technical opinion on a
solution, not a certification of these metrics, and it does not transfer to a
deployer. ``docs/cnil-opinion.md`` states the scope exactly. Earlier versions
of this docstring said "formally validated", which overstated it.

Everything in this package operates on *speaker embeddings* (x-vectors) or on
precomputed cosine score matrices. Anonymization and embedding extraction are
upstream steps performed with external toolkits; see ``docs/reproduction.md``.
"""

from legal_eval.__about__ import __version__
from legal_eval.audio_integrity import (
    AudioIntegrity,
    active_fraction,
    check_anonymized,
    clipped_fraction,
    optimistic_bias_note,
)
from legal_eval.comparison import (
    DisclosureResult,
    calibration_samples,
    fit_calibration,
    is_separable,
    local_information_disclosure,
    similarity_rank_disclosure,
)
from legal_eval.embeddings import (
    average_embeddings,
    build_conversations,
    build_speaker_embeddings,
    build_test_embeddings,
    l2_normalize,
)
from legal_eval.inference import (
    AttributeDisclosure,
    attribute_disclosure,
    attribute_disclosure_by_speaker,
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
from legal_eval.preflight import (
    CheckResult,
    ceilings,
    chance_levels,
    failed,
    run_preflight,
    unmarked_claims,
)
from legal_eval.pseudo_speaker import (
    PseudoSpeakerAudit,
    audit_pseudo_speakers,
    chance_identification_rate,
    population_for_chance,
    verification_threshold,
)
from legal_eval.scoring import cosine_score_matrix
from legal_eval.sharding import balance, partition, shard_of
from legal_eval.sweeps import (
    SweepResult,
    eer_sweep,
    linkability_sweep,
    singling_out_sweep,
)
from legal_eval.trial_metrics import (
    ZebraProfile,
    category_tag,
    cllr,
    gomez_barrero_linkability,
    min_cllr,
    optimal_llr,
    pav,
    zebra,
)
from legal_eval.vpc import (
    benchmark_vpc_run,
    load_vpc_embeddings,
)

__all__ = [
    "ZebraProfile",
    "category_tag",
    "cllr",
    "gomez_barrero_linkability",
    "min_cllr",
    "optimal_llr",
    "pav",
    "zebra",
    "AudioIntegrity",
    "active_fraction",
    "check_anonymized",
    "clipped_fraction",
    "optimistic_bias_note",
    "CheckResult",
    "ceilings",
    "chance_levels",
    "failed",
    "run_preflight",
    "unmarked_claims",
    "AttributeDisclosure",
    "attribute_disclosure",
    "attribute_disclosure_by_speaker",
    "verification_threshold",
    "population_for_chance",
    "chance_identification_rate",
    "audit_pseudo_speakers",
    "__version__",
    "average_embeddings",
    "balance",
    "benchmark_vpc_run",
    "build_conversations",
    "build_speaker_embeddings",
    "build_test_embeddings",
    "calibration_samples",
    "cosine_score_matrix",
    "eer_sweep",
    "fit_calibration",
    "is_separable",
    "l2_normalize",
    "linkability",
    "linkability_sweep",
    "load_paper_results",
    "local_information_disclosure",
    "load_score_matrix",
    "load_vpc_embeddings",
    "load_spk2utt",
    "load_xvectors",
    "partition",
    "read_results",
    "rocch_eer",
    "save_score_matrix",
    "shard_of",
    "singling_out",
    "similarity_rank_disclosure",
    "singling_out_sweep",
    "DisclosureResult",
    "PseudoSpeakerAudit",
    "PublishedCurve",
    "SweepResult",
    "write_results",
]
