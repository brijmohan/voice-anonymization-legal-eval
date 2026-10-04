#!/usr/bin/env python3
"""Quickstart: the three metrics on synthetic embeddings, in one file.

Run with::

    python examples/01_quickstart.py

Nothing here needs data or a GPU. It shows the shape of a real evaluation:
build enrollment and test embeddings from disjoint utterances, score them, and
sweep each metric over the number of speakers the attacker must search.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script inside examples/ puts examples/ on sys.path, not the repo
# root, so an uninstalled clone cannot import the package. Append (not insert)
# the repo root as a fallback: a proper `pip install -e .` still takes priority.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus
from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
from legal_eval.metrics import TRIVIAL_SINGLING_OUT
from legal_eval.scoring import cosine_score_matrix
from legal_eval.sweeps import eer_sweep, linkability_sweep, singling_out_sweep

N_SPEAKERS = 80
N_UTTERANCES = 12
CONVERSATION_LENGTHS = (1, 3)


def evaluate(embeddings, enroll_utts, test_utts, conversation_length):
    """Return the three metrics at the full population size."""
    enroll = build_speaker_embeddings(enroll_utts, embeddings)
    test, _ = build_test_embeddings(test_utts, embeddings, conversation_length, "first")
    matrix = cosine_score_matrix(enroll, test)

    counts = [N_SPEAKERS]
    linkability = linkability_sweep(
        matrix, speaker_counts=counts, conversation_length=conversation_length
    ).mean()[N_SPEAKERS]
    eer = eer_sweep(
        matrix, speaker_counts=counts, conversation_length=conversation_length
    ).mean()[N_SPEAKERS]
    singling_out = singling_out_sweep(
        enroll,
        test_utts,
        embeddings,
        conversation_length=conversation_length,
        speaker_counts=counts,
        n_folds=3,
    ).mean()[N_SPEAKERS]
    return singling_out, linkability, 1.0 - eer


def main() -> None:
    spk2utt, original = synthetic_corpus(
        n_speakers=N_SPEAKERS, n_utterances=N_UTTERANCES, seed=0
    )
    # Enrollment and test must never share an utterance, or the metrics measure
    # recording identity rather than speaker identity.
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=N_UTTERANCES // 2)

    conditions = {
        "original": original,
        "weakly anonymized": simulate_anonymization(original, strength=0.55, seed=1),
        "strongly anonymized": simulate_anonymization(original, strength=0.95, seed=2),
    }

    print(f"Synthetic corpus: {N_SPEAKERS} speakers, {N_UTTERANCES} utterances each")
    print(f"Chance level: Singling Out {TRIVIAL_SINGLING_OUT:.2f}, "
          f"Linkability {1 / N_SPEAKERS:.3f}, 1-EER 0.50\n")

    header = f"{'condition':<22}{'L':>3}{'SinglingOut':>13}{'Linkability':>13}{'1-EER':>9}"
    print(header)
    print("-" * len(header))
    for name, embeddings in conditions.items():
        for length in CONVERSATION_LENGTHS:
            singling_out, linkability, one_minus_eer = evaluate(
                embeddings, enroll_utts, test_utts, length
            )
            print(
                f"{name:<22}{length:>3}{singling_out:>13.3f}"
                f"{linkability:>13.3f}{one_minus_eer:>9.3f}"
            )

    print(
        "\nLinkability separates the three conditions sharply and rises with the\n"
        "conversation length, while 1-EER moves far less. That gap is the point of\n"
        "the paper: the EER understates how much the residual risk varies."
    )


if __name__ == "__main__":
    main()
