"""Command line interface.

Typical reproduction, starting from Kaldi-style data directories holding
``spk2utt`` and ``xvector.h5``:

.. code-block:: console

    # 1. score every enrollment speaker against every test speaker, for one L
    legal-eval score-matrix --enroll-dir data/cv11-A --test-dir data/cv11-B \
        --conversation-length 1 --output scores/informed_L1.npy

    # 2. sweep the metrics over the number of speakers
    legal-eval linkability --score-matrix scores/informed_L1.npy \
        --output results/linkability_informed_L1.json
    legal-eval eer --score-matrix scores/informed_L1.npy \
        --output results/eer_informed_L1.json

    # 3. Singling Out works from embeddings rather than a score matrix
    legal-eval singling-out --enroll-dir data/cv11-B --test-dir data/cv11-A \
        --conversation-length 1 --output results/singling_out_informed_L1.json

    # 4. draw the figure
    legal-eval plot --results-dir results --output figures/paper_figure.pdf

``legal-eval demo`` runs all of it on synthetic embeddings, which is the quickest
way to check an installation.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger("legal_eval")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument(
        "--n-runs", type=int, default=5, help="number of random draws per point"
    )
    parser.add_argument(
        "--speaker-counts",
        type=str,
        default=None,
        help=(
            "comma-separated speaker counts, or 'linear' / 'geometric' for the "
            "paper's grids. Defaults to the metric's usual grid."
        ),
    )


def _parse_speaker_counts(spec: str | None, maximum: int) -> list[int] | None:
    from legal_eval.sweeps import default_speaker_counts, geometric_speaker_counts

    if spec is None:
        return None
    if spec == "linear":
        return default_speaker_counts(maximum)
    if spec == "geometric":
        return geometric_speaker_counts(maximum)
    return [int(part) for part in spec.split(",") if part.strip()]


def _load_kaldi_dir(directory: Path, min_utterances: int = 0):
    """Load ``spk2utt`` and the matching x-vectors from a Kaldi-style directory."""
    from legal_eval.io import load_spk2utt, load_xvectors, utterances_of

    spk2utt = load_spk2utt(directory / "spk2utt", min_utterances=min_utterances)
    if not spk2utt:
        raise SystemExit(f"no speakers found in {directory / 'spk2utt'}")

    for name in ("xvector.h5", "xvectors.h5", "xvector.npz", "xvectors.npz"):
        candidate = directory / name
        if candidate.exists():
            logger.info("loading embeddings from %s", candidate)
            return spk2utt, load_xvectors(candidate, utterances_of(spk2utt))
    raise SystemExit(f"no xvector.h5 or xvector.npz found in {directory}")


def cmd_score_matrix(args: argparse.Namespace) -> int:
    from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
    from legal_eval.io import save_score_matrix
    from legal_eval.scoring import cosine_score_matrix

    enroll_spk2utt, enroll_embeddings = _load_kaldi_dir(args.enroll_dir)
    test_spk2utt, test_embeddings = _load_kaldi_dir(
        args.test_dir, min_utterances=args.conversation_length
    )

    logger.info("averaging %d enrollment speakers", len(enroll_spk2utt))
    enroll = build_speaker_embeddings(enroll_spk2utt, enroll_embeddings)

    logger.info(
        "building test embeddings for %d speakers at L=%d",
        len(test_spk2utt),
        args.conversation_length,
    )
    test, chosen = build_test_embeddings(
        test_spk2utt,
        test_embeddings,
        conversation_length=args.conversation_length,
        selection=args.selection,
        rng=np.random.default_rng(args.seed),
    )

    matrix = cosine_score_matrix(
        enroll,
        test,
        metadata={
            "conversation_length": args.conversation_length,
            "selection": args.selection,
            "seed": args.seed,
            "attacker": args.attacker,
            "enroll_dir": str(args.enroll_dir),
            "test_dir": str(args.test_dir),
            "utterances_per_test_speaker": {s: len(u) for s, u in chosen.items()},
        },
    )
    save_score_matrix(matrix, args.output)
    logger.info("wrote %s with shape %s", args.output, matrix.scores.shape)
    return 0


def cmd_linkability(args: argparse.Namespace) -> int:
    from legal_eval.io import load_score_matrix, write_results
    from legal_eval.sweeps import linkability_sweep

    matrix = load_score_matrix(args.score_matrix)
    counts = _parse_speaker_counts(args.speaker_counts, matrix.scores.shape[0])
    result = linkability_sweep(
        matrix,
        speaker_counts=counts,
        n_runs=args.n_runs,
        seed=args.seed,
        conversation_length=int(matrix.metadata.get("conversation_length", 1)),
        estimator=args.estimator,
        metadata={"attacker": matrix.metadata.get("attacker", "unknown")},
        progress=True,
    )
    write_results(result.to_dict(), args.output)
    _report(result)
    return 0


def cmd_eer(args: argparse.Namespace) -> int:
    from legal_eval.io import load_score_matrix, write_results
    from legal_eval.sweeps import eer_sweep

    matrix = load_score_matrix(args.score_matrix)
    counts = _parse_speaker_counts(args.speaker_counts, matrix.scores.shape[0])
    result = eer_sweep(
        matrix,
        speaker_counts=counts,
        n_runs=args.n_runs,
        seed=args.seed,
        conversation_length=int(matrix.metadata.get("conversation_length", 1)),
        max_nontarget_per_speaker=args.max_nontarget_per_speaker,
        metadata={"attacker": matrix.metadata.get("attacker", "unknown")},
        progress=True,
    )
    write_results(result.to_dict(), args.output)
    _report(result, transform=lambda v: 1.0 - v, label="1-EER")
    return 0


def cmd_singling_out(args: argparse.Namespace) -> int:
    from legal_eval.embeddings import build_speaker_embeddings
    from legal_eval.io import write_results
    from legal_eval.sweeps import singling_out_sweep

    enroll_spk2utt, enroll_raw = _load_kaldi_dir(
        args.enroll_dir, min_utterances=args.n_enroll_utterances
    )
    test_spk2utt, test_raw = _load_kaldi_dir(
        args.test_dir, min_utterances=2 * args.conversation_length
    )

    rng = np.random.default_rng(args.seed)
    speakers = list(enroll_spk2utt)
    if args.n_enroll_speakers and args.n_enroll_speakers < len(speakers):
        picked = rng.choice(len(speakers), size=args.n_enroll_speakers, replace=False)
        speakers = [speakers[int(i)] for i in sorted(picked)]
    trimmed = {s: enroll_spk2utt[s][: args.n_enroll_utterances] for s in speakers}
    logger.info(
        "using %d enrollment speakers, %d utterances each",
        len(trimmed),
        args.n_enroll_utterances,
    )
    enroll = build_speaker_embeddings(trimmed, enroll_raw)

    counts = _parse_speaker_counts(args.speaker_counts, len(test_spk2utt))
    result = singling_out_sweep(
        enroll,
        test_spk2utt,
        test_raw,
        conversation_length=args.conversation_length,
        speaker_counts=counts,
        n_runs=args.n_runs,
        n_folds=args.n_folds,
        max_calibration=args.max_calibration,
        seed=args.seed,
        vary_folds=not args.freeze_folds,
        metadata={"attacker": args.attacker},
        progress=True,
    )
    write_results(result.to_dict(), args.output)
    _report(result)
    return 0


#: ``<metric>_<attacker>_L<length>.json``, the layout ``plot`` expects.
_RESULT_NAME = re.compile(
    r"^(?P<metric>singling_out|linkability|eer)_(?P<attacker>[a-z_]+)_L(?P<length>\d+)$"
)


def cmd_plot(args: argparse.Namespace) -> int:
    from legal_eval.io import read_results
    from legal_eval.plotting import plot_paper_figure
    from legal_eval.sweeps import SweepResult

    nested: dict[str, dict[int, dict[str, SweepResult]]] = {}
    found = 0
    for path in sorted(Path(args.results_dir).glob("*.json")):
        match = _RESULT_NAME.match(path.stem)
        if not match:
            logger.warning("skipping %s: name does not match the expected pattern", path.name)
            continue
        result = SweepResult.from_dict(read_results(path))
        nested.setdefault(match["metric"], {}).setdefault(
            int(match["length"]), {}
        )[match["attacker"]] = result
        found += 1

    if not found:
        raise SystemExit(
            f"no result files matching <metric>_<attacker>_L<length>.json in {args.results_dir}"
        )

    lengths = (
        [int(v) for v in args.conversation_lengths.split(",")]
        if args.conversation_lengths
        else sorted({length for by_length in nested.values() for length in by_length})
    )
    plot_paper_figure(nested, args.output, conversation_lengths=lengths)
    logger.info("plotted %d result files to %s", found, args.output)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole pipeline on synthetic embeddings."""
    from legal_eval.demo import simulate_anonymization, split_corpus, synthetic_corpus
    from legal_eval.embeddings import build_speaker_embeddings, build_test_embeddings
    from legal_eval.io import write_results
    from legal_eval.plotting import plot_paper_figure
    from legal_eval.scoring import cosine_score_matrix
    from legal_eval.sweeps import eer_sweep, linkability_sweep, singling_out_sweep

    output = Path(args.output_dir)
    (output / "results").mkdir(parents=True, exist_ok=True)

    logger.info("generating a synthetic corpus of %d speakers", args.n_speakers)
    spk2utt, original = synthetic_corpus(
        n_speakers=args.n_speakers, n_utterances=args.n_utterances, seed=args.seed
    )
    enroll_utts, test_utts = split_corpus(spk2utt, n_first=args.n_utterances // 2)

    # Singling Out needs 2L utterances per test speaker for a test conversation
    # plus a calibration one, so only keep the lengths the corpus can support.
    n_test_utterances = args.n_utterances - args.n_utterances // 2
    lengths = [length for length in (1, 3) if 2 * length <= n_test_utterances]
    if not lengths:
        raise SystemExit(
            f"--n-utterances {args.n_utterances} leaves {n_test_utterances} test "
            "utterances per speaker; at least 2 are needed. Try --n-utterances 12."
        )

    conditions = {
        "original": original,
        "informed": simulate_anonymization(original, strength=0.55, seed=args.seed + 1),
        "ignorant": simulate_anonymization(original, strength=0.95, seed=args.seed + 2),
    }
    nested: dict[str, dict[int, dict[str, object]]] = {}

    for attacker, embeddings in conditions.items():
        enroll = build_speaker_embeddings(enroll_utts, embeddings)
        for length in lengths:
            test, _ = build_test_embeddings(test_utts, embeddings, length, "first")
            matrix = cosine_score_matrix(
                enroll, test, metadata={"conversation_length": length, "attacker": attacker}
            )
            counts = [n for n in (2, 5, 10, 20, 40, args.n_speakers) if n <= args.n_speakers]

            link = linkability_sweep(
                matrix, speaker_counts=counts, n_runs=args.n_runs, seed=args.seed,
                conversation_length=length, metadata={"attacker": attacker},
            )
            eer = eer_sweep(
                matrix, speaker_counts=counts, n_runs=args.n_runs, seed=args.seed,
                conversation_length=length, metadata={"attacker": attacker},
            )
            sing = singling_out_sweep(
                enroll, test_utts, embeddings, conversation_length=length,
                speaker_counts=counts, n_runs=args.n_runs, n_folds=args.n_folds,
                seed=args.seed, metadata={"attacker": attacker},
            )
            for result in (link, eer, sing):
                name = f"{result.metric}_{attacker}_L{length}.json"
                write_results(result.to_dict(), output / "results" / name)
                nested.setdefault(result.metric, {}).setdefault(length, {})[attacker] = result

            logger.info(
                "%-9s L=%d  linkability %.3f  1-EER %.3f  singling out %.3f",
                attacker, length,
                link.mean()[counts[-1]], 1.0 - eer.mean()[counts[-1]],
                sing.mean()[counts[-1]],
            )

    figure = output / "demo_figure.png"
    plot_paper_figure(nested, figure, conversation_lengths=lengths)
    logger.info("wrote results to %s and the figure to %s", output / "results", figure)
    return 0


def _report(result, transform=None, label: str | None = None) -> None:
    """Print a short table of the sweep."""
    means, stds = result.mean(), result.std()
    name = label or result.metric
    print(f"\n{name} (L={result.conversation_length})")
    print(f"{'speakers':>10}  {'mean':>8}  {'std':>8}")
    for count in result.speaker_counts:
        value = transform(means[count]) if transform else means[count]
        print(f"{count:>10}  {value:>8.4f}  {stds[count]:>8.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal-eval",
        description=(
            "Legally validated evaluation of voice anonymization: Singling Out, "
            "Linkability and ROCCH-EER (Interspeech 2025)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("score-matrix", help="score enrollment against test speakers")
    p.add_argument("--enroll-dir", type=Path, required=True, help="Kaldi dir for enrollment")
    p.add_argument("--test-dir", type=Path, required=True, help="Kaldi dir for test")
    p.add_argument("--conversation-length", type=int, default=1, help="L")
    p.add_argument(
        "--selection", choices=("reference", "first", "random"), default="reference",
        help="which utterances form a test conversation",
    )
    p.add_argument("--attacker", default="unknown", help="attacker label recorded in metadata")
    p.add_argument("--output", type=Path, required=True, help="output .npy path")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_score_matrix)

    p = subparsers.add_parser("linkability", help="sweep Linkability over N'")
    p.add_argument("--score-matrix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--estimator", choices=("sampling", "exact"), default="sampling",
        help="'exact' returns the closed form, with no sampling noise",
    )
    _add_common(p)
    p.set_defaults(func=cmd_linkability)

    p = subparsers.add_parser("eer", help="sweep ROCCH-EER over N'")
    p.add_argument("--score-matrix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-nontarget-per-speaker", type=int, default=500)
    _add_common(p)
    p.set_defaults(func=cmd_eer)

    p = subparsers.add_parser("singling-out", help="sweep Singling Out over N")
    p.add_argument("--enroll-dir", type=Path, required=True, help="Kaldi dir for enrollment")
    p.add_argument("--test-dir", type=Path, required=True, help="Kaldi dir for test")
    p.add_argument("--conversation-length", type=int, default=1, help="L")
    p.add_argument("--n-enroll-speakers", type=int, default=495)
    p.add_argument(
        "--n-enroll-utterances", type=int, default=30,
        help="utterances averaged into each enrollment embedding",
    )
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--max-calibration", type=int, default=9, help="M")
    p.add_argument(
        "--freeze-folds", action="store_true",
        help="reuse one conversation split across folds, as the original code did",
    )
    p.add_argument("--attacker", default="unknown")
    p.add_argument("--output", type=Path, required=True)
    _add_common(p)
    p.set_defaults(func=cmd_singling_out)

    p = subparsers.add_parser("plot", help="draw the figure from result files")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conversation-lengths", type=str, default=None, help="e.g. 1,3,30")
    p.set_defaults(func=cmd_plot)

    p = subparsers.add_parser("demo", help="run everything on synthetic embeddings")
    p.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    p.add_argument("--n-speakers", type=int, default=60)
    p.add_argument("--n-utterances", type=int, default=12)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
