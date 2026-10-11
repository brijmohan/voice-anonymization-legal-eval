r"""The inference criterion: what can be deduced about a person, not who they are.

Article 29 Working Party Opinion 05/2014 sets three tests for anonymity. This
package implemented two of them, singling out and linkability, for its first two
years. This module is the third.

Why membership inference is the wrong instrument
------------------------------------------------
The Opinion asks about the possibility of deducing, with significant
probability, **the value of an attribute** from other attributes. Membership
inference asks whether a record was present in a dataset. Those are different
questions: a system can resist membership inference and still reveal a speaker's
age band and region, and a system can leak membership while revealing no
attribute at all, which is closer to singling out. So the instrument here is
**attribute inference**: train a classifier for a sensitive attribute on
anonymized output and ask how much better than guessing it does.

Prior art, and what is new here
-------------------------------
Measuring attribute leakage with a classifier is not new and this module should
not pretend otherwise.

* **Noe, Mohammadamini, Matrouf, Parcollet, Nautsch and Bonastre**, "Adversarial
  Disentanglement of Speaker Representation for Attribute-Driven Privacy
  Preservation", Interspeech 2021, doi:10.21437/interspeech.2021-1712. Introduces
  attribute-driven privacy preservation, notes that neural speaker embeddings
  carry "age, sex, physical state, education level or ethnicity", and measures
  concealment of the sex attribute with a classifier. The method here follows
  theirs.
* **Vali and Backstrom**, "Privacy PORCUPINE", Interspeech 2024,
  doi:10.21437/interspeech.2024-117. Shows that the occurrence frequency of
  codebook elements in a vector-quantized bottleneck has an uneven information
  rate, which leaks. Independent support for the warning about quantization in
  ``docs/constitution.md`` section 4.3.
* **Feng, Peri and Narayanan**, "User-Level Differential Privacy against
  Attribute Inference Attack of Speech Emotion Recognition on Federated
  Learning", Interspeech 2022, doi:10.21437/interspeech.2022-10060. Names the
  attack and measures leakage under a differential-privacy mechanism.

Two differences, and they are the contribution rather than the classifier:

1. **Measured on the released artifact, not on the representation.** Noe et al.
   measure on the speaker embedding. A controller releases *audio*, so what
   matters legally is what survives into features extracted from the rendered
   output. Vocoder drift (Panariello et al., Interspeech 2023) is the reason
   these differ.
2. **Reported against the prior, in the Opinion's own terms.** A raw accuracy is
   not interpretable: 71% accuracy on an attribute whose majority class is 70%
   is nothing, and the same number on a balanced three-way attribute is a lot.
   The quantity reported is the share of the available headroom an attacker
   captures, so 0 means nothing was learned and 1 means the attribute is fully
   determined, which is what "deduce with significant probability" needs.

What this cannot tell you
-------------------------
A null result bounds **this** classifier on **these** labels, nothing more. The
labels also matter: Common Voice's demographics are optional and self-reported,
with age in ten-year bands, so a weak result may be label noise rather than
privacy. Report the label coverage alongside the number or the number means
little.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttributeDisclosure:
    """How much of an attribute an attacker recovers from released features.

    Attributes:
        attribute: Name of the attribute measured.
        accuracy: Cross-validated accuracy of the attacker's classifier.
        balanced_accuracy: Mean per-class recall. The one to quote when classes
            are imbalanced, which demographic labels always are.
        prior: Accuracy of always predicting the majority class. The chance
            level, and the only thing that makes ``accuracy`` interpretable.
        majority_class: The class that prior predicts.
        advantage: ``(accuracy - prior) / (1 - prior)``, the share of available
            headroom captured. 0 is no disclosure, 1 is full determination.
            Negative means the classifier did worse than guessing, which is
            noise, not protection.
        per_class_recall: Recall per class. A rare class recovered well is a
            worse legal outcome than the aggregate suggests, because a rare
            attribute value is more identifying, so ``summary`` reports the
            most exposed minority value rather than the best overall recall.
        n_classes: Number of distinct labels.
        n_labelled: How many samples carried a label.
        n_unlabelled: How many were dropped for having none. Report it: a
            result computed on a third of the population is not a result about
            the population.
    """

    attribute: str
    accuracy: float
    balanced_accuracy: float
    prior: float
    majority_class: str
    advantage: float
    per_class_recall: dict[str, float] = field(default_factory=dict)
    n_classes: int = 0
    n_labelled: int = 0
    n_unlabelled: int = 0

    def summary(self) -> str:
        """One block, with the verdict stated rather than left to the reader."""
        if self.advantage <= 0.02:
            verdict = "no disclosure beyond the prior"
        elif self.advantage < 0.2:
            verdict = "weak disclosure"
        elif self.advantage < 0.6:
            verdict = "substantial disclosure"
        else:
            verdict = "the attribute is largely determined"
        coverage = (
            self.n_labelled / (self.n_labelled + self.n_unlabelled)
            if (self.n_labelled + self.n_unlabelled)
            else 0.0
        )
        # The majority class is usually the best recovered and is the least
        # interesting: predicting it is what the prior already does. What
        # matters legally is a MINORITY value recovered well, because a rare
        # attribute value is more identifying than a common one.
        minority = {
            c: r for c, r in self.per_class_recall.items()
            if c != self.majority_class
        }
        exposed = (
            max(minority.items(), key=lambda kv: kv[1]) if minority else ("", 0.0)
        )
        return (
            f"{self.attribute}: {self.n_classes} classes, "
            f"{self.n_labelled} labelled ({coverage:.0%} of samples)\n"
            f"  accuracy {self.accuracy:.4f} against a prior of {self.prior:.4f} "
            f"({self.majority_class})\n"
            f"  balanced accuracy {self.balanced_accuracy:.4f}\n"
            f"  advantage {self.advantage:+.4f} -> {verdict}\n"
            f"  most exposed minority value: {exposed[0]} "
            f"at recall {exposed[1]:.4f}"
        )


def _softmax_regression(
    x: np.ndarray, y: np.ndarray, n_classes: int,
    l2: float = 1e-2, steps: int = 200, lr: float = 0.5,
) -> np.ndarray:
    """Multinomial logistic regression by gradient descent, numpy only.

    Deliberately simple. The core of this package must work with numpy alone,
    and the point of the measurement is whether an attribute is recoverable at
    all, which a linear probe answers. A stronger attacker would only raise the
    number, so a linear probe gives a **lower bound** on disclosure, which is
    the conservative direction for a privacy claim.

    Returns:
        Weight matrix of shape ``(n_features + 1, n_classes)``.
    """
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros((x.shape[1], n_classes))
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y] = 1.0
    for _ in range(steps):
        logits = x @ w
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        grad = x.T @ (p - onehot) / len(x) + l2 * w
        w -= lr * grad
    return w


def _predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (np.hstack([x, np.ones((len(x), 1))]) @ w).argmax(axis=1)


def attribute_disclosure(
    features: np.ndarray,
    labels: np.ndarray,
    attribute: str = "attribute",
    n_folds: int = 5,
    seed: int = 0,
    missing: tuple[str, ...] = ("", "?", "unknown", "none", "nan", "NA"),
) -> AttributeDisclosure:
    """Measure how much of ``labels`` is recoverable from ``features``.

    Groups are **not** accounted for: if several rows belong to one speaker and
    that speaker appears in both folds, the classifier can memorise the speaker
    rather than learn the attribute, and the number is inflated. Pass one row
    per speaker, or use :func:`attribute_disclosure_by_speaker`.

    Args:
        features: ``(n_samples, dim)`` features extracted from the released
            artifact, not from an internal representation.
        labels: ``(n_samples,)`` attribute values. Missing values are dropped
            and counted.
        attribute: Name, for reporting.
        n_folds: Cross-validation folds.
        seed: Shuffle seed.
        missing: Label values treated as absent.

    Returns:
        The :class:`AttributeDisclosure`.

    Raises:
        ValueError: If fewer than two classes survive, so there is nothing to
            predict.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError(
            f"{len(features)} features but {len(labels)} labels"
        )

    lowered = {m.lower() for m in missing}
    keep = np.array(
        [str(v).strip().lower() not in lowered for v in labels], dtype=bool
    )
    n_unlabelled = int((~keep).sum())
    x, y_raw = features[keep], labels[keep]

    classes = sorted({str(v) for v in y_raw})
    if len(classes) < 2:
        raise ValueError(
            f"attribute '{attribute}' has {len(classes)} class(es) after "
            "dropping missing labels, so there is nothing to predict"
        )
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[str(v)] for v in y_raw])

    # Standardise, or a feature with a large scale dominates the linear probe.
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    folds = np.array_split(order, n_folds)
    predicted = np.empty(len(x), dtype=int)
    for i in range(n_folds):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        if len(np.unique(y[train])) < 2:
            predicted[test] = np.bincount(y[train]).argmax()
            continue
        w = _softmax_regression(x[train], y[train], len(classes))
        predicted[test] = _predict(x[test], w)

    accuracy = float((predicted == y).mean())
    recalls = {}
    for c, i in index.items():
        mask = y == i
        recalls[c] = float((predicted[mask] == i).mean()) if mask.any() else 0.0
    balanced = float(np.mean(list(recalls.values())))

    counts = np.bincount(y, minlength=len(classes))
    majority = int(counts.argmax())
    prior = float(counts[majority] / len(y))
    headroom = 1.0 - prior
    advantage = float((accuracy - prior) / headroom) if headroom > 0 else 0.0

    return AttributeDisclosure(
        attribute=attribute,
        accuracy=accuracy,
        balanced_accuracy=balanced,
        prior=prior,
        majority_class=classes[majority],
        advantage=advantage,
        per_class_recall=recalls,
        n_classes=len(classes),
        n_labelled=int(len(y)),
        n_unlabelled=n_unlabelled,
    )


def attribute_disclosure_by_speaker(
    features: np.ndarray,
    labels: np.ndarray,
    speakers: np.ndarray,
    attribute: str = "attribute",
    n_folds: int = 5,
    seed: int = 0,
) -> AttributeDisclosure:
    """As :func:`attribute_disclosure`, but one row per speaker.

    Averages each speaker's features and takes their modal label, so a speaker
    cannot appear on both sides of a fold. Without this, a classifier that
    recognises the speaker scores well on the attribute for the wrong reason,
    and the measurement reports speaker identity as attribute leakage.

    Args:
        features: ``(n_samples, dim)``.
        labels: ``(n_samples,)`` attribute value per row.
        speakers: ``(n_samples,)`` speaker id per row.
        attribute: Name, for reporting.
        n_folds: Cross-validation folds.
        seed: Shuffle seed.

    Returns:
        The :class:`AttributeDisclosure`, computed over speakers.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    speakers = np.asarray(speakers, dtype=object)
    unique = sorted({str(s) for s in speakers})
    rows, values = [], []
    for s in unique:
        mask = np.array([str(v) == s for v in speakers], dtype=bool)
        rows.append(features[mask].mean(axis=0))
        vals, counts = np.unique(labels[mask].astype(str), return_counts=True)
        values.append(vals[counts.argmax()])
    return attribute_disclosure(
        np.stack(rows), np.array(values, dtype=object),
        attribute=attribute, n_folds=n_folds, seed=seed,
    )
