# The metrics

Definitions, the calibration rule, and how each maps onto the paper's equations.

Throughout, `s(·,·)` is cosine similarity between speaker embeddings, and an
embedding for a *conversation* of length `L` is the L2-normalised mean of the `L`
utterance embeddings. Single-utterance embeddings are left as the extractor
produced them, since it already normalises.

## Singling Out, `π_sing`

> *"it is not possible to single out an individual record"* (Opinion 05/2014)

Built on the predicate singling out (PSO) framework of Cohen and Nissim. The
attacker holds one enrollment embedding `x_enroll` and a threshold, defining

```
p(x_test) = 1{ s(x_test, x_enroll) > s_thresh }
```

Given a set `X` of `N` test embeddings, one per speaker, **isolation succeeds
when the predicate fires on exactly one of them**, whether or not it is the
attacker's own speaker. That last clause matters: singling out is about carving
one record out of a crowd, not about identifying whose record it is.

```
π_sing = Pr_{X, x_enroll} { ∃! i : p(x_test_i) = 1 }
```

### The calibration rule

The threshold is not chosen, it is calibrated, and this is what makes the number
interpretable. It is set so the predicate fires on `1/N` of a held-out
calibration set:

```
E_{x_calib} { p(x_calib) } = 1/N
```

With `M` calibration conversations per speaker and `N` speakers there are `M·N`
calibration scores, of which `M` must pass. So the threshold goes midway between
the `M`-th and `(M+1)`-th highest calibration scores. The paper uses `M = 9`,
giving the "average of the 9th and 10th similarity scores".

A predicate with expectation `1/N` that carries *no* information about the data
isolates with probability

```
N · (1/N) · (1 - 1/N)^(N-1)  ⟶  exp(-1) ≈ 0.368
```

which is the trivial baseline every Singling Out curve is read against. A system
is PSO-secure when the measured rate does not significantly exceed it.
`tests/test_singling_out.py::test_trivial_attacker_isolates_at_exp_minus_one`
asserts the implementation lands on this value.

Speakers with fewer than `2L` utterances cannot supply both a test and a
calibration conversation and are excluded, which is why the largest usable `N`
falls below the full population at `L = 30`.

### Data layout

Per fold, each test speaker's utterances are shuffled; the first `L` become its
test conversation, and up to `M = min(9, C(K-L, L))` calibration conversations
are taken from the remaining `K-L` utterances. Test and calibration conversations
of a speaker therefore never share an utterance.

## Linkability, `π_link`

> *"it is not possible to link records relating to the same individual"* (Opinion 05/2014)

Linkage succeeds when the true enrollment speaker outscores every other candidate:

```
π_link = Pr { s(x_i_test, x_i_enroll) > max_{j≠i} s(x_i_test, x_j_enroll) }
```

The candidate set is the true speaker plus `N'-1` others drawn without
replacement from the enrollment population. Chance level is `1/N'`.

> This is **not** the linkability metric of Gomez-Barrero et al., and not
> Sidekit's `Dsys`. Those measure the separation of mated and nonmated score
> distributions; this one is a closed-set identification rate. The paper says so
> in a footnote, and early versions of the internal code used the other
> definition. See [provenance.md](provenance.md).

### Why the fast path is exact

The literal implementation draws a candidate set per test speaker, per `N'`, per
run. On the paper's grid that is 220 × 5 × 4,949 draws of up to 22,023 indices,
and it dominates the runtime.

It is avoidable. Linkage succeeds exactly when none of the sampled candidates
beats the true speaker, so the only thing that matters about test speaker `i` is

```
m_i = #{ j ≠ i : s(x_i_test, x_j_enroll) > s(x_i_test, x_i_enroll) }
```

the number of enrollment speakers that outscore the true one. Drawing `N'-1` of
the `n-1` others without replacement and asking whether any of those `m_i`
beaters came along *is* a hypergeometric experiment:

```
Pr{ success } = C(n-1-m_i, N'-1) / C(n-1, N'-1)
```

So one `Hypergeometric(m_i, n-1-m_i, N'-1)` draw per test speaker replaces the
whole subset construction, with the identical sampling distribution. `m_i` is
computed once and reused across the sweep. The sweep drops from hours to seconds.

Two estimators are exposed:

- `estimator="sampling"` (default) draws as above, reproducing the original
  experiment's sampling behaviour including its run-to-run spread.
- `estimator="exact"` returns the closed-form probability. Same expectation, no
  sampling noise, one value instead of `n_runs`. Useful when the error bar is
  measuring the estimator rather than anything about the system.

`tests/test_linkability.py` asserts both against a literal subset-sampling
implementation.

### Ties

The original code decided linkage with `np.argmax` over `[true, *others]`, which
returns the first maximum and so resolves a tie in the true speaker's favour.
That convention is kept: only *strictly* higher scores count as beaters. With
float cosine scores ties are vanishingly rare, but the choice is deliberate
rather than incidental.

## ROCCH-EER

The equal error rate read off the convex hull of the ROC, rather than the raw
empirical EER. The hull is the set of operating points reachable by randomised
mixtures of two thresholds, so the ROCCH-EER is the error rate of the best
calibrated decision rule those scores support. It removes the dependence on score
quantisation, which is the right baseline for a privacy claim.

The paper plots `1 - EER` so that, like the other two metrics, higher means more
risk.

This is an independent implementation: the ROC is traced from the sorted scores,
its lower-left convex hull is taken by monotone chain, and the crossing with
`Pmiss = Pfa` is interpolated. The BOSARIS toolkit the original experiments used
through Sidekit is licensed for non-commercial use only and is not vendored here.
`tests/test_eer.py` pins it against a brute-force search over randomised rules and
against the analytic EER of equal-variance Gaussians, `Φ(-d/2)`.

### The EER sweep

Target trials are each test item against its own enrollment speaker; nontarget
trials are against others from the candidate set. The number of nontarget trials
per test item is capped (default 500) because the EER depends on the
*distribution* of nontarget scores, not on how many were drawn, and that
distribution does not change with `N'`. The paper's own finding that the EER is
nearly flat in `N'` is this observation. The cap keeps large `N'` affordable
without biasing the estimate.

## Attacker models

From Srivastava et al. (2020), as used in the Voice Privacy Challenges. Note the
naming differs from the VPC:

| This paper | Knowledge | Extractor trained on | VPC name |
|---|---|---|---|
| **Ignorant** | Unaware data is anonymized | Original speech | Ignorant |
| **Semi-Informed** | Aware, but not of the exact system | A *similar* system's output (VPC 2022 B1.a) | no equivalent |
| **Informed** | Full knowledge and access | The *same* system's output (VPC 2024 B1) | Semi-Informed |
| *Original* | n/a | Original speech, tested on original speech | n/a |

The *Informed* attacker is the worst case and is what a conservative compliance
argument should be built on.
