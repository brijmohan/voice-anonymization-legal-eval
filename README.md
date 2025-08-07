# voice-anonymization-legal-eval

A small Python library implementing two privacy metrics for anonymized speech
embeddings:

- **Singling Out** – probability that a predicate isolates exactly one speaker.
- **Linkability** – probability that anonymized utterances can be correctly
  linked to enrollment recordings.

The implementation follows the definitions from *Legally validated evaluation
framework for voice anonymization*.

## Testing

Tests require `pytest` and `numpy`.
Run them with:

```bash
python -m pytest -q
```

If the dependencies are missing, install them via `pip install numpy pytest`.
