#!/usr/bin/env python3
"""Build the reference library from one or more BibTeX files.

The library is a derived artifact, so it needs a generator: a categorised
bibliography maintained by hand drifts from the sources it was built from, and
nothing says so. Run this and the two files in docs/ are rebuilt.

The taxonomy follows the shape of this research rather than a generic topic
list: the legal frame, the quantities, the evaluation design, the systems, what
leaks regardless of the system, the components, the data, and the adjacent
literature. Rules are ordered and the first match wins, so the specific sits
above the general. A fallback at the end catches anything that matches only the
word "anonymization", and it is named for what it is: an entry landing there is
a prompt to add a family, not a category.

Inputs are given on the command line, so this is usable by anybody with their
own bibliographies. The ones behind the committed output are a doctoral thesis
bibliography and the bibliography of a paper in preparation, neither of which
is public, which is why the output is committed and the inputs are not.

Two things the rules get wrong if you let them. BibTeX braces have to be
stripped before matching, because "Privacy and Utility of {X}-Vector" does not
contain the substring "of x-vector". And a topic filter that drops entries by
subject has to let the privacy core override it, or a paper on anonymizing
code-switching speech is discarded along with the code-switching ASR work.

Usage:
    python scripts/build_bibliography.py \\
        --bib ../thesis/References/references.bib \\
        --bib ../paper/mybib.bib \\
        --keep-whole ../paper/mybib.bib \\
        --out-dir docs
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

SECTIONS = [
    ("A", "The legal frame", [
        ("legal-instruments", "Instruments and regulatory guidance"),
        ("legal-formalization", "Formalizing the legal criteria"),
        ("privacy-theory", "What privacy is, and comparative law"),
    ]),
    ("B", "Measuring privacy", [
        ("metrics-verification", "The verification family"),
        ("metrics-population", "Population and ranking metrics"),
        ("metrics-worstcase", "Worst case and per individual"),
        ("metrics-comparison", "Comparing metrics against each other"),
        ("math-methods", "Mathematical machinery"),
    ]),
    ("C", "Evaluation design, and its critics", [
        ("evaluation-critique", "Critiques of how privacy is evaluated"),
        ("attacker-models", "Attacker models and informed attacks"),
        ("challenges", "Challenges, evaluation plans and results"),
    ]),
    ("D", "Anonymization systems", [
        ("systems-signal", "Signal processing"),
        ("systems-xvector", "Speaker embedding replacement"),
        ("systems-bottleneck", "Phonetic and quantized bottlenecks"),
        ("systems-tts-vc", "Voice conversion and synthesis"),
        ("systems-codec", "Neural audio codec and language models"),
        ("systems-adversarial", "Disentangled, adversarial and invariant representations"),
        ("systems-early", "De-identification before the challenge era"),
        ("systems-other", "Anonymization, family not yet assigned"),
    ]),
    ("E", "What leaks anyway", [
        ("residual-channels", "What survives the voiceprint"),
        ("content-privacy", "Verbal content and metadata"),
        ("attribute-privacy", "Attribute inference, the third criterion"),
        ("pseudo-speaker", "Pseudo-speaker identifiability"),
    ]),
    ("F", "Components and background", [
        ("speaker-recognition", "Speaker recognition"),
        ("speech-generation", "Speech generation"),
        ("asr", "Recognition used for utility"),
    ]),
    ("G", "Data", [("datasets", "Corpora")]),
    ("H", "Adjacent literature and context", [
        ("sdc", "Statistical disclosure control and the k-anonymity family"),
        ("dp-adjacent", "Differential privacy"),
        ("crypto", "Encrypted and distributed speech processing"),
        ("other-modalities", "Anonymization in other modalities"),
        ("deployment-context", "Deployment context and user perceptions"),
    ]),
    ("Z", "Unclassified", [("unclassified", "Not yet placed")]),
]
RULES = [
    # A. Pinned first: this project's own publications.
    ("A", "legal-formalization",
     r"legally validated evaluation framework|singling out|formaliz\w+ the gdpr"
     r"|predicate singling|formal guarantee|unifying re-identification"
     r"|legal (notion|definition)"),
    ("A", "legal-instruments",
     r"gdpr|general data protection|article 29|working party|opinion 05/2014"
     r"|edpb|cnil|iso/iec|eprivacy|data protection (law|board|directive|act)"
     r"|anonymisation (code|guidance)|legal guarantee|right to be forgotten"),
    ("A", "privacy-theory",
     r"right to privacy|what is privacy|history and definition of privacy"
     r"|privacy law in|global trends in privacy|fundamental right"
     r"|privacy protection.*survey"),
    # B. The quantities.
    ("B", "metrics-comparison",
     r"comparative study of speech anonymization metrics|comparison of .*metrics"
     r"|metric commensurab"),
    ("B", "metrics-worstcase",
     r"zebra|zero.evidence|worst.case|per.speaker analysis|individual(ised)? risk"
     r"|privacy log.odds"),
    ("B", "metrics-population",
     r"linkabilit|unlinkabilit|similarity rank|rank disclosure"
     r"|biometric (template )?protection (schemes? )?evaluat"),
    ("B", "metrics-verification",
     r"\bcllr\b|calibrat|likelihood.ratio cost|equal error|\bdet\b curve"
     r"|convex hull|application.independent|bayes error"
     r"|evaluation of (speaker|biometric) (recognition|verification)"),
    ("B", "math-methods",
     r"isotonic|pool adjacent|hypergeometric|bootstrap|affinity propagation"
     r"|clustering by passing messages|mean opinion score"),
    # C. How the evaluation is built, and who objects.
    ("C", "challenges",
     r"voiceprivacy|voice privacy challenge|challenge (evaluation )?plan"
     r"|introducing the voiceprivacy|\bspsc\b|symposium on security and privacy in speech"
     r"|voice conversion challenge|multi.target speaker detection"),
    ("C", "evaluation-critique",
     r"false sense of privacy|overestimat|evaluating and designing|reality or illusion"
     r"|pitfall|a critical|limitations of|benchmark for multi.speaker|re.?think"),
    ("C", "attacker-models",
     r"informed attacker|ignorant attacker|semi.informed|lazy.informed"
     r"|attack(er)? model|against informed|threat model|attack landscape"),
    # D. The systems, most specific first.
    ("D", "systems-codec",
     r"audio codec|soundstream|encodec|codec language model|streaming anonymi"
     r"|speechtokenizer"),
    ("D", "systems-bottleneck",
     r"bottleneck|phonetic representation|\basr.bn\b|vector.quantiz|neural discrete"
     r"|porcupine|quantiz\w+ bottleneck"),
    ("D", "systems-xvector",
     r"x.?vector based (speaker )?anonymi|design choices for x.?vector"
     r"|privacy and utility of x.?vector|speaker anonymi\w+ (using|with|via) "
     r".*(embedding|vector)|pseudo.?speaker"),
    ("D", "systems-signal",
     r"mcadams|vtln|vocal tract length|formant|pitch shift|spectral warp"
     r"|signal.processing.based|linear prediction|voice.indistinguishability"),
    ("D", "systems-adversarial",
     r"adversarial (representation|training|neural|regulariz|disentangl)"
     r"|domain.adversarial|speaker.invariant|invariant (training|feature)"
     r"|anonymized representations|information obfuscat|generative adversarial privacy"
     r"|disentangled and interpretable|learning of disentangled"),
    ("D", "systems-early",
     r"de.?identification|voicemask|hidebehind|voice transformation"
     r"|diphone recognition|reversible speaker|autoencoders and adversarial"
     r"|speech data publishing|urban sound recordings"),
    ("D", "systems-tts-vc",
     r"voice conversion|knn.?vc|nearest neighbour|synthesis based|tts.based"
     r"|serial disentangl|\bmusa\b|cascaded|diffusion|ddpo"),
    # E. What leaks regardless of the system.
    ("E", "content-privacy",
     r"content privacy|content masking|content anonymi|authorship attribution"
     r"|speech transcript|named entity|privacy.transformed|content style"
     r"|spoken content|character.level vocoder|phonetic search|text representation"
     r"|i am not what i write"),
    ("E", "attribute-privacy",
     r"attribute|soft biometric|emotion|age estimation|gender (recognition|classif)"
     r"|health|patholog|dysarthria|depression|personal attribute leakage"
     r"|paralinguistic|affective|speech characterisation|speech characterization"),
    ("E", "residual-channels",
     r"prosody|vocoder drift|rhythm|intonation|speaking (style|rate)|accent"
     r"|is not identity|intelligibilit"),
    # F. Components.
    ("F", "speaker-recognition",
     r"speaker (verification|recognition|identification|embedding|diariz|detection)"
     r"|x.?vector|i.?vector|ecapa|plda|time delay neural|d.?vector"
     r"|wav2vec|hubert|wavlm|self.supervised|voiceprint biometric|biometric identification"),
    ("F", "speech-generation",
     r"vocoder|hifi.?gan|bigvgan|wavenet|wavernn|melgan|tacotron|fastspeech"
     r"|source.filter|glow|neural (speech )?synthesis|text.to.speech|festival"
     r"|cheaptrick|speech synthesis|grapheme to phoneme|bandwidth extension"
     r"|speech enhancement|spectral envelope|\bgelp\b|acoustic scene synthesis"),
    ("F", "asr",
     r"speech recognition|acoustic model|language model|whisper|\bctc\b|transducer"
     r"|word error|lattice|kaldi|end.to.end|analysis window duration"),
    # G, H.
    ("G", "datasets",
     r"corpus|corpora|database|dataset|librispeech|libritts|common voice"
     r"|voxceleb|vctk|iemocap|voxpopuli|switchboard|timit|found data"),
    ("H", "sdc",
     r"k.?anonym|l.?diversity|t.?closeness|statistical disclosure"
     r"|data publishing|medical records|anonymized network traces|big data privacy"),
    ("H", "dp-adjacent",
     r"differential(ly)? privat|differential privacy|local privacy|minimax"
     r"|membership (inference|privacy)|reconstruction attack|synthetic data"),
    ("H", "crypto",
     r"homomorphic|encrypted|secure (and private|computation|multiparty)"
     r"|voiceguard|privacy.preserving (nearest neighbor|machine learning)"
     r"|federated"),
    ("H", "other-modalities",
     r"face anonymization|deepprivacy|medical image|image synthesis"),
    ("H", "deployment-context",
     r"smart (speaker|home)|alexa|amazon echo|personal assistant"
     r"|privacy (perception|attitude|concern)|usable privacy"),
    # Last resort, and named for what it is. Anything still matching only the
    # word "anonymization" is an anonymization system this taxonomy has not
    # given a family to, which is a prompt to add one rather than a category.
    ("D", "systems-other",
     r"anonymi|de.?identif|privacy.preserving|privacy aware|protecting privacy"),
]
#: The working set: what this project actually rests on, matched on title so a
#: key rename cannot silently drop a note. Each note says what the work
#: establishes and where this repository uses it.
WORKING_SET = [
    (r"opinion 05/2014", "The three conjunctive criteria the whole framework implements. Singling out, linkability and inference, and inference means attribute values rather than membership."),
    (r"formalizing the gdpr", "Predicate singling out, the formal basis of our Singling Out metric. Also proves differential privacy implies PSO security while k-anonymity does not, which is why the k-anonymity literature in section H is background rather than a solution."),
    (r"opinion 28/2024", "A regulator's anonymity test for models rather than datasets: extraction probability negligible, query risk insignificant. The closest thing to a threshold the field has been given."),
    (r"unifying re-identification", "Puts re-identification, attribute inference and reconstruction on one footing under differential privacy. Directly relevant to reporting our three criteria together."),
    (r"biometric information protection", "The standards family ZEBRA was already offered to, which is the route a voice anonymization metric would travel."),
    (r"privacy enhancing data de-identification framework", "The existing de-identification standard. Any proposal of ours has to say how it relates to this."),
    (r"legally validated evaluation framework", "This project's own paper. Singling Out and Linkability, and the finding that the EER is close to blind to residual re-identification risk."),
    (r"false sense of privacy", "The methodological critique that settled our population question: evaluation overestimates privacy by assuming a weak adversary. Their recommendation to reduce the number of identities, against our own measurements, produced finding 3.12."),
    (r"privacy zebra", "The worst-case metric. The only quantity in the field about one individual rather than an average, and it reached the standardisation argument six years before we did."),
    (r"comparative study of speech anonymization metrics", "The precursor to our metric commensurability work. Worth reading before claiming novelty."),
    (r"similarity rank disclosure", "A 2026 rank-based proposal, one of the four competing replacements for the EER that motivated the comparison."),
    (r"per-speaker analysis of re-identification", "Average-case metrics hide large disparities between individuals. The empirical case for reporting a worst case alongside a mean."),
    (r"attribute-driven privacy", "Prior art for the third criterion: attribute concealment measured with a classifier, which is the method our inference measure follows."),
    (r"porcupine", "Vector-quantized bottlenecks leak attributes through codebook occurrence. Relevant because the ASR-BN baseline we evaluate has exactly such a bottleneck."),
    (r"prosody is not identity", "Prosody is identity-bearing and must be perturbed rather than passed through. One of the residual channels an EER-based evaluation does not see."),
    (r"vocoder drift", "The vocoder does not faithfully render the embedding it is given, and the error is reversible. Why the lazy-informed and semi-informed conditions differ."),
    (r"neural audio codec language models", "The origin of the VPC 2026 B4 baseline, which is the one system we have not yet been able to afford to evaluate."),
    (r"introducing the voiceprivacy", "The origin of the attacker taxonomy we inherit, ignorant, lazy-informed and semi-informed, and of the enrollment and trial split."),
    (r"voiceprivacy 2024 challenge evaluation plan", "The evaluation plan our dataset decisions are inherited from."),
    (r"third voiceprivacy", "The 2026 challenge, whose baselines B2 to B5 are the systems under test here."),
    (r"voiceprivacy 2026 challenge evaluation plan", "The plan our conformance audit is against: Track 1 baselines, the WavLM-ECAPA attacker, and mixed-gender trials."),
    (r"anonymizing speech: evaluating and designing", "A thesis whose subject is the evaluation design itself rather than a new anonymizer. The closest prior work to this project's aim."),
    (r"mcadams", "The B2 baseline, and the system our CV27 measurements show does not move any criterion."),
    (r"common voice", "The corpus the CV27 subsets are drawn from. Read the datasheet before using the gender labels."),
    (r"librispeech", "The corpus the field evaluates on, and the reason two baselines failed on Common Voice: read speech contains no clip with zero voiced frames."),
    (r"ecapa", "The attacker's embedding extractor in the 2026 challenge, as WavLM-ECAPA. Every number this project reports about that challenge is conditional on this model."),
    (r"content anonymization for privacy in long-form", "Verbal content as an identifying channel, which is the assumption obligation G1 flags as not safe to take for granted."),
    (r"personal attribute leakage in federated", "Gender, age, accent, emotion and dysarthria inferred without access to raw speech. The breadth of the third criterion in one paper."),
    (r"authorship attribution models distinguish speakers", "Speakers identifiable from transcripts alone, which is the G1 channel measured directly."),
]

#: Section titles, derived so there is one ordered source of truth. Defining
#: SECTIONS twice, once ordered and once as a lookup, is how the two fell out
#: of step while this was being written.
SECTION_TITLE = {letter: title for letter, title, _ in SECTIONS}
SUBSECTION_TITLE = {sub: title
                    for _, _, subs in SECTIONS for sub, title in subs}

#: Relevant to voice privacy. Broad on purpose: curation is the taxonomy's job,
#: not this filter's.
KEEP = re.compile(
    r"anonym|privacy|private|de-?identif|pseudonym|speaker (verif|recogni|embedd|identif)"
    r"|x-?vector|i-?vector|ecapa|voice conversion|speech synthesis|vocoder|wavenet|hifi-?gan"
    r"|bottleneck|disentangl|adversarial|gdpr|legal|regulat|data protection"
    r"|calibrat|likelihood ratio|equal error|linkab|zebra|cllr|biometric|re-?identif"
    r"|differential privacy|k-anonymity|membership inference|voiceprivacy|voice privacy"
    r"|attribute|emotion|prosody|utility|intelligib|asv|tts", re.I)

#: Other research that shares a bibliography with this one but not a subject.
DROP = re.compile(
    r"code.?switch|low.?resource|indian language|phone merg|homophone|pronunciation"
    r"|hindi|tamil|telugu|gujarati|marathi|bengali|articulatory gesture", re.I)

#: The privacy core, which overrides a subject drop. Without this, a 2025 paper
#: on anonymizing code-switching speech is discarded with the ASR work.
PRIVACY = re.compile(r"anonym|privacy|private|de-?identif|pseudonym|re-?identif", re.I)

#: Corpora and toolkits, which the topic filter would otherwise discard as off
#: subject while the library needs them as a category.
CORPUS = re.compile(
    r"corpus|corpora|database|dataset|librispeech|libritts|common voice|voxceleb"
    r"|vctk|iemocap|voxpopuli|challenge|toolkit|kaldi|speechbrain|wav2vec|hubert"
    r"|wavlm|whisper", re.I)

#: Titles too malformed to place.
UNUSABLE = {"icassp 2025"}

#: Entries added by hand, because they were missing from every
#: bibliography this project had. They are embedded rather than taken as
#: another input file so that the generator can reproduce its own output
#: given only the private sources. Each one was checked against the
#: publisher or the preprint server; where a field could not be confirmed
#: the entry says so instead of guessing.
ADDED = r"""
@techreport{art29wp2014opinion05,
  title = {Opinion 05/2014 on Anonymisation Techniques},
  author = {{Article 29 Data Protection Working Party}},
  institution = {European Commission},
  number = {WP216},
  year = {2014},
  month = apr,
  keywords = {A/legal-instruments},
  note = {The three criteria this framework implements: singling out, linkability, inference}
}

@techreport{edpb2024opinion28,
  title = {Opinion 28/2024 on Certain Data Protection Aspects Related to the Processing of Personal Data in the Context of {AI} Models},
  author = {{European Data Protection Board}},
  institution = {European Data Protection Board},
  number = {28/2024},
  year = {2024},
  month = dec,
  keywords = {A/legal-instruments}
}

@techreport{iso27559,
  title = {{ISO/IEC} 27559:2022 Information Security, Cybersecurity and Privacy Protection: Privacy Enhancing Data De-identification Framework},
  author = {{ISO/IEC}},
  institution = {International Organization for Standardization},
  year = {2022},
  keywords = {A/legal-instruments}
}

@techreport{iso20889,
  title = {{ISO/IEC} 20889:2018 Privacy Enhancing Data De-identification Terminology and Classification of Techniques},
  author = {{ISO/IEC}},
  institution = {International Organization for Standardization},
  year = {2018},
  keywords = {A/legal-instruments}
}

@techreport{iso24745,
  title = {{ISO/IEC} 24745 Information Security, Cybersecurity and Privacy Protection: Biometric Information Protection},
  author = {{ISO/IEC}},
  institution = {International Organization for Standardization},
  year = {2022},
  keywords = {A/legal-instruments},
  note = {The standards family ZEBRA was offered for incorporation into}
}

@article{cohen2020singlingout,
  title = {Towards Formalizing the {GDPR}'s Notion of Singling Out},
  author = {Cohen, Aloni and Nissim, Kobbi},
  journal = {Proceedings of the National Academy of Sciences},
  year = {2020},
  archiveprefix = {arXiv},
  eprint = {1904.06009},
  keywords = {A/legal-formalization}
}

@article{khalili2025unifying,
  title = {Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy},
  journal = {arXiv preprint},
  year = {2025},
  archiveprefix = {arXiv},
  eprint = {2507.06969},
  keywords = {A/legal-formalization}
}

@article{hanisch2024falsesense,
  title = {A False Sense of Privacy: Towards a Reliable Evaluation Methodology for the Anonymization of Biometric Data},
  author = {Hanisch, Simon and Todt, Julian and Patino, Jose and Evans, Nicholas and Strufe, Thorsten},
  journal = {Proceedings on Privacy Enhancing Technologies},
  volume = {2024},
  number = {1},
  pages = {116--132},
  year = {2024},
  doi = {10.56553/popets-2024-0008},
  archiveprefix = {arXiv},
  eprint = {2304.01635},
  keywords = {C/evaluation-critique}
}

@inproceedings{nautsch2020zebra,
  title = {The Privacy {ZEBRA}: Zero Evidence Biometric Recognition Assessment},
  author = {Nautsch, Andreas and Patino, Jose and Tomashenko, Natalia and Yamagishi, Junichi and No{\'e}, Paul-Gauthier and Bonastre, Jean-Fran{\c c}ois and Todisco, Massimiliano and Evans, Nicholas},
  booktitle = {Interspeech 2020},
  year = {2020},
  doi = {10.21437/Interspeech.2020-1815},
  archiveprefix = {arXiv},
  eprint = {2005.09413},
  keywords = {B/metrics-worstcase}
}

@inproceedings{noe2021attributedriven,
  title = {Adversarial Disentanglement of Speaker Representation for Attribute-Driven Privacy Preservation},
  author = {No{\'e}, Paul-Gauthier and Mohammadamini, Mohammad and Matrouf, Driss and Parcollet, Titouan and Nautsch, Andreas and Bonastre, Jean-Fran{\c c}ois},
  booktitle = {Interspeech 2021},
  year = {2021},
  doi = {10.21437/interspeech.2021-1712},
  keywords = {E/attribute-privacy}
}

@inproceedings{vali2024porcupine,
  title = {Privacy {PORCUPINE}: Anonymization of Speaker Attributes Using Occurrence Normalization for Space-Filling Vector Quantization},
  author = {Vali, Mohammad Hassan and B{\"a}ckstr{\"o}m, Tom},
  booktitle = {Interspeech 2024},
  year = {2024},
  doi = {10.21437/interspeech.2024-117},
  keywords = {E/attribute-privacy}
}

@inproceedings{meyer2023prosody,
  title = {Prosody Is Not Identity: A Speaker Anonymization Approach Using Prosody Cloning},
  author = {Meyer, Sarina and Lux, Florian and Koch, Julia and Denisov, Pavel and Tilli, Pascal and Vu, Ngoc Thang},
  booktitle = {ICASSP 2023},
  year = {2023},
  doi = {10.1109/icassp49357.2023.10096607},
  keywords = {E/residual-channels}
}

@inproceedings{panariello2023vocoderdrift,
  title = {Vocoder Drift in x-Vector-Based Speaker Anonymization},
  author = {Panariello, Michele and Todisco, Massimiliano and Evans, Nicholas},
  booktitle = {Interspeech 2023},
  year = {2023},
  doi = {10.21437/interspeech.2023-448},
  keywords = {E/residual-channels}
}

@article{panariello2024nac,
  title = {Speaker Anonymization Using Neural Audio Codec Language Models},
  author = {Panariello, Michele and Tavares Nogueira, Francesco and Todisco, Massimiliano and Evans, Nicholas},
  journal = {ICASSP 2024},
  year = {2024},
  archiveprefix = {arXiv},
  eprint = {2309.14129},
  keywords = {D/systems-codec},
  note = {The origin of the VPC 2026 B4 baseline}
}

@article{chandra2026srd,
  title = {Similarity Rank Disclosure},
  author = {Chandra and Petten{\`o} and Evans, Nicholas and Panariello, Michele and Todisco, Massimiliano},
  journal = {arXiv preprint},
  year = {2026},
  archiveprefix = {arXiv},
  eprint = {2605.07291},
  keywords = {B/metrics-population},
  note = {Author list and title to confirm against the published version}
}

@article{dufour2026perspeaker,
  title = {A Large-Scale Per-Speaker Analysis of Re-identification Risk in Speech Anonymization},
  author = {Dufour and Magron and Rouvier and Vincent, Emmanuel},
  journal = {Interspeech 2026},
  year = {2026},
  archiveprefix = {arXiv},
  eprint = {2606.07210},
  keywords = {B/metrics-worstcase},
  note = {Author initials to confirm against the published version}
}

@article{tomashenko2026third,
  title = {The Third {VoicePrivacy} Challenge},
  author = {Tomashenko, Natalia and Miao, Xiaoxiao and Champion, Pierre and Meyer, Sarina and Panariello, Michele},
  journal = {arXiv preprint},
  year = {2026},
  archiveprefix = {arXiv},
  eprint = {2601.11846},
  keywords = {C/challenges}
}

@article{miao2024multispeaker,
  title = {A Benchmark for Multi-speaker Anonymization},
  author = {Miao, Xiaoxiao and Tao and Zeng and Wang},
  journal = {arXiv preprint},
  year = {2024},
  archiveprefix = {arXiv},
  eprint = {2407.05608},
  keywords = {C/evaluation-critique},
  note = {Author initials to confirm against the published version}
}

@article{attributeleakage2025federated,
  title = {Personal Attribute Leakage in Federated Speech Models},
  journal = {arXiv preprint},
  year = {2025},
  archiveprefix = {arXiv},
  eprint = {2510.13357},
  keywords = {E/attribute-privacy},
  note = {Gender, age, accent, emotion and dysarthria inferred from weight differentials}
}

@article{spokencontent2024vocoder,
  title = {Preserving Spoken Content in Voice Anonymisation with Character-Level Vocoder Conditioning},
  journal = {arXiv preprint},
  year = {2024},
  archiveprefix = {arXiv},
  eprint = {2408.04306},
  keywords = {E/content-privacy}
}

@article{musa2024,
  title = {{MUSA}: Multi-lingual Speaker Anonymization via Serial Disentanglement},
  journal = {arXiv preprint},
  year = {2024},
  archiveprefix = {arXiv},
  eprint = {2407.11629},
  keywords = {D/systems-tts-vc}
}

@article{ddpovc2026,
  title = {{DDPO-VC}: Speaker De-Identification via Diffusion Denoising Policy Optimization},
  journal = {arXiv preprint},
  year = {2026},
  archiveprefix = {arXiv},
  eprint = {2606.15313},
  keywords = {D/systems-tts-vc}
}

@inproceedings{meyer2025codeswitching,
  title = {First Steps Towards Voice Anonymization for Code-Switching Speech},
  author = {Meyer, Sarina},
  booktitle = {Interspeech 2025},
  year = {2025},
  keywords = {D/systems-tts-vc},
  note = {Full author list to confirm}
}

@techreport{vpc2026plan,
  title = {The {VoicePrivacy} 2026 Challenge Evaluation Plan},
  author = {{VoicePrivacy Challenge organisers}},
  institution = {VoicePrivacy Challenge},
  year = {2026},
  month = sep,
  url = {https://www.voiceprivacychallenge.org/vp2026/docs/VPC_2026_sept01.pdf},
  keywords = {C/challenges}
}
"""

FIELDS = ("title", "author", "booktitle", "journal", "school", "institution",
          "publisher", "volume", "number", "pages", "year", "month", "doi",
          "url", "archiveprefix", "eprint", "keywords", "note")


def flatten(s: str) -> str:
    """Strip BibTeX braces and LaTeX escapes so patterns can match."""
    s = re.sub(r"[{}]", "", s or "")
    s = re.sub(r"\\[a-zA-Z]+\{?", " ", s)
    s = re.sub(r"[$\\]", "", s)
    return " ".join(s.split())


def parse_bib(text: str) -> list[dict]:
    """Every entry in a BibTeX file, as a dict of the fields we use."""
    out = []
    for block in re.split(r"\n(?=@)", text):
        m = re.match(r"@(\w+)\s*\{\s*([^,]+),", block.strip(), re.I)
        if not m:
            continue
        e = {"type": m.group(1).lower(), "key": m.group(2).strip()}
        for field in FIELDS:
            fm = re.search(field + r"\s*=\s*(?:[{](.*?)[}]|[\"](.*?)[\"]|([a-z]+))\s*,?\s*\n",
                           block, re.I | re.S)
            if fm:
                value = fm.group(1) or fm.group(2) or fm.group(3) or ""
                e[field] = " ".join(value.split())
        out.append(e)
    return out


def norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


#: A key that is mostly digits is a repository identifier rather than a name.
#: Both describe the same work, and a reader citing from this library wants
#: srivastava2022taslp rather than srivastavahal03197376.
OPAQUE_KEY = re.compile(r"\d{5,}")


def combine(kept: dict, other: dict) -> dict:
    """Merge two records of the same work.

    Picking one and discarding the other loses fields that only the discarded
    record had, which is how a DOI goes missing from an entry that had one.

    Args:
        kept: The record already held.
        other: A second record of the same title.

    Returns:
        One record carrying every field either had, preferring a hand checked
        source for a conflict, and the more readable of the two keys.
    """
    hand = "added by hand"
    primary, secondary = ((other, kept) if other.get("source") == hand
                          else (kept, other))
    out = dict(secondary)
    out.update({k: v for k, v in primary.items() if v})
    # The key is chosen on readability, not on which record won.
    keys = [k for k in (kept["key"], other["key"]) if k]
    readable = [k for k in keys if not OPAQUE_KEY.search(k)]
    out["key"] = (readable or keys)[0]
    return out


def classify(entry: dict) -> tuple[str, str]:
    """The one place in the taxonomy this entry belongs."""
    blob = " ".join([entry.get("title_clean", ""), entry.get("venue", ""),
                     flatten(entry.get("note", "")), entry.get("key", "")]).lower()
    for section, sub, pattern in RULES:
        if re.search(pattern, blob, re.I):
            return section, sub
    return "Z", "unclassified"


def relevant(entry: dict) -> bool:
    """Whether a filtered source's entry belongs to this body of research."""
    blob = " ".join(str(entry.get(k, "")) for k in
                    ("title", "booktitle", "journal", "note"))
    if DROP.search(blob) and not PRIVACY.search(blob):
        return False
    return bool(KEEP.search(blob) or CORPUS.search(blob))


def short_authors(author: str | None) -> str:
    """Surnames only, for the index. The bib keeps the full list."""
    if not author:
        return "Anon."
    author = re.sub(r"[{}\\]", "", author)
    parts = [a.strip() for a in re.split(r"\s+and\s+", author) if a.strip()]
    surnames = [a.split(",")[0].strip() if "," in a else a.split()[-1]
                for a in parts]
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]}"
    return f"{surnames[0]} et al."


def identifier(entry: dict, prefix: str = ", ") -> str:
    if entry.get("doi"):
        return f"{prefix}doi:{entry['doi']}"
    if entry.get("eprint"):
        return f"{prefix}arXiv:{entry['eprint']}"
    return ""


def collect(sources: list[tuple[Path, bool]]) -> list[dict]:
    """Parse, filter, dedupe and classify every source.

    Args:
        sources: ``(path, keep_whole)`` pairs. A source kept whole is not
            subject to the relevance filter, which is right for the
            bibliography of a paper that is itself part of this research.

    Returns:
        Classified entries with unique keys.
    """
    merged: dict[str, dict] = {}
    readable: list[tuple[str, str, bool]] = [
        (p.name, p.read_text(encoding="utf-8", errors="replace"), whole)
        for p, whole in sources
    ]
    # Last, so a hand checked entry wins the dedupe against a terser copy.
    readable.append(("added by hand", ADDED, True))
    for origin, text, keep_whole in readable:
        for e in parse_bib(text):
            if not keep_whole and not relevant(e):
                continue
            e["title_clean"] = flatten(e.get("title", ""))
            e["venue"] = flatten(e.get("booktitle") or e.get("journal")
                                 or e.get("institution") or e.get("school")
                                 or e.get("publisher") or "")
            if not e["title_clean"] or e["title_clean"].lower() in UNUSABLE:
                continue
            e["source"] = origin
            n = norm(e["title_clean"])
            if n not in merged:
                merged[n] = e
            else:
                merged[n] = combine(merged[n], e)

    out = list(merged.values())
    seen: set[str] = set()
    for e in out:
        if "keywords" in e and re.fullmatch(r"[A-Z]/[a-z-]+", e["keywords"]):
            e["section"], e["subsection"] = e["keywords"].split("/")
        else:
            e["section"], e["subsection"] = classify(e)
        e["keywords"] = f"{e['section']}/{e['subsection']}"
        base = re.sub(r"[^A-Za-z0-9]", "", e["key"]) or "ref"
        key, i = base, 2
        while key in seen:
            key, i = f"{base}{i}", i + 1
        seen.add(key)
        e["key"] = key

    for pattern, note in WORKING_SET:
        hits = [e for e in out if "working_note" not in e
                and re.search(pattern, e["title_clean"] + " " + e["venue"], re.I)]
        if not hits:
            print(f"  working set pattern matched nothing: {pattern}",
                  file=sys.stderr)
            continue
        # Prefer the richest record of the work.
        hits.sort(key=len, reverse=True)
        hits[0]["working_note"] = note
    return out


def write_bib(entries: list[dict], path: Path) -> None:
    """The machine-readable library, with the taxonomy in each entry."""
    lines = [
        "% The reference library for this body of research.",
        "%",
        "% Generated by scripts/build_bibliography.py. Every entry carries a",
        "% keywords field holding its place in the taxonomy, so the",
        "% categorisation travels with the data rather than living only in",
        "% docs/bibliography.md.",
        "%",
        "% Metadata provenance: entries taken from our own bibliographies are",
        "% reproduced as they were recorded. Entries added by hand were checked",
        "% against the publisher or the preprint server, except where a note",
        "% says a field is still to be confirmed. Nothing was reconstructed",
        "% from memory.",
        "%",
        "% Do not edit by hand. tests/test_bibliography.py checks that this file",
        "% and docs/bibliography.md agree.",
        "",
    ]
    for section, _, subs in SECTIONS:
        order = {sub: i for i, (sub, _) in enumerate(subs)}
        rows = sorted((e for e in entries if e["section"] == section),
                      key=lambda e: (order.get(e["subsection"], 99),
                                     e.get("year", "9999"), e["key"]))
        for e in rows:
            lines.append(f"@{e['type']}{{{e['key']},")
            written = [f"  {f} = {{{e[f]}}}," for f in FIELDS if e.get(f)]
            written[-1] = written[-1].rstrip(",")
            lines.extend(written)
            lines.extend(["}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_index(entries: list[dict], path: Path) -> None:
    """The categorised index, with notes on the part that is load bearing."""
    by_year: Counter = Counter((e.get("year") or "????")[:4] for e in entries)

    def period(lo, hi):
        return sum(n for y, n in by_year.items()
                   if y.isdigit() and lo <= int(y) < hi)

    def count(**kw):
        return sum(1 for e in entries
                   if all(e.get(k) == v for k, v in kw.items()))

    added = sum(1 for e in entries if e.get("source") == "added by hand")
    md: list[str] = []
    w = md.append
    w("# The reference library")
    w("")
    w("Every work this body of research rests on, in one place, categorised by")
    w("the question it answers rather than by topic. The machine-readable form")
    w("is [`references.bib`](references.bib), where each entry carries its place")
    w("in this taxonomy in its `keywords` field. Both files are generated by")
    w("`scripts/build_bibliography.py`.")
    w("")
    w(f"**{len(entries)} entries.** {added} were added by hand while building")
    w("this library, because they were missing from every bibliography we had.")
    w("That absence is worth stating plainly: the thesis bibliography stops at")
    w("2021 and the current paper's bibliography is about content privacy, so")
    w("between them they omitted Opinion 05/2014 and Cohen and Nissim, the two")
    w("documents this framework is built on. A library assembled only from what")
    w("we had already cited would have had a hole exactly where the foundations")
    w("are.")
    w("")
    w("## How to read it")
    w("")
    w("**The working set** is what this project actually depends on, and each")
    w("entry there says what the work establishes and where we use it.")
    w("Everything else is categorised background: real references, placed, but")
    w("without a note each. Distinguishing the two is the point, because a")
    w("library where every entry looks equally load bearing tells a reader")
    w("nothing about what to read first.")
    w("")
    w("Metadata added by hand was checked against the publisher or the preprint")
    w("server. A few entries carry a note saying a field is still to be")
    w("confirmed, which is deliberate: an unverified author initial is recorded")
    w("as unverified rather than guessed.")
    w("")
    w("Deliberately excluded: work on low-resource ASR, code-switching and")
    w("pronunciation modelling, which shares a bibliography with this research")
    w("but not a subject. One exception is kept, a 2025 paper on anonymizing")
    w("code-switching speech, because there the two subjects meet.")
    w("")
    w("## The taxonomy")
    w("")
    w("| | Section | Entries |")
    w("|---|---|---|")
    for section, title, _ in SECTIONS:
        n = count(section=section)
        if n:
            w(f"| {section} | {title} | {n} |")
    w("")
    w(f"Coverage by period: {period(0, 2015)} before 2015, "
      f"{period(2015, 2022)} from 2015 to 2021, "
      f"{period(2022, 3000)} from 2022 onward.")
    w("")
    w("## The working set")
    w("")
    w("The references the engineering in this repository rests on.")
    w("")
    for section, title, subs in SECTIONS:
        order = {sub: i for i, (sub, _) in enumerate(subs)}
        rows = sorted((e for e in entries if e["section"] == section
                       and "working_note" in e),
                      key=lambda e: (order.get(e["subsection"], 99),
                                     e.get("year", "")))
        if not rows:
            continue
        w(f"### {section}. {title}")
        w("")
        for e in rows:
            venue = f" {e['venue']}." if e.get("venue") else ""
            w(f"**{short_authors(e.get('author'))} ({e.get('year', 'n.d.')}).** "
              f"{e['title_clean']}.{venue}{identifier(e, ' ')}")
            w("")
            w(f"> {e['working_note']}")
            w("")
            w(f"`{e['key']}`")
            w("")
    w("## The full library")
    w("")
    for section, title, subs in SECTIONS:
        if not count(section=section):
            continue
        w(f"### {section}. {title}")
        w("")
        for sub, sub_title in subs:
            rows = sorted((e for e in entries if e["subsection"] == sub),
                          key=lambda e: (e.get("year", "9999"),
                                         short_authors(e.get("author"))))
            if not rows:
                continue
            w(f"#### {sub_title} ({len(rows)})")
            w("")
            for e in rows:
                venue = f" {e['venue']}" if e.get("venue") else ""
                star = " **[working set]**" if "working_note" in e else ""
                w(f"- {short_authors(e.get('author'))} "
                  f"({e.get('year', 'n.d.')}). {e['title_clean']}."
                  f"{venue}{identifier(e)}. `{e['key']}`{star}")
            w("")
    w("## Gaps worth filling")
    w("")
    w("Stated so they are not mistaken for coverage.")
    w("")
    w(f"- **Section B holds {count(section='B')} entries against section A's "
      f"{count(section='A')} and section D's {count(section='D')}**, which is")
    w("  odd for the part of the field this project is actually about. The")
    w("  reason is real rather than an artefact of this library: proposals to")
    w("  replace the equal error rate are almost all from 2024 onward, and four")
    w("  of them arrived in one year.")
    w(f"- **The pseudo-speaker subsection holds "
      f"{count(subsection='pseudo-speaker')} entries**, which is obligation G3.")
    w("  `legal_eval.pseudo_speaker` implements a measure of it, so either we")
    w("  are ahead of the literature or we have not found it.")
    w(f"- **The attacker-model subsection holds "
      f"{count(subsection='attacker-models')} entries.** For a criterion whose")
    w("  regulatory obligation is to state the attacker concretely, that is a")
    w("  thin literature to lean on.")
    w("- **ISO/IEC 24745 is referenced but not read.** It is the standards")
    w("  family a voice anonymization metric would actually travel to, and the")
    w("  entry here is a placeholder for the real thing.")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bib", action="append", type=Path, required=True,
                        help="a BibTeX source, repeatable")
    parser.add_argument("--keep-whole", action="append", type=Path, default=[],
                        help="a source exempt from the relevance filter, for a "
                             "bibliography that is itself part of this research")
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--report", action="store_true",
                        help="print the distribution instead of writing")
    args = parser.parse_args(argv)

    whole = {p.resolve() for p in args.keep_whole}
    missing = [p for p in args.bib if not p.exists()]
    if missing:
        print(f"no such file: {', '.join(str(p) for p in missing)}",
              file=sys.stderr)
        return 1

    entries = collect([(p, p.resolve() in whole) for p in args.bib])

    unplaced = [e for e in entries if e["section"] == "Z"]
    if unplaced:
        print(f"{len(unplaced)} entries could not be placed:", file=sys.stderr)
        for e in unplaced[:10]:
            print(f"  {e['title_clean'][:80]}", file=sys.stderr)

    counts = Counter((e["section"], e["subsection"]) for e in entries)
    print(f"{len(entries)} entries, "
          f"{sum(1 for e in entries if 'working_note' in e)} in the working set")
    for section, title, _ in SECTIONS:
        rows = [(s, n) for (sec, s), n in counts.items() if sec == section]
        if not rows:
            continue
        print(f"\n{section}. {title}  ({sum(n for _, n in rows)})")
        for sub, n in sorted(rows, key=lambda r: -r[1]):
            print(f"    {SUBSECTION_TITLE.get(sub, sub):<52} {n}")

    if args.report:
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_bib(entries, args.out_dir / "references.bib")
    write_index(entries, args.out_dir / "bibliography.md")
    print(f"\nwrote {args.out_dir / 'references.bib'} and "
          f"{args.out_dir / 'bibliography.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
