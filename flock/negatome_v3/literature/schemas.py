# The output contracts: the Pydantic models and the JSON schemas sent, one per
# stage that asks a model for structured output.
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ReportedPair(BaseModel):
    """One protein pair the screen found discussed in the paper."""

    model_config = ConfigDict(extra='forbid')

    protein_a: str = Field(
        description='First protein, named exactly as the paper writes it.',
    )
    protein_b: str = Field(
        description='Second protein, named exactly as the paper writes it.',
    )
    relationship: Literal['no_interaction', 'interaction', 'both'] = Field(
        description=(
            'What the paper reports about this pair: no_interaction if it reports '
            'they do not interact, interaction if it reports they do, both if it '
            'reports both.'
        ),
    )
    confidence: Literal['high', 'medium', 'low'] = Field(
        description='How confident you are in this reading of the paper.',
    )
    # min_length emits minItems: 1, which structured outputs does honour: only the
    # complex array constraints are unsupported. It puts 'at least one excerpt' on
    # the wire rather than leaving it to the description alone, so a pair with no
    # evidence cannot come back for the grounding gate to have nothing to check.
    excerpts: list[str] = Field(
        min_length=1,
        description=(
            'Verbatim sentences copied character-for-character from the paper text '
            'above, supporting this reading. At least one is required.'
        ),
    )


# Every docstring here is emitted verbatim as its schema's description and ships to
# the model on each request, so notes about the contract itself are comments: the
# wrapper class exists because structured outputs constrains to an object rather than
# a bare array, and recursive schemas are unsupported.
class ScreenResult(BaseModel):
    """Every pair the screen reports for one paper."""

    model_config = ConfigDict(extra='forbid')

    pairs: list[ReportedPair] = Field(
        description=(
            'Every protein pair the paper discusses an interaction or lack of '
            'interaction for. Empty if the paper discusses none.'
        ),
    )


# No confidence field: asking for one was measured and rejected. The evidence
# span stays, since it is what makes a pick auditable.
class AccessionPick(BaseModel):
    """Which UniProt entry a paper means by one protein name."""

    model_config = ConfigDict(extra='forbid')

    choice: int = Field(
        description=(
            'The number of the candidate the paper means, or 0 if it cannot be '
            'determined from the text.'
        ),
    )
    evidence: str = Field(
        description=(
            'The span of the paper text that decided it, copied character for '
            'character. Empty if the choice is 0.'
        ),
    )
    reason: str = Field(
        description='One sentence explaining the choice.',
    )


# The curation vocabularies, as Literals rather than free strings: an
# out-of-vocabulary verdict is one apply cannot act on.
MappingVerdict = Literal[
    'correct', 'wrong_paralog', 'wrong_protein', 'wrong_species',
    'fragment_or_isoform_mismatch', 'affinity_tag_or_reporter',
    'cannot_determine',
]
Relationship = Literal[
    'non_interaction', 'positive_interaction', 'both_reported',
    'not_discussed', 'no_evidence',
]
EvidenceLocation = Literal[
    'abstract', 'introduction', 'results', 'discussion',
    'results_and_discussion', 'figure_caption', 'table_caption',
    'supplementary', 'other', 'none',
]
# not_a_protein_pair and indirect_or_functional_only both name a sentence that
# reads as a clean negative until you notice what is being denied: a protein
# against a promoter region, or a depletion experiment rather than a binding test.
Qualifier = Literal[
    'full_length_pair', 'domain_fragment_only', 'mutant_only',
    'different_paralog', 'cited_prior_work', 'not_a_protein_pair',
    'indirect_or_functional_only', 'unclear',
]


# Descriptions are one line each: the rubric ships as the system prompt and is
# where the definitions live, so restating them here would be two copies to keep
# in step.
class CurationVerdict(BaseModel):
    """One curated judgement on one candidate non-interacting pair."""

    model_config = ConfigDict(extra='forbid')

    pair: str = Field(
        description='The exact pair string from the packet, e.g. "P12345|Q67890".',
    )
    mapping_verdict: MappingVerdict = Field(
        description='Whether the recorded accessions are the proteins the paper means.',
    )
    mapping_note: str = Field(
        description=(
            'The protein the accession should be, or an empty string when the '
            'mapping is correct.'
        ),
    )
    relationship: Relationship = Field(
        description='What the paper reports about this pair.',
    )
    evidence_sentence: str = Field(
        description=(
            'The single best sentence supporting the relationship call, copied '
            'character for character from the paper text. Empty if none.'
        ),
    )
    evidence_location: EvidenceLocation = Field(
        description='Which part of the paper the evidence sentence came from.',
    )
    qualifier: Qualifier = Field(
        description='What the negation is actually about: the full-length pair, or less.',
    )
    usable_as_negative: bool = Field(
        description='Whether this pair belongs in the benchmark as a negative.',
    )
    confidence: str = Field(
        description='high, medium or low.',
    )
    reasoning: str = Field(
        description='One to three sentences, naming the section relied on.',
    )


class CurationResult(BaseModel):
    """Every verdict for one paper."""

    model_config = ConfigDict(extra='forbid')

    verdicts: list[CurationVerdict] = Field(
        description=(
            'One verdict per pair in the packet. Cover every pair - do not skip '
            'any, and do not judge the same pair twice.'
        ),
    )
