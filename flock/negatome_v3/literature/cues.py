# Negation-cue detection over sentences, and the fingerprint of the cue list.
from __future__ import annotations

import hashlib
import json
import re

# Negation cues for protein-interaction claims, grouped by phrasing family.
#
# Lifted unchanged from TERN-2296's candidates.py, where the list only had to
# surface sentences for a curator to read. Here it decides which paragraphs the
# screen is sent at all, so its coverage sets a hard ceiling on achievable recall.
# Promoting it from a sort key to a filter is defensible only because that ceiling
# is measured: 82 of 90 recall-gold rows survive cue-block packaging, against 89
# of 90 for the whole body minus methods.
NEGATION_CUE_PATTERNS = {
    'explicit_no_interaction': (
        r'\b(?:did|do|does|was|were|is|are|could|can)\s*n[o\']?t\s+'
        r'(?:\w+\s+){0,3}?(?:interact|bind|bound|associate|'
        r'co-?immunoprecipitate|co-?precipitate|co-?localis|co-?localiz|'
        r'complex)\w*',
        r'\bno\s+(?:detectable|significant|appreciable|apparent|specific|'
        r'measurable|direct|physical)?\s*(?:interaction|binding|association|'
        r'complex\s+formation)\b',
        r'\b(?:absence|lack)\s+of\s+(?:any\s+)?(?:detectable\s+|significant\s+|'
        r'direct\s+)?(?:interaction|binding|association|complex)\b',
        r'\bnot\s+(?:a\s+)?(?:direct\s+)?(?:interactor|binding\s+partner|'
        r'substrate|target)\b',
    ),
    'failed_assay': (
        r'\bfail(?:ed|s|ure)?\s+to\s+(?:\w+\s+){0,2}?(?:interact|bind|associate|'
        r'co-?immunoprecipitate|co-?precipitate|detect|pull\s*down|'
        r'immunoprecipitate)\w*',
        r'\b(?:unable|inability)\s+to\s+(?:\w+\s+){0,2}?(?:interact|bind|'
        r'associate|detect)\w*',
        r'\bwas\s+not\s+(?:co-?)?(?:immunoprecipitated|precipitated|detected|'
        r'pulled\s+down|recovered|retained)\b',
    ),
    'negative_result': (
        r'\b(?:negative|no)\s+(?:result|signal|band|reaction)s?\b',
        r'\b(?:undetectable|not\s+detected|below\s+(?:the\s+)?detection)\b',
        r'\b(?:did\s+not|does\s+not|failed\s+to)\s+(?:\w+\s+){0,3}?'
        r'(?:show|display|exhibit|yield|produce)\s+(?:any\s+)?'
        r'(?:interaction|binding|association|signal)\b',
    ),
    # Specificity claims are routinely written elliptically, with the verb omitted
    # after the negation: "APPL1 bound Rab5 but not Rab4, Rab7 or Rab11". The
    # contrast_negative patterns all require an interaction verb *after* the
    # negation and so miss this form, which is one of the commonest ways a
    # non-interacting pair enters the literature.
    #
    # The protein-name constraint in the second pattern carries (?-i:) because
    # everything here is compiled with re.IGNORECASE, which would otherwise void the
    # leading [A-Z] and let the pattern fire on a list of lowercase common nouns:
    # "binding was observed not with lipids, sugars or salts". Scoping the flag
    # inside the pattern rather than compiling this one separately keeps the
    # constraint visible to cue_fingerprint(), which hashes the pattern text.
    'elliptical_exclusion': (
        r'\b(?:interact|bind|bound|binding|associat|precipitat|complex)\w*'
        r'[^.]{0,120}?\b(?:but|and|although|whereas|though)\s+not\b',
        r'\b(?:interact|bind|bound|binding|associat)\w*[^.]{0,120}?'
        r'\bnot\s+(?:with|to)?\s*(?-i:[A-Z][A-Za-z0-9-]{1,14})'
        r'(?:\s*,\s*(?-i:[A-Z][A-Za-z0-9-]{1,14}))*'
        r'(?:\s*,?\s*(?:or|nor|and)\s+(?-i:[A-Z][A-Za-z0-9-]{1,14}))',
        r'\b(?:specific(?:ally)?|selectiv(?:e|ely))\s+(?:\w+\s+){0,3}?'
        r'(?:interact|bind|bound|binding|associat)\w*[^.]{0,80}?\bnot\b',
    ),
    'contrast_negative': (
        r'\b(?:whereas|while|but|however|in\s+contrast)\b[^.]{0,80}?\b'
        r'(?:no|not|neither)\b[^.]{0,40}?\b(?:interact|bind|bound|associat)\w*',
        r'\bneither\b[^.]{0,60}?\bnor\b[^.]{0,60}?\b(?:interact|bind|bound|associat)\w*',
    ),
}

# Abbreviation periods are masked before splitting rather than excluded by
# lookbehind, because Python requires fixed-width lookbehind and these differ in
# length. "et al." and "Fig." cause most of the spurious splits here.
ABBREVIATIONS = (
    'et al', 'e.g', 'i.e', 'cf', 'ca', 'approx', 'vs', 'viz',
    'Fig', 'Figs', 'fig', 'figs', 'Eq', 'Ch', 'Ref', 'ref', 'No', 'no',
    'Dr', 'Prof', 'Mr', 'Ms', 'St', 'Inc', 'Co', 'Ltd',
    'pp', 'vol', 'Vol', 'min', 'sec', 'hr', 'wt', 'temp',
)
PERIOD_PLACEHOLDER = '\x00'
ABBREVIATION_RE = re.compile(
    r'\b(' + '|'.join(
        re.escape(abbr).replace(r'\.', '[.]') for abbr in ABBREVIATIONS
    ) + r')\.',
)
DECIMAL_RE = re.compile(r'(?<=\d)\.(?=\d)')
# A period after a single capital is an initial only where an initial can appear:
# opening a block, or after a sentence end or another capital. The lookbehind is
# what stops it swallowing a genuine sentence end after a one-letter protein or
# construct name - "the knockout of X. The band reappeared" - which would merge
# the two sentences and let an interaction verb in one pair with a negation in the
# next. Erring towards splitting can only cost a cue match; erring the other way
# manufactures one.
INITIAL_RE = re.compile(r'(?<![a-z] )\b([A-Z])\.')
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

# Compiled once, in declaration order, so a corpus-wide scan does not re-enter the
# regex cache for every sentence.
_COMPILED_CUES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for patterns in NEGATION_CUE_PATTERNS.values()
    for pattern in patterns
)


def masked_sentences(text: str) -> list[str]:
    """Split a text block into sentences, leaving the masked periods in place.

    This is the form the cue patterns are matched against. They bound their
    wildcards with [^.], so a period surviving inside a sentence - a decimal in an
    assay value, an abbreviation, an initial - truncates the span and kills the
    match: "a Kd of 0.5 uM but not Rab4, Rab7 or Rab11" carries no cue while the
    same sentence with "5 uM" does. Assay values with decimals are ubiquitous here,
    and the loss is invisible because the paper is simply recorded as a skip.
    """
    if not text:
        return []
    masked = ABBREVIATION_RE.sub(
        lambda match: match.group(1).replace(
            '.', PERIOD_PLACEHOLDER,
        ) + PERIOD_PLACEHOLDER,
        text,
    )
    masked = DECIMAL_RE.sub(PERIOD_PLACEHOLDER, masked)
    masked = INITIAL_RE.sub(rf'\1{PERIOD_PLACEHOLDER}', masked)
    sentences = [
        sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(masked)
    ]
    return [sentence for sentence in sentences if sentence]


def split_sentences(text: str) -> list[str]:
    """Split a text block into non-empty sentences, as written."""
    return [
        sentence.replace(PERIOD_PLACEHOLDER, '.')
        for sentence in masked_sentences(text)
    ]


def block_has_cue(text: str) -> bool:
    """Whether any sentence in a block carries a negation cue.

    Cue detection is a property of a sentence rather than of the whole block: the
    patterns bound their wildcards with [^.] so a match cannot straddle a sentence
    boundary, and running them over concatenated paragraph text would let an
    interaction verb in one sentence pair with a negation in the next.
    """
    return any(
        pattern.search(sentence)
        for sentence in masked_sentences(text)
        for pattern in _COMPILED_CUES
    )


def cue_fingerprint() -> str:
    """Hash the cue list and the sentence splitter that feeds it.

    Both are included because both decide which paragraphs a cue-filtered run
    selects: editing a splitter regex moves that selection without the cue list
    changing at all, and a run_id that could not see it would let two different
    configurations collide on one identity.
    """
    payload = {
        'patterns': {
            family: list(patterns)
            for family, patterns in NEGATION_CUE_PATTERNS.items()
        },
        'abbreviations': list(ABBREVIATIONS),
        'splitter': [
            DECIMAL_RE.pattern,
            INITIAL_RE.pattern,
            SENTENCE_SPLIT_RE.pattern,
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]
