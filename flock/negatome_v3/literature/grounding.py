# Strict-substring grounding gate for the verbatim excerpts the screen returns.
from __future__ import annotations

import html
import re
import unicodedata

# Lifted from ~/dev/orca/orca/integrity/grounding.py, minus its docling
# PDF-extraction fixups: this corpus is JATS XML, so those artifacts do not arise
# and every extra fold is a chance to rewrite one side of the comparison but not
# the other. Dashes are folded because NFKC does not decompose them and protein
# names are full of hyphens.
_DASHES = ('−', '‐', '‑', '‒', '–', '—', '―')
_APPROX = ('≈', '∼', '≃', '≅', '≉')
_FOLD_TABLE = str.maketrans({
    **{char: '-' for char in _DASHES},
    **{char: '~' for char in _APPROX},
    '×': 'x',
})
_WHITESPACE_RE = re.compile(r'\s+')


def normalise(text: str) -> str:
    """Canonicalise text so an excerpt and its source compare faithfully.

    Whitespace collapse is what lets an excerpt spanning a paragraph break match:
    packaging joins blocks with blank lines, which normalise to one space.
    """
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.translate(_FOLD_TABLE)
    return _WHITESPACE_RE.sub(' ', text).strip().casefold()


def ungrounded_excerpts(excerpts: list[str], source_text: str) -> list[str]:
    """Return the excerpts that are not verbatim spans of the text sent.

    Strict substring after normalisation, never fuzzy: a near-miss is a failure,
    because the point is to catch excerpts the model composed rather than copied.
    Checked against what was sent rather than the original JATS, since the model
    cannot quote what it was never shown. The source is normalised once here and
    not memoised - a paper is screened once and never revisited, so a cache keyed
    on whole papers would retain a gigabyte to serve lookups that never repeat.

    Args:
        excerpts: The verbatim spans claimed by the model.
        source_text: The exact text sent to the model.

    Returns:
        The subset that are not genuine spans, in input order. An empty excerpt is
        a substring of anything, so it is reported as ungrounded rather than passed.

    Raises:
        ValueError: If source_text is empty, which is a bug rather than a reason to
            pass the excerpt.
    """
    if not source_text:
        raise ValueError(
            'Empty source_text: nothing to ground excerpts against.',
        )
    normalised_source = normalise(source_text)
    return [
        item for item in excerpts
        if not item.strip() or normalise(item) not in normalised_source
    ]
