from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator

# The only label other modules need by name.
SECTION_ABSTRACT = 'abstract'

# Canonical section labels, matched against the uppercased title with any leading
# numbering stripped. Order matters: combined headings must be tested before the
# single-topic ones, or "RESULTS AND DISCUSSION" matches "RESULTS" first.
#
# 'results_and_discussion' is a real category, not a fallback: ~9% of the OA
# articles in the calibration set merge the two under one heading. 'other' covers
# articles whose sections are topic-named rather than canonical, common in pre-2000
# papers, which would otherwise be silently miscounted.
SECTION_TITLE_PATTERNS = (
    ('results_and_discussion', re.compile(r'^RESULTS?\s*(AND|&|/)\s*DISCUSSION')),
    ('results_and_discussion', re.compile(r'^DISCUSSION\s*(AND|&|/)\s*RESULTS?')),
    (
        'methods', re.compile(
            r'^(MATERIALS?\s*(AND|&)?\s*METHODS?|METHODS?|EXPERIMENTAL'
            r'\s*(PROCEDURES?|SECTION|METHODS?)?|PATIENTS?\s*AND\s*METHODS?|'
            r'METHODOLOGY|MATERIALS?)\b',
        ),
    ),
    ('results', re.compile(r'^RESULTS?\b')),
    ('discussion', re.compile(r'^DISCUSSION\b')),
    ('conclusion', re.compile(r'^(CONCLUSIONS?|SUMMARY|CONCLUDING)\b')),
    ('introduction', re.compile(r'^(INTRODUCTION|BACKGROUND)\b')),
    (
        'supplementary', re.compile(
            r'^(SUPPLEMENTARY|SUPPLEMENTAL|SUPPORTING|ASSOCIATED\s+DATA|'
            r'ADDITIONAL\s+(DATA|FILE))',
        ),
    ),
)

# Sections carrying no evidence about protein interactions.
EXCLUDED_TITLE_PATTERN = re.compile(
    r'^(REFERENCES?|SELECTED\s+REFERENCES|BIBLIOGRAPHY|ACKNOWLEDG(E)?MENTS?|'
    r'AUTHORS?[’\']?\s+(CONTRIBUTIONS?|INFORMATION)|CONTRIBUTOR\s+INFORMATION|'
    r'COMPETING\s+INTERESTS?|CONFLICTS?\s+OF\s+INTEREST|DISCLOSURES?|'
    r'ABBREVIATIONS?|LIST\s+OF\s+ABBREVIATIONS|FOOTNOTES?|FUNDING|'
    r'PRE-?PUBLICATION\s+HISTORY|AVAILABILITY|ETHICS|CONSENT|'
    r'AUTHOR\s+SUMMARY)\b',
)

# sec-type values that settle the classification without consulting the title.
SEC_TYPE_MAP = {
    'results': 'results',
    'discussion': 'discussion',
    'intro': 'introduction',
    'introduction': 'introduction',
    'background': 'introduction',
    'methods': 'methods',
    'materials|methods': 'methods',
    'materials-methods': 'methods',
    'conclusions': 'conclusion',
    'conclusion': 'conclusion',
    'supplementary-material': 'supplementary',
    'ref-list': 'excluded',
    'ack': 'excluded',
    'fn-group': 'excluded',
    'contrib-info': 'excluded',
    'COI-statement': 'excluded',
    'abbreviations': 'excluded',
    'associated-data': 'supplementary',
}

# Leading section numbering, e.g. "1.", "2.3", "IV." before the real title.
LEADING_NUMBER_RE = re.compile(r'^\s*([0-9]+(\.[0-9]+)*|[IVXLC]+)[.)]?\s+')
WHITESPACE_RE = re.compile(r'\s+')

# Elements whose own text is not part of the paragraph containing them: a table's
# cells and a figure's label and caption are inlined into the surrounding sentence
# otherwise, which mangles it and defeats sentence splitting. Captions are collected
# separately as their own blocks.
INLINE_SKIP_TAGS = frozenset({
    'fig', 'table-wrap', 'table', 'graphic', 'media',
})

# Back-matter containers carrying no evidence about protein interactions. These
# mirror the 'excluded' entries in SEC_TYPE_MAP, which only ever reaches a <sec>;
# in <back> the same content sits in tags of its own.
EXCLUDED_CONTAINER_TAGS = frozenset({
    'ref-list', 'ack', 'fn-group', 'glossary', 'bio', 'author-notes', 'notes',
})

# A section's paragraphs, captions and subsections are all nested rather than direct
# children a good part of the time: enumerated results sit in <list>, highlighted
# claims in <boxed-text> or <statement>, appendices under <app-group>/<app>, and a
# figure is routinely declared inside the paragraph that first cites it. The stop
# sets say where each search must not follow, so nothing is collected twice - a
# caption is not also a paragraph, and a nested section's paragraphs belong to that
# section's own walk.
PARAGRAPH_TAGS = frozenset({'p'})
SECTION_TAGS = frozenset({'sec'})
PARAGRAPH_STOP_TAGS = SECTION_TAGS | INLINE_SKIP_TAGS | EXCLUDED_CONTAINER_TAGS
CAPTION_STOP_TAGS = SECTION_TAGS | EXCLUDED_CONTAINER_TAGS
SECTION_STOP_TAGS = EXCLUDED_CONTAINER_TAGS

# Captions are separate evidence sites: a "no interaction detected" result is often
# stated only in a caption, which a paragraph-only parse would miss entirely. The
# label is the block's title and block_type carries the distinction, so the block
# keeps the canonical section it was found in.
CAPTION_TAGS = (
    ('fig', 'figure_caption'),
    ('table-wrap', 'table_caption'),
)


def classify_section(title: str | None, sec_type: str | None) -> str:
    """Map a JATS section heading to a canonical section label.

    sec-type is preferred where present but is absent on roughly 70% of top-level
    sections in this corpus, so the title carries most of the load. Titles matching
    nothing canonical return 'other' rather than being forced into a neighbour.
    """
    if sec_type:
        mapped = SEC_TYPE_MAP.get(sec_type.strip().lower())
        if mapped is None:
            mapped = SEC_TYPE_MAP.get(sec_type.strip())
        if mapped:
            return mapped
    if not title:
        return 'other'
    normalised = LEADING_NUMBER_RE.sub('', title.strip().upper())
    if EXCLUDED_TITLE_PATTERN.match(normalised):
        return 'excluded'
    for label, pattern in SECTION_TITLE_PATTERNS:
        if pattern.match(normalised):
            return label
    return 'other'


def element_text(element: ET.Element, drop_citation_xrefs: bool = True) -> str:
    """Extract normalised text from a JATS element, including inline markup.

    Bibliographic cross-references are dropped by default: rendered inline they
    become bracketed numerals that fragment sentence splitting without carrying
    meaning. Figure and table cross-references are kept, since "as shown in
    Figure 3" is part of the claim.
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if node.tag == 'xref' and drop_citation_xrefs and node.get('ref-type') == 'bibr':
            if node.tail:
                parts.append(node.tail)
            return
        if node.tag in INLINE_SKIP_TAGS:
            if node.tail:
                parts.append(node.tail)
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
        if node.tail:
            parts.append(node.tail)

    if element.text:
        parts.append(element.text)
    for child in element:
        walk(child)
    text = ''.join(parts)
    # Citation stripping leaves orphaned brackets and spaces before punctuation.
    text = re.sub(r'\[\s*[,\s–-]*\]', '', text)
    text = re.sub(r'\(\s*[,;\s]*\)', '', text)
    text = WHITESPACE_RE.sub(' ', text).strip()
    return re.sub(r'\s+([.,;:])', r'\1', text)


def _iter_nested(
        parent: ET.Element,
        tags: frozenset[str],
        stop: frozenset[str],
) -> Iterator[ET.Element]:
    """Yield descendants carrying one of these tags, without entering a stop tag.

    A match is yielded rather than descended into, so the search returns the
    outermost element of each kind and never one nested inside another of its own.
    """
    for child in parent:
        if child.tag in stop:
            continue
        if child.tag in tags:
            yield child
        else:
            yield from _iter_nested(child, tags, stop)


def _walk_sections(
        parent: ET.Element,
        path: tuple[str, ...],
        inherited_label: str | None,
        rows: list[dict],
) -> None:
    """Recursively collect paragraphs and captions from a JATS section tree."""
    section_path = ' > '.join(path)
    for paragraph in _iter_nested(parent, PARAGRAPH_TAGS, PARAGRAPH_STOP_TAGS):
        text = element_text(paragraph)
        if not text:
            continue
        rows.append({
            'section_path': section_path,
            'section_title': path[-1] if path else '',
            'canonical_section': inherited_label or 'other',
            'block_type': 'paragraph',
            'text': text,
        })

    for tag, block_type in CAPTION_TAGS:
        for element in _iter_nested(
            parent, frozenset({tag}), CAPTION_STOP_TAGS,
        ):
            caption = element.find('caption')
            text = element_text(caption) if caption is not None else ''
            if not text:
                continue
            rows.append({
                'section_path': section_path,
                'section_title': (element.findtext('label') or '').strip(),
                'canonical_section': inherited_label or 'other',
                'block_type': block_type,
                'text': text,
            })

    for section in _iter_nested(parent, SECTION_TAGS, SECTION_STOP_TAGS):
        title = (section.findtext('title') or '').strip()
        label = classify_section(title, section.get('sec-type'))
        # A subsection inherits its parent's label unless the parent was
        # unclassifiable, which is how topic-titled subsections under a proper
        # RESULTS heading stay attributed to results.
        if label == 'other' and inherited_label not in (None, 'other'):
            effective = inherited_label
        else:
            effective = label
        if effective == 'excluded':
            continue
        _walk_sections(section, path + (title,), effective, rows)


def unstructured_abstract_block(text: str) -> dict:
    """One block for an abstract carrying no <sec> tree of its own.

    Shared with the corpus stage, which wraps a search-response abstract in this same
    shape for papers Europe PMC holds no full text for. Keeping one definition is what
    makes those papers indistinguishable from a parsed abstract downstream, rather
    than only intended to be.
    """
    return {
        'section_path': '',
        'section_title': 'ABSTRACT',
        'canonical_section': SECTION_ABSTRACT,
        'block_type': 'paragraph',
        'text': text,
    }


def parse_article(xml_bytes: bytes) -> list[dict]:
    """Parse JATS full-text XML into section-tagged text blocks.

    Returns:
        Dicts with section_path, section_title, canonical_section, block_type, text
        and block_index, covering the structured abstract, body paragraphs, and
        figure and table captions. Reference lists, acknowledgements and other
        non-evidential sections are dropped.

    Raises:
        ET.ParseError: If the XML is malformed.
    """
    root = ET.fromstring(xml_bytes)
    rows: list[dict] = []

    for abstract in root.findall('.//front//abstract'):
        # Structured abstracts carry their own <sec> tree; both forms reduce to
        # blocks labelled as abstract so the annotation schema stays flat.
        abstract_rows: list[dict] = []
        _walk_sections(abstract, (), SECTION_ABSTRACT, abstract_rows)
        for row in abstract_rows:
            row['canonical_section'] = SECTION_ABSTRACT
        direct = element_text(abstract) if not abstract_rows else ''
        if direct:
            abstract_rows.append(unstructured_abstract_block(direct))
        rows.extend(abstract_rows)

    # <back> is walked as well as <body>, because appendices and back-matter
    # supplementary sections carry results. EXCLUDED_CONTAINER_TAGS is what keeps the
    # reference list, acknowledgements and author notes out of it.
    for tag in ('.//body', './/back'):
        element = root.find(tag)
        if element is not None:
            _walk_sections(element, (), None, rows)

    for index, row in enumerate(rows):
        row['block_index'] = index
    return rows
