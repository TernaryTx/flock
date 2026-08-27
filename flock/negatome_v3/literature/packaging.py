# Which text a run sends for each paper, built from the parsed full-text blocks.
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from flock.negatome_v3.literature import cues
from flock.negatome_v3.literature.jats import SECTION_ABSTRACT

INPUT_FULLTEXT_MINUS_METHODS = 'ft_minus_methods'
INPUT_CUE_WINDOW = 'cue_window'
INPUT_ABSTRACT = 'abstract'

# The block table's key, and what iter_papers groups on. Not the pmcid: most papers
# screened on their abstract have none, because Europe PMC holds no full text for
# them. paper_id is the pmcid where there is one and the pmid otherwise.
PAPER_KEY = 'paper_id'

# Methods is ~18% of body characters and almost never carries an interaction
# claim. Nothing else is gated on: TERN-2296 measured that targeting specific
# sections caps achievable recall at 88%.
DROPPED_SECTIONS = ('methods',)

# Blocks are joined with a blank line so paragraph boundaries survive into the
# prompt. The grounding gate collapses whitespace, so an excerpt spanning a
# boundary still matches.
BLOCK_SEPARATOR = '\n\n'


@dataclass(frozen=True)
class InputLevel:
    """One packaging of a paper: which blocks are sent, under which parameters.

    A level is data rather than a callable so that every parameter which changes
    the text sent is visible to fingerprint(), and so to the run_id. Without that,
    two runs with different cue lists would share a run_id and silently overwrite
    each other in the ledger.
    """

    name: str
    slug: str
    keep_sections: tuple[str, ...] = ()
    dropped_sections: tuple[str, ...] = ()
    cue_filtered: bool = False
    window_radius: int = 0

    def select_sections(self, paper: pd.DataFrame) -> pd.DataFrame:
        """Apply this level's section rule, before any cue filtering."""
        selected = paper
        if self.keep_sections:
            kept = selected.canonical_section.isin(self.keep_sections)
            selected = selected[kept]
        if self.dropped_sections:
            dropped = selected.canonical_section.isin(self.dropped_sections)
            selected = selected[~dropped]
        return selected

    def cue_mask(self, sections: pd.DataFrame) -> pd.Series:
        """Which blocks carry a negation cue, plus their neighbours."""
        carries = [cues.block_has_cue(text) for text in sections.text]
        keep = list(carries)
        for position, hit in enumerate(carries):
            if not hit or not self.window_radius:
                continue
            low = max(0, position - self.window_radius)
            high = min(len(carries), position + self.window_radius + 1)
            for neighbour in range(low, high):
                keep[neighbour] = True
        return pd.Series(keep, index=sections.index, dtype=bool)

    def fingerprint(self) -> dict:
        """The parameters that decide what text this level produces.

        The cue digest is included only for a cue-filtered level, so editing the
        cue list does not move the run_id of packagings that never consult it.
        """
        payload: dict = {
            'name': self.name,
            'keep_sections': list(self.keep_sections),
            'dropped_sections': list(self.dropped_sections),
            'cue_filtered': self.cue_filtered,
            'window_radius': self.window_radius,
        }
        if self.cue_filtered:
            payload['cues'] = cues.cue_fingerprint()
        return payload


# cue_window is how a paper with usable full text is packaged: only the paragraphs
# carrying a negation cue, 17.4% of body characters for 82 of 90 recall-gold rows
# against 89 of 90 for ft_minus_methods, which is what makes the run affordable.
# window_radius stays 0 because widening to +-1 buys 4 of those rows for 2.05x the
# characters. ft_minus_methods is kept as the ceiling it is judged against.
#
# abstract is how every other paper is packaged. It has no recall figure and will not
# get one - TERN-2296 put f_abstract at 11% and no curated negative is
# abstract-located - so what justifies it is retrieval: those papers were asked for on
# the condition that the negation phrase sits in the abstract, so the claim is in the
# text by construction. It is deliberately not cue_filtered, since the query has
# already vouched for a cue-matched phrase and filtering could only discard the block
# that phrase is in.
INPUT_LEVELS = {
    INPUT_FULLTEXT_MINUS_METHODS: InputLevel(
        name=INPUT_FULLTEXT_MINUS_METHODS,
        slug='ft',
        dropped_sections=DROPPED_SECTIONS,
    ),
    INPUT_CUE_WINDOW: InputLevel(
        name=INPUT_CUE_WINDOW,
        slug='cue',
        dropped_sections=DROPPED_SECTIONS,
        cue_filtered=True,
        window_radius=0,
    ),
    INPUT_ABSTRACT: InputLevel(
        name=INPUT_ABSTRACT,
        slug='abs',
        keep_sections=(SECTION_ABSTRACT,),
    ),
}


def get_level(name: str) -> InputLevel:
    """Resolve an input level by name.

    Raises:
        ValueError: If the name is not a known level.
    """
    if name not in INPUT_LEVELS:
        raise ValueError(
            f'Unknown input level {name!r}; expected one of {tuple(INPUT_LEVELS)}.',
        )
    return INPUT_LEVELS[name]


def build_paper_input(
        paper: pd.DataFrame,
        level: InputLevel,
        paper_id: str = '<unknown>',
) -> str:
    """Package one paper's blocks into the text sent to the model.

    The two ways this produces no text mean different things. No blocks at all at
    this level is a data problem and raises. A cue-filtered level that selects
    nothing returns the empty string: 10 of 182 papers in the calibration set
    contain no negation cue anywhere, and the runner records those as skips rather
    than errors, since they need no request.

    Args:
        paper: One paper's blocks; sorted by block_index here rather than assumed.
        level: The packaging to apply.
        paper_id: Named for the error message only. Passed in rather than read off
            the frame, which carries only the columns packaging reads.

    Raises:
        ValueError: If the paper has no text at this level before cue filtering.
    """
    ordered = paper.sort_values('block_index')
    sections = level.select_sections(ordered)
    if not BLOCK_SEPARATOR.join(sections.text).strip():
        raise ValueError(f'No {level.name} text for {paper_id}.')

    if not level.cue_filtered:
        return BLOCK_SEPARATOR.join(sections.text)
    return BLOCK_SEPARATOR.join(sections[level.cue_mask(sections)].text)


def iter_papers(
        path: str | Path,
        batch_size: int = 20_000,
        key_column: str = PAPER_KEY,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Stream a block table one paper at a time.

    The corpus block table does not fit in memory - about 20 kB of parquet per
    paper, so several gigabytes over a full corpus before pandas expands it.
    Reading it in row-group batches and grouping consecutive rows keeps a run
    bounded by one paper plus one batch. Only the columns packaging reads are
    pulled out.

    This requires the table to be written grouped by the key column, which the corpus
    stage guarantees by sorting each retrieval route before it writes. Rather than
    trust that, a key reappearing after its group has been emitted raises: packaging a
    paper twice from two fragments would send the model half a paper and record it as
    whole.

    Args:
        path: Local parquet written by the corpus stage.
        batch_size: Rows per read batch. Each batch is materialised whole and
            pyarrow coalesces row groups up to this size, so 20,000 rows (a few
            hundred papers, tens of MB) rather than the ten-times-larger batch that
            reads no faster and holds hundreds of MB to serve one paper.
        key_column: Column identifying a paper. Pass 'pmcid' to read a block table
            written before paper_id existed, e.g. the TERN-2296 parse.

    Yields:
        (paper key, that paper's blocks) in file order.

    Raises:
        FileNotFoundError: If the parquet does not exist.
        ValueError: If the table is not grouped by the key column.
    """
    blocks_path = Path(path)
    if not blocks_path.exists():
        raise FileNotFoundError(
            f'No full-text block table at {blocks_path}. Build it with '
            f'python -m flock.negatome_v3.literature.corpus, or download the '
            f'published corpus blocks parquet from S3.',
        )
    parquet_file = pq.ParquetFile(blocks_path)
    emitted: set[str] = set()
    buffered: list[pd.DataFrame] = []
    current: str | None = None

    def flush() -> tuple[str, pd.DataFrame]:
        if current in emitted:
            raise ValueError(
                f'{current} appears in more than one place in {path}. '
                f'The block table must be sorted by {key_column}.',
            )
        emitted.add(str(current))
        return str(current), pd.concat(buffered)

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[key_column, 'block_index', 'canonical_section', 'text'],
    ):
        frame = record_batch.to_pandas()
        for key, group in frame.groupby(key_column, sort=False):
            if current is not None and key != current:
                yield flush()
                buffered = []
            current = str(key)
            buffered.append(group)
    if buffered:
        yield flush()
