# Assemble the screening corpus: negation queries -> filtered papers + parsed blocks.
from __future__ import annotations

import argparse
import logging
import os
import xml.etree.ElementTree as ET
from datetime import date as date_type
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.negatome_v3.literature import europepmc
from flock.negatome_v3.literature import RUN_ROOT
from flock.negatome_v3.literature import SOURCE_ABSTRACT
from flock.negatome_v3.literature import SOURCE_FULLTEXT
from flock.negatome_v3.literature import TEXT_SOURCES
from flock.negatome_v3.literature.jats import parse_article
from flock.negatome_v3.literature.jats import SECTION_ABSTRACT
from flock.negatome_v3.literature.jats import unstructured_abstract_block
from flock.paths import get_literature_stage_prefix
from flock.paths import LITERATURE_STAGE_VERSIONS
from flock.paths import make_dated_filename
from flock.paths import NEGATOME_LITERATURE_RAW_S3

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = RUN_ROOT / 'corpus'
PAPERS_NAME = 'papers'
BLOCKS_NAME = 'blocks'
CORPUS_STAGE = 'corpus'

# Formulaic phrasings of an interaction claim, one query per phrasing family, aligned
# with the cue families in cues.py that decide which paragraphs a matched paper sends.
#
# ** The index drops no, not, a and any from a phrase, so no query here can select on
# negation - "no interaction" ties exactly with bare "interaction". ** Negation
# filtering happens entirely in the cue regexes at packaging time, so a phrase is
# only worth adding if what survives the drop is still specific, and every phrase
# must be matched by at least one cue regex or cue_window cuts the sentence that
# retrieved the paper. not_directly_interact covers the adverb gap: phrase search is
# exact-adjacency while the cue patterns allow three words between the negation and
# the verb.
#
# Validated against the 650 known Negatome v2 papers under TERN-2298 - retrieves 504,
# against a ~50% floor. Hit counts and the rejected phrases are in the ticket.
NEGATION_QUERIES = {
    'does_not_interact': '"does not interact" OR "do not interact" OR "did not interact"',
    'not_directly_interact': (
        '"does not directly interact" OR "did not directly interact" '
        'OR "do not directly interact" OR "does not directly bind" '
        'OR "did not directly bind" OR "does not physically interact" '
        'OR "did not physically interact" OR "does not physically bind" '
        'OR "does not functionally interact" OR "did not specifically bind"'
    ),
    'no_detectable_interaction': '"no detectable interaction"',
    'interaction_not_detected': '"interaction was not detected"',
    'absence_of_interaction': '"absence of interaction" OR "lack of interaction" OR "absence of any interaction"',
    'does_not_bind': '"does not bind" OR "do not bind" OR "did not bind"',
    'failed_to_bind': '"failed to bind" OR "unable to bind"',
    'failed_coip': '"failed to co-immunoprecipitate" OR "did not co-immunoprecipitate" OR "was not co-immunoprecipitated" OR "does not coprecipitate"',
    'no_association': '"did not associate with" OR "does not associate with"',
    'no_complex': '"did not form a complex" OR "does not form a complex"',
}

# Publication types carrying no primary evidence, excluded at query time. Reviews
# are deliberately not excluded: a review can restate a negative result, and
# curation decides whether a report is usable evidence.
EXCLUDED_PUB_TYPES = (
    'Editorial', 'Letter', 'News', 'Comment',
    'Retracted Publication', 'Retraction of Publication',
)

# The second pass on retractions, because Europe PMC's pub-type vocabulary is not
# fully stable across sources and a retracted paper reaching curation would be a
# defensibility problem rather than a rounding error.
RETRACTION_MARKERS = ('retract', 'withdraw')

# The corpus arrives by two routes, added under TERN-2298. Papers Europe PMC will
# serve as full text are read that way; roughly half of what a phrase query returns is
# indexed but not downloadable, so for those the query asks instead that the negation
# phrase sit in the abstract, which is then the whole of what can be screened.
# OPEN_ACCESS:Y partitions the two, so a paper arrives by exactly one route and the
# hit counts never double count. An abstract cannot establish construct-only,
# wrong-paralog or bridged-through-a-third-protein status, so text_source rides along
# with every paper and every pair rather than being dropped once the corpus is built.
OPEN_ACCESS_FILTER = 'OPEN_ACCESS:Y'
ABSTRACT_FIELD = 'ABSTRACT'

# Some pre-2000 articles return HTTP 200 from fullTextXML but hold only a stub with
# no paragraph content, because Europe PMC has a page scan rather than marked-up text.
# Counting those as full-text coverage would overstate what the screen can read, so
# they fall back to the abstract like any other paper with no full text.
MIN_USABLE_BODY_CHARS = 2000

# The per-paper outcome, named so an empty run returns the same frame a populated one
# does. parsed_abstract_chars is deliberately not abstract_chars: that name is already
# taken by the search API's abstract length, and two frames setting it left the merge
# with abstract_chars_x and abstract_chars_y and neither usable.
OUTCOME_COLUMNS = (
    'paper_id', 'fulltext_status', 'parse_ok', 'parse_error', 'n_blocks',
    'body_chars', 'parsed_abstract_chars', 'usable_fulltext',
)

# Fixed rather than inferred: the table is written row group by row group, and an
# inferred schema can disagree between groups - an all-empty section_title in one
# would land as null and fail the append.
BLOCK_SCHEMA = pa.schema([
    # What packaging groups on: the pmcid where there is one and the pmid otherwise,
    # since Europe PMC holds no PMCID for most abstract-route papers. pmcid stays as a
    # column of its own because it is the identifier the full-text cache and every
    # external PMC link are keyed on, and paper_id no longer always carries it.
    ('paper_id', pa.string()),
    ('pmcid', pa.string()),
    ('block_index', pa.int32()),
    ('section_path', pa.string()),
    ('section_title', pa.string()),
    ('canonical_section', pa.string()),
    ('block_type', pa.string()),
    ('text', pa.string()),
])


def build_query(clause: str, excluded_pub_types: tuple[str, ...] = EXCLUDED_PUB_TYPES) -> str:
    """Wrap one negation clause in the corpus-wide filters."""
    excluded = ' OR '.join(
        f'PUB_TYPE:"{pub_type}"' for pub_type in excluded_pub_types
    )
    return f'({clause}) AND LANG:eng AND HAS_ABSTRACT:Y NOT ({excluded})'


def source_query(
        clause: str,
        text_source: str,
        excluded_pub_types: tuple[str, ...] = EXCLUDED_PUB_TYPES,
) -> str:
    """Restrict a negation clause to one retrieval route.

    The abstract restriction scopes the clause alone and is applied before the corpus
    filters wrap it. Scoping the whole filtered query instead would push LANG:eng and
    the pub-type exclusions inside the field and search the abstract text for them,
    which returns nothing and reads as an empty route.

    Raises:
        ValueError: If the text source is not one of TEXT_SOURCES.
    """
    if text_source == SOURCE_FULLTEXT:
        filtered = build_query(clause, excluded_pub_types)
        return f'({filtered}) AND {OPEN_ACCESS_FILTER}'
    if text_source == SOURCE_ABSTRACT:
        scoped = build_query(
            f'{ABSTRACT_FIELD}:({clause})', excluded_pub_types,
        )
        return f'{scoped} NOT {OPEN_ACCESS_FILTER}'
    raise ValueError(
        f'Unknown text source {text_source!r}; expected one of {TEXT_SOURCES}.',
    )


def hit_counts(
        queries: dict[str, str],
        cache_dir: str = europepmc.DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Report each query's hit count, unsplit and per route, without sweeping it.

    Three requests per query. This is how the filter syntax gets verified against the
    live API before a sweep commits to it: a clause Europe PMC does not understand
    shows up here as a wildly wrong count rather than as a silently truncated corpus
    days later. The route columns sum to less than hit_count on purpose - the
    difference is the closed-access papers whose phrase is not in the abstract, which
    nothing can screen.
    """
    rows = []
    for name, clause in queries.items():
        unsplit = build_query(clause)
        row: dict = {'query': name, 'query_string': unsplit}
        for label, query in (
            ('hit_count', unsplit),
            (SOURCE_FULLTEXT, source_query(clause, SOURCE_FULLTEXT)),
            (SOURCE_ABSTRACT, source_query(clause, SOURCE_ABSTRACT)),
        ):
            for page in europepmc.search_pages(
                query, cache_dir=cache_dir, result_type='idlist', max_pages=1,
            ):
                row[label] = page.get('hitCount')
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_queries(
        queries: dict[str, str],
        text_source: str = SOURCE_FULLTEXT,
        cache_dir: str = europepmc.DEFAULT_CACHE_DIR,
        max_pages: int | None = None,
) -> pd.DataFrame:
    """Run every query to exhaustion within one route, one row per paper.

    A paper matched by several queries keeps all of their names: which query brought
    in the papers that eventually yielded pairs is the question the next iteration
    of the query set turns on.

    Args:
        queries: Mapping of query name to negation clause.
        text_source: Which route to sweep. The two partition on OPEN_ACCESS:Y, so
            sweeping both and concatenating cannot duplicate a paper.
        cache_dir: Local cache root.
        max_pages: Pages per query, for a scoped trial run. None sweeps fully.
    """
    by_key: dict[str, dict] = {}
    matched: dict[str, set[str]] = {}

    for name, clause in queries.items():
        query = source_query(clause, text_source)
        n_hits = 0
        for page in europepmc.search_pages(
            query, cache_dir=cache_dir, max_pages=max_pages,
        ):
            for record in page.get('resultList', {}).get('result', []):
                # Keyed on the Europe PMC id, the only identifier every source
                # record carries; pmid and pmcid are both absent on some records.
                identifier = str(record.get('id') or '')
                if not identifier:
                    continue
                n_hits += 1
                if identifier not in by_key:
                    flat = europepmc.flatten_metadata_record(
                        identifier, record,
                    )
                    # The abstract is kept for both routes and costs most of a
                    # gigabyte over a full corpus, which is worth it because it is the
                    # only copy that survives the sweep. The abstract route has no
                    # other text at all, and on the full-text route it is what a
                    # paper whose fetch 404s or whose parse raises falls back to -
                    # dropping it there left those papers with no blocks, and a paper
                    # with no blocks is never visited by the screen. main() drops the
                    # column before writing the papers table, so it costs memory
                    # during the sweep only.
                    by_key[identifier] = flat
                    matched[identifier] = set()
                matched[identifier].add(name)
        logger.info(
            'Query %s: %d hits, %d unique papers so far',
            name, n_hits, len(by_key),
        )

    papers = pd.DataFrame(list(by_key.values()))
    if papers.empty:
        return papers
    papers['matched_queries'] = papers.identifier.map(
        lambda key: ';'.join(sorted(matched[key])),
    )
    papers['n_queries'] = papers.identifier.map(lambda key: len(matched[key]))
    papers['text_source'] = text_source
    return papers


def sweep_sources(
        queries: dict[str, str],
        text_sources: list[str],
        cache_dir: str = europepmc.DEFAULT_CACHE_DIR,
        max_pages: int | None = None,
) -> pd.DataFrame:
    """Sweep every route and return the corpus as one frame.

    The per-route frames are concatenated and dropped here rather than in main(), so
    they are not still alive through the hours-long fetch that follows. An all-empty
    sweep returns a frame with no columns at all, which filter_papers passes through.
    """
    swept = [
        sweep_queries(
            queries, text_source=text_source, cache_dir=cache_dir,
            max_pages=max_pages,
        )
        for text_source in text_sources
    ]
    populated = [frame for frame in swept if not frame.empty]
    if not populated:
        return pd.DataFrame()
    return pd.concat(populated, ignore_index=True)


def clean_abstract(value: object) -> str:
    """The search response's abstract as stripped text, '' if the record had none.

    Guards the missing case explicitly because str() of a float NaN is the string
    'nan', which would enter the block table as a paper's entire abstract.
    """
    return '' if pd.isna(value) else str(value).strip()


def assign_paper_id(papers: pd.DataFrame) -> pd.Series:
    """Build the key every later stage joins and groups on.

    The pmcid where there is one and the pmid otherwise. A PMCID always carries its
    PMC prefix and a PMID is bare digits, so one column holds both without collision.
    The fallback is what the abstract route needs: most of those papers have no PMCID
    at all, since Europe PMC holds no full text for them.
    """
    pmcid = papers.pmcid.fillna('').astype(str).str.strip()
    pmid = papers.pmid.fillna('').astype(str).str.strip()
    return pmcid.where(pmcid != '', pmid)


def filter_papers(
        papers: pd.DataFrame,
        retraction_markers: tuple[str, ...] = RETRACTION_MARKERS,
) -> pd.DataFrame:
    """Drop papers that cannot serve as evidence, and reduce to one row per paper_id.

    The sweep dedupes on the Europe PMC id, the only identifier every source record
    carries, but every step after this one joins on paper_id instead.
    """
    if papers.empty:
        return papers
    kept = papers.copy()

    pub_types = kept.pub_types.fillna('').str.lower()
    retracted = pd.Series(False, index=kept.index)
    for marker in retraction_markers:
        retracted = retracted | pub_types.str.contains(marker)
    logger.info(
        'Dropping %d retracted or withdrawn papers',
        int(retracted.sum()),
    )
    kept = kept[~retracted]

    no_abstract = ~kept.has_abstract.fillna(False)
    logger.info('Dropping %d papers with no abstract', int(no_abstract.sum()))
    kept = kept[~no_abstract]

    kept['paper_id'] = assign_paper_id(kept)
    # A record with neither identifier cannot be keyed, joined to an outcome or given
    # a custom_id, so it goes here rather than as a null key in the block table.
    unkeyed = kept.paper_id == ''
    n_unkeyed = int(unkeyed.sum())
    logger.info('Dropping %d papers with neither a PMCID nor a PMID', n_unkeyed)
    kept = kept[~unkeyed]

    # A paper_id reaching here twice - two source records for one article, which the
    # id-keyed dedupe upstream cannot see - would write that paper's blocks twice, and
    # iter_papers would read the two copies as one group and send the model the paper
    # interleaved with itself at twice the tokens. The empty keys are already gone,
    # which matters because duplicated() treats every missing key as equal.
    repeated = kept.paper_id.duplicated(keep=False)
    if repeated.any():
        # Union the query attribution rather than keeping the surviving row's.
        unioned = kept[repeated].groupby('paper_id').matched_queries.apply(
            lambda values: ';'.join(
                sorted({name for value in values for name in value.split(';')}),
            ),
        )
        matched = kept.loc[repeated, 'paper_id'].map(unioned)
        kept.loc[repeated, 'matched_queries'] = matched
        kept.loc[repeated, 'n_queries'] = matched.str.count(';') + 1
    extra = kept.paper_id.duplicated()
    logger.info('Dropping %d records repeating a paper_id', int(extra.sum()))
    kept = kept[~extra]

    logger.info('%d papers kept of %d swept', len(kept), len(papers))
    return kept.reset_index(drop=True)


def build_block_table(
        papers: pd.DataFrame,
        blocks_path: str | Path,
        cache_dir: str = europepmc.DEFAULT_CACHE_DIR,
        min_usable_body_chars: int = MIN_USABLE_BODY_CHARS,
        row_group_size: int = 50_000,
        fetch_workers: int = europepmc.DEFAULT_FETCH_WORKERS,
) -> pd.DataFrame:
    """Write every paper's blocks into one table and report the per-paper outcome.

    Full-text-route papers are fetched and their JATS parsed. A fetch is attempted for
    every PMCID regardless of what isOpenAccess and inEPMC say, because records exist
    with inEPMC=Y, a valid PMCID and a 404 from fullTextXML: the recorded status is the
    honest coverage figure and the flags are kept only so the disagreement can be
    quantified. Where that fetch or parse yields no usable body and no abstract of its
    own, the search response's abstract is written as the paper's one block, so it can
    still be screened at the abstract level instead of leaving the run unremarked.

    The fetches run concurrently and their results arrive in the order the rows were
    sorted into, so parsing and writing stay here on one thread and the table is the
    same whatever fetch_workers is. At corpus scale this is the difference between an
    afternoon and a day: a sweep is bounded by round trips, not by this machine.

    Everything else is screened on the abstract already in hand from the search
    response, written as a single block so packaging reads both through one code path.
    That is the abstract route by construction, and also the handful of full-text-route
    papers Europe PMC indexes without a PMCID: there is nothing to fetch for those, and
    selecting them by what they are not is what keeps them from falling between the two
    selections.

    Every paper reaching here therefore leaves with at least one block, which is what
    lets the screen treat the block table as the corpus.

    Blocks are written row group by row group rather than accumulated: at corpus scale
    the table runs to gigabytes. Each route is sorted by paper_id and written in turn,
    which is enough for packaging's requirement that a paper's rows be contiguous,
    since a paper takes exactly one route.

    Returns:
        Per-paper outcome, one row per paper written, with the columns of
        OUTCOME_COLUMNS whether or not any paper was attempted.
    """
    # A sweep that matched nothing arrives here as a frame with no columns at all,
    # which filter_papers passes through by design, so the column access below is not
    # safe to reach. An empty block table is still written: a later stage failing to
    # find the file cannot tell an empty corpus from an unfinished one.
    if papers.empty or 'paper_id' not in papers.columns:
        logger.warning('No papers to attempt; writing an empty block table')
        with pq.ParquetWriter(str(blocks_path), BLOCK_SCHEMA):
            pass
        return pd.DataFrame(columns=list(OUTCOME_COLUMNS))

    on_fulltext_route = papers.text_source == SOURCE_FULLTEXT
    fetchable = on_fulltext_route & papers.pmcid.notna()
    to_fetch = papers[fetchable].sort_values('paper_id')
    # The complement rather than the abstract route by name. Selecting both sides
    # positively left the full-text-route papers with no PMCID in neither: no outcome
    # row, no blocks, and so invisible to a screen that walks the block table. There
    # were 49 of them in the first corpus sweep, and the summary rounded 132,677 of
    # 132,726 to 100.0%, which is how a silent drop stays silent.
    abstract_only = papers[~fetchable].sort_values('paper_id')
    unfetchable = int((on_fulltext_route & ~fetchable).sum())
    logger.info(
        'Fetching full text for %d papers; %d screened on their abstract, of which '
        '%d reached the full-text route with no PMCID to fetch',
        len(to_fetch), len(abstract_only), unfetchable,
    )

    outcomes: list[dict] = []
    buffered: list[dict] = []

    def flush(writer: pq.ParquetWriter, force: bool = False) -> None:
        if buffered and (force or len(buffered) >= row_group_size):
            writer.write_table(
                pa.Table.from_pylist(buffered, schema=BLOCK_SCHEMA),
            )
            buffered.clear()

    fetched = europepmc.fetch_full_text_ordered(
        (str(pmcid) for pmcid in to_fetch.pmcid),
        cache_dir=cache_dir, max_workers=fetch_workers,
    )
    with pq.ParquetWriter(str(blocks_path), BLOCK_SCHEMA) as writer:
        for position, (row, result) in enumerate(
            zip(to_fetch.itertuples(), fetched), start=1,
        ):
            pmcid, status, xml = result
            # The pool yields in the order it was given, which is this frame's order.
            # A mismatch here means that guarantee has broken, and every paper after it
            # would have its blocks filed under another article - checked rather than
            # trusted because nothing downstream could detect it.
            if pmcid != str(row.pmcid):
                raise RuntimeError(
                    f'Fetch result for {pmcid} arrived against row {row.pmcid}; the '
                    'ordered fetch is out of step with the papers frame.',
                )
            blocks: list[dict] = []
            parse_error = None
            if xml:
                try:
                    blocks = parse_article(xml)
                except ET.ParseError as error:
                    logger.warning('%s: XML parse error (%s)', pmcid, error)
                    parse_error = str(error)

            body_chars = sum(
                len(block['text']) for block in blocks
                if block['canonical_section'] != SECTION_ABSTRACT
            )
            usable = body_chars >= min_usable_body_chars
            # A 404 or a parse error leaves no blocks at all, and the screen walks the
            # block table, so such a paper would leave the run silently rather than
            # fall back to its abstract. The search response's abstract stands in.
            #
            # Only when there is no usable body AND the parse produced no abstract of
            # its own: a paper with usable full text is screened at cue_window, which
            # selects any block carrying a cue, so adding an abstract there would
            # change what the settled packaging sends.
            has_abstract = any(
                block['canonical_section'] == SECTION_ABSTRACT for block in blocks
            )
            search_abstract = clean_abstract(row.abstract_text)
            if not usable and not has_abstract and search_abstract:
                blocks.append({
                    **unstructured_abstract_block(search_abstract),
                    'block_index': len(blocks),
                })
            outcomes.append({
                'paper_id': row.paper_id,
                'fulltext_status': status,
                'parse_ok': bool(xml) and parse_error is None,
                'parse_error': parse_error,
                'n_blocks': len(blocks),
                'body_chars': body_chars,
                'parsed_abstract_chars': sum(
                    len(block['text']) for block in blocks
                    if block['canonical_section'] == SECTION_ABSTRACT
                ),
                'usable_fulltext': usable,
            })
            for block in blocks:
                buffered.append({
                    'paper_id': row.paper_id,
                    'pmcid': pmcid,
                    'block_index': int(block['block_index']),
                    'section_path': block['section_path'],
                    'section_title': block['section_title'],
                    'canonical_section': block['canonical_section'],
                    'block_type': block['block_type'],
                    'text': block['text'],
                })
            flush(writer)
            if position % 1000 == 0:
                logger.info('  %d/%d fetched', position, len(to_fetch))

        for row in abstract_only.itertuples():
            abstract = clean_abstract(row.abstract_text)
            # usable_fulltext is False rather than null: there genuinely is none. It
            # is what the screen reads to pick the abstract packaging for this paper.
            outcomes.append({
                'paper_id': row.paper_id,
                'fulltext_status': None,
                'parse_ok': False,
                'parse_error': None,
                'n_blocks': 1 if abstract else 0,
                'body_chars': 0,
                'parsed_abstract_chars': len(abstract),
                'usable_fulltext': False,
            })
            if not abstract:
                continue
            buffered.append({
                'paper_id': row.paper_id,
                'pmcid': None if pd.isna(row.pmcid) else str(row.pmcid),
                'block_index': 0,
                **unstructured_abstract_block(abstract),
            })
            flush(writer)
        flush(writer, force=True)

    return pd.DataFrame(outcomes)


def log_corpus_summary(papers: pd.DataFrame) -> None:
    """Log the coverage numbers a screening run is scoped against.

    Broken down by route, because the two are screened on different text for different
    money and only the full-text one has a measured recall, so one coverage percentage
    over the whole corpus would describe nothing that exists.
    """
    total = len(papers)
    if not total:
        return
    logger.info('Corpus: %d papers', total)
    for text_source in TEXT_SOURCES:
        on_route = papers[papers.text_source == text_source]
        if on_route.empty:
            continue
        logger.info('  %s route: %d papers', text_source, len(on_route))
        if text_source == SOURCE_ABSTRACT:
            chars = on_route.parsed_abstract_chars.fillna(0)
            logger.info(
                '    abstract chars: median %.0f, %d with no abstract text',
                chars.median(), int((chars == 0).sum()),
            )
            continue
        usable = on_route.usable_fulltext.fillna(False)
        for label, mask in (
            ('with a PMCID', on_route.pmcid.notna()),
            ('full text 200', on_route.fulltext_status == 200),
            ('usable full text', usable),
        ):
            logger.info(
                '    %-17s %6d (%.1f%%)',
                label, int(mask.sum()), 100 * mask.mean(),
            )
        body_chars = on_route.loc[usable, 'body_chars']
        logger.info(
            '    body chars among usable: median %.0f',
            body_chars.median() if usable.any() else 0,
        )


def main() -> None:
    """Assemble the literature Negatome v3 screening corpus from Europe PMC.

    Runs each negation query to exhaustion over Europe PMC's free-text index by both
    retrieval routes, deduplicates the hits to one row per paper while keeping which
    queries matched, drops the records that cannot serve as evidence, then fetches and
    parses JATS full text for the papers that have it and takes the search response's
    abstract for the rest. Writes a dated papers table and blocks table and uploads
    them under the corpus stage prefix. The corpus is the union of the two routes;
    --text-sources scopes it to one for a diagnostic.

    Full-text availability is measured rather than inferred: every PMCID is fetched
    regardless of its isOpenAccess and inEPMC flags, because records exist whose flags
    say yes and whose full text 404s. Retrieval and usability are reported separately,
    since some pre-2000 articles return a stub with no marked-up body.

    Use --hit-counts first. It costs three requests per query - the unsplit count and
    one per route - and verifies the filter syntax against the live API; a clause
    Europe PMC does not understand shows up there rather than as a quietly truncated
    corpus. The query set itself was validated against the known Negatome v2 papers
    under TERN-2298 - see NEGATION_QUERIES.

    --queries and --max-pages scope a trial sweep but do not change where its output
    goes: both tables still upload under the corpus stage prefix, where the resolver
    picks the newest file. Pass --no-upload with them unless the partial corpus is
    meant to become the current one.

    Every response is cached locally, so a resumed sweep or fetch costs nothing and
    the exact bytes Europe PMC returned stay available for provenance. The cache
    makes a sweep a frozen snapshot; delete its search subdirectory to re-sweep
    against a moved index. Only the two tables are uploaded - the cache is far larger
    and mirroring it to the raw prefix is a deliberate separate step.

    The full-text fetches run concurrently, which is what makes a corpus-scale sweep an
    afternoon rather than a day. --fetch-workers changes the wall clock only: results
    are consumed in the order the papers were sorted into, so the tables do not depend
    on it.
    """
    setup_logging()
    parser = parse_args()
    args = parser.parse_args()

    queries = NEGATION_QUERIES
    if args.queries:
        queries = {name: NEGATION_QUERIES[name] for name in args.queries}

    if args.hit_counts:
        print(hit_counts(queries, cache_dir=args.cache_dir).to_string(index=False))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    version = LITERATURE_STAGE_VERSIONS[CORPUS_STAGE]
    run_date = args.date or date_type.today().isoformat()
    papers_path = output_dir / make_dated_filename(
        PAPERS_NAME, version, '.parquet', run_date,
    )
    blocks_path = output_dir / make_dated_filename(
        BLOCKS_NAME, version, '.parquet', run_date,
    )

    papers = sweep_sources(
        queries, args.text_sources, cache_dir=args.cache_dir,
        max_pages=args.max_pages,
    )
    logger.info(
        'Swept %d unique papers across the %s routes',
        len(papers), ' and '.join(args.text_sources),
    )
    papers = filter_papers(papers)
    if papers.empty:
        logger.warning(
            'No papers survived the sweep and filters; nothing to write. Check the '
            'query syntax with --hit-counts.',
        )
        return

    outcomes = build_block_table(
        papers, blocks_path, cache_dir=args.cache_dir,
        min_usable_body_chars=args.min_usable_body_chars,
        fetch_workers=args.fetch_workers,
    )
    # build_block_table is the only consumer of the abstract text, and it has just
    # written each one into the block table, which is where every later stage reads
    # it. Keeping the column would put a second copy of every abstract-route abstract
    # in the papers table - most of a hundred MB at corpus scale - that nothing reads.
    papers = papers.drop(columns=['abstract_text'], errors='ignore')
    papers = papers.merge(outcomes, on='paper_id', how='left')
    log_corpus_summary(papers)

    papers.to_parquet(papers_path, index=False)
    logger.info('Wrote %d papers to %s', len(papers), papers_path)
    logger.info(
        'Wrote blocks to %s (%.1f MB)', blocks_path,
        os.path.getsize(blocks_path) / 1024 ** 2,
    )

    if not args.no_upload:
        prefix = get_literature_stage_prefix(CORPUS_STAGE)
        upload_file_to_s3(str(papers_path), prefix)
        upload_file_to_s3(str(blocks_path), prefix)
        logger.info('Uploaded both tables to %s', prefix)
        logger.info(
            'Raw Europe PMC responses are cached under %s; mirror them to %s for '
            'provenance when the sweep is final',
            args.cache_dir, NEGATOME_LITERATURE_RAW_S3,
        )


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Assemble the literature Negatome v3 screening corpus from Europe PMC.',
    )
    parser.add_argument(
        '--output-dir', default=str(DEFAULT_OUTPUT_DIR),
        help='Local directory for the papers and blocks tables.',
    )
    parser.add_argument(
        '--cache-dir', default=europepmc.DEFAULT_CACHE_DIR,
        help='Local cache root for raw Europe PMC responses.',
    )
    parser.add_argument(
        '--date', help='Date stamp for the output filenames (YYYY-MM-DD).',
    )
    parser.add_argument(
        '--queries', nargs='*', choices=sorted(NEGATION_QUERIES),
        help='Run only these queries. Defaults to all of them. Pair it with '
             '--no-upload: a scoped sweep still uploads to the corpus prefix.',
    )
    parser.add_argument(
        '--text-sources', nargs='*', choices=list(TEXT_SOURCES),
        default=list(TEXT_SOURCES),
        help='Retrieval routes to sweep. fulltext is open access and read in full, '
             'abstract is closed access with the phrase in the abstract. Both by '
             'default: the corpus is the union.',
    )
    parser.add_argument(
        '--max-pages', type=int,
        help='Pages per query, for a scoped trial sweep. Pair it with --no-upload: '
             'a scoped sweep still uploads to the corpus prefix.',
    )
    parser.add_argument(
        '--min-usable-body-chars', type=int, default=MIN_USABLE_BODY_CHARS,
        help='Body characters required for retrieved text to count as usable.',
    )
    parser.add_argument(
        '--fetch-workers', type=int, default=europepmc.DEFAULT_FETCH_WORKERS,
        help='Concurrent full-text fetches. Results are consumed in order, so this '
             'changes only how long the sweep takes, never what it writes.',
    )
    parser.add_argument(
        '--hit-counts', action='store_true',
        help='Report each query hit count and exit; three requests per query, the '
             'unsplit count and one per route.',
    )
    parser.add_argument(
        '--no-upload', action='store_true',
        help='Write the tables locally without uploading them.',
    )
    return parser


if __name__ == '__main__':
    main()
