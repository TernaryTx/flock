# The agentic curation pass: what actually decides which candidate pairs enter
# the benchmark as negatives.
from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import TextIO

import anthropic
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from anthropic.types import Message

from flock import REPO_ROOT
from flock.aws import upload_file_to_s3
from flock.logging_utils import setup_logging
from flock.negatome_v3.literature import batch
from flock.negatome_v3.literature import CORPUS_BLOCKS_PATH
from flock.negatome_v3.literature import PAIRS_ROOT
from flock.negatome_v3.literature import runs
from flock.negatome_v3.literature.accessions import read_source_identity
from flock.negatome_v3.literature.accessions import require_matching_source
from flock.negatome_v3.literature.accessions import source_provenance_path
from flock.negatome_v3.literature.accessions import write_source_provenance
from flock.negatome_v3.literature.config import CURATION_MAX_TOKENS
from flock.negatome_v3.literature.config import CURATION_MODEL
from flock.negatome_v3.literature.config import CURATION_PROMPT_VERSION
from flock.negatome_v3.literature.config import CURATION_RESULT_MODEL
from flock.negatome_v3.literature.config import CurationConfig
from flock.negatome_v3.literature.grounding import ungrounded_excerpts
from flock.negatome_v3.literature.packaging import BLOCK_SEPARATOR
from flock.negatome_v3.literature.packaging import DROPPED_SECTIONS
from flock.negatome_v3.literature.pairs import print_counts
from flock.negatome_v3.literature.pairs import read_kept_pairs
from flock.negatome_v3.literature.pairs import write_table
from flock.negatome_v3.literature.vocabulary import CURATION_JUDGED
from flock.negatome_v3.literature.vocabulary import CURATION_LABELS
from flock.negatome_v3.literature.vocabulary import CURATION_NAME
from flock.negatome_v3.literature.vocabulary import CURATION_NO_ANSWER
from flock.negatome_v3.literature.vocabulary import CURATION_NONE
from flock.negatome_v3.literature.vocabulary import CURATION_NOT_JUDGED
from flock.negatome_v3.literature.vocabulary import CURATION_ORDER
from flock.negatome_v3.literature.vocabulary import CURATION_SELF_PAIR
from flock.negatome_v3.literature.vocabulary import CURATION_STAGE
from flock.negatome_v3.literature.vocabulary import CURATION_UNGROUNDED
from flock.negatome_v3.literature.vocabulary import NO_CLAIM_RELATIONSHIPS
from flock.negatome_v3.literature.vocabulary import STATUS_ORDER
from flock.negatome_v3.literature.vocabulary import STATUS_RESOLVED
from flock.negatome_v3.literature.vocabulary import USABLE_CONDITIONS
from flock.paths import get_literature_stage_prefix
from flock.paths import LITERATURE_STAGE_VERSIONS
from flock.paths import make_dated_filename
from flock.pricing import INTRO_RATES_END
from flock.pricing import token_cost
from flock.provenance import build_output_provenance
from flock.provenance import write_output_provenance
from flock.uniprot import fetch_uniprot_tsv

logger = logging.getLogger(__name__)

# The stage's own tables, beside the pair tables they are built from.
CURATION_INPUTS_PATH = PAIRS_ROOT / 'curation_inputs.parquet'
CURATION_VERDICTS_PATH = PAIRS_ROOT / 'curation_verdicts.parquet'
CARDS_CACHE_PATH = PAIRS_ROOT / 'uniprot_cards.parquet'

# What a card describes. Named here rather than in flock.uniprot because it is
# this stage's choice of what the model needs to check a mapping.
CARD_FIELDS = (
    'accession,id,protein_name,gene_names,organism_name,organism_id,length,reviewed'
)

# Which price list to cost a run at, as --rates takes it.
RATE_TIERS = ('intro', 'standard')

# Models return – and × as the six literal characters of an escape often
# enough to matter. Unescaped before the grounding check, never after, so the span
# compared is the span the model meant to quote.
ESCAPED_UNICODE_RE = re.compile(r'\\+u([0-9a-fA-F]{4})')

# How much body text one packet carries at most. A handful of papers are longer
# than any request can hold. Truncation is marked in the text and recorded in the
# input table, and grounding is checked against what was sent, so a quote from
# beyond the cut is ungrounded rather than silently accepted.
TEXT_CHAR_LIMIT = 150000
TRUNCATION_NOTE = (
    '\n\n## [truncated] The remainder of this paper was too long to include and '
    'is not available. Judge only on the text above and on what the methods tool '
    'returns.'
)


def canonical_pair(accession_a: str, accession_b: str) -> str:
    """Render the key a pair is identified by, in the packet and in a verdict.

    Args:
        accession_a: One side's accession.
        accession_b: The other side's accession.

    Returns:
        The two accessions sorted and joined with a pipe.
    """
    return '|'.join(sorted([accession_a, accession_b]))


def add_pair_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the pair key a verdict is joined back on.

    The input build and apply must agree on this key exactly or the verdicts stop
    joining, so both go through here rather than spelling it out twice.

    Args:
        frame: A pair table, modified in place.

    Returns:
        The same frame, carrying a pair column.
    """
    frame['pair'] = [
        canonical_pair(str(accession_a), str(accession_b))
        for accession_a, accession_b in zip(
            frame['accession_a'], frame['accession_b'],
        )
    ]
    return frame


def unescape_literal(text: str) -> str:
    """Turn literal backslash-u escapes back into the characters they name.

    Args:
        text: A model-supplied string.

    Returns:
        The string with any \\uXXXX sequences replaced by their character.
    """
    return ESCAPED_UNICODE_RE.sub(
        lambda match: chr(int(match.group(1), 16)), text,
    )


def collect_curation_pairs(
        frame: pd.DataFrame,
        drop_self_pairs: bool = True,
) -> pd.DataFrame:
    """Reduce a pair table to the resolved pairs curation judges.

    Args:
        frame: A pair table, already filtered to the rows the pair stage kept.
        drop_self_pairs: Drop pairs carrying one accession on both sides. There
            is no A-B interaction to adjudicate on those, so curating them buys
            nothing. Pairs whose two written names collapse onto one accession
            are logged separately from a name reported against itself, since that
            can also be a mapping error worth reading by hand.

    Returns:
        The resolved rows, with the pair key each verdict is joined back on.
    """
    resolved = add_pair_key(
        frame[frame['pair_status'] == STATUS_RESOLVED].copy(),
    )
    if drop_self_pairs:
        self_paired = resolved['accession_a'].eq(resolved['accession_b'])
        same_name = self_paired & resolved['norm_a'].eq(resolved['norm_b'])
        logger.info(
            'dropping %d pairs with one accession on both sides: %d where the '
            'screen named the same protein twice, %d where two names map onto '
            'one entry',
            int(self_paired.sum()), int(same_name.sum()),
            int((self_paired & ~same_name).sum()),
        )
        resolved = resolved[~self_paired]
    return resolved.reset_index(drop=True)


def get_cards_tsv(accessions: list[str], timeout_s: int) -> str:
    """Ask UniProt for one batch of cards.

    The packet's cards and the lookup tool's answer come through here together,
    so the two cannot drift into describing the same accession differently.

    Args:
        accessions: Accessions to describe.
        timeout_s: Request timeout.

    Returns:
        The endpoint's TSV, headers included.

    Raises:
        requests.RequestException: On any transport or HTTP failure.
    """
    return fetch_uniprot_tsv(accessions, CARD_FIELDS, timeout_s)


def fetch_cards(
        accessions: list[str],
        cache_path: Path = CARDS_CACHE_PATH,
        timeout_s: int = 60,
        chunk_size: int = 100,
) -> dict[str, dict]:
    """Fetch a UniProt card per accession, reusing anything already cached.

    The packet carries a card for every side, so the lookup tool is a second
    opinion rather than the only one.

    Args:
        accessions: Accessions to describe.
        cache_path: Parquet of cards fetched by an earlier build.
        timeout_s: Request timeout.
        chunk_size: Accessions per request. The endpoint takes a comma-separated
            list, so this bounds URL length.

    Returns:
        accession to card dict. Accessions UniProt does not return are absent,
        and their side goes into the packet with an empty card - visible to the
        model as a card it cannot check the mapping against.
    """
    cached: dict[str, dict] = {}
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        # Every value is a string, as it was in the TSV the cards were parsed
        # from: a card goes into the packet as JSON, and a parquet round trip
        # that turned Length into a numpy integer would fail to serialise.
        cached = {
            card['Entry']: card
            for card in frame.astype(str).to_dict('records')
        }
        logger.info('%d cards already cached', len(cached))

    wanted = [item for item in accessions if item and item not in cached]
    for start in range(0, len(wanted), chunk_size):
        chunk = wanted[start:start + chunk_size]
        lines = get_cards_tsv(chunk, timeout_s).strip().split('\n')
        if len(lines) < 2:
            logger.warning('no cards returned for %d accessions', len(chunk))
            continue
        header = lines[0].split('\t')
        for line in lines[1:]:
            card = dict(zip(header, line.split('\t')))
            cached[card['Entry']] = card
        logger.info(
            'fetched cards for %d/%d accessions', min(
                start + chunk_size,
                len(wanted),
            ), len(wanted),
        )

    if wanted and cached:
        write_table(pd.DataFrame(list(cached.values())), str(cache_path))
    missing = [item for item in accessions if item and item not in cached]
    if missing:
        logger.warning(
            '%d accessions have no UniProt card, e.g. %s', len(missing),
            ', '.join(missing[:5]),
        )
    return cached


def read_paper_texts(
        blocks_path: str | Path,
        paper_ids: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Read body text and methods text for the papers being curated.

    Scanned row group by row group rather than loaded: the block table holds the
    whole corpus, of which this stage wants a fraction.

    Args:
        blocks_path: Local parquet of parsed blocks from the corpus stage.
        paper_ids: The papers to read.

    Returns:
        (body text with methods dropped, methods text) keyed by paper_id. Methods
        is dropped from the packet for the same reason the screen drops it -
        about 18% of body characters, almost never carrying an interaction claim
        - and is served by a tool instead, so a verdict that turns on the assay
        can still reach it.
    """
    handle = pq.ParquetFile(str(blocks_path))
    columns = [
        'paper_id', 'block_index', 'canonical_section', 'section_title', 'text',
    ]
    # Filtered in Arrow and converted once at the end rather than converted per
    # row group and filtered in pandas, which would build a Python string for
    # every row it then throws away.
    wanted = pa.array(sorted(paper_ids))
    tables = []
    for group in range(handle.num_row_groups):
        table = handle.read_row_group(group, columns=columns)
        table = table.filter(
            pc.is_in(table.column('paper_id'), value_set=wanted),
        )
        if table.num_rows:
            tables.append(table)
        if group % 20 == 0:
            logger.info(
                'scanned %d/%d row groups',
                group, handle.num_row_groups,
            )
    blocks = (
        pa.concat_tables(tables).to_pandas() if tables
        else pd.DataFrame(columns=columns)
    )

    body: dict[str, str] = {}
    methods: dict[str, str] = {}
    for paper_id, paper in blocks.groupby('paper_id'):
        ordered = paper.sort_values('block_index')
        dropped = ordered['canonical_section'].isin(DROPPED_SECTIONS)
        # The screen's separator, not a copy of it: the grounding gate
        # collapses exactly this whitespace, and a quote spanning a block
        # boundary only matches because the two agree.
        body[str(paper_id)] = BLOCK_SEPARATOR.join(
            f'## [{row.canonical_section}] '
            f'{row.section_title or row.canonical_section}'
            f'{BLOCK_SEPARATOR}{row.text}'
            for row in ordered[~dropped].itertuples()
        )
        methods[str(paper_id)] = BLOCK_SEPARATOR.join(
            ordered[dropped]['text'].tolist(),
        )
    return body, methods


def chosen_side(row: object, suffix: str) -> dict:
    """How one side's accession was arrived at, as the packet states it.

    Args:
        row: One pair's row, as a namedtuple.
        suffix: 'a' or 'b'.

    Returns:
        The provenance block for that side. An empty source is rendered as the
        thing it means rather than as a blank the model has to interpret.
    """
    return {
        'name_source': str(getattr(row, f'name_source_{suffix}')) or 'name_matched_directly',
        'species_source': str(getattr(row, f'species_source_{suffix}')) or 'no_species_check_ran',
        'candidates_considered': int(getattr(row, f'n_candidates_{suffix}')),
        'assignment': str(getattr(row, f'provenance_{suffix}')),
    }


def build_pairs_payload(
        paper_pairs: pd.DataFrame,
        cards: dict[str, dict],
) -> list[dict]:
    """Render one paper's pairs as the packet describes them.

    How each accession was chosen travels with it, because that is what tells the
    model where to look hardest: a side whose species_source is empty had no
    organism check run against it at all.

    Args:
        paper_pairs: This paper's resolved pairs.
        cards: UniProt cards, keyed by accession.

    Returns:
        One entry per pair, in table order.
    """
    payload = []
    for row in paper_pairs.itertuples():
        payload.append({
            'pair': str(row.pair),
            'written_name_a': str(row.name_a),
            'written_name_b': str(row.name_b),
            'accession_a': str(row.accession_a),
            'accession_b': str(row.accession_b),
            'card_a': cards.get(str(row.accession_a), {}),
            'card_b': cards.get(str(row.accession_b), {}),
            'how_accession_a_was_chosen': chosen_side(row, 'a'),
            'how_accession_b_was_chosen': chosen_side(row, 'b'),
            'screen_excerpts': [str(item) for item in row.excerpts],
        })
    return payload


def build_inputs(
        frame: pd.DataFrame,
        blocks_path: str | Path,
        cards: dict[str, dict],
        max_chars: int = TEXT_CHAR_LIMIT,
) -> pd.DataFrame:
    """Build the table of papers to curate, one row per request.

    The text is carried in the table rather than re-read at request time, so what
    a verdict is grounded against is the text the run actually sent and a corpus
    rebuild cannot quietly change it underneath a resume.

    Args:
        frame: The resolved pairs, from collect_curation_pairs.
        blocks_path: Local parquet of parsed blocks from the corpus stage.
        cards: UniProt cards, keyed by accession.
        max_chars: Longest body text a packet carries. A few papers are longer
            than any request can hold.

    Returns:
        The input table: one row per paper, carrying its pairs, its text and its
        methods.
    """
    body, methods = read_paper_texts(
        blocks_path, set(frame['paper_id'].astype(str)),
    )
    rows = []
    for grouped_id, paper_pairs in frame.groupby('paper_id', sort=True):
        paper_id = str(grouped_id)
        text = body.get(paper_id, '')
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + TRUNCATION_NOTE
        rows.append({
            'custom_id': paper_id,
            'paper_id': paper_id,
            'pmid': str(paper_pairs['pmid'].iloc[0]),
            'pmcid': str(paper_pairs['pmcid'].iloc[0]),
            'n_pairs': len(paper_pairs),
            'pairs_json': json.dumps(build_pairs_payload(paper_pairs, cards)),
            'text': text,
            'methods': methods.get(paper_id, ''),
            'text_truncated': truncated,
            'text_chars_full': len(body.get(paper_id, '')),
        })
    inputs = pd.DataFrame(rows)
    if inputs['text_truncated'].any():
        logger.warning(
            '%d papers were truncated to %d characters',
            int(inputs['text_truncated'].sum()), max_chars,
        )
    empty = inputs['text'].eq('')
    if empty.any():
        logger.warning(
            '%d papers have no body text in the block table and would be judged '
            'on the packet alone; they are dropped', int(empty.sum()),
        )
    return inputs[~empty].reset_index(drop=True)


def build_packet(item: dict) -> dict:
    """Assemble the self-contained packet for one paper.

    The serialised JSON is what the model reads, so the key order is part of the
    packet rather than an implementation detail.

    Args:
        item: One row of the input table, as a dict.

    Returns:
        The packet, ready to serialise.
    """
    return {
        'pmid': item['pmid'],
        'pmcid': item['pmcid'],
        'n_pairs': int(item['n_pairs']),
        'pairs': json.loads(item['pairs_json']),
        'paper_text': item['text'],
    }


def build_messages(packet: dict) -> list[dict]:
    """Render the packet as the single user turn the loop starts from.

    The packet block carries a cache breakpoint, so every turn after the first
    re-reads it at cache rates rather than paying for the paper again.

    Args:
        packet: The paper's packet.

    Returns:
        The messages array.
    """
    return [{
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': (
                'Curate the pairs in this packet.\n\n'
                f"```json\n{json.dumps(packet, indent=1)}\n```"
            ),
            'cache_control': {'type': 'ephemeral'},
        }],
    }]


def usage_dict(usage: object) -> dict:
    """Flatten one response's usage, keeping the thinking split when present.

    Args:
        usage: The SDK's usage object.

    Returns:
        A plain dict of token counts. thinking_tokens is 0 on the streaming path,
        which does not populate output_tokens_details; billing is unaffected,
        since output_tokens is what is charged.
    """
    fields = (
        'input_tokens', 'output_tokens',
        'cache_creation_input_tokens', 'cache_read_input_tokens',
    )
    flat = {field: getattr(usage, field, 0) or 0 for field in fields}
    details = getattr(usage, 'output_tokens_details', None)
    flat['thinking_tokens'] = getattr(details, 'thinking_tokens', 0) or 0
    return flat


def lookup_uniprot(
        accessions: list[str],
        timeout_s: int = 30,
        max_accessions: int = 50,
) -> str:
    """Answer a uniprot_lookup tool call.

    Args:
        accessions: Accessions to describe.
        timeout_s: Request timeout.
        max_accessions: Enough for any packet's cards. Bounds URL length rather
            than anything the model can ask for.

    Returns:
        A TSV table, or an error line the model can read and work around. A
        failed lookup is never raised: it would cost the whole paper, and the
        packet already carries a card for every side.
    """
    if not accessions:
        return 'No accessions supplied.'
    try:
        text = get_cards_tsv(accessions[:max_accessions], timeout_s)
    except requests.RequestException as error:
        return f'UniProt lookup failed: {error}'
    return text.strip() or 'No entries returned.'


def ground_verdicts(verdicts: list[dict], source_text: str) -> list[dict]:
    """Mark each verdict with whether its quoted sentence is real.

    An ungrounded verdict is voided rather than corrected: the sentence is the
    only part of the answer that can be checked against the paper, so a verdict
    that misquotes it has nothing left standing behind it.

    A verdict that reports no claim either way is the exception: there is nothing
    for it to quote, so an empty sentence is consistent with what it says rather
    than a failure to copy.

    Args:
        verdicts: The model's verdicts for one paper.
        source_text: Exactly the text that reached the model - the packet's body
            text, plus the methods section only if the model asked for it. A
            quote matching methods it never requested is recall, not reading,
            which is the failure this stage is meant to catch.

    Returns:
        The verdicts, each carrying grounded and curation_outcome.
    """
    # Every sentence checked in one call, because ungrounded_excerpts normalises
    # the source once per call and the source here is a whole paper: per-verdict
    # calls would re-normalise 150,000 characters for each of the paper's pairs,
    # on the worker threads the API reads are sharing.
    sentences = [
        unescape_literal(str(verdict.get('evidence_sentence', '')))
        for verdict in verdicts
    ]
    ungrounded = (
        set(ungrounded_excerpts(sentences, source_text)) if source_text
        else set(sentences)
    )
    grounded = []
    for verdict, sentence in zip(verdicts, sentences):
        item = dict(verdict)
        relationship = str(item.get('relationship', ''))
        no_claim = (
            not sentence.strip() and relationship in NO_CLAIM_RELATIONSHIPS
        )
        item['grounded'] = no_claim or sentence not in ungrounded
        item['curation_outcome'] = (
            CURATION_JUDGED if item['grounded'] else CURATION_UNGROUNDED
        )
        grounded.append(item)
    return grounded


def verdict_is_usable(verdict: dict) -> bool:
    """Whether one verdict clears the benchmark gate, on its own fields.

    The scalar half of usable_from_verdict, sharing its conditions so a verdict
    judged mid-run and the same verdict judged at publish cannot disagree.

    Args:
        verdict: One grounded verdict.

    Returns:
        True if the mapping, the relationship, the qualifier and the grounding
        all hold.
    """
    fields_hold = all(
        str(verdict.get(field, '')) == expected
        for field, expected in USABLE_CONDITIONS
    )
    return fields_hold and bool(verdict.get('grounded'))


def dedupe_verdicts(verdicts: list[dict]) -> tuple[list[dict], int]:
    """Keep one verdict per pair, resolving a repeat conservatively.

    A paper's answer can carry the same pair twice. Where the repeats disagree
    the stricter one wins - a pair one verdict calls unusable does not become
    usable because another says so - and where they agree the first is kept.

    Strictness is the recomputed gate, not the model's own usable_as_negative:
    that flag is the one thing this stage declines to trust anywhere else, and
    letting it decide which of two verdicts survives would put it back in the
    path it was taken out of.

    Args:
        verdicts: One paper's verdicts, already grounded.

    Returns:
        (the deduplicated verdicts in first-seen order, how many were dropped).
    """
    kept: dict[str, dict] = {}
    dropped = 0
    for verdict in verdicts:
        pair = str(verdict.get('pair', ''))
        existing = kept.get(pair)
        if existing is None:
            kept[pair] = verdict
            continue
        dropped += 1
        if verdict_is_usable(existing) and not verdict_is_usable(verdict):
            kept[pair] = verdict
    return list(kept.values()), dropped


def pair_coverage(
        verdicts: list[dict],
        expected: set[str],
) -> tuple[list[str], list[str]]:
    """Which of the packet's pairs an answer missed, and which it invented.

    The schema cannot state this: it accepts any list of verdicts, including an
    empty one, so an answer covering three of a paper's eight pairs is valid
    JSON. Left unchecked it is also a finished paper - the record carries no
    error, so no later invocation re-attempts it, and the five pairs it never
    judged reach the published table as though the model had declined them.

    Args:
        verdicts: The paper's verdicts, already deduplicated.
        expected: The pair keys the packet asked about.

    Returns:
        (pairs in the packet with no verdict, pairs judged that were not in the
        packet). A pair in the second list is usually the model rewriting the key
        rather than copying it, which silently strands the real pair.
    """
    returned = {str(verdict.get('pair', '')) for verdict in verdicts}
    return sorted(expected - returned), sorted(returned - expected)


def parse_answer(text: str) -> list[dict]:
    """Validate one paper's answer against the schema.

    Args:
        text: The model's final text block.

    Returns:
        The verdicts as plain dicts.

    Raises:
        ValueError: If the text is not the schema's shape, which is a failure of
            the request rather than of the paper and so is retryable.
    """
    parsed = CURATION_RESULT_MODEL.model_validate_json(text)
    return [verdict.model_dump() for verdict in parsed.verdicts]


def call_model(
        client: anthropic.Anthropic,
        config: CurationConfig,
        messages: list[dict],
) -> Message:
    """Send one turn.

    Streamed, not because the output is read incrementally but because the SDK
    refuses a non-streaming request whose max_tokens could run past ten minutes,
    which this effort level's does.

    Args:
        client: API client.
        config: The run configuration.
        messages: The conversation so far.

    Returns:
        The final message of the stream.
    """
    # config.params and config.tools are deep copies already, so popping and
    # mutating here cannot reach the module constants behind them.
    params = config.params
    output_config = params.pop('output_config', {})
    output_config['format'] = {
        'type': 'json_schema', 'schema': config.output_json_schema,
    }
    with client.messages.stream(
        model=config.model,
        max_tokens=config.max_tokens,
        system=[{
            'type': 'text',
            'text': config.system_prompt,
            'cache_control': {'type': 'ephemeral'},
        }],
        messages=messages,
        tools=config.tools,
        output_config=output_config,
        **params,
    ) as stream:
        return stream.get_final_message()


def blank_record(item: dict, model: str) -> dict:
    """Every field a record carries, with nothing answered yet.

    Both the answered path and the failed-outside-the-loop path start here, so a
    field added to one cannot go missing from the other - which would turn every
    out-of-loop failure into a KeyError in status, i.e. break the reporting of
    exactly the failures being reported.

    Args:
        item: One row of the input table, as a dict.
        model: Who was asked.

    Returns:
        The record, ready to be filled in.
    """
    return {
        'custom_id': str(item['custom_id']),
        'paper_id': str(item['paper_id']),
        'pmid': str(item['pmid']),
        'model': model,
        'n_pairs': int(item['n_pairs']),
        'verdicts': [],
        'n_duplicates': 0,
        'stop_reason': '',
        'methods_served': False,
        'tool_calls': [],
        'turns': 0,
        'usage': [],
        'cost_usd': 0.0,
        'seconds': 0.0,
        'error': None,
    }


def curate_paper(
        client: anthropic.Anthropic,
        config: CurationConfig,
        item: dict,
        max_turns: int = 8,
        tier: str = 'intro',
) -> dict:
    """Run one paper's tool loop and return its record.

    Tool calls are served locally: UniProt live, and the methods section the
    packet left out. Whether methods was served is recorded and decides what the
    evidence is grounded against, since a quote from a section the model never
    asked for is recall rather than reading.

    Args:
        client: API client.
        config: The run configuration.
        item: One row of the input table, as a dict.
        max_turns: Turn ceiling. Reaching it is recorded as a failure, not as an
            empty answer, so the paper is re-attempted rather than published as
            having no verdicts.
        tier: Which price list to cost the run at.

    Returns:
        The record for this paper: its verdicts, what it cost, and how it failed
        if it did.
    """
    messages = build_messages(build_packet(item))
    # Accumulated in the record itself rather than in locals copied out at the
    # end, so blank_record stays the one statement of what a record carries.
    record: dict = blank_record(item, config.model)
    started = time.monotonic()

    for _ in range(max_turns):
        try:
            response = call_model(client, config, messages)
        except anthropic.APIError as error:
            # Recorded rather than raised: one paper failing must not end the
            # run, and a failed record is what the next invocation re-attempts.
            record['error'] = f'{type(error).__name__}: {error}'
            break
        record['usage'].append(usage_dict(response.usage))
        record['stop_reason'] = response.stop_reason or ''

        if response.stop_reason != 'tool_use':
            text = next(
                (block.text for block in response.content if block.type == 'text'),
                '',
            )
            if not text.strip():
                # Zero output tokens, no answer, billed for input and no error
                # from the API. Recorded as an error here so the paper is retried
                # rather than published as a paper with nothing to say.
                record['error'] = f'empty answer (stop_reason {response.stop_reason})'
                break
            try:
                verdicts = parse_answer(text)
            except ValueError as error:
                record['error'] = f'ValidationError: {error}'
                break
            source = item['text']
            if record['methods_served']:
                source = f"{source}{BLOCK_SEPARATOR}{item['methods']}"
            verdicts, duplicates = dedupe_verdicts(
                ground_verdicts(verdicts, source),
            )
            record['n_duplicates'] = duplicates
            missing, unexpected = pair_coverage(
                verdicts,
                {
                    str(pair['pair'])
                    for pair in json.loads(item['pairs_json'])
                },
            )
            if missing or unexpected:
                # An error rather than a short answer, and the verdicts are left
                # off the record: a paper is either answered whole or re-attempted
                # whole, since a half-answer recorded as done is a half-answer
                # published as final.
                record['error'] = (
                    f'incomplete answer: {len(missing)} of '
                    f"{record['n_pairs']} pairs have no verdict "
                    f'({", ".join(missing[:3])}), {len(unexpected)} verdicts '
                    f'name pairs not in the packet ({", ".join(unexpected[:3])})'
                )
                break
            record['verdicts'] = verdicts
            break

        messages.append({'role': 'assistant', 'content': response.content})
        results = []
        for block in response.content:
            if block.type != 'tool_use':
                continue
            record['tool_calls'].append(block.name)
            if block.name == 'uniprot_lookup':
                content = lookup_uniprot(
                    [
                        str(accession)
                        for accession in block.input.get('accessions', [])
                    ],
                )
            elif block.name == 'paper_methods':
                record['methods_served'] = True
                # The whole section, uncut. Grounding widens to exactly what was
                # served, so any cap here would let a quote from past it read as
                # verbatim when the model never saw the text - the recall this
                # stage exists to catch.
                content = str(item['methods'])
                if not content:
                    content = 'No methods section is available for this paper.'
            else:
                content = f'Unknown tool: {block.name}'
            results.append({
                'type': 'tool_result',
                'tool_use_id': block.id,
                'content': content,
            })
        messages.append({'role': 'user', 'content': results})
    else:
        record['error'] = f'hit the {max_turns}-turn ceiling with no answer'

    record['turns'] = len(record['usage'])
    # No batch tier: a multi-turn tool loop cannot go through the Batch API, so
    # the 50% discount does not exist for this stage.
    record['cost_usd'] = sum(
        token_cost(usage, config.model, intro=tier == 'intro')
        for usage in record['usage']
    )
    record['seconds'] = round(time.monotonic() - started, 1)
    return record


def packet_pairs(inputs: pd.DataFrame, custom_ids: set[str]) -> dict[str, set[str]]:
    """The pair keys each named paper's packet asks about.

    Args:
        inputs: The input table.
        custom_ids: The papers to read, which is only ever the ones already
            answered - parsing every paper's pairs when nothing is recorded
            against them would be work with nothing to check.

    Returns:
        custom_id to its packet's pair keys.
    """
    wanted = inputs[inputs['custom_id'].astype(str).isin(custom_ids)]
    return {
        str(row.custom_id): {
            str(pair['pair']) for pair in json.loads(row.pairs_json)
        }
        for row in wanted.itertuples()
    }


def completed_custom_ids(run_dir: Path, inputs: pd.DataFrame) -> set[str]:
    """The papers whose recorded answer still covers the packet they were sent.

    Done is decided by re-checking the coverage rule against what is on disk
    rather than by trusting that a record without an error was complete when it
    was written. A record predating a change to that rule, or written before the
    check existed at all, would otherwise be resumed past forever while apply
    reports its unjudged pairs and refuses to publish - a state no re-run could
    repair.

    Args:
        run_dir: The run directory.
        inputs: The input table the run is answering.

    Returns:
        The custom_ids that need no further work.
    """
    answered = {
        str(record['custom_id']): record['verdicts']
        for record in runs.latest_records(
            run_dir, lambda item: {
                'custom_id': item['custom_id'],
                'error': item.get('error'),
                'verdicts': [
                    {'pair': verdict['pair']} for verdict in item['verdicts']
                ],
            },
        )
        if not record['error']
    }
    expected = packet_pairs(inputs, set(answered))
    done = set()
    stale = 0
    for custom_id, verdicts in answered.items():
        missing, unexpected = pair_coverage(
            verdicts, expected.get(custom_id, set()),
        )
        if missing or unexpected:
            stale += 1
            continue
        done.add(custom_id)
    if stale:
        logger.warning(
            '%d recorded answers no longer cover their packet and are being '
            're-attempted', stale,
        )
    return done


def pending_positions(inputs: pd.DataFrame, done: set[str]) -> list[int]:
    """The row positions still to curate.

    Positions rather than the items themselves: an item carries the whole paper,
    so a run would otherwise hold a second copy of every one of them alongside
    the table they came from.

    Args:
        inputs: The input table.
        done: custom_ids already answered.

    Returns:
        Positions into the table, in table order.
    """
    return [
        position
        for position, custom_id in enumerate(inputs['custom_id'].astype(str))
        if custom_id not in done
    ]


def build_item(inputs: pd.DataFrame, position: int) -> dict:
    """Build one paper's request item from its row.

    Args:
        inputs: The input table.
        position: The row's position in it.

    Returns:
        Everything one request needs, and everything its answer is checked
        against.
    """
    row = inputs.iloc[position]
    return {
        'custom_id': str(row['custom_id']),
        'paper_id': str(row['paper_id']),
        'pmid': str(row['pmid']),
        'pmcid': str(row['pmcid']),
        'n_pairs': int(row['n_pairs']),
        'pairs_json': str(row['pairs_json']),
        'text': str(row['text']),
        'methods': str(row['methods']),
    }


def select_papers(
        inputs: pd.DataFrame,
        pending: list[int],
        papers: tuple[str, ...],
) -> list[int]:
    """Narrow the pending positions to named papers.

    Args:
        inputs: The input table.
        pending: Positions still to curate.
        papers: paper_ids or pmids to keep. Either identifies a paper, since a
            paper without a PMCID is keyed on its pmid and one with both carries
            the two in different columns.

    Returns:
        The subset of pending positions naming those papers.

    Raises:
        ValueError: If any name matches no pending row. A typo would otherwise
            quietly curate fewer papers than asked for, and a name that is
            already answered would look the same as one that does not exist.
    """
    wanted = set(papers)
    custom_ids = inputs['custom_id'].astype(str).tolist()
    pmids = inputs['pmid'].astype(str).tolist()
    selected = []
    found: set[str] = set()
    # Selected and accounted for in one pass, so the rule that picks a paper and
    # the rule that decides a name was found cannot be edited apart.
    for position in pending:
        hit = wanted & {custom_ids[position], pmids[position]}
        if hit:
            found |= hit
            selected.append(position)
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(
            f'{len(missing)} named papers are not pending in this run: '
            f'{", ".join(missing)}. They are either absent from the input table '
            f'or already answered.',
        )
    return selected


def next_records_index(run_dir: Path) -> int:
    """The chunk index a new invocation's records file takes.

    Derived from the files already in the records directory rather than from the
    manifest's batch list, which this stage has none of: there are no batches to
    poll, only local invocations, and inventing entries there to advance a
    counter would put ids in the manifest that address nothing.

    Args:
        run_dir: The run directory.

    Returns:
        One past the highest index already written, or 0.
    """
    existing = sorted((run_dir / runs.RECORDS_SUBDIR).glob('*.jsonl'))
    return max((int(path.stem) for path in existing), default=-1) + 1


class RunLedger:
    """The shared state of a threaded run: spend, records and progress.

    One lock covers all three. They are written together on every paper and the
    spend ceiling is only meaningful if it is read and updated atomically with
    the record it belongs to.

    Args:
        handle: Open records file for this invocation.
        max_cost: Dollars this invocation may spend before it stops starting
            papers.
        total: How many papers this invocation set out to curate, for the
            progress line.
    """

    def __init__(self, handle: TextIO, max_cost: float, total: int) -> None:
        self.handle = handle
        self.max_cost = max_cost
        self.total = total
        self.spent = 0.0
        self.n_done = 0
        self.n_failed = 0
        self.n_skipped = 0
        self._lock = threading.Lock()

    def over_budget(self) -> bool:
        with self._lock:
            return self.spent >= self.max_cost

    def skip(self) -> None:
        with self._lock:
            self.n_skipped += 1

    def record(self, record: dict) -> None:
        """Write one paper's record, count it, and print its line."""
        with self._lock:
            self.handle.write(json.dumps(record) + '\n')
            self.handle.flush()
            self.spent += record['cost_usd']
            self.n_done += 1
            if record['error']:
                self.n_failed += 1
            flag = f"  FAILED {record['error']}" if record['error'] else ''
            print(
                f"[{self.n_done}/{self.total}] {record['pmid']}: "
                f"{record['turns']} turns, {len(record['tool_calls'])} tool "
                f"calls, {len(record['verdicts'])}/{record['n_pairs']} verdicts, "
                f"${record['cost_usd']:.4f}, {record['seconds']:.0f}s | running "
                f"${self.spent:.2f}{flag}",
                flush=True,
            )


def run_curation(
        config: CurationConfig,
        inputs: pd.DataFrame,
        run_dir: Path,
        workers: int = 8,
        max_cost: float = 2500.0,
        max_turns: int = 8,
        tier: str = 'intro',
        limit: int = 0,
        papers: tuple[str, ...] = (),
) -> RunLedger:
    """Curate every paper not already answered, several at a time.

    Args:
        config: The run configuration.
        inputs: The input table.
        run_dir: The run directory.
        workers: How many papers are in flight at once. A paper takes about a
            minute, so this is what makes the full run finish overnight.
        max_cost: Dollars this invocation may spend before it stops starting new
            papers. Papers already in flight finish.
        max_turns: Turn ceiling per paper.
        tier: Which price list to cost the run at.
        limit: Stop after this many papers, 0 for all of them.
        papers: Curate only these, by paper_id or pmid. For a pilot aimed at
            named papers rather than at whatever sorts first - which is how a
            handful of assignments known to be wrong get put in front of the
            rubric's mapping half.

    Returns:
        The ledger, for the caller to report.
    """
    # A record carrying an error is not done, and neither is one whose verdicts
    # no longer cover its packet: the next invocation re-attempts both, which is
    # what makes this stage's retry the same command as its run.
    done = completed_custom_ids(run_dir, inputs)
    if done:
        print(f'{len(done):,} papers already answered, resuming past them')
    pending = pending_positions(inputs, done)
    if papers:
        pending = select_papers(inputs, pending, papers)
    if limit:
        pending = pending[:limit]
    total = len(pending)
    print(f'{total:,} papers to curate, {workers} at a time, ceiling ${max_cost:,.2f}')

    chunk_index = next_records_index(run_dir)
    client = batch.get_client()
    with open(runs.records_path(run_dir, chunk_index), 'w') as handle:
        ledger = RunLedger(handle, max_cost, total)

        def curate(position: int) -> None:
            if ledger.over_budget():
                ledger.skip()
                return
            item = build_item(inputs, position)
            try:
                record = curate_paper(client, config, item, max_turns, tier)
            except Exception as error:  # noqa: BLE001 - see below
                # Anything curate_paper does not already record: one paper must
                # not be able to end the run, and a recorded failure is what the
                # next invocation re-attempts. The error is written into the
                # record, so nothing is swallowed.
                logger.exception('%s failed outside the loop', item['pmid'])
                record = blank_record(item, config.model)
                record['error'] = f'{type(error).__name__}: {error}'
            ledger.record(record)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(curate, pending))

    manifest = runs.read_manifest(run_dir)
    manifest.setdefault('sessions', []).append({
        'chunk_index': chunk_index,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'n_papers': ledger.n_done,
        'n_failed': ledger.n_failed,
        'n_skipped': ledger.n_skipped,
        'cost_usd': round(ledger.spent, 4),
        'model': config.model,
        'rate_tier': tier,
    })
    runs.write_manifest(run_dir, manifest)
    return ledger


def merged_records(run_dirs: list[Path]) -> list[dict]:
    """The latest record per paper across several runs, later runs winning.

    A paper the settled configuration cannot answer - one whose verdicts plus
    thinking exceed max_tokens - is answerable under a larger budget, and a
    larger budget is a different run_id by construction. Merging here is what
    lets those answers reach the published table without pretending they came
    from the run that failed on them.

    Args:
        run_dirs: Run directories, in precedence order: a paper answered in more
            than one is taken from the last that answered it.

    Returns:
        One record per paper.
    """
    merged: dict[str, dict] = {}
    for run_dir in run_dirs:
        records = runs.latest_records(run_dir)
        logger.info('%s: %d records', run_dir.name, len(records))
        for record in records:
            custom_id = str(record['custom_id'])
            incumbent = merged.get(custom_id)
            # A later run wins on the papers it answered, which is the point of
            # merging, and loses on the ones it failed: an errored record
            # carries no verdicts, so letting one supersede an answer already in
            # hand would turn a judged paper into a gap that --allow-incomplete
            # would go on to publish. It is still stored when nothing better
            # exists, so a paper no run could answer stays counted as a failure
            # rather than dropping out of the accounting altogether.
            if incumbent and record['error'] and not incumbent['error']:
                continue
            merged[custom_id] = record
    return list(merged.values())


# What a run folded in with --extra-run-id may differ from the primary on: the
# label that makes it a separate run at all, the output ceiling it exists to
# raise, and the run_id both of those derive into. Everything else the config
# freezes - the rubric, the schema, the tool definitions, the model, the packet
# version - is what a verdict means, so a run differing there answered a
# different question and cannot stand in for the primary.
MERGEABLE_CONFIG_KEYS = ('label', 'max_tokens', 'run_id')


def require_mergeable_run(
        run_dir: Path,
        primary: Path,
        source: str,
        allow_unverified: bool = False,
) -> bool:
    """Refuse to fold in a run that answered a different question.

    Records merge on custom_id, which is the paper id, so any two runs over
    overlapping papers collide by construction - that collision is what makes
    --extra-run-id work at all. What makes it safe is that both runs were built
    from the same pair table and the same input packet and answered under the
    same rubric, none of which the merge can check for itself.

    Args:
        run_dir: The run being folded in.
        primary: The run it is being folded into.
        source: The pair table apply was pointed at.
        allow_unverified: Proceed when this run recorded no source binding.

    Returns:
        Whether this run's source was checked against a recorded binding, for
        the caller to combine with every other contributing run's.

    Raises:
        RuntimeError: If it was built from a different input table, or under a
            configuration differing anywhere but MERGEABLE_CONFIG_KEYS.
    """
    verified = require_matching_source(
        run_dir, source, allow_unverified=allow_unverified,
    )
    recorded = runs.read_manifest(run_dir).get('corpus')
    expected = runs.read_manifest(primary).get('corpus')
    if recorded != expected:
        raise RuntimeError(
            f'{run_dir.name} was built from a different input table than '
            f'{primary.name} ({recorded} against {expected}), so the packets '
            f'it answered are not the ones the primary run was asked about.',
        )
    config = runs.read_run_config(run_dir)
    primary_config = runs.read_run_config(primary)
    differing = sorted(
        key for key in set(config) | set(primary_config)
        if key not in MERGEABLE_CONFIG_KEYS and config.get(key) != primary_config.get(key)
    )
    if differing:
        raise RuntimeError(
            f'{run_dir.name} differs from {primary.name} on {differing}, and '
            f'only {list(MERGEABLE_CONFIG_KEYS)} may differ. A run answering '
            f'under a different rubric, schema, tool set or packet version is '
            f'answering a different question, so its verdicts cannot be merged '
            f"in as though they were the primary run's.",
        )
    return verified


def load_verdicts(records: list[dict]) -> pd.DataFrame:
    """Read the verdicts out of a run's records, one row per (paper, pair).

    Args:
        records: One record per paper, from merged_records.

    Returns:
        One row per verdict, carrying the paper it came from and what answering
        it cost. Papers that failed contribute no rows and are counted separately
        by their record.
    """
    rows = []
    for record in records:
        for verdict in record['verdicts']:
            rows.append({
                'paper_id': str(record['paper_id']),
                'pmid': str(record['pmid']),
                'model': str(record['model']),
                'methods_served': bool(record['methods_served']),
                **verdict,
            })
    frame = pd.DataFrame(rows)
    logger.info('%d verdicts over %d papers', len(frame), len(records))
    return frame


def run_records(records: list[dict]) -> pd.DataFrame:
    """The per-paper accounting table, without the verdicts.

    Args:
        records: One record per paper, from merged_records.

    Returns:
        One row per paper answered, for cost and failure accounting.
    """
    return pd.DataFrame([
        {
            'custom_id': record['custom_id'],
            'paper_id': record['paper_id'],
            'pmid': record['pmid'],
            'n_pairs': record['n_pairs'],
            'n_verdicts': len(record['verdicts']),
            'n_duplicates': record['n_duplicates'],
            'turns': record['turns'],
            'n_tool_calls': len(record['tool_calls']),
            'methods_served': record['methods_served'],
            'cost_usd': record['cost_usd'],
            'seconds': record['seconds'],
            'stop_reason': record['stop_reason'],
            'error': record['error'],
        }
        for record in records
    ])


def usable_from_verdict(verdict: pd.DataFrame) -> pd.Series:
    """Recompute the benchmark gate from the verdict's own fields.

    The model reports usable_as_negative itself and the rubric defines it, but a
    published negative should not rest on the model applying its own definition
    consistently. A pair enters only if the mapping is correct on both sides, the
    relationship is a non-interaction, the qualifier is the full-length pair, and
    the quoted evidence is real.

    Args:
        verdict: The verdict table.

    Returns:
        The recomputed flag, aligned to the table.
    """
    usable = verdict['grounded'].astype(bool)
    for field, expected in USABLE_CONDITIONS:
        usable &= verdict[field].eq(expected)
    return usable


def apply_verdicts(
        frame: pd.DataFrame,
        verdicts: pd.DataFrame,
        answered_papers: set[str],
) -> pd.DataFrame:
    """Fold the verdicts into the pair table.

    Joined on (paper_id, pair) rather than on the row order of either table: the
    same accession pair can appear twice in one paper under two written names,
    and both rows get the verdict, since the pair is what was judged.

    Args:
        frame: The pair table.
        verdicts: The verdict table, from load_verdicts, carrying the recomputed
            usable column.
        answered_papers: The papers that came back with an answer. A resolved
            pair from one of these that has no verdict was left out of an answer
            the model did give; one from any other paper was never judged at all.
            The two are different failures and are recorded as such.

    Returns:
        The table with a curation verdict on every resolved row.

    Raises:
        ValueError: If two verdicts share a (paper_id, pair), which would make
            the join ambiguous. dedupe_verdicts runs per paper before a record is
            written, so this means two records for one paper survived.
    """
    columns = [
        'mapping_verdict', 'mapping_note', 'relationship', 'evidence_sentence',
        'evidence_location', 'qualifier', 'confidence', 'reasoning',
    ]
    frame = add_pair_key(frame.copy())
    indexed = verdicts.set_index([
        verdicts['paper_id'].astype(str).rename('paper_key'),
        verdicts['pair'].astype(str).rename('pair_key'),
    ])
    if indexed.index.has_duplicates:
        duplicated = indexed.index[indexed.index.duplicated()].tolist()
        raise ValueError(
            f'{len(duplicated)} (paper, pair) keys carry more than one verdict, '
            f'e.g. {duplicated[:3]}.',
        )
    keys = pd.MultiIndex.from_arrays([
        frame['paper_id'].astype(str), frame['pair'].astype(str),
    ])

    for column in columns:
        frame[column] = indexed[column].reindex(keys).fillna('').to_numpy()
    frame['curation_outcome'] = (
        indexed['curation_outcome'].reindex(
            keys,
        ).fillna(CURATION_NONE).to_numpy()
    )
    # Cast to the nullable boolean before reindexing rather than filling an
    # object column afterwards: a reindex that introduces NaN into a bool column
    # makes it object dtype, and filling that is deprecated behaviour pandas
    # warns about on every apply.
    for source_column, target in (
        ('usable_as_negative', 'model_usable'),
        ('usable', 'usable_as_negative'),
    ):
        frame[target] = (
            indexed[source_column].astype('boolean').reindex(keys)
            .fillna(False).astype(bool).to_numpy()
        )

    missing = frame['pair_status'].eq(STATUS_RESOLVED)
    missing &= frame['curation_outcome'].eq(CURATION_NONE)
    # A pair the input build deliberately excluded is not a pair that failed to
    # answer. Both are unjudged, but only one is worth chasing, and the published
    # table would otherwise report them as having had no answer from an API they
    # were never sent to.
    self_paired = frame['accession_a'].eq(frame['accession_b'])
    answered = frame['paper_id'].astype(str).isin(answered_papers)
    frame.loc[missing & self_paired, 'curation_outcome'] = CURATION_SELF_PAIR
    missing &= ~self_paired
    frame.loc[missing & answered, 'curation_outcome'] = CURATION_NOT_JUDGED
    frame.loc[missing & ~answered, 'curation_outcome'] = CURATION_NO_ANSWER
    return frame


def require_complete_run(
        frame: pd.DataFrame,
        records: pd.DataFrame,
        allow_incomplete: bool = False,
) -> dict[str, int]:
    """Refuse to publish a table the run has not finished answering.

    The dated file this stage writes is the canonical curated table, and the
    latest one is what downstream resolves to, so publishing a pilot or a
    cost-capped invocation would make an incomplete pass current. Every gap is
    labelled in the table rather than hidden, but a label does not stop the
    upload, so the upload is stopped here.

    Ungrounded verdicts do not count: a voided verdict is a finished judgement on
    that pair. Nor do self-pairs, which were deliberately never sent.

    allow_incomplete exists for the one gap re-running cannot close: a paper the
    API refuses on bio-category grounds returns the same refusal every time, so
    the run is as finished as it will ever be. It downgrades the refusal to a
    counted gap rather than removing the check, and the counts it returns are
    written into the published provenance so a table carrying gaps says so.

    Args:
        frame: The curated pair table.
        records: The run's per-paper records.
        allow_incomplete: Publish anyway, recording the gaps.

    Returns:
        The gap counts, keyed by what they are: papers carrying an error, and
        resolved pairs at each unanswered outcome.

    Raises:
        RuntimeError: If any paper failed or any resolved pair has no verdict,
            unless allow_incomplete is set.
    """
    resolved = frame['pair_status'].eq(STATUS_RESOLVED)
    unanswered = frame.loc[resolved, 'curation_outcome'].value_counts()
    gaps = {
        outcome: int(unanswered.get(outcome, 0))
        for outcome in (CURATION_NO_ANSWER, CURATION_NOT_JUDGED)
    }
    failed = int(records['error'].notna().sum())
    counts = {'papers_failed': failed, **gaps}
    if not (failed or any(gaps.values())):
        return counts
    if allow_incomplete:
        return counts
    raise RuntimeError(
        f'This run is not finished: {failed:,} papers carry an error, '
        f'{gaps[CURATION_NO_ANSWER]:,} resolved pairs had no answer and '
        f'{gaps[CURATION_NOT_JUDGED]:,} were in a packet but absent from its '
        f'answer. The dated curated table is the canonical one, so uploading '
        f'this would make an incomplete pass current. Re-run curation until '
        f'status reports none left, pass --allow-incomplete if the gaps are '
        f'ones no re-run can close, or drop --upload to write the table '
        f'locally.',
    )


def _run_id(args: argparse.Namespace) -> str:
    """The run a subcommand addresses: named directly, or derived from the config."""
    return args.run_id or CurationConfig(
        prompt_version=args.prompt_version, label=args.label,
        max_tokens=args.max_tokens,
    ).run_id


def _run_inputs(args: argparse.Namespace) -> None:
    """Build the table of papers to curate."""
    source = read_kept_pairs(args.source)
    frame = collect_curation_pairs(
        source, drop_self_pairs=not args.keep_self_pairs,
    )
    resolved = int((source['pair_status'] == STATUS_RESOLVED).sum())
    print(
        f'{len(frame):,} resolved pairs over {frame.paper_id.nunique():,} papers '
        f'({resolved - len(frame):,} dropped as one accession on both sides)',
    )
    accessions = sorted(
        set(frame['accession_a'].astype(str)) | set(
            frame['accession_b'].astype(str),
        ),
    )
    print(f'fetching UniProt cards for {len(accessions):,} accessions')
    cards = fetch_cards(accessions, Path(args.cards))
    inputs = build_inputs(frame, args.blocks, cards, max_chars=args.max_chars)
    out_path = write_table(inputs, args.out)

    chars = inputs['text'].str.len()
    print(f'\n{len(inputs):,} papers, {int(inputs.n_pairs.sum()):,} pairs')
    print(
        f'{int(inputs.text_truncated.sum()):,} papers truncated to '
        f'{args.max_chars:,} chars',
    )
    print(
        f'paper text: mean {chars.mean():,.0f} chars, median '
        f'{chars.median():,.0f}, max {chars.max():,.0f} '
        f'(~{chars.sum() / 3.7 / 1e6:,.1f}M tokens in total)',
    )
    print(f'pairs per paper: median {inputs.n_pairs.median():.0f}, '
          f'max {inputs.n_pairs.max()}')
    print(
        f'methods available for '
        f'{int(inputs.methods.str.len().gt(0).sum()):,} papers',
    )

    # The input table cannot carry which pair table it was built from - its
    # columns are the request payload - so the identity goes beside it, run
    # copies it into the manifest, and apply refuses a source that disagrees.
    provenance_path = write_source_provenance(
        out_path, args.source, str(frame['uniprot_release'].iloc[0]),
    )
    print(f'\nwrote {out_path}')
    print(f'wrote {provenance_path}')


def _run_run(args: argparse.Namespace) -> None:
    """Curate every paper not already answered."""
    config = CurationConfig(
        prompt_version=args.prompt_version, label=args.label,
        max_tokens=args.max_tokens,
    )
    inputs = pd.read_parquet(args.inputs)
    if inputs.empty:
        raise ValueError(f'No papers in {args.inputs}.')

    run_dir = runs.make_run_dir(config.run_id)
    runs.write_run_config(run_dir, config)
    manifest = runs.read_manifest(run_dir)
    # Both bindings go through the shared check, so they are validated and
    # persisted in the same manifest write: recording one and rejecting the
    # resume afterwards would leave the run bound to a table its own verdicts
    # never came from.
    bindings = {'corpus': runs.corpus_identity(args.inputs)}
    recorded_source = read_source_identity(args.inputs)
    if recorded_source:
        bindings['pairs_source'] = recorded_source
    elif 'pairs_source' in manifest:
        # A run whose first invocation found no sidecar records no binding at
        # all; a sidecar that has since gone is the other way round, and
        # resuming without it would let apply check --source against a table the
        # earlier verdicts were never built from, and pass.
        raise RuntimeError(
            f'{config.run_id} recorded a source pair table and '
            f'{source_provenance_path(args.inputs).name} has since gone. Restore '
            f'it rather than resuming, or apply cannot check its --source.',
        )
    else:
        logger.warning(
            'No %s beside the input table, so apply cannot check its --source. '
            'Rebuild the inputs to record it.',
            source_provenance_path(args.inputs).name,
        )
    runs.bind_manifest(manifest, bindings)
    runs.write_manifest(run_dir, manifest)

    if args.rates == 'intro' and date.today() > INTRO_RATES_END:
        logger.warning(
            'Intro rates ended %s, so every cost this run reports is about a '
            'third too low. Pass --rates standard for a real figure.',
            INTRO_RATES_END.isoformat(),
        )

    print(f'run_id {config.run_id}')
    ledger = run_curation(
        config, inputs, run_dir,
        workers=args.workers, max_cost=args.max_cost, max_turns=args.max_turns,
        tier=args.rates, limit=args.limit, papers=tuple(args.papers or ()),
    )
    print(
        f'\n{ledger.n_done:,} papers answered, {ledger.n_failed:,} failed, '
        f'{ledger.n_skipped:,} not started, ${ledger.spent:,.2f} spent',
    )
    if ledger.n_failed:
        print(
            'Re-run this command to re-attempt the failures; a record with an '
            'error is not treated as done.',
        )
    if ledger.n_skipped:
        print(f'The ${args.max_cost:,.2f} ceiling stopped this invocation. '
              f'Raise --max-cost and re-run to continue.')


def _run_status(args: argparse.Namespace) -> None:
    """Report what a run has answered, what it cost, and what is left."""
    run_dir = runs.require_run_dir(_run_id(args))
    records = run_records(merged_records([run_dir]))
    if records.empty:
        print(f'{run_dir.name}: nothing recorded yet')
        return
    answered_papers = records['error'].isna()
    answered = records[answered_papers]
    # The row count off the parquet footer, not the table: status is meant to be
    # run repeatedly mid-run, and reading it whole would materialise every paper's
    # text beside the run already holding its own copy.
    n_inputs = pq.ParquetFile(args.inputs).metadata.num_rows
    remaining = n_inputs - len(answered)

    print(f'{run_dir.name}')
    print(f'  {len(answered):,} of {n_inputs:,} papers answered, '
          f'{len(records) - len(answered):,} failed')
    print(f'  ${records.cost_usd.sum():,.2f} spent so far')
    if len(answered):
        per_paper = float(answered['cost_usd'].mean())
        print(f'  {int(answered.n_verdicts.sum()):,} verdicts over '
              f'{int(answered.n_pairs.sum()):,} pairs in those papers')
        print(f'  ${per_paper:.4f}/paper, {answered.turns.mean():.2f} turns, '
              f'{int(answered.n_tool_calls.sum()):,} tool calls, '
              f'{answered.seconds.mean():.0f}s/paper')
        print(f'  {remaining:,} papers left, ~${per_paper * remaining:,.0f} to '
              f'finish')
    if int(records.n_duplicates.sum()):
        print(
            f'  {int(records.n_duplicates.sum()):,} duplicate verdicts deduplicated',
        )
    failures = records[~answered_papers]
    if len(failures):
        print('\nfailures:')
        print_counts(
            failures['error'].str.split(':').str[0], width=40, total=len(records),
        )


def _run_apply(args: argparse.Namespace) -> None:
    """Fold the verdicts into the pair table and write the dated output."""
    run_id = _run_id(args)
    extra_run_ids = list(args.extra_run_id or ())
    run_dir = runs.require_run_dir(run_id)
    extra_dirs = [runs.require_run_dir(item) for item in extra_run_ids]
    # Every run contributing a record is checked before any of them is read,
    # and the published source_verified is the weakest of them: a table is only
    # as checked as the least checked run that put a verdict in it.
    source_verified = require_matching_source(
        run_dir, args.source,
        allow_unverified=args.allow_unverified_source,
    )
    for extra_dir in extra_dirs:
        source_verified &= require_mergeable_run(
            extra_dir, run_dir, args.source,
            allow_unverified=args.allow_unverified_source,
        )
    # The extra runs come last, so a paper the primary run failed on and an
    # extra run answered is taken from the extra run.
    records = merged_records([run_dir, *extra_dirs])
    verdicts = load_verdicts(records)
    if verdicts.empty:
        raise ValueError(f'{run_id} has no verdicts to apply.')
    verdicts['usable'] = usable_from_verdict(verdicts)
    frame = read_kept_pairs(args.source)
    papers = run_records(records)
    answered_papers = papers['error'].isna()
    frame = apply_verdicts(
        frame, verdicts,
        set(papers.loc[answered_papers, 'paper_id'].astype(str)),
    )
    # Checked before anything is written, so an --upload of a run that is not
    # finished fails on the command that asked for it rather than after it has
    # produced a table someone might reach for anyway.
    # Counted either way so the provenance records them; only an --upload
    # refuses on them.
    published_gaps = require_complete_run(
        frame, papers,
        allow_incomplete=args.allow_incomplete or not args.upload,
    )
    write_table(verdicts, args.verdicts_out)

    stamp = args.date or date.today().isoformat()
    out_path = write_table(
        frame,
        str(
            Path(args.out_dir) / make_dated_filename(
                CURATION_NAME, LITERATURE_STAGE_VERSIONS[CURATION_STAGE],
                '.parquet', stamp,
            ),
        ),
    )
    provenance_path = out_path.with_suffix('.provenance.json')
    write_output_provenance(
        provenance_path,
        build_output_provenance(
            workflow='flock.negatome_v3.literature.curation.apply',
            parameters={
                'source': args.source,
                'allow_unverified_source': bool(args.allow_unverified_source),
                'allow_incomplete': bool(args.allow_incomplete),
                'run_id': run_id, 'date': stamp,
            },
            input_paths={'pairs': args.source},
            extra={
                'uniprot_release': frame['uniprot_release'].iloc[0],
                'curation_run_id': run_id,
                'additional_run_ids': extra_run_ids,
                # False means the source table was taken on the operator's word
                # rather than checked against what the run was built from, which
                # anything reading this table downstream has to be able to see.
                'source_verified': source_verified,
                # Non-zero counts here mean the table was published under
                # --allow-incomplete and carries the gaps it names.
                'published_gaps': published_gaps,
                'model': CURATION_MODEL,
                'n_papers_answered': int(answered_papers.sum()),
                'n_papers_failed': int(len(papers) - answered_papers.sum()),
                'n_verdicts': int(len(verdicts)),
                'n_ungrounded': int((~verdicts['grounded']).sum()),
                'n_duplicates_dropped': int(papers['n_duplicates'].sum()),
                'cost_usd': round(float(papers['cost_usd'].sum()), 2),
                'n_usable_negatives': int(frame['usable_as_negative'].sum()),
            },
            source_name='flock',
            repo_root=REPO_ROOT,
        ),
    )

    resolved = frame[frame['pair_status'] == STATUS_RESOLVED]
    print('\ncuration outcome, over the resolved pairs:')
    print_counts(
        resolved['curation_outcome'], width=38, total=len(resolved),
        labels=CURATION_LABELS, order=CURATION_ORDER,
    )
    judged = resolved[resolved['curation_outcome'] == CURATION_JUDGED]
    print('\nmapping verdict, over the judged pairs:')
    print_counts(judged['mapping_verdict'], width=38, total=len(judged))
    print('\nrelationship:')
    print_counts(judged['relationship'], width=38, total=len(judged))
    print('\nqualifier:')
    print_counts(judged['qualifier'], width=38, total=len(judged))
    disagreements = judged['model_usable'].ne(judged['usable_as_negative'])
    disagreed = int(disagreements.sum())
    print(f'\n{int(frame.usable_as_negative.sum()):,} usable negatives '
          f'({disagreed:,} pairs where the model\'s own flag and the recomputed '
          f'gate disagree)')
    print('\npair status of the table written:')
    print_counts(
        frame['pair_status'], width=38, total=len(frame), order=STATUS_ORDER,
    )
    print(f'\nwrote {out_path} ({len(frame):,} pairs)')
    print(f'wrote {provenance_path}')
    if args.upload:
        prefix = get_literature_stage_prefix(CURATION_STAGE)
        upload_file_to_s3(str(out_path), prefix)
        upload_file_to_s3(str(provenance_path), prefix)
        print(f'uploaded both to {prefix}')


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Curate the resolved pairs paper by paper with a tool-using model, '
            'and fold the verdicts into the pair table.'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    def add_config_args(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            '--prompt-version', default=CURATION_PROMPT_VERSION,
            help='Prompt stem.',
        )
        target.add_argument(
            '--label', default='',
            help='Extra slug in the run_id, e.g. pilot.',
        )
        target.add_argument(
            '--max-tokens', type=int, default=CURATION_MAX_TOKENS,
            help=(
                'Output ceiling per turn, thinking included, up to 128000 on '
                'this model. Raising it is a different run_id, which is how a '
                'paper too pair-dense to answer under the settled value is '
                're-answered without disturbing the runs that succeeded.'
            ),
        )

    def add_run_id_arg(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            '--run-id', default='',
            help=(
                'Address an existing run directly, for when an edit to the '
                'rubric, the tools or the schema has moved the run_id.'
            ),
        )

    inputs = subparsers.add_parser(
        'inputs', help='Build the table of papers to curate.',
    )
    inputs.set_defaults(run=_run_inputs)
    inputs.add_argument(
        '--source', required=True,
        help=(
            'The pair table to curate the resolved pairs of - the accessions '
            'stage output once the model pick has been applied.'
        ),
    )
    inputs.add_argument(
        '--blocks', default=str(CORPUS_BLOCKS_PATH),
        help='Local corpus blocks parquet, for the body and methods text.',
    )
    inputs.add_argument(
        '--cards', default=str(CARDS_CACHE_PATH),
        help='Where UniProt cards are cached between builds.',
    )
    inputs.add_argument(
        '--max-chars', type=int, default=TEXT_CHAR_LIMIT,
        help=(
            'Longest body text a packet carries, about 40,000 tokens. Anything '
            'past it is cut and the cut is marked in the text.'
        ),
    )
    inputs.add_argument(
        '--keep-self-pairs', action='store_true',
        help=(
            'Keep pairs carrying one accession on both sides, which are dropped '
            'by default: there is no A-B interaction on them to judge.'
        ),
    )
    inputs.add_argument(
        '--out', default=str(CURATION_INPUTS_PATH),
        help='Where to write the input table.',
    )

    run = subparsers.add_parser(
        'run', help='Curate every paper the run has not already answered.',
    )
    run.set_defaults(run=_run_run)
    add_config_args(run)
    run.add_argument(
        '--inputs', default=str(CURATION_INPUTS_PATH),
        help='The input table to curate.',
    )
    run.add_argument(
        '--workers', type=int, default=8,
        help='Papers in flight at once.',
    )
    run.add_argument(
        '--max-cost', type=float, default=2500.0,
        help=(
            'Dollars this invocation may spend before it stops starting papers. '
            'Papers already in flight finish.'
        ),
    )
    run.add_argument(
        '--max-turns', type=int, default=8,
        help='Turn ceiling per paper.',
    )
    run.add_argument(
        '--rates', default='intro', choices=RATE_TIERS,
        help=(
            'Which price list to cost the run at. Introductory rates end '
            f'{INTRO_RATES_END.isoformat()}.'
        ),
    )
    run.add_argument(
        '--limit', type=int, default=0,
        help='Stop after this many papers, for a pilot. 0 runs all of them.',
    )
    run.add_argument(
        '--papers', nargs='*', default=None,
        help=(
            'Curate only these papers, by paper_id or pmid, for a pilot aimed '
            'at named papers rather than at whatever sorts first.'
        ),
    )

    status = subparsers.add_parser(
        'status', help='Report progress, spend and failures.',
    )
    status.set_defaults(run=_run_status)
    add_config_args(status)
    add_run_id_arg(status)
    status.add_argument(
        '--inputs', default=str(CURATION_INPUTS_PATH),
        help='The input table, for the denominator.',
    )

    apply_parser = subparsers.add_parser(
        'apply', help='Fold the verdicts into the pair table.',
    )
    apply_parser.set_defaults(run=_run_apply)
    add_config_args(apply_parser)
    add_run_id_arg(apply_parser)
    apply_parser.add_argument(
        '--source', required=True,
        help='The pair table the inputs were built from.',
    )
    apply_parser.add_argument(
        '--extra-run-id', action='append', default=None,
        help=(
            'Fold another run\'s answers in alongside this one, repeatable and '
            'last-wins. For papers the primary run could not answer that a run '
            'under a different --max-tokens did.'
        ),
    )
    apply_parser.add_argument(
        '--allow-unverified-source', action='store_true',
        help=(
            'Apply even though the run recorded no source pair table. Only for '
            'a run predating that record, and only once you have confirmed '
            '--source is what its inputs were built from.'
        ),
    )
    apply_parser.add_argument(
        '--allow-incomplete', action='store_true',
        help=(
            'Upload even though the run carries gaps. Only for gaps no re-run '
            'can close, such as a paper the API refuses on bio-category '
            'grounds. The counts are written into the published provenance.'
        ),
    )
    apply_parser.add_argument(
        '--verdicts-out', default=str(CURATION_VERDICTS_PATH),
        help='Where to write the flat verdict table.',
    )
    apply_parser.add_argument(
        '--out-dir', default=str(PAIRS_ROOT),
        help='Directory for the dated curated pair table.',
    )
    apply_parser.add_argument(
        '--date', default='',
        help='Date stamp for the output filename. Defaults to today.',
    )
    apply_parser.add_argument(
        '--upload', action='store_true',
        help='Upload the table and its provenance to the stage prefix in S3.',
    )
    return parser


def main() -> None:
    """Curate the resolved pairs, one paper at a time, with a tool-using model.

    The stage the benchmark's negatives come out of. Each paper is sent as a
    self-contained packet - its pairs, a UniProt card per accession, how each
    accession was chosen, and the paper's text with methods dropped - and the
    model answers over a multi-turn loop it can spend on live UniProt lookups and
    on the methods section the packet withheld. A multi-turn loop is not a shape
    the Batch API has, so this stage sends its own requests and there is no 50%
    batch discount to be had.

    Four subcommands: inputs builds the packets, run answers them and is also the
    retry (a record carrying an error is not treated as done, so re-running the
    same command re-attempts it), status reports spend and progress mid-run, and
    apply folds the verdicts into the pair table and publishes it.

    A verdict whose quoted sentence is not a verbatim span of the text the model
    was shown is voided, not corrected: the quote is the only part of an answer
    that can be checked against the paper.

    Reads FLOCK_ANTHROPIC_API_KEY and fails hard if it is unset, so a run cannot
    silently bill the wrong account. This module is executed by the user, not by
    an agent.
    """
    setup_logging()
    parser = parse_args()
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
