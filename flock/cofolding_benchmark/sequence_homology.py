from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import date

import pandas as pd

from flock import FLOCK_VERSION
from flock import REPO_ROOT
from flock.aws import upload_file_to_s3
from flock.cofolding_benchmark import CUTOFF_DATE
from flock.cofolding_benchmark.date_flags import undirected_pair
from flock.http_utils import stream_to_file
from flock.logging_utils import setup_logging
from flock.paths import COFOLDING_BENCHMARK_S3
from flock.paths import make_dated_filename
from flock.pdb_metadata import PdbUniprotData
from flock.pdb_release_dates import PdbReleaseDateData
from flock.provenance import build_output_provenance
from flock.provenance import write_output_provenance
from flock.uniprot import fetch_uniprot_tsv


PDB_SEQRES_URL = 'https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz'

# MMseqs2 lives in its own conda env, as PRODIGY does — see flock.prodigy. Held
# as an env name rather than a binary path so the module runs anywhere the env
# exists; a plain mmseqs already on PATH is used in preference to spawning one.
MMSEQS_ENV = 'mmseqs2'

# Deliberately looser than either rule, so build_accession_entry_index can
# re-cut any threshold without searching again.
SEARCH_MIN_SEQ_ID = 0.25
SEARCH_COVERAGE = 0.3

HIT_COLUMNS = [
    'query', 'target', 'fident', 'alnlen', 'qlen', 'tlen',
    'qcov', 'tcov', 'evalue', 'bits',
]
HIT_FORMAT = ','.join(HIT_COLUMNS)


def build_pdb_sequence_db(
        work_dir: str,
        min_length: int = 30,
) -> tuple[str, dict[str, list[str]]]:
    """Collapse the PDB seqres protein chains onto unique sequences.

    Identical seqres strings collapse to one representative so the MMseqs2
    target database stays small - 1.15M records become ~174k sequences - and
    the representative -> chain expansion is kept so a hit traces back to every
    (pdb_id, chain) carrying that sequence.

    A database already present in work_dir is reused rather than rebuilt. The
    seqres file is reissued weekly, so rebuilding mid-project would silently
    change which entries can supply evidence between one run and the next.

    Args:
        work_dir: Directory to download into and write the FASTA to.
        min_length: Shortest chain to keep, in residues. Below this a match
            carries no homology signal and inflates the search.

    Returns:
        Tuple of (fasta path, representative id -> list of '<pdb>_<chain>').
    """
    logger = logging.getLogger(__name__)
    os.makedirs(work_dir, exist_ok=True)
    fasta_path = os.path.join(work_dir, 'pdb_prot_unique.fasta')
    chains_path = os.path.join(work_dir, 'rep_to_chains.json')
    if os.path.exists(fasta_path) and os.path.exists(chains_path):
        with open(chains_path) as handle:
            existing = json.load(handle)
        logger.info(
            'Reusing %s (%d unique sequences)', fasta_path, len(existing),
        )
        return fasta_path, existing

    # SIFTS cannot stand in for this: PdbUniprotData carries chain-to-accession
    # mappings and residue ranges but no sequences, and 9.3% of PDB protein
    # chains have no SIFTS mapping at all. Backlog: move this into
    # flock.pdb_metadata as a PdbMetadata subclass so it is versioned and
    # cached like the rest.
    seqres_path = os.path.join(work_dir, 'pdb_seqres.txt.gz')
    if not os.path.exists(seqres_path):
        logger.info('Downloading %s', PDB_SEQRES_URL)
        stream_to_file(PDB_SEQRES_URL, seqres_path)

    sequence_to_chains: dict[str, list[str]] = defaultdict(list)
    n_protein = 0
    n_short = 0
    header = None
    with gzip.open(seqres_path, 'rt') as handle:
        for line in handle:
            if line.startswith('>'):
                header = line[1:].strip()
                continue
            if header is None:
                continue
            sequence = line.strip().upper()
            if 'mol:protein' not in header:
                header = None
                continue
            n_protein += 1
            if len(sequence) < min_length:
                n_short += 1
                header = None
                continue
            sequence_to_chains[sequence].append(header.split()[0])
            header = None

    rep_to_chains: dict[str, list[str]] = {}
    with open(fasta_path, 'w') as out:
        for index, (sequence, chains) in enumerate(sequence_to_chains.items()):
            rep = f'S{index}'
            rep_to_chains[rep] = chains
            out.write(f'>{rep}\n{sequence}\n')
    with open(chains_path, 'w') as handle:
        json.dump(rep_to_chains, handle)

    logger.info(
        '%d protein chains (%d shorter than %d dropped) -> %d unique sequences',
        n_protein, n_short, min_length, len(rep_to_chains),
    )
    return fasta_path, rep_to_chains


def write_query_fasta(
        accessions: list[str],
        fasta_path: str,
        batch_size: int = 100,
) -> list[str]:
    """Fetch full-length UniProt sequences and write them as the query FASTA.

    Sequences come from the UniProt REST endpoint rather than from a local
    human-proteome FASTA: these accessions are only about 46% human, so the
    local route would silently cover half the set.

    Args:
        accessions: UniProt accessions to fetch. Deduplicated, order preserved.
        fasta_path: Where to write the FASTA.
        batch_size: Accessions per request.

    Returns:
        Accessions the endpoint returned no sequence for.
    """
    logger = logging.getLogger(__name__)
    unique = list(dict.fromkeys(accessions))
    sequences: dict[str, str] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        text = fetch_uniprot_tsv(batch, fields='accession,sequence')
        for line in text.splitlines()[1:]:
            if not line.strip():
                continue
            entry, _, sequence = line.partition('\t')
            if sequence.strip():
                sequences[entry.strip()] = sequence.strip()
        if start % (batch_size * 10) == 0:
            logger.info('  fetched %d / %d', len(sequences), len(unique))

    with open(fasta_path, 'w') as out:
        for accession, sequence in sequences.items():
            out.write(f'>{accession}\n{sequence}\n')

    missing = [a for a in unique if a not in sequences]
    logger.info(
        'Wrote %d sequences to %s (%d accessions had none)',
        len(sequences), fasta_path, len(missing),
    )
    return missing


def fasta_accessions(fasta_path: str) -> set[str] | None:
    """Read the accessions a cached query FASTA covers.

    Args:
        fasta_path: Path to a FASTA written by write_query_fasta.

    Returns:
        The accessions it holds, or None if the file does not exist.
    """
    if not os.path.exists(fasta_path):
        return None
    with open(fasta_path) as handle:
        return {
            line[1:].strip().split()[0]
            for line in handle if line.startswith('>')
        }


def write_hits_manifest(
        hits_path: str,
        accessions: list[str],
        target_fasta: str,
) -> str:
    """Record what a hit TSV was searched over, so it can be reused safely.

    A hit TSV holds only hits, so it cannot say which queries were searched
    and simply matched nothing. Without this a stale file silently turns
    unsearched pairs into homolog=False, which reads downstream as clean.

    Args:
        hits_path: The hit TSV this manifest describes.
        accessions: Every query accession the search covered.
        target_fasta: The sequence database searched against.

    Returns:
        Path to the manifest written beside the hit TSV.
    """
    manifest_path = hits_path + '.manifest.json'
    with open(manifest_path, 'w') as handle:
        json.dump(
            {
                'accessions': sorted(accessions),
                'target_fasta': os.path.basename(target_fasta),
                'target_sequences': sum(
                    1 for line in open(target_fasta) if line.startswith('>')
                ),
                'search_min_seq_id': SEARCH_MIN_SEQ_ID,
                'search_coverage': SEARCH_COVERAGE,
            }, handle,
        )
    return manifest_path


def check_hits_manifest(hits_path: str, accessions: list[str]) -> None:
    """Refuse a reused hit TSV that does not cover the current query set.

    Args:
        hits_path: The hit TSV being reused.
        accessions: Every query accession this run needs searched.

    Raises:
        FileNotFoundError: If the hit TSV has no manifest beside it.
        ValueError: If the manifest does not cover every accession, or was
            searched at floors above the ones this run re-cuts from.
    """
    logger = logging.getLogger(__name__)
    manifest_path = hits_path + '.manifest.json'
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f'{hits_path} has no {os.path.basename(manifest_path)} beside it, '
            'so there is no way to tell which queries it searched. Re-run the '
            'search instead of reusing it.',
        )
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    missing = set(accessions) - set(manifest['accessions'])
    if missing:
        raise ValueError(
            f'{hits_path} was searched over {len(manifest["accessions"])} '
            f'accessions and is missing {len(missing)} this run needs, e.g. '
            f'{sorted(missing)[:3]}. Re-run the search.',
        )
    id_too_strict = manifest['search_min_seq_id'] > SEARCH_MIN_SEQ_ID
    cov_too_strict = manifest['search_coverage'] > SEARCH_COVERAGE
    if id_too_strict or cov_too_strict:
        raise ValueError(
            f'{hits_path} was searched at id>={manifest["search_min_seq_id"]} '
            f'cov>={manifest["search_coverage"]}, above this run\'s floors '
            f'{SEARCH_MIN_SEQ_ID}/{SEARCH_COVERAGE}; it cannot be re-cut down.',
        )
    logger.info(
        'Reusing %s (%d accessions, %d target sequences)',
        hits_path, len(manifest['accessions']), manifest['target_sequences'],
    )


def run_mmseqs_search(
        query_fasta: str,
        target_fasta: str,
        hits_path: str,
        tmp_dir: str,
        threads: int = 8,
        sensitivity: float = 5.7,
        max_seqs: int = 20000,
        env: str = MMSEQS_ENV,
) -> str:
    """Run MMseqs2 easy-search of the query accessions against PDB seqres.

    Args:
        query_fasta: Query sequences, one record per accession.
        target_fasta: The unique-sequence PDB database.
        hits_path: Where to write the hit TSV.
        tmp_dir: MMseqs2 scratch directory.
        threads: Worker threads.
        sensitivity: MMseqs2 -s. 5.7 is its own sensitive default.
        max_seqs: Hits kept per query before alignment. Large because one
            accession can match thousands of PDB chains.
        env: Conda env holding mmseqs, used only when it is not on PATH.

    Returns:
        hits_path.

    Raises:
        subprocess.CalledProcessError: If MMseqs2 exits non-zero.
    """
    logger = logging.getLogger(__name__)
    on_path = shutil.which('mmseqs')
    launcher = (
        [on_path] if on_path
        else ['conda', 'run', '--no-capture-output', '-n', env, 'mmseqs']
    )
    command = [
        *launcher, 'easy-search', query_fasta, target_fasta,
        hits_path, tmp_dir,
        '--min-seq-id', str(SEARCH_MIN_SEQ_ID),
        '-c', str(SEARCH_COVERAGE), '--cov-mode', '0',
        '-e', '1e-3', '--max-seqs', str(max_seqs), '-s', str(sensitivity),
        '--threads', str(threads), '--format-output', HIT_FORMAT,
    ]
    logger.info('Running %s', ' '.join(command))
    subprocess.run(command, check=True, capture_output=True, text=True)
    return hits_path


def build_chain_accession_map(sifts: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map every (PDB entry, chain) to its SIFTS UniProt accession.

    Args:
        sifts: The SIFTS chain-to-UniProt table (PdbUniprotData().data).

    Returns:
        Dict keyed by (uppercase entry id, chain id).
    """
    pdb = sifts['PDB'].astype(str).str.upper()
    return dict(zip(zip(pdb, sifts['Chain']), sifts['Uniprot_Acc']))


def pre_cutoff_entries(
        date_lookup: dict[str, str],
        cutoff: str = CUTOFF_DATE,
) -> set[str]:
    """Entries with a known release date on or before the cutoff.

    Deliberately excludes entries with no date rather than defaulting them to
    leaked, which is what pair_in_training_set does. That default was written
    for obsoleted entries, which are old; here an unknown date overwhelmingly
    means a release newer than the cached snapshot, so treating it as leaked
    would flag structures no model could have seen. The homolog rule already
    reaches far, and it should not reach on missing data.

    Args:
        date_lookup: Uppercase entry id -> release date (YYYY-MM-DD).
        cutoff: Release-date cutoff (YYYY-MM-DD).

    Returns:
        Set of uppercase entry ids.
    """
    return {
        entry for entry, release_date in date_lookup.items()
        if release_date is not None and str(release_date) <= cutoff
    }


def build_accession_entry_index(
        hits: pd.DataFrame,
        rep_to_chains: dict[str, list[str]],
        chain_to_accession: dict[tuple[str, str], str],
        entries: set[str],
        min_identity: float = 0.40,
        min_coverage: float = 0.5,
) -> dict[str, dict[str, set[str]]]:
    """Index each accession to the pre-cutoff entries holding a homolog of it.

    The value for an entry is the set of SIFTS accessions of the chains matched
    there, which is what makes the two-sided test in flag_homolog_pairs able to
    require two genuinely distinct proteins.

    Args:
        hits: MMseqs2 hit table with the HIT_COLUMNS names.
        rep_to_chains: Representative id -> '<pdb>_<chain>' labels.
        chain_to_accession: (entry, chain) -> SIFTS accession.
        entries: PDB entries eligible as evidence, i.e. pre-cutoff.
        min_identity: Fractional identity a hit must exceed.
        min_coverage: Floor on both query and target coverage.

    Returns:
        accession -> {entry -> set of SIFTS accessions matched there}.
    """
    identity_ok = hits['fident'] > min_identity
    query_ok = hits['qcov'] >= min_coverage
    target_ok = hits['tcov'] >= min_coverage
    kept = hits[identity_ok & query_ok & target_ok]
    index: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set),
    )
    for accession, rep in zip(kept['query'], kept['target']):
        for label in rep_to_chains.get(rep, ()):
            entry, _, chain = label.partition('_')
            entry = entry.upper()
            if entry not in entries:
                continue
            matched = chain_to_accession.get((entry, chain))
            if matched is not None:
                index[accession][entry].add(matched)
    return index


def flag_homolog_pairs(
        pairs: list[tuple[str, str]],
        index: dict[str, dict[str, set[str]]],
) -> dict[tuple[str, str], str]:
    """Find pairs with a pre-cutoff entry holding a homolog of both sides.

    The two matched chains must resolve to DIFFERENT SIFTS accessions. Keying
    distinctness on the accession rather than the chain letter is what stops
    two sides that are mutual homologs - RAB27B and NRAS both matching HRAS in
    6D56 - from flagging each other through a single protein, which
    establishes nothing. That class is about a third of naive flags, and
    neither a chain-ID nor a sequence test removes it.

    What this does NOT establish: that the two chains touch, that they touch
    through the interface the negative is about, or that the homology covers
    the interface rather than a shared unrelated domain. It is evidence a model
    could have memorised a related complex, not that the complex is the one
    under test.

    Args:
        pairs: Undirected accession pairs to test.
        index: Output of build_accession_entry_index.

    Returns:
        Mapping of flagged pair -> the PDB entry id that flagged it.
    """
    flagged: dict[tuple[str, str], str] = {}
    for side_a, side_b in pairs:
        left = index.get(side_a)
        right = index.get(side_b)
        if not left or not right:
            continue
        if len(left) > len(right):
            left, right = right, left
        for entry, left_accessions in left.items():
            right_accessions = right.get(entry)
            if right_accessions is None:
                continue
            if len(left_accessions | right_accessions) > 1:
                flagged[(side_a, side_b)] = entry
                break
    return flagged


def build_copresence_index(
        pairs: list[tuple[str, str]],
        sifts: pd.DataFrame,
) -> dict[tuple[str, str], set[str]]:
    """Map each pair to the PDB entries holding BOTH of its accessions.

    The exact-co-presence rule, and the direct counterpart of the source-PDB
    sets the structural sources supply for their own pairs. A negative with no
    deposited complex of its own still leaks if some entry happens to contain
    both proteins, and this is the only way to see that for a pair whose source
    never named a structure.

    Not subsumed by the homology rule, so both are kept and OR'd. It fires on
    60.3% of PDB-source negatives against 0.16% of literature ones, because the
    PDB source is drawn from entries that by construction hold both proteins.

    Args:
        pairs: Undirected accession pairs to test.
        sifts: The SIFTS chain-to-UniProt table (PdbUniprotData().data).

    Returns:
        Mapping of pair -> set of entry ids containing both sides. Pairs with
        no shared entry are absent.
    """
    wanted = {side for pair in pairs for side in pair}
    relevant = sifts[sifts['Uniprot_Acc'].isin(wanted)]
    entries_by_accession: dict[str, set[str]] = defaultdict(set)
    for accession, entry in zip(
        relevant['Uniprot_Acc'], relevant['PDB'].astype(str).str.upper(),
    ):
        entries_by_accession[accession].add(entry)

    shared: dict[tuple[str, str], set[str]] = {}
    for side_a, side_b in pairs:
        both = entries_by_accession.get(side_a, set()) & entries_by_accession.get(
            side_b, set(),
        )
        if both:
            shared[(side_a, side_b)] = both
    return shared


def read_pair_file(path: str) -> list[tuple[str, str]]:
    """Read undirected accession pairs from a two-column negative source file.

    Args:
        path: TSV or CSV carrying the pair columns, provenance comments allowed.

    Returns:
        Sorted-tuple pairs, deduplicated.
    """
    separator = '\t' if path.endswith('.txt') or path.endswith('.tsv') else ','
    frame = pd.read_csv(path, sep=separator, comment='#')
    left, right = frame.columns[0], frame.columns[1]
    return sorted({
        undirected_pair(a, b) for a, b in zip(frame[left], frame[right])
    })


def annotate_negative_leakage(
        pairs: list[tuple[str, str]],
        work_dir: str,
        min_identity: float = 0.40,
        min_coverage: float = 0.5,
        cutoff: str = CUTOFF_DATE,
        threads: int = 8,
        reuse_hits: str | None = None,
) -> pd.DataFrame:
    """Run both leakage rules over a set of negative pairs.

    Args:
        pairs: Undirected accession pairs.
        work_dir: Directory for the sequence database, query FASTA and hits.
        min_identity: Fractional identity floor for the homology rule.
        min_coverage: Query and target coverage floor for the homology rule.
        cutoff: Release-date cutoff (YYYY-MM-DD).
        threads: MMseqs2 worker threads.
        reuse_hits: Path to an existing hit TSV to skip the search.

    Returns:
        One row per pair with uniprot_a, uniprot_b, copresence_in_training_set,
        homolog_in_training_set, the PDB entry behind each flag, and
        copresence_entries: every entry holding both sides, undated.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(work_dir, exist_ok=True)

    # One SIFTS load for the run: PdbUniprotData has no cache, so constructing
    # it per helper re-downloads and re-parses the whole ~1M-row table.
    sifts = PdbUniprotData().data
    shared_entries = build_copresence_index(pairs, sifts)
    date_lookup = PdbReleaseDateData().get_release_date_dict()
    entries = pre_cutoff_entries(date_lookup, cutoff)
    logger.info(
        '%d pairs share at least one PDB entry; %d entries are pre-cutoff',
        len(shared_entries), len(entries),
    )

    fasta_path, rep_to_chains = build_pdb_sequence_db(work_dir)
    wanted = sorted({side for pair in pairs for side in pair})
    if reuse_hits is not None:
        check_hits_manifest(reuse_hits, wanted)
        hits_path = reuse_hits
    else:
        query_fasta = os.path.join(work_dir, 'query.fasta')
        # Existence is not enough: query.fasta is a function of --pairs, so a
        # cached one from a narrower run would leave the extra accessions
        # unsearched and they would be written out as homolog=False, which
        # reads downstream as "checked and clean" rather than "never checked".
        cached = fasta_accessions(query_fasta)
        if cached is not None and not set(wanted) - cached:
            logger.info(
                'Reusing query sequences at %s (%d accessions)',
                query_fasta, len(cached),
            )
        else:
            if cached is not None:
                logger.info(
                    'Rebuilding %s: %d of %d accessions are missing from it',
                    query_fasta, len(set(wanted) - cached), len(wanted),
                )
            unresolved = write_query_fasta(wanted, query_fasta)
            if unresolved:
                logger.warning(
                    '%d of %d accessions have no UniProt sequence and cannot '
                    'be homology-checked', len(unresolved), len(wanted),
                )
        hits_path = run_mmseqs_search(
            query_fasta, fasta_path, os.path.join(work_dir, 'hits.tsv'),
            os.path.join(work_dir, 'tmp'), threads=threads,
        )
        write_hits_manifest(hits_path, wanted, fasta_path)

    hits = pd.read_csv(hits_path, sep='\t', names=HIT_COLUMNS)
    logger.info(
        '%d hit rows over %d queries', len(hits), hits['query'].nunique(),
    )
    index = build_accession_entry_index(
        hits, rep_to_chains, build_chain_accession_map(sifts), entries,
        min_identity=min_identity, min_coverage=min_coverage,
    )
    homologs = flag_homolog_pairs(pairs, index)
    logger.info(
        '%d of %d pairs flagged by homology at id>%.2f cov>=%.2f',
        len(homologs), len(pairs), min_identity, min_coverage,
    )

    records = []
    for side_a, side_b in pairs:
        shared = sorted(shared_entries.get((side_a, side_b), set()))
        leaked = sorted(set(shared) & entries)
        records.append({
            'uniprot_a': side_a,
            'uniprot_b': side_b,
            'copresence_in_training_set': bool(leaked),
            'copresence_entry': leaked[0] if leaked else None,
            # Every entry holding both sides, undated. The leakage flags only
            # ever need the pre-cutoff ones, but annotate.py reads this column
            # to find source structures for literature negatives, and a clean
            # negative's entries are by definition all post-cutoff — so the
            # date-filtered column above is empty for exactly the pairs that
            # reach the benchmark.
            'copresence_entries': ','.join(shared),
            'homolog_in_training_set': (side_a, side_b) in homologs,
            'homolog_entry': homologs.get((side_a, side_b)),
        })
    return pd.DataFrame(records)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Flag negative pairs whose proteins, or close homologs of them, '
            'appear together in a pre-cutoff PDB entry.'
        ),
    )
    parser.add_argument(
        '--pairs', required=True, nargs='+',
        help='Negative source files to annotate (two accession columns each).',
    )
    parser.add_argument(
        '--work-dir', required=True,
        help='Directory for the sequence database, query FASTA and hits.',
    )
    parser.add_argument(
        '--out-dir', default=None,
        help=(
            'Directory to write the dated leakage table and its provenance '
            'into. Defaults to --work-dir.'
        ),
    )
    parser.add_argument(
        '--date', default=None,
        help='Date stamp for the output filename (YYYY-MM-DD). Defaults to today.',
    )
    parser.add_argument(
        '--upload', action='store_true',
        help='Upload the table and its provenance to the cofolding prefix.',
    )
    parser.add_argument(
        '--min-identity', type=float, default=0.40,
        help='Fractional identity floor for the homology rule.',
    )
    parser.add_argument(
        '--min-coverage', type=float, default=0.5,
        help='Query and target coverage floor for the homology rule.',
    )
    parser.add_argument(
        '--threads', type=int, default=8,
        help='MMseqs2 worker threads.',
    )
    parser.add_argument(
        '--reuse-hits', default=None,
        help='Existing MMseqs2 hit TSV to reuse instead of searching again.',
    )
    return parser


def main() -> None:
    """Build the negative-side leakage index the cofolding benchmark consults.

    Negatives carry no structure of their own to date-check unless their source
    named one, so both Negatome sources need a leakage rule computed from the
    accessions alone. Two are applied and OR'd downstream: exact co-presence in
    a pre-cutoff entry, and a pre-cutoff entry holding chains matching both
    sides above the identity and coverage floors.

    The identity floor defaults to 0.40 to match PPI3D's cluster_data_40, the
    only stated identity threshold anywhere in the pipeline. PINDER's cluster
    ids are not a sequence cut and supply no threshold to match: over 400
    sampled clusters, within-cluster identity runs from a 28% 10th percentile
    to a 92% 90th, and 12% of same-cluster pairs have no detectable alignment.

    Writes one row per pair with both flags, the PDB entry behind each, and
    copresence_entries — every entry holding both sides, regardless of date,
    which is what annotate.py needs to find source structures for negatives
    whose own source named none.
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = parse_args()
    args = parser.parse_args()
    # The search floors bound what the index can re-cut to. Below them the
    # requested threshold is unreachable and would be silently clamped, with
    # the log still reporting the value that was asked for.
    if args.min_identity < SEARCH_MIN_SEQ_ID:
        parser.error(
            f'--min-identity {args.min_identity} is below the search floor '
            f'{SEARCH_MIN_SEQ_ID}; lower SEARCH_MIN_SEQ_ID and search again.',
        )
    if args.min_coverage < SEARCH_COVERAGE:
        parser.error(
            f'--min-coverage {args.min_coverage} is below the search floor '
            f'{SEARCH_COVERAGE}; lower SEARCH_COVERAGE and search again.',
        )

    pairs = sorted(
        {pair for path in args.pairs for pair in read_pair_file(path)},
    )
    logger.info(
        '%d distinct undirected pairs across %d files',
        len(pairs), len(args.pairs),
    )

    table = annotate_negative_leakage(
        pairs, args.work_dir,
        min_identity=args.min_identity, min_coverage=args.min_coverage,
        threads=args.threads, reuse_hits=args.reuse_hits,
    )
    out_dir = args.out_dir or args.work_dir
    os.makedirs(out_dir, exist_ok=True)
    filename = make_dated_filename(
        'negative_leakage', FLOCK_VERSION, '.parquet',
        args.date or date.today().isoformat(),
    )
    out_path = os.path.join(out_dir, filename)
    table.to_parquet(out_path, index=False)
    either = (
        table['copresence_in_training_set'] | table['homolog_in_training_set']
    )
    logger.info(
        'Wrote %s: %d co-presence, %d homolog, %d either',
        out_path,
        int(table['copresence_in_training_set'].sum()),
        int(table['homolog_in_training_set'].sum()),
        int(either.sum()),
    )

    # The table decides which negatives reach the benchmark, so the record has
    # to name every input and both floors - a parquet alone cannot carry them.
    provenance_path = out_path.replace('.parquet', '.provenance.json')
    write_output_provenance(
        provenance_path,
        build_output_provenance(
            workflow='flock.cofolding_benchmark.sequence_homology',
            parameters={
                'min_identity': args.min_identity,
                'min_coverage': args.min_coverage,
                'search_min_seq_id': SEARCH_MIN_SEQ_ID,
                'search_coverage': SEARCH_COVERAGE,
                'cutoff': CUTOFF_DATE,
            },
            input_paths={
                f'pairs_{i}': path for i, path in enumerate(args.pairs)
            },
            extra={
                'n_pairs': len(pairs),
                'n_copresence': int(table['copresence_in_training_set'].sum()),
                'n_homolog': int(table['homolog_in_training_set'].sum()),
                'pdb_seqres_url': PDB_SEQRES_URL,
            },
            source_name='flock',
            repo_root=REPO_ROOT,
        ),
    )
    logger.info('Wrote %s', provenance_path)

    if args.upload:
        for path in (out_path, provenance_path):
            logger.info('Uploading to %s', COFOLDING_BENCHMARK_S3)
            upload_file_to_s3(path, COFOLDING_BENCHMARK_S3)


if __name__ == '__main__':
    main()
