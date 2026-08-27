# The final filter over the curated pairs, and the Negatome source it writes.
from __future__ import annotations

import argparse
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from flock import NEGATOME_LIT_VERSION
from flock import REPO_ROOT
from flock.aws import download_file_from_s3
from flock.aws import upload_file_to_s3
from flock.compile_negatome import deduplicate_pairs
from flock.compile_negatome import load_negatome
from flock.compile_negatome import SOURCE_PDB
from flock.intact_filter import load_intact_pairs
from flock.intact_filter import pair_membership
from flock.negatome_v3.literature import PAIRS_ROOT
from flock.negatome_v3.literature.pairs import print_counts
from flock.negatome_v3.literature.pairs import write_table
from flock.negatome_v3.literature.vocabulary import CURATION_JUDGED
from flock.negatome_v3.literature.vocabulary import NEGATIVES_DROP_ORDER
from flock.negatome_v3.literature.vocabulary import NEGATIVES_STEM
from flock.paths import get_flock_pinder_path
from flock.paths import get_flock_ppi3d_path
from flock.paths import get_intact_pairs_path
from flock.paths import get_negatome_pdb_path
from flock.paths import make_dated_filename
from flock.paths import NEGATOME_S3
from flock.provenance import build_output_provenance
from flock.provenance import write_csv_with_provenance
from flock.provenance import write_output_provenance

# The per-side columns that have to travel with their accession when a row is
# flipped into sorted order.
SIDE_COLUMNS = ('accession', 'name', 'viral', 'provenance')


def normalise_sides(frame: pd.DataFrame) -> pd.DataFrame:
    """Flip each row so side a holds the lexicographically smaller accession.

    The curated table's `pair` column is already sorted but accession_a and
    accession_b are not, so a name or a viral flag read straight off side a
    describes whichever protein the screen happened to write first. Every
    per-side column is flipped together, which is what keeps a name attached to
    its own accession.

    Args:
        frame: Rows from the curated pair table.

    Returns:
        A copy with the per-side columns in sorted-accession order.
    """
    frame = frame.copy()
    flip = frame['accession_a'] > frame['accession_b']
    for column in SIDE_COLUMNS:
        left, right = f'{column}_a', f'{column}_b'
        flipped = frame.loc[flip, [right, left]].values
        frame.loc[flip, [left, right]] = flipped
    return frame


def collect_pairs(curated: pd.DataFrame) -> pd.DataFrame:
    """Collapse the curated table to one row per distinct usable pair.

    Only judged rows are read. A pair is carried forward if any paper judged it
    usable as a negative, and the count of papers judging otherwise is carried
    with it so the contradiction filter has something to act on.

    Args:
        curated: The curated pair table, one row per pair per paper.

    Returns:
        One row per distinct pair, with uniprot_a <= uniprot_b, the per-side
        columns, the paper counts and an empty drop_reason.
    """
    judged = curated[curated['curation_outcome'] == CURATION_JUDGED]
    judged = normalise_sides(judged)
    against = (
        judged[~judged['usable_as_negative']]
        .groupby('pair')['paper_id'].nunique()
    )
    usable = judged[judged['usable_as_negative']]
    grouped = usable.groupby('pair', sort=True)
    pairs = grouped.agg(
        uniprot_a=('accession_a', 'first'),
        uniprot_b=('accession_b', 'first'),
        name_a=('name_a', 'first'),
        name_b=('name_b', 'first'),
        viral_a=('viral_a', 'first'),
        viral_b=('viral_b', 'first'),
        provenance_a=('provenance_a', 'first'),
        provenance_b=('provenance_b', 'first'),
        n_papers_usable=('paper_id', 'nunique'),
        paper_ids=(
            'paper_id', lambda ids: ','.join(sorted(set(ids.astype(str)))),
        ),
    ).reset_index()
    pairs['n_papers_against'] = (
        pairs['pair'].map(against).fillna(0).astype(int)
    )
    pairs['drop_reason'] = ''
    return pairs


def known_negative_pairs(
        pdb_date: str | None = None,
) -> tuple[set[tuple[str, str]], str]:
    """Load every pair the in-house PDB Negatome source already holds.

    Read from the source file rather than from the compiled negatome_pairs
    table, which is IntAct-filtered and so names only a fraction of what the
    PDB source holds.

    The published Negatome 2.0 Manual set is deliberately not consulted. This
    filter deduplicates against what the dataset actually contains, and Manual
    is no longer compiled in - the literature source supersedes it. Checking a
    pair against a source that contributes no rows would delete a curated
    negative rather than deduplicate it. Overlap with the published Negatome
    2.0 is a property to report, not a reason to drop a pair: this is a new
    dataset, not an update to theirs.

    Args:
        pdb_date: Date override for the PDB source (YYYY-MM-DD).

    Returns:
        Tuple of (set of (uniprot_a, uniprot_b) tuples with a <= b, the PDB
        source filename actually read).
    """
    pdb_filename = get_negatome_pdb_path(date=pdb_date).rsplit('/', 1)[-1]
    existing = load_negatome(pdb_date=pdb_date, include=(SOURCE_PDB,))
    existing = deduplicate_pairs(existing)
    pairs = set(zip(existing['uniprot_a'], existing['uniprot_b']))
    return pairs, pdb_filename


def known_positive_pairs(
        pinder_date: str | None = None,
        ppi3d_date: str | None = None,
) -> tuple[set[tuple[str, str]], dict[str, str]]:
    """Load every pair the structural positive sources record as interacting.

    Read from the two source tables rather than from the assembled benchmark.
    The benchmark drops any pair that is also a negative, so a curated pair
    contradicted by a structure would be missing from exactly the frame meant
    to catch it. Neither source table is order-normalised, so both are sorted
    here.

    Both tables default to the latest in S3 and both decide drops, so the
    resolved filenames are returned for the provenance record rather than left
    implicit.

    Args:
        pinder_date: Date override for the PINDER pair table (YYYY-MM-DD).
        ppi3d_date: Date override for the PPI3D pair table (YYYY-MM-DD).

    Returns:
        Tuple of (set of (uniprot_a, uniprot_b) tuples with a <= b, mapping of
        provenance key -> the filename actually read).
    """
    paths = {
        'pinder_source': get_flock_pinder_path(date=pinder_date),
        'ppi3d_source': get_flock_ppi3d_path(date=ppi3d_date),
    }
    pairs: set[tuple[str, str]] = set()
    filenames = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for key, s3_path in paths.items():
            folder, filename = s3_path.rsplit('/', 1)
            filenames[key] = filename
            local_path = os.path.join(tmpdir, filename)
            download_file_from_s3(folder + '/', filename, local_path)
            frame = pd.read_csv(local_path, comment='#')
            sides = frame[['Target', 'Partner']]
            pairs.update(zip(sides.min(axis=1), sides.max(axis=1)))
    return pairs, filenames


def apply_negative_filters(
        pairs: pd.DataFrame,
        intact_pairs: set[tuple[str, str]],
        positive_pairs: set[tuple[str, str]],
        known_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """Write drop_reason for the four filters that decide what ships.

    Applied in a fixed order and recorded rather than deleted, so the funnel can
    be read off the table afterwards. The order matters only for attribution:
    the filters overlap, and a pair is charged to the first one that catches it.

    The structural-positive filter is not made redundant by the IntAct one:
    IntAct misses pairs whose only positive evidence is a deposited structure.
    It overlaps assemble_flock's conflict rule, which drops any negative that
    is also a positive, but runs here so the drop is attributed and counted in
    this source's own funnel rather than silently absorbed downstream.

    Args:
        pairs: One row per distinct pair, from collect_pairs.
        intact_pairs: Positive pairs from IntAct.
        positive_pairs: Pairs from the PINDER and PPI3D positive sources.
        known_pairs: Pairs already in the Negatome.

    Returns:
        The table with drop_reason written.
    """
    pairs = pairs.copy()
    # np.select applies the conditions in order, so the executed attribution
    # order is NEGATIVES_DROP_ORDER itself rather than a copy of it.
    conditions = [
        pairs['n_papers_against'] > 0,
        pair_membership(pairs, intact_pairs),
        pair_membership(pairs, positive_pairs),
        pair_membership(pairs, known_pairs),
    ]
    pairs['drop_reason'] = np.select(
        conditions, NEGATIVES_DROP_ORDER, default='',
    )
    return pairs


def report_overlap(pairs: pd.DataFrame) -> None:
    """Print how far the three filters overlap, so the funnel is not read as a sum.

    Args:
        pairs: The filtered pair table, carrying drop_reason.
    """
    dropped = pairs[pairs['drop_reason'] != '']
    print(f'\n{len(dropped):,} of {len(pairs):,} pairs dropped, charged to the '
          f'first filter that caught them:')
    print_counts(
        dropped['drop_reason'], width=20, total=len(pairs),
        order=NEGATIVES_DROP_ORDER,
    )


def write_source_file(
        pairs: pd.DataFrame,
        out_dir: str,
        stamp: str,
        version: str,
        sources: dict[str, str],
) -> Path:
    """Write the two-column TSV that compile_negatome reads.

    Args:
        pairs: The kept pairs.
        out_dir: Directory to write into.
        stamp: Date stamp for the filename.
        version: Version string for the filename.
        sources: Provenance labels for the header lines.

    Returns:
        The path written.
    """
    filename = make_dated_filename(NEGATIVES_STEM, version, '.txt', stamp)
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_provenance(
        pairs[['uniprot_a', 'uniprot_b']], str(path), sources=sources,
        sep='\t', header=True,
    )
    return path


def main() -> None:
    """Filter the curated pairs down to the Negatome source the benchmark uses.

    Four filters run over the distinct pairs any paper judged usable as a
    negative. A pair another paper judged not usable is dropped rather than
    resolved in favour of usable: the two readings are of different papers and
    nothing in the table says which is right. A pair IntAct records as a
    positive is dropped, following Blohm et al. 2014, which costs some of the
    third-protein-bridged negatives the rubric was written to keep - IntAct
    stores complex-derived binary pairs, so a bridged association is a positive
    there. A pair PINDER or PPI3D solved a structure for is dropped on the same
    reasoning, and catches what IntAct misses: a pair whose only positive
    evidence is a deposited complex. A pair the in-house PDB source already
    holds is dropped rather than merged, since the source column is not where
    this dataset's overlap between its own sources gets measured; the published
    Negatome 2.0 is not consulted at all, because it contributes no rows and
    overlap with it is a figure to report rather than a reason to drop.

    The virus-virus drop the ticket also names is already applied upstream, by
    pairs.apply_final_filters, and no pair reaching here has both sides viral.

    Writes literature_negatives_<version>_<date>.txt beside the other Negatome
    sources, plus a local audit table carrying every pair with the reason it was
    dropped.
    """
    parser = parse_args()
    args = parser.parse_args()

    curated = pd.read_parquet(args.source)
    pairs = collect_pairs(curated)
    print(f'{len(pairs):,} distinct pairs judged usable by at least one paper')

    intact_path = get_intact_pairs_path(date=args.intact_date)
    intact_filename = intact_path.rsplit('/', 1)[-1]
    intact_pairs = load_intact_pairs(path=intact_path)
    positive_pairs, positive_filenames = known_positive_pairs(
        pinder_date=args.pinder_date, ppi3d_date=args.ppi3d_date,
    )
    print(f'{len(positive_pairs):,} pairs in the structural positive sources')
    known_pairs, pdb_filename = known_negative_pairs(pdb_date=args.pdb_date)
    print(f'{len(known_pairs):,} pairs already in the PDB Negatome source')

    pairs = apply_negative_filters(
        pairs, intact_pairs, positive_pairs, known_pairs,
    )
    kept = pairs[pairs['drop_reason'] == ''].reset_index(drop=True)
    report_overlap(pairs)
    print(f'\n{len(kept):,} pairs kept')

    stamp = args.date or date.today().isoformat()
    # Every input that decides a drop, named by the file actually read. All
    # four filters move independently and three of them default to the latest
    # in S3, so without this the funnel cannot be reproduced from the record.
    sources = {
        'curated_source': Path(args.source).name,
        'intact_source': intact_filename,
        'pdb_negatome_source': pdb_filename,
        **positive_filenames,
    }
    audit_name = f'literature_negatives_audit_{args.version}_{stamp}.parquet'
    audit_out = args.audit_out or str(PAIRS_ROOT / audit_name)
    audit_path = write_table(pairs, audit_out)
    out_path = write_source_file(
        kept, args.out_dir, stamp, args.version, sources,
    )
    provenance_path = out_path.with_suffix('.provenance.json')
    write_output_provenance(
        provenance_path,
        build_output_provenance(
            workflow='flock.negatome_v3.literature.negatives',
            parameters={
                'source': args.source, 'date': stamp,
                'version': args.version,
            },
            input_paths={'curated_pairs': args.source},
            extra={
                **sources,
                'n_pairs_usable': len(pairs),
                'n_kept': len(kept),
                'n_dropped_by_reason': {
                    reason: int((pairs['drop_reason'] == reason).sum())
                    for reason in NEGATIVES_DROP_ORDER
                },
            },
            source_name='flock',
            repo_root=REPO_ROOT,
        ),
    )
    print(f'\nwrote {out_path} ({len(kept):,} pairs)')
    print(f'wrote {provenance_path}')
    print(f'wrote {audit_path} (all {len(pairs):,} pairs, with drop_reason)')
    if args.upload:
        upload_file_to_s3(str(out_path), NEGATOME_S3)
        print(f'uploaded {out_path.name} to {NEGATOME_S3}')


def parse_args() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Filter the curated literature pairs into the Negatome source file '
            'the benchmark builds from.'
        ),
    )
    parser.add_argument(
        '--source', required=True,
        help='The curated pair table written by the curation stage.',
    )
    parser.add_argument(
        '--audit-out',
        default=None,
        help=(
            'Where to write every usable pair with its drop_reason. Defaults '
            'to a date-stamped file beside the source, so one build does not '
            'overwrite the only explanation of the previous one.'
        ),
    )
    parser.add_argument(
        '--out-dir', default=str(PAIRS_ROOT),
        help='Directory for the dated source file and its provenance.',
    )
    parser.add_argument(
        '--version', default=NEGATOME_LIT_VERSION,
        help='Version string for the published filename.',
    )
    parser.add_argument(
        '--date', default='',
        help='Date stamp for the output filename. Defaults to today.',
    )
    parser.add_argument(
        '--intact-date', default=None,
        help='Pin the IntAct snapshot. Defaults to the latest in S3.',
    )
    parser.add_argument(
        '--pdb-date', default=None,
        help='Pin the PDB Negatome source. Defaults to the latest in S3.',
    )
    parser.add_argument(
        '--pinder-date', default=None,
        help='Pin the PINDER pair table. Defaults to the latest in S3.',
    )
    parser.add_argument(
        '--ppi3d-date', default=None,
        help='Pin the PPI3D pair table. Defaults to the latest in S3.',
    )
    parser.add_argument(
        '--upload', action='store_true',
        help='Upload the source file to the Negatome prefix in S3.',
    )
    return parser


if __name__ == '__main__':
    main()
