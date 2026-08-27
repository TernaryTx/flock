from __future__ import annotations

import gzip
import logging
import os
from functools import lru_cache
from tempfile import NamedTemporaryFile

import pandas as pd
import requests

from flock.http_utils import stream_to_file
from flock.http_utils import UNIPROT_RELEASE_HEADER
from flock.pdb_metadata import PdbMetadata

logger = logging.getLogger(__name__)

_UNIPROT_STREAM_URL = 'https://rest.uniprot.org/uniprotkb/stream'

# UniProt return-field name -> our column name; order sets the TSV column order.
# Field names verified against rest.uniprot.org/configure/uniprotkb/result-fields.
_FIELD_COLUMNS = {
    'accession': 'accession',
    'id': 'entry_name',
    'protein_name': 'protein_names',
    'gene_names': 'gene_names',
    'organism_name': 'organism',
    'organism_id': 'organism_id',
    'xref_orthodb': 'orthodb',
    'xref_eggnog': 'eggnog',
}

# Every column a cached table must carry: the downloaded fields plus the two
# post_process derives.
_TABLE_COLUMNS = [*_FIELD_COLUMNS.values(), 'is_viral', 'uniprot_release']

# Taxonomy ID 10239 is Viruses, the superkingdom. The table's organism_id is a
# leaf taxon, so viral membership cannot be derived from it without the full
# taxonomy tree - it has to be queried separately.
_VIRAL_ACCESSIONS_URL = (
    f'{_UNIPROT_STREAM_URL}'
    f'?query=reviewed:true+AND+taxonomy_id:10239'
    f'&fields=accession&format=list'
)


def fetch_viral_accessions(timeout: int = 60) -> tuple[set[str], str]:
    """Downloads the accessions of every reviewed viral UniProtKB entry.

    Args:
        timeout: Seconds to wait for the UniProt response.

    Returns:
        The viral accessions, and the UniProt release they were served from.

    Raises:
        requests.HTTPError: If the server returns an error status.
    """
    response = requests.get(_VIRAL_ACCESSIONS_URL, timeout=timeout)
    response.raise_for_status()
    return set(response.text.split()), response.headers.get(UNIPROT_RELEASE_HEADER, '')


class UniprotReviewedData(PdbMetadata):
    """Reviewed (Swiss-Prot) UniProtKB entries - accession, names, organism and orthology cross-references.

    The whole reviewed table is downloaded as a gzipped TSV, tagged with a
    viral flag, and kept as a dated CSV in S3 by the base class. Around 575k
    entries over 14.9k organisms, roughly 100 MB uncompressed.

    Consumers that need to reproduce a result should record get_release()
    alongside the dated filename they loaded: the reviewed release moves every
    few weeks and entries are added, renamed and demerged between releases, so
    the release is what a measured number is actually tied to.
    """
    url = (
        f'{_UNIPROT_STREAM_URL}'
        f'?query=reviewed:true'
        f'&fields={",".join(_FIELD_COLUMNS)}'
        f'&format=tsv&compressed=true'
    )

    def retrieve_new_data(self) -> None:
        """Downloads the reviewed table, tags it, and caches a dated copy in S3.

        Overrides the base class, whose urlretrieve call has no timeout at all
        and would let a stalled UniProt stream hang a job indefinitely. Also
        records the release the table was served from, for post_process to
        stamp onto the rows before the table reaches S3.
        """
        with NamedTemporaryFile(delete=False) as temp_file:
            self._release = stream_to_file(self.url, temp_file.name)
            df = self.read_local_data(temp_file.name)
            os.remove(temp_file.name)
        df = self.post_process(df)
        self.data = df
        self._save_and_upload_df(df)

    def post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags viral entries and stamps the rows with their UniProt release.

        The reviewed release rolls over every few weeks, and both queries have
        to land on the same one or the is_viral flag describes rows from a
        different snapshot. The releases are compared rather than assumed
        equal, and raising here aborts the refresh before anything reaches S3.

        Args:
            df: The parsed reviewed table.

        Returns:
            The table with added is_viral and uniprot_release columns.

        Raises:
            ValueError: If the release rolled over between the two downloads.
        """
        accessions, viral_release = fetch_viral_accessions()
        if viral_release != self._release:
            raise ValueError(
                f'UniProt rolled over mid-refresh: the table came from release '
                f'{self._release!r} but the viral list from {viral_release!r}. '
                f'Re-run to build the table from a single release.',
            )
        df['is_viral'] = df['accession'].isin(accessions)
        df['uniprot_release'] = self._release
        return df

    def fetch_latest_bucket_data(self) -> pd.DataFrame:
        """Loads the latest cached table, rebuilding it if it predates a column.

        The base class takes whichever dated CSV sorts newest, with no schema
        check, so a file written before a column existed would otherwise
        surface as a KeyError deep inside a consumer. Rebuilding rather than
        raising keeps the class self-healing: the refresh is written under
        today's date and supersedes the stale file, where raising would leave
        every consumer broken until someone deleted an S3 object by hand.

        Returns:
            DataFrame of the latest data, freshly downloaded if the cached copy
            was missing a column.
        """
        df = super().fetch_latest_bucket_data()
        missing = [
            column for column in _TABLE_COLUMNS if column not in df.columns
        ]
        if missing:
            logger.warning(
                'Cached %s table is missing columns %s - rebuilding from UniProt.',
                self.get_class_name(), missing,
            )
            self.retrieve_new_data()
            return self.data
        return df

    @staticmethod
    def read_local_data(file_name: str) -> pd.DataFrame:
        """Reads the gzipped reviewed UniProtKB TSV.

        The TSV header row is discarded in favour of our own column names, so a
        change to UniProt's header wording cannot silently rename a column. Its
        width is still checked first: pandas would otherwise promote the surplus
        leading column of a wider-than-expected response to the index and shift
        every value one place left, without warning. Empty fields become NaN
        rather than empty strings, matching how the base class reads the cached
        CSV back from S3.

        Args:
            file_name: Local path to the gzipped TSV.

        Returns:
            DataFrame with the columns named in _FIELD_COLUMNS.

        Raises:
            ValueError: If the response does not have the expected column count.
        """
        columns = list(_FIELD_COLUMNS.values())
        with gzip.open(file_name, 'rt') as handle:
            header = handle.readline().rstrip('\n').split('\t')
        if len(header) != len(columns):
            raise ValueError(
                f'Expected {len(columns)} columns from UniProt but the response '
                f'has {len(header)}: {header}. The requested field list and the '
                f'response no longer agree.',
            )
        return pd.read_csv(
            file_name,
            sep='\t',
            header=0,
            names=columns,
            index_col=False,
            compression='gzip',
        )

    def get_release(self) -> str:
        """Returns the UniProt release the loaded table was built from.

        Every row carries the same value, since a table is only ever assembled
        from one release. This is what a consumer should record in its own
        provenance: two dated files can hold the same release, and the entries
        move between releases rather than between download dates.
        """
        return self.data['uniprot_release'].iloc[0]

    def get_viral_accessions(self) -> set[str]:
        """Returns the accessions of reviewed entries belonging to viral organisms."""
        return set(self.data.loc[self.data['is_viral'], 'accession'])


@lru_cache(maxsize=1)
def get_reviewed_data() -> UniprotReviewedData:
    """Returns a process-wide shared UniprotReviewedData.

    Constructing the class loads roughly 100 MB from S3 and holds a DataFrame
    several times that size, so callers should share one instance through this
    accessor rather than constructing their own.

    Returns:
        The cached UniprotReviewedData instance.
    """
    return UniprotReviewedData()
