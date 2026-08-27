from __future__ import annotations

import os.path
import re
from datetime import datetime
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import pandas as pd

from flock.aws import list_files_from_s3
from flock.aws import load_dataframe_from_s3
from flock.aws import upload_file_to_s3
from flock.paths import PDB_METADATA_CACHE_S3


class PdbMetadata:
    """Base class for downloading, caching, and processing whole-PDB metadata tables.

    Subclasses declare a source `url`; the first instantiation downloads and
    parses it, then caches a dated CSV under PDB_METADATA_CACHE_S3. Later
    instantiations load the latest cached CSV instead of re-fetching.
    """
    url: str = ''

    def __init__(self) -> None:
        self.data = self.fetch_latest_bucket_data()

    def retrieve_new_data(self) -> None:
        """Download new data from the source URL and cache a versioned copy in S3."""
        with NamedTemporaryFile(delete=False) as temp_file:
            local_file, _ = urlretrieve(self.url, filename=temp_file.name)
            df = self.read_local_data(local_file)
            os.remove(local_file)
        self.data = df
        self._save_and_upload_df(df)

    def _save_and_upload_df(self, df: pd.DataFrame) -> None:
        """Write df to a versioned CSV in a temp directory and upload it to S3."""
        with TemporaryDirectory() as temp_dir:
            local_csv = os.path.join(
                temp_dir,
                f'{self.get_class_name()}_{datetime.today().strftime("%d%m%y")}.csv',
            )
            df.to_csv(local_csv, index=False)
            upload_file_to_s3(local_csv, self.s3_bucket_path)

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    @property
    def s3_bucket_path(self) -> str:
        return PDB_METADATA_CACHE_S3 + f'{self.get_class_name()}/'

    def get_latest_bucket_version(self) -> str:
        """Return the S3 path of the latest cached version of this data, if any."""
        def _extract_date(file: str) -> datetime:
            match = re.search(r'_(\d{6})\.csv$', file)
            if match:
                return datetime.strptime(match.group(1), '%d%m%y')
            return datetime.min

        try:
            files = list_files_from_s3(self.s3_bucket_path)
        except CalledProcessError:
            files = []
        files = [file for file in files if file.endswith('.csv')]
        sorted_files = sorted(files, key=_extract_date, reverse=True)
        return sorted_files[0] if sorted_files else ''

    def fetch_latest_bucket_data(self) -> pd.DataFrame:
        """Load the latest cached data from S3, fetching fresh data if none exists."""
        latest_file = self.get_latest_bucket_version()
        if latest_file:
            return load_dataframe_from_s3(
                s3_path=os.path.dirname(latest_file),
                filename=os.path.basename(latest_file),
            )
        self.retrieve_new_data()
        return self.data

    @staticmethod
    def read_local_data(file_name: str) -> pd.DataFrame:
        return pd.read_csv(file_name)


class PdbUniprotData(PdbMetadata):
    """PDB-UniProt mapping: PDB, Chain, UniProt accession, and residue ranges."""
    url = 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz'

    @staticmethod
    def read_local_data(file_name: str) -> pd.DataFrame:
        return pd.read_csv(
            file_name, sep='\t', header=1, compression='gzip',
            names=[
                'PDB', 'Chain', 'Uniprot_Acc', 'Res_Count_Beg', 'Res_Count_End',
                'PDB_Beg', 'PDB_End', 'Uniprot_Beg', 'Uniprot_End',
            ],
        )
