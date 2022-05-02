""" This module downloads necessary dataset via FTP"""
import os
import shutil
from ftplib import FTP
from typing import Optional

FTP_HOST = 'ftp.ebi.ac.uk'
BASE_URL = 'pub/databases/opentargets/platform/21.11/output/etl/parquet/'
DATASETS_URL_DICT = {
    'diseases': None,
    'targets': None,
    'evidences': 'evidence/sourceId=eva',
}


def fetch(ftp: FTP, name: str, url: Optional[str] = None) -> None:
    """
    Fetches datasets from the given url and saves to a directory with specified name
    """
    ftp.cwd(BASE_URL + (url if url else name))
    files = ftp.nlst()
    for idx, f in enumerate(files):
        print(f'Downloading {name} files: {idx}/{len(files) - 1}')
        ftp.retrbinary('RETR ' + f, open(f'./datasets/{name}/{f}', 'wb').write)
    ftp.cwd('/')


def recreate_dirs() -> None:
    """
    Purges existing dataset dir and recreate the structure
    """
    # remove old `datasets` dir
    dataset_dir = './datasets'
    shutil.rmtree(dataset_dir, ignore_errors=True)

    # create the new dir and sub_dirs
    for folder in DATASETS_URL_DICT.keys():
        os.makedirs(os.path.join(dataset_dir, folder))


def download_datasets() -> None:
    """
    Purges and recreate existing `datasets` dir and downloads datasets
    """
    recreate_dirs()

    # make a ftp connection
    ftp = FTP(FTP_HOST)
    ftp.login()

    # fetch datasets from the ftp
    for name, url in DATASETS_URL_DICT.items():
        fetch(ftp, name, url)


if __name__ == "__main__":
    pass
