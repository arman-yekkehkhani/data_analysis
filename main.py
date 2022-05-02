import argparse
import functools
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from download_datasets import download_datasets

PARQUET_ENGINE = 'pyarrow'

evidence_cols = ['diseaseId', 'targetId', 'score']
disease_cols = ['id', 'name']
target_cols = ['id', 'approvedSymbol']


def top3(x):
    """
    returns top 3 greatest elements
    """
    if len(x.values) < 3:
        return list(x.values)
    else:
        idx = np.argpartition(x.values, -3)[-3:]
        return x.values[idx].tolist()


def compute_stats(df: DataFrame) -> DataFrame:
    """
    Groups evidence df by ['diseaseId', 'targetId'], then compute median and 3 greatest scores
    """
    result = (df
              .groupby(['diseaseId', 'targetId'])
              .agg({'score': ['median', top3]})
              .reset_index(level=[0, 1])
              )
    result.columns = ['diseaseId', 'targetId', 'median', 'top3']
    return result


def compute_join_stats(evd_df: DataFrame, dss_df: DataFrame, tgt_df: DataFrame) -> DataFrame:
    """
    Join evidences, diseases and target dataframes, then sort in ascending order by the median
    """
    return (evd_df
            .merge(dss_df,
                   left_on='diseaseId',
                   right_on='id',
                   how='inner')
            .merge(tgt_df,
                   left_on='targetId',
                   right_on='id',
                   how='inner')
            .filter(['diseaseId', 'targetId', 'score', 'name', 'approvedSymbol'])
            .groupby(['diseaseId', 'targetId'])
            .agg(median=('score', 'median'))
            .sort_values(by=['median'])
            .reset_index(level=[0, 1])
            )


def has_n_common_neigh(df: DataFrame, n: int):
    """
    Returns true if a tgt-tgt pair has greater or equal to n neighbours
    """
    df['has_n_neighbour'] = df.apply(lambda x: len(x['diseaseId_x'] & x['diseaseId_y']) >= n, axis=1)
    return df


def count_n_common_neighbors(evd_df: DataFrame, n_neigh: int, partitions: int):
    """
    Returns total count of target-target pairs having more than n_neigh diseases in common
    """
    df = (evd_df
          .filter(['diseaseId', 'targetId'])
          .groupby('targetId')
          .agg(set)
          )

    # cartesian product
    df = df.merge(df, how='cross')

    # partition df
    start_idx = [i * (len(df) // partitions) for i in range(partitions)]
    end_idx = start_idx[1:]
    end_idx.append(len(df) - 1)
    dfs = [df.iloc[s:f, :] for (s, f) in zip(start_idx, end_idx)]

    copier = functools.partial(has_n_common_neigh, n=n_neigh)
    with mp.Pool(mp.cpu_count()) as pool:
        res = pd.concat(pool.map(copier, dfs))

    # subtract same tgt-tgt count, and divide by 2 for repeated results
    return (res['has_n_neighbour'].sum() - evd_df['targetId'].nunique()) / 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    if args.overwrite or not os.path.isdir('datasets'):
        download_datasets()

    # load data
    evd_df = pd.read_parquet('./datasets/evidences', engine=PARQUET_ENGINE, columns=evidence_cols)
    dss_df = pd.read_parquet('./datasets/diseases', engine=PARQUET_ENGINE, columns=disease_cols)
    tgt_df = pd.read_parquet('./datasets/targets', engine=PARQUET_ENGINE, columns=target_cols)

    # compute stats of evidences df and save
    stat_df = compute_stats(evd_df)
    stat_df.to_json('evidence_stats.json', orient='records', lines=True)

    # compute stats of target-disease df and save
    dss_tgt_df = compute_join_stats(evd_df, dss_df, tgt_df)
    dss_tgt_df.to_json('disease_target.json', orient='records', lines=True)

    # c = Copier()
    print(f'Number of target-target pairs share a connection to at least two diseases'
          f' : {count_n_common_neighbors(evd_df, 2, mp.cpu_count())}')
