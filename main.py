import argparse
import math
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
    returns top 3 greatest elements from a dataframe column
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
              .dropna()
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
            .dropna()
            .groupby(['diseaseId', 'targetId'])
            .agg(median=('score', 'median'))
            .sort_values(by=['median'])
            .reset_index(level=[0, 1])
            .merge(dss_df,
                   left_on='diseaseId',
                   right_on='id',
                   how='inner')
            .merge(tgt_df,
                   left_on='targetId',
                   right_on='id',
                   how='inner')
            .filter(['diseaseId', 'targetId', 'name', 'median', 'approvedSymbol'])
            )


def count_common_elm(df: DataFrame):
    """
    count common elements between `left` and `right` columns of a dataframe
    """
    res = list(filter(lambda x: x >= 2, map(lambda x, y: len(set(x) & set(y)), df['left'], df['right'])))
    return len(res)


def cartesian_prod_idx(size: int):
    """
    Return cartesian_product of indices of two dfs, excluding x-x, retaining one of x-y and y-x
    """
    return np.hstack(
        [np.vstack([np.ones(size - i - 1, dtype=np.int8) * i, np.arange(i + 1, size, dtype=np.int8)]) for i in
         range(size)]).T


def cartesian_product(partition_size: int, *dfs):
    """
    Returns cartesian_product of dfs of the same size divided in to partitions
    """
    idx = cartesian_prod_idx((len(dfs[0])))
    p_idx = [i * partition_size for i in range(math.ceil(len(idx) / partition_size) + 1)]
    return [pd.DataFrame(
        np.column_stack([df.values[idx[s:f, i]] for i, df in enumerate(dfs)]),
        columns=['left', 'right']
    ) for s, f in zip(p_idx[:-1], p_idx[1:])]


def calc_common_neighbors(evd_df: DataFrame, n_neigh: int, partition_size: int):
    """
    Returns total count of target-target pairs having more than n_neigh diseases in common
    """
    df = (evd_df
          .filter(['diseaseId', 'targetId'])
          .groupby('targetId')
          .agg(set)
          )

    df['dss_num'] = df.apply(lambda x: len(x['diseaseId']), axis=1)
    df = df[df.dss_num >= n_neigh].filter(['diseaseId'])

    dfs = cartesian_product(partition_size, df, df)
    with mp.Pool(mp.cpu_count()) as pool:
        res = list(pool.map(count_common_elm, dfs))

    return sum(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
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

    import time

    t0 = time.time()
    print(f'Number of target-target pairs share a connection to at least two diseases'
          f' : {calc_common_neighbors(evd_df, n_neigh=2, partition_size=100_000)},'
          f' done in {time.time() - t0:.2f}s')
