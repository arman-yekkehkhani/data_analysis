import argparse
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
        return x.values
    else:
        idx = np.argpartition(x.values, -3)[-3:]
        return x.values[idx].tolist()


def compute_stats(df: DataFrame) -> DataFrame:
    """
    Groups evidence df by ['diseaseId', 'targetId'], then compute median and 3 greatest scores
    """
    return (df
            .groupby(['diseaseId', 'targetId'])
            .agg({'score': ['median', top3]})
            .reset_index(level=[0, 1])
            )


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


def count_n_common_neighbors(df: DataFrame, n_neigh: int, partitions: int):
    df = (df
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

    def f(df: DataFrame):
        df['has_n_neighbour'] = df.apply(lambda x: len(x['diseaseId_x'] & x['diseaseId_y']) > n_neigh, axis=1)
        return df

    with mp.Pool(mp.cpu_count()) as pool:
        res = pd.concat(pool.map(f, dfs))

    # subtract same tgt-tgt count
    return res['has_n_neighbour'].sum() - df['targetId'].nunique()


def spark():
    # psutils -> get cpus
    # local[numbr of cpus]
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()

    if args.overwrite or not os.path.isdir('datasets'):
        download_datasets()

    # load data
    evd_df = pd.read_parquet('./evidences', engine=PARQUET_ENGINE, columns=evidence_cols)
    dss_df = pd.read_parquet('./diseases', engine=PARQUET_ENGINE, columns=disease_cols)
    tgt_df = pd.read_parquet('./targets', engine=PARQUET_ENGINE, columns=target_cols)

    # compute stats of evidences df and save
    stat_df = compute_stats(evd_df)
    stat_df.to_json('evidence_stats.json', orient='records', lines=True)

    # compute stats of target-disease df and save
    dss_tgt_df = compute_join_stats(evd_df, dss_df, tgt_df)
    dss_tgt_df.to_json('disease_target.json', orient='records', lines=True)

    print(f'Number of target-target pairs share a connection to at least two diseases'
          f' : {count_n_common_neighbors(evd_df, 2, mp.cpu_count())}')
