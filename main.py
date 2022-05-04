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


def count_common_neigh(df: DataFrame):
    """
    # TODO:
    """
    df['n_neigh'] = df.apply(lambda x: len(x['left'] & x['right']), axis=1)
    return df


def cartesian_prod_idx(size=100):
    # TODO:
    return np.hstack(
        [np.vstack([np.ones(size - i - 1, dtype=np.int8) * i, np.arange(i + 1, size, dtype=np.int8)]) for i in
         range(size)]).T


def cartesian_product(*dfs, partitions):
    """
    TODO
    :param dfs:
    :param partitions:
    :return:
    """
    # TODO: generalize to multi df
    idx = cartesian_prod_idx((len(dfs[0])))
    start_idx = [i * (len(idx) // partitions) for i in range(partitions)]
    end_idx = start_idx[1:]
    end_idx.append(len(idx) - 1)
    return [pd.DataFrame(
        np.column_stack([df.values[idx[s:f, i]] for i, df in enumerate(dfs)]),
        columns=['left', 'right']
    ) for s, f in zip(start_idx, end_idx)]


def calc_common_neighbors(evd_df: DataFrame, n_neigh: int, partitions: int):
    """
    Returns total count of target-target pairs having more than n_neigh diseases in common
    """
    df = (evd_df
          .head(800)
          .filter(['diseaseId', 'targetId'])
          .groupby('targetId')
          .agg(set)
          )

    df['dss_num'] = df.apply(lambda x: len(x['diseaseId']), axis=1)
    df = df[df.dss_num >= n_neigh].filter(['diseaseId'])

    dfs = cartesian_product(df, df, partitions=partitions)
    with mp.Pool(mp.cpu_count()) as pool:
        res = pd.concat(pool.map(count_common_neigh, dfs))

    print(res[res.n_neigh >= n_neigh])
    return len(res[res.n_neigh >= n_neigh])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--common_neighbors', action='store_true')
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

    if args.common_neighbors:
        import time

        t0 = time.time()
        print(f'Number of target-target pairs share a connection to at least two diseases'
              f' : {calc_common_neighbors(evd_df, n_neigh=2, partitions=mp.cpu_count())},'
              f' done in {time.time() - t0:.2f}s')
