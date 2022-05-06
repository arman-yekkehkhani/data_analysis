import unittest
from collections import namedtuple

import pandas as pd

from main import compute_join_stats, calc_common_neighbors, compute_stats

Evidence = namedtuple('Evidence', ['targetId', 'diseaseId', 'score'])
Target = namedtuple('Target', ['id', 'approvedSymbol'])
Disease = namedtuple('Disease', ['id', 'name'])

# prepare test dataset
evidences = [
    Evidence('T1', 'D1', 0.2),
    Evidence('T1', 'D1', 0.3),
    Evidence('T1', 'D1', 0.4),
    Evidence('T1', 'D2', 0.2),
    Evidence('T1', 'D2', 0.3),
    Evidence('T2', 'D1', None),
    Evidence('T3', 'D1', 0.2),
    Evidence('T3', 'D1', None),
    Evidence('T4', 'D1', 0.2),
    Evidence('T5', 'D1', 0.2),
    Evidence('T5', 'D2', 0.2),

]

targets = [
    Target('T1', 'T1_SYM'),
    Target('T2', 'T2_SYM'),
    Target('T3', 'T3_SYM'),
    Target('T4', 'T4_SYM'),
]

diseases = [
    Disease('D1', 'D1_NAME'),
    Disease('D2', 'D2_NAME'),
]

# load dataset
evd_df = pd.DataFrame(evidences)
tgt_df = pd.DataFrame(targets)
dss_df = pd.DataFrame(diseases)

# Expected Results
evd_stat = pd.DataFrame([
    {"diseaseId": "D1", "targetId": "T1", "median": 0.3, "top3": [0.2, 0.3, 0.4]},
    {"diseaseId": "D1", "targetId": "T3", "median": 0.2, "top3": [0.2]},
    {"diseaseId": "D1", "targetId": "T4", "median": 0.2, "top3": [0.2]},
    {"diseaseId": "D1", "targetId": "T5", "median": 0.2, "top3": [0.2]},
    {"diseaseId": "D2", "targetId": "T1", "median": 0.25, "top3": [0.2, 0.3]},
    {"diseaseId": "D2", "targetId": "T5", "median": 0.2, "top3": [0.2]}
])

dss_tgt = pd.DataFrame([
    {"diseaseId": "D1", "targetId": "T3", "name": "D1_NAME", "median": 0.2, "approvedSymbol": "T3_SYM"},
    {"diseaseId": "D1", "targetId": "T4", "name": "D1_NAME", "median": 0.2, "approvedSymbol": "T4_SYM"},
    {"diseaseId": "D1", "targetId": "T1", "name": "D1_NAME", "median": 0.3, "approvedSymbol": "T1_SYM"},
    {"diseaseId": "D2", "targetId": "T1", "name": "D2_NAME", "median": 0.25, "approvedSymbol": "T1_SYM"},
])


class Test(unittest.TestCase):

    def test_evidence_stats(self):
        compute_stats(evd_df).to_json(orient='records')
        self.assertEqual(compute_stats(evd_df).to_json(), evd_stat.to_json())

    def test_disease_targets_stats(self):
        self.assertEqual(compute_join_stats(evd_df, dss_df, tgt_df).to_json(), dss_tgt.to_json())

    def test_count_common_neighbors(self):
        self.assertEqual(calc_common_neighbors(evd_df, n_neigh=2, partition_size=10), 1)


if __name__ == '__main__':
    unittest.main()
