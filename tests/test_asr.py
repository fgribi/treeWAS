import unittest
import pandas as pd

from treewas.asr import fitch_parsimony
from treewas.tree import TreeWrapper


class TestFitchParsimony(unittest.TestCase):

    def setUp(self):
        self.tree = TreeWrapper("((A,B)Node1,(C,D)Node2)Root;", format=1)
        self.genes = pd.DataFrame({
            'A': [1, 0, 1, 1],
            'B': [1, 1, 0, 1],
            'C': [0, 0, 1, 1],
            'D': [0, 1, 0, 0],
            'Node1': [1, 0, 0, 1],
            'Node2': [0, 0, 0, 1],
            'Root': [0, 0, 0, 1],
        }, dtype=bool)

    def test_fitch_parsimony(self):
        expected_reconstruction = pd.DataFrame({
            'A': [1, 0, 1, 1],
            'B': [1, 1, 0, 1],
            'C': [0, 0, 1, 1],
            'D': [0, 1, 0, 0],
            'Node1': [1, 0.5, 0.5, 1],
            'Node2': [0, 0.5, 0.5, 1],
            'Root': [0.5, 0.5, 0.5, 1],
        }, dtype=float)

        expected_scores = pd.Series([1, 2, 2, 1])
        reconstruction, scores = fitch_parsimony(self.genes.loc[:, "A":"D"], self.tree, get_scores=True)

        pd.testing.assert_frame_equal(expected_reconstruction, reconstruction)
        pd.testing.assert_series_equal(expected_scores, scores, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
