import unittest
import pandas as pd
import numpy as np

from treewas.treewas import (
    subsequent_score,
    simultaneous_score,
    terminal_score,
)
from treewas.tree import TreeWrapper


class TestAssociationScores(unittest.TestCase):

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
        }, index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]),
            dtype=bool)
        self.binary_traits = pd.DataFrame([
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
        ], index=['A', 'B', 'C', 'D', 'Node1', 'Node2', 'Root'],
            columns=["trait1", "trait2", "trait3"], dtype=bool)
        self.continuous_traits = pd.DataFrame([
            [5.3, 1.2, 1.0],
            [1.2, 3.3, 4.0],
            [0.4, 1.2, 3.3],
            [2.4, 3.4, 1.2],
            [2.5, 2.6, 2.9],
            [5.9, 4.1, 3.1],
            [3.0, 1.9, 2.4],
        ], index=['A', 'B', 'C', 'D', 'Node1', 'Node2', 'Root'],
            columns=["trait1", "trait2", "trait3"])
        self.categorical_traits = pd.DataFrame([
            ["x", "y", "x"],
            ["y", "z", "x"],
            ["z", "z", "x"],
            ["x", "x", "z"],
            ["y", "x", "z"],
            ["z", "x", "y"],
            ["z", "y", "y"],
        ], index=['A', 'B', 'C', 'D', 'Node1', 'Node2', 'Root'],
            columns=["trait1", "trait2", "trait3"])
        self.binary_traits_nans = self.binary_traits.astype(float)
        self.binary_traits_nans.loc['A', 'trait1'] = np.nan
        self.binary_traits_nans.loc['C', 'trait2'] = np.nan
        self.binary_traits_nans.loc['Node1', 'trait3'] = np.nan

        self.categorical_traits_nans = self.categorical_traits.copy(deep=True)
        self.categorical_traits_nans.loc['A', 'trait1'] = np.nan
        self.categorical_traits_nans.loc['C', 'trait2'] = np.nan
        self.categorical_traits_nans.loc['Node1', 'trait3'] = np.nan

    def test_terminal_scores_binary(self):
        # Expected output calculated using the original version
        expected_result_binary = pd.DataFrame([
            [1, -0.5, 0],
            [0, 0.5, 1.0],
            [0, -0.5, -1.0],
            [0.5, -1, -0.5],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))
        result = terminal_score(self.genes.loc[:, "A":"D"],
                                self.binary_traits.loc["A": "D"],
                                sign=True)
        pd.testing.assert_frame_equal(result, expected_result_binary)

    def test_terminal_score_binary_nan(self):
        expected_result = pd.DataFrame([
            [0.75, -0.75, 0],
            [0.25, 0.25, 1.0],
            [-0.25, -0.25, -1.0],
            [0.25, -0.75, -0.5],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))
        result = terminal_score(self.genes.loc[:, "A":"D"],
                                self.binary_traits_nans.loc["A": "D"],
                                sign=True)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_terminal_score_continuous(self):
        expected_result = pd.DataFrame([
            [0.3775510, -0.02272727, 0.08333333],
            [-0.2142857, 0.97727273, 0.15],
            [0.2142857, -0.97727273, -0.15],
            [-0.1224490, -0.52272727, 0.35],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))

        result = terminal_score(self.genes.loc[:, "A":"D"],
                                self.continuous_traits.loc["A": "D"],
                                trait_type="continuous",
                                sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result,
                                      check_exact=False,
                                      atol=1e-6)

    def test_terminal_score_categorical(self):
        expected_result = pd.DataFrame([
            [0.7071068, 0.7071068, 0.5773503],
            [0.7071068, 0.7071068, 0.5773503],
            [0.7071068, 0.7071068, 0.5773503],
            [0.5773503, 1, 1],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))

        result = terminal_score(self.genes.loc[:, "A":"D"],
                                self.categorical_traits.loc["A": "D"],
                                trait_type="categorical",
                                sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result,
                                      check_exact=False,
                                      atol=1e-6)

    def test_terminal_score_categorical_nan(self):
        expected_result = pd.DataFrame([
            [0.8660254, 0.8660254, 0.5773503],
            [0.8660254, 0.8660254, 0.5773503],
            [0.8660254, 0.8660254, 0.5773503],
            [0.8660254, 0.8660254, 1],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]),
            dtype=float)

        result = terminal_score(self.genes.loc[:, "A":"D"],
                                self.categorical_traits_nans.loc["A": "D"],
                                trait_type="categorical",
                                sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result,
                                      check_exact=False,
                                      atol=1e-6)

    def test_simultaneous_score_binary(self):
        expected_result = pd.DataFrame([
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, -2],
            [0, -1, 0]
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)
        result = simultaneous_score(self.genes,
                                    self.binary_traits,
                                    self.tree.edge_df[["parent", "child"]],
                                    sign=True)
        pd.testing.assert_frame_equal(expected_result, result)

    def test_simultaneous_score_binary_nan(self):
        expected_result = pd.DataFrame([
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, -1, 0]
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)

        result = simultaneous_score(self.genes,
                                    self.binary_traits_nans,
                                    self.tree.edge_df[["parent", "child"]],
                                    sign=True)
        pd.testing.assert_frame_equal(expected_result, result)

    def test_simultaneous_score_continuous(self):
        expected_result = pd.DataFrame([
            [-0.09090909, 0.24137793, 0.16666667],
            [-0.87272727, 5.551115e-17, -0.2666667],
            [-0.49090909, -1.482759, -0.5666667],
            [0.63636364, 0.2413793, 0.6333333],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))
        result = simultaneous_score(self.genes,
                                    self.continuous_traits,
                                    self.tree.edge_df[["parent", "child"]],
                                    trait_type="continuous",
                                    sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result,
                                      check_exact=False,
                                      atol=1e-6)

    def test_simultaneous_score_categorical(self):
        expected_result = pd.DataFrame([
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 2],
            [1, 0, 1],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)

        result = simultaneous_score(self.genes,
                                    self.categorical_traits,
                                    self.tree.edge_df[["parent", "child"]],
                                    trait_type="categorical",
                                    sign=True)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_simultaneous_score_categorical_nan(self):
        expected_result = pd.DataFrame([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)

        result = simultaneous_score(self.genes,
                                    self.categorical_traits_nans,
                                    self.tree.edge_df[["parent", "child"]],
                                    trait_type="categorical",
                                    sign=True)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_subsequent_score_binary_nan(self):
        expected_result= pd.DataFrame([
            [0.7222222, -0.3888889, -0.3333333],
            [0.3333333, 0.2222222, -0.1666667],
            [0.1666667, 0.1666667, -0.3888889],
            [-0.1666667, -0.3888889, 0.1666667]
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)
        result = subsequent_score(self.genes,
                                  self.binary_traits_nans,
                                  self.tree.edge_df[["parent", "child"]],
                                  sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result,
                                      check_exact=False,
                                      atol=1e-6)

    def test_subsequent_score_categorical_nan(self):
        expected_result = pd.DataFrame([
            [0.9258201, 0.4364358, 0.6546537],
            [0.7319251, 0.6546537, 0.6546537],
            [0.4140393, 0.5855400, 0.6546537],
            [0.9258201, 0.4140393, 0.9258201]
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)
        result = subsequent_score(self.genes,
                                  self.categorical_traits_nans,
                                  self.tree.edge_df[["parent", "child"]],
                                  trait_type="categorical",
                                  sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result,
                                      check_exact=False,
                                      atol=1e-6)

    def test_subsequent_score(self):
        expected_result_binary = pd.DataFrame([
            [0.8888889, -0.2222222, -0.1666667],
            [0.1666667, 0.3888889, -0.3333333],
            [0.1666667, 0.1666667, -0.7777778],
            [0, -0.5555556, 0.5]
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]), dtype=float)
        expected_result_continuous = pd.DataFrame([
            [-0.17777778, -0.20498084, -0.01296296],
            [-0.14848485, 0.09195402, -0.07592593],
            [-0.06363636, -0.23754789, -0.14259259],
            [0.05656566, -0.04406130, 0.20185185],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))
        expected_result_categorical = pd.DataFrame([
            [0.8416254, 0.1666667, 0.5651942],
            [0.5477226, 0.4281744, 0.4281744],
            [0.4281744, 0.5477226, 0.7302967],
            [0.6454972, 0.4714045, 0.6454972],
        ], columns=["trait1", "trait2", "trait3"],
            index=pd.Index(["orthogene1", "orthogene2", "orthogene3", "orthogene4"]))
        result = subsequent_score(self.genes,
                                  self.binary_traits,
                                  self.tree.edge_df[["parent", "child"]],
                                  sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result_binary,
                                      check_exact=False,
                                      atol=1e-6)
        result = subsequent_score(self.genes,
                                  self.continuous_traits,
                                  self.tree.edge_df[["parent", "child"]],
                                  trait_type="continuous",
                                  sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result_continuous,
                                      check_exact=False,
                                      atol=1e-6)
        result = subsequent_score(self.genes,
                                  self.categorical_traits,
                                  self.tree.edge_df[["parent", "child"]],
                                  trait_type="categorical",
                                  sign=True)
        pd.testing.assert_frame_equal(result,
                                      expected_result_categorical,
                                      check_exact=False,
                                      atol=1e-6)


if __name__ == '__main__':
    unittest.main()
