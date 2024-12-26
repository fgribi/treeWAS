"""
A module for ancestral state reconstruction.
"""

import numpy as np
import pandas as pd
from treewas.tree import TreeWrapper

# Make pandas shut up about setting values on copies
pd.options.mode.chained_assignment = None


def fitch_downpass(genes, tree):
    """ The first pass of Fitch's parsimony algorithm.

    Returns preliminary ancestral state sets and per locus parsimony scores.

    :param genes: A DataFrame containing the observed binary genotypes. Each row represents a genetic locus and each
        column represents an individual from the sample
    :type genes: pandas.DataFrame
    :param tree: The reconstructed phylogeny. Names of leaf nodes need to correspond to the names of the individuals.
        Has to be bifurcating.
    :type tree: TreeWrapper
    :return: A tuple containing a DataFrame with the ancestral state sets determined during the first pass.
        Ambiguous states are represented as the number 2. The second item in the tuple is a Series with the per locus
        parsimony scores.
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    """
    n_loci, __ = genes.shape
    scores = pd.Series(np.zeros(n_loci, dtype=int), index=genes.index)

    # Consider NAs as ambiguous
    reconstructed = genes.fillna(2).astype(np.int8)

    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            left = reconstructed[node.children[0].name]
            right = reconstructed[node.children[1].name]

            # Increase the score if the two states do not match, i.e. the intersection of the state sets is zero.
            scores += (left + right == 1)

            # Use 2 to represent the union of 0 and 1.
            # The following essentially assigns the parent node the intersection of the state sets given that it is not
            # empty, otherwise it assigns the union.
            reconstructed[node.name] = ((left == right) * left +
                                        (left + right == 1) * 2 +
                                        ((left == 2) & (right != 2)) * right +
                                        ((left != 2) & (right == 2)) * left)
    return reconstructed, scores


def fitch_uppass(reconstructed, tree):
    """ The second pass of Fitch's parsimony algorithm.

    Returns probabilities of being in a particular ancestral state.

    :param reconstructed: A DataFrame containing the reconstructed ancestral state sets. Ambiguous states should be
        represented as the number two. Each row represents a genetic locus and each column represents an individual from
        the sample
    :type reconstructed: pandas.DataFrame
    :param tree: The reconstructed phylogeny. Names of leaf nodes need to correspond to the names of the individuals.
        Has to be bifurcating.
    :type tree: TreeWrapper
    :return: A DataFrame with the probabilities of being in a particular ancestral state.
    :rtype: pandas.DataFrame
    """
    for node in tree.traverse("preorder"):
        if not node.is_leaf() and not node.is_root():
            current = reconstructed.loc[:, node.name]
            ancestor = reconstructed.loc[:, node.up.name]
            left = reconstructed.loc[:, node.children[0].name]
            right = reconstructed.loc[:, node.children[1].name]

            diminished = (current == 2)
            encompassing = ~diminished & (ancestor == 2)
            encompassing = encompassing & ((right == 1 - current) | (left == 1 - current))

            # No need to account for expanding ambiguity for binary states
            reconstructed.loc[diminished, node.name] = ancestor.loc[diminished]
            reconstructed.loc[encompassing, node.name] = 2
    reconstructed = reconstructed.astype(float)
    reconstructed[reconstructed == 2] = 0.5
    return reconstructed


def fitch_parsimony(genes, tree, get_scores=False):
    """ Perform Fitch's parsimony algorithm

    Reconstructs ancestral state sets using Fitch's algorithm [1]_ and if ``get_scores``
    is ``True``, the per locus parsimony scores will be returned.

    For genes where  the ancestral state is ambiguous, the entry is set to `0.5`, otherwise the entry is either one
    or zero.

    .. note:: Only works for binary states and bifurcating trees.

    :param genes: A DataFrame containing the observed binary genotypes. Entries either have to be boolean or only zeros
        and ones. Each row represents a genetic locus and each column represents an individual from the sample.
    :type genes: pandas.DataFrame
    :param tree: The reconstructed phylogeny. Names of leaf nodes need to correspond to the names of the individuals.
        Has to be bifurcating.
    :type tree: TreeWrapper
    :param get_scores: If set to True, returns the per locus parsimony scores
    :type get_scores: bool
    :return: A DataFrame with reconstructed states and if ``get_scores`` is ``True`` a Series with the per-site
        parsimony score
    :rtype: pandas.DataFrame | tuple[pandas.DataFrame, pandas.Series]
    :raise NotImplementedError: if ``genes`` is not binary

    References
    ----------
    .. [1] W. M. Fitch, *Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology*,
        Systematic Biology,  1971, https://doi.org/10.1093/sysbio/20.4.406.
    """

    unique_values = pd.unique(genes.dropna().values.ravel())
    if unique_values.size != 2:
        raise NotImplementedError("Current implementation of Fitch's parsimony only works for binary states."
                                  " Consider using a different tool for the reconstruction of ancestral states.")
        # TODO implement

    reconstructed, scores = fitch_downpass(genes, tree)
    reconstructed = fitch_uppass(reconstructed, tree)
    res = (reconstructed, scores) if get_scores else reconstructed
    return res
