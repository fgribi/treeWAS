import numpy as np
import pandas as pd
from .tree import TreeWrapper


def fitch_downpass(genes, tree):
    n_loci, __ = genes.shape
    scores = pd.Series(np.zeros(n_loci, dtype=int))
    reconstructed = genes.astype("Int8")

    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            left = reconstructed[node.children[0].name]
            right = reconstructed[node.children[1].name]

            # Increase the score if the two states do not match
            scores += (left + right == 1)

            # Use 2 to represent the union of 0 and 1
            reconstructed[node.name] = ((left == right) * left +
                                        (left + right == 1) * 2 +
                                        ((left == 2) & (right != 2)) * right +
                                        ((left != 2) & (right == 2)) * left)
    return reconstructed, scores


def fitch_uppass(reconstructed, tree):
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
            current.loc[diminished] = ancestor.loc[diminished]  # account for diminished ambiguity
            current.loc[encompassing] = 2  # account for encompassing ambiguity
    reconstructed = reconstructed.astype(float)
    reconstructed[reconstructed == 2] = 0.5
    return reconstructed


def fitch_parsimony(genes, tree, get_scores=False):
    """ Perform Fitch's parsimony algorithm

    Reconstructs ancestral state sets using Fitch's algorithm [1]_ and if ``get_scores``
    is ``True``, the per locus parsimony scores will be returned.

    For genes where  the ancestral state is ambiguous, the entry is set to `0.5`, otherwise the entry is either one
    or zero.

    .. note:: Currently only works for binary states.

    :param genes: A table containing the observed binary genotypes. Each row represents a genetic locus and each column
        represents an individual from the sample
    :type genes: pandas.DataFrame
    :param tree: The reconstructed phylogeny. Names of leaf nodes need to correspond to the names of the individuals.
        Has to be bifurcating.
    :type tree: TreeWrapper
    :param get_scores: If set to True, returns the per locus parsimony scores
    :type get_scores: bool
    :return: The reconstructed states and possibly the per-site parsimony score
    :rtype: :class: `pd.DataFrame` | tuple

    References
    ----------
    .. [1] W. M. Fitch, *Toward Defining the Course of Evolution: Minimum Change for a Specific Tree Topology*,
        Systematic Biology,  1971, https://doi.org/10.1093/sysbio/20.4.406.
    """

    reconstructed, scores = fitch_downpass(genes, tree)
    reconstructed = fitch_uppass(reconstructed, tree)
    res = (reconstructed, scores) if get_scores else reconstructed
    return res
