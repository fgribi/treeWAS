import fire
import os
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from itertools import combinations

from treewas.asr import fitch_parsimony, fitch_downpass
from treewas.tree import TreeWrapper

TRAIT_TYPES = ["discrete", "continuous", "categorical"]
RECONSTRUCTION_TYPES = ["parsimony", "ML"]


def _nandot(a, b):
    """Matrix dot product without NaN propagation."""
    a = a[:, :, np.newaxis]
    b = b[np.newaxis, :, :]
    dot_product = np.nansum(a * b, axis=1)
    return dot_product


def _normalize(data):
    return (0 + data - data.min()) / (0 + data.max() - data.min())


def _get_anc_dec(edges, data, axis):
    if axis == 0:
        anc = data.loc[:, edges["parent"]]
        desc = data.loc[:, edges["child"]]
    else:
        anc = data.loc[edges["parent"], :]
        desc = data.loc[edges["child"], :]

    return anc.to_numpy(dtype=float), desc.to_numpy(dtype=float)



def _row_chi_square(row: pd.Series, col: pd.Series):
    # Calculating the contigency table as below is significantly faster than using pd.crosstab
    row_mask = row.isna().to_numpy(dtype=bool)
    col_mask = col.isna().to_numpy(dtype=bool)
    mask = ~(row_mask | col_mask)
    filtered_row = row[mask]
    filtered_col = col[mask]
    unique_row, row_indices = np.unique(filtered_row, return_inverse=True)
    unique_col, col_indices = np.unique(filtered_col, return_inverse=True)
    contingency_table = np.zeros((len(unique_row), len(unique_col)), dtype=int)
    np.add.at(contingency_table, (row_indices, col_indices), 1)

    # Skip chi-squared if table is degenerate
    if contingency_table.size == 0 or np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
        return 0

    score, *__ = chi2_contingency(contingency_table, correction=False)

    return (score / len(col)) ** 0.5


def _chi_square(genes: pd.DataFrame, traits: pd.DataFrame) -> pd.DataFrame:
    n_loci, __ = genes.shape
    __, n_traits = traits.shape
    scores = np.zeros((n_loci, n_traits))

    # TODO: find a better way if possible. Pandas apply is very inefficient.
    for i, trait in enumerate(traits.columns):
            scores[:, i] = genes.apply(lambda x: _row_chi_square(x, traits.loc[:, trait]), axis=1)

    return pd.DataFrame(scores)


def _non_categorical_terminal_score(genes, traits):
    __, n_ind = genes.shape
    g = genes.to_numpy(dtype=float)
    t = traits.to_numpy(dtype=float)
    scores = (_nandot(g, t) -
              _nandot(g, 1 - t) -
              _nandot(1 - g, t) +
              _nandot(1 - g, 1 - t))
    scores = scores / n_ind
    scores_df = pd.DataFrame(scores)
    return scores_df


def terminal_score(genes, traits, trait_type="discrete", sign=True):
    """Calculate treeWAS terminal score (score 1).

    The column names of the gene dataframe need to be the same as the index of the trait dataframe. The order also needs
    to match.

    :param genes: Table containing binary genetic data. Rows represent loci and columns individuals.
    :type genes: pandas.DataFrame
    :param traits: Table containing phenotypic data. Rows represent individuals and columns traits.
    :type traits: pandas.DataFrame
    :param trait_type: String indicating the type of trait, either ``"discrete"``, ``"continuous"`` or ``"categorical"``
    :type trait_type: str
    :param sign: Flag indicating whether the returned scores should be signed. (default is True)
    :type sign: bool
    :return: Table with the terminal scores, where rows represent loci and columns represent traits.
    :rtype: pandas.DataFrame
    :raise ValueError: If specified trait_type is not admissible
    """
    if trait_type not in TRAIT_TYPES:
        raise ValueError(f"Invalid trait type specified. Must be one of {TRAIT_TYPES}")


    if trait_type == "categorical":
        scores = _chi_square(genes, traits)
    else:
        traits = _normalize(traits)
        scores = _non_categorical_terminal_score(genes, traits)

    scores = scores.set_index(genes.index)
    scores.columns = traits.columns

    return scores if sign else np.abs(scores)


def _categorical_simultaneous_score(diff_genes, edges, rec_genes, rec_traits):
    unique_values = pd.unique(rec_traits.values.ravel())
    categories = unique_values[~pd.isna(unique_values)]
    n_loci, __ = rec_genes.shape
    __, n_traits = rec_traits.shape
    scores = np.zeros((n_loci, n_traits))

    # Examine only changes between two specific categorical values at a time
    for pair in combinations(categories, 2):
        pair_traits = rec_traits.replace(pair, [0, 1])
        pair_traits[~pair_traits.isin({0, 1})] = np.nan
        anc_pair_traits, desc_pair_traits = _get_anc_dec(edges, pair_traits, axis=1)
        diff_pair_traits = anc_pair_traits - desc_pair_traits
        diff_pair_traits[np.isnan(diff_pair_traits)] = 0
        scores += np.abs(diff_genes @ diff_pair_traits)
    return scores


def simultaneous_score(rec_genes, rec_traits, edges, trait_type="discrete", sign=True):
    """Calculate treeWAS simultaneous score (score 2).

    :param rec_genes: Binary table containing the empirical genotype and the reconstructed genotype at
        all internal nodes of the tree.
    :type rec_genes: pandas.DataFrame
    :param rec_traits: Table containing the empirical traits and the reconstructed traits at all
            internal nodes of the tree.
    :type rec_traits: pandas.DataFrame
    :param edges:  A dataframe with parent and child nodes for each edge
    :type edges: pandas.DataFrame
    :param trait_type: String indicating the type of trait, either ``"discrete"``, ``"continuous"`` or ``"categorical"``
    :type trait_type: str
    :param sign: Flag indicating whether the returned scores should be signed. (default is True)
    :type sign: bool
    :return: ``DataFrame`` containing the score, rows representing the loci and columns represent traits
    :rtype: pandas.DataFrame
    :raise ValueError: If specified trait_type is not admissible
    """
    if trait_type not in TRAIT_TYPES:
        raise ValueError(f"Invalid trait type specified. Must be one of {TRAIT_TYPES}")

    anc_genes, desc_genes = _get_anc_dec(edges, rec_genes, axis=0)
    diff_genes = anc_genes - desc_genes

    if trait_type == "categorical":
        scores = _categorical_simultaneous_score(diff_genes, edges, rec_genes, rec_traits)
    else:
        rec_traits = _normalize(rec_traits)
        anc_traits, desc_traits = _get_anc_dec(edges, rec_traits, axis=1)
        diff_traits = anc_traits - desc_traits
        scores = _nandot(diff_genes, diff_traits)

    scores = pd.DataFrame(scores, index=rec_genes.index, columns=rec_traits.columns)
    return scores if sign else np.abs(scores)


def _non_categorical_subsequent_score(edges, rec_genes, rec_traits):
    n_edges, __ = edges.shape
    anc_genes, desc_genes = _get_anc_dec(edges, rec_genes, axis=0)
    anc_traits, desc_traits = _get_anc_dec(edges, rec_traits, axis=1)
    anc_genes = anc_genes[:, :, np.newaxis]
    desc_genes = desc_genes[:, :, np.newaxis]
    anc_traits = anc_traits[np.newaxis, :, :]
    desc_traits = desc_traits[np.newaxis, :, :]
    scores = (4 / 3 * anc_genes * anc_traits +
              2 / 3 * desc_genes * anc_traits +
              2 / 3 * anc_genes * desc_traits +
              4 / 3 * desc_genes * desc_traits -
              anc_genes - desc_genes - anc_traits - desc_traits + 1)
    scores = scores / n_edges
    scores = np.nansum(scores, axis=1)
    scores = pd.DataFrame(scores)
    return scores


def subsequent_score(rec_genes, rec_traits, edges, trait_type="discrete", sign=True):
    """Calculate treeWAS subsequent score (score 3).

    :param rec_genes: Binary table containing the empirical genotype and the reconstructed genotype at all internal
            nodes of the tree.
    :type rec_genes: pandas.DataFrame
    :param rec_traits: Table containing the empirical traits and the reconstructed traits at all
            internal nodes of the tree.
    :type rec_traits: pandas.DataFrame
    :param edges: A dataframe with parent and child nodes for each edge
    :type edges: pandas.DataFrame
    :param trait_type: String indicating the type of trait,
    :type trait_type: str
    :param sign: Flag indicating whether the returned scores should be signed. (default is True)
    :type sign: bool
    :return: Table containing the score, rows representing the loci and columns represent traits
    :rtype: pandas.DataFrame
    :raise ValueError: If specified trait_type is not admissible
    """
    if trait_type not in TRAIT_TYPES:
        raise ValueError(f"Invalid trait type specified. Must be one of {TRAIT_TYPES}")

    if trait_type == "categorical":
        scores = _chi_square(rec_genes, rec_traits)
    else:
        rec_traits = _normalize(rec_traits)
        scores = _non_categorical_subsequent_score(edges, rec_genes, rec_traits)

    scores = scores.set_index(rec_genes.index)
    scores.columns = rec_traits.columns

    return scores if sign else np.abs(scores)


def _non_polymorphic(data, leaf_names):
    # Much faster to just convert the pandas DataFrame to numpy and do it like this than to use nunique
    leaf_data = data[leaf_names].to_numpy()
    return np.all(leaf_data == leaf_data[:, [0]], axis=1)


def simulate_loci(n_sim, dist, tree, node_names, seed=None):
    """Simulate loci to be used for estimating null distribution of association scores.

    :param n_sim: Number of loci to be simulated. Should be at least then times the number of empirical loci.
    :type n_sim: int
    :param dist: Homoplasy distribution for sampling per locus number of substitutions.
    :type dist: numpy.ndarray
    :param tree: Reconstructed phylogenetic tree.
    :type tree: TreeWrapper
    :param seed: Seed for pseudo-random number generator. (default is None)
    :type seed: int
    """

    def _mutate_loci(data, loci):
        n = len(loci)
        roots = rng.integers(0, 2, n, dtype=bool)
        data.loc[:, tree.name] = roots

        # Draw the branches for the substitutions for each locus
        subs = np.zeros((len(length_dist), n), dtype=bool)
        for i, locus in enumerate(loci):
            sub_ind = rng.choice(len(length_dist), size=n_subs[locus], p=length_dist, replace=False)
            subs[sub_ind, i] = True

        # Mutate the loci where it is indicated by subs
        for edge, node in enumerate(tree.traverse("preorder")):
            if not node.is_root():
                data.loc[:, node.name] = data[node.up.name]
                current_subs = subs[edge - 1]
                data.loc[current_subs, node.name] = ~data.loc[current_subs, node.up.name]

        return data

    rng = np.random.default_rng(seed)
    sim_data = pd.DataFrame(np.zeros((n_sim, len(node_names)), dtype=bool), columns=node_names)
    branch_lengths = tree.edge_df["length"]
    length_dist = branch_lengths / branch_lengths.sum()
    leaf_names = tree.get_leaf_names()

    # Draw number of subs from homoplasy distribution. Add one, since the distribution does not include probability for
    # zero substitutions
    n_subs = rng.choice(dist.size, size=n_sim, p=dist) + 1
    sim_data = _mutate_loci(sim_data, np.arange(n_sim))
    non_polymorphic = _non_polymorphic(sim_data, leaf_names)

    # The simulated loci should all be polymorphic
    while non_polymorphic.any():
        non_polymorphic_loci = np.where(non_polymorphic)[0]
        sim_data.loc[non_polymorphic] = _mutate_loci(sim_data.loc[non_polymorphic], non_polymorphic_loci)
        non_polymorphic = _non_polymorphic(sim_data, leaf_names)

    return sim_data


def _get_distribution(scores: pd.Series) -> np.ndarray:
    """Constructs a distribution from a ``pd.Series`` containing discrete values."""
    counts = scores.value_counts().sort_index()
    full_counts = counts.reindex(range(0, scores.max() + 1), fill_value=0)

    # Scores that are zero are disregarded
    full_counts = full_counts.iloc[1:]
    dist = full_counts / full_counts.sum()
    return dist.to_numpy()


def _reconstruct(states, tree, reconstruction, datatype="discrete"):

    if reconstruction not in  RECONSTRUCTION_TYPES:
        raise ValueError(f"Reconstruction method must be one of {RECONSTRUCTION_TYPES}")

    if datatype == "continuous":
        raise NotImplementedError("Ancestral state reconstruction for continuous states is not yet implemented.")
        # TODO implement

    if datatype == "categorical":
        raise NotImplementedError("Ancestral state reconstruction for categorical states is not yet implemented.")
        # TODO implement

    if reconstruction == "ML":
        raise NotImplementedError("Ancestral state reconstruction using maximum-likelihood is not yet implemented.")
        # TODO implement

    if reconstruction == "parsimony" and datatype == "discrete":
        return fitch_parsimony(states, tree, get_scores=True)

def _handle_reconstruction(genes, traits, tree, trait_type, homoplasy_distribution, gene_reconstruction, trait_reconstruction,
                           test):
    rec_genes = genes
    rec_traits = traits
    homoplasy_scores = None

    # Terminal test does not require reconstruction
    if "simultaneous" in test or "subsequent" in test:
        # reconstruct genes if not user provided
        if gene_reconstruction is not None:
            rec_genes, homoplasy_scores = _reconstruct(genes, tree, gene_reconstruction)

        # reconstruct traits if not user provided
        if trait_reconstruction is not None:
            rec_traits, __ = _reconstruct(traits.transpose(), tree, trait_reconstruction, trait_type)
            rec_traits = rec_traits.transpose()

    # If no user-provided distribution is available, check if the scores have been calculated, otherwise do so
    if homoplasy_distribution is None:
        if homoplasy_scores is None:
            __, homoplasy_scores = fitch_downpass(genes, tree)
        homoplasy_distribution = _get_distribution(homoplasy_scores)

    return homoplasy_distribution, rec_genes, rec_traits

def _get_scores(rec_genes, rec_traits, edges,  leaf_names, test, trait_type):
    score_functions = {
        "terminal": terminal_score,
        "simultaneous": simultaneous_score,
        "subsequent": subsequent_score
    }

    scores = {}
    for test_type in test:
        if test_type in score_functions:
            if test_type == "terminal":
                score = score_functions[test_type](rec_genes.loc[:, leaf_names],
                                                   rec_traits.loc[leaf_names, :],
                                                   trait_type)
            else:
                score = score_functions[test_type](rec_genes, rec_traits, edges, trait_type)
            scores[f"{test_type}"] = score

    return scores


def _get_p_values_ecdf(emp_scores: pd.DataFrame, sim_scores: pd.DataFrame) -> pd.DataFrame:
    """Calculates p-values for the empirical scores using the ECDF."""
    # TODO: implement using scipy or other library?
    np_scores = np.abs(emp_scores.to_numpy())
    dist= np.abs(sim_scores.to_numpy())
    p_values = np.mean(np_scores[:, np.newaxis, :] <= dist[np.newaxis, :, :], axis=1)
    return pd.DataFrame(p_values, index=emp_scores.index, columns=emp_scores.columns)

def _get_p_values(emp_scores, sim_scores, test, p_value_by):

    if not p_value_by in ("density", "count"):
        raise ValueError("Specified p-value calculation method is not permissible.")

    if not p_value_by == "count":
        raise NotImplementedError("Currently, the only permissible p-value calculation method is \"count\".")

    p_values = {}
    for test_type in test:
        p_values[f"{test_type}"] = _get_p_values_ecdf(emp_scores[test_type], sim_scores[test_type])
    return p_values


def treewas(genes,
            traits,
            tree,
            trait_type,
            n_sim = None,
            homoplasy_distribution = None,
            test = ("terminal", "simultaneous", "subsequent"),
            gene_reconstruction = None,
            sim_gene_reconstruction = "parsimony",
            trait_reconstruction = None,
            p_value_by="count",
            seed=None):
    """Run the *treeWAS* algorithm

    :param tree: A :class:`TreeWrapper` object with the reconstructed phylogeny of the samples. All nodes have to be
        named and the names should correspond to the names of the samples.
    :type tree: TreeWrapper
    :param genes: Table containing binary genetic data. Columns represent isolates and rows represent genes.
    :type genes: :class:pandas.DataFrame
    :param traits: Table containing trait data, Columns represent traits and rows represent isolates.
    :type traits: :class:pandas.DataFrame
    :param trait_type: A string indicating the type of trait, one of ``discrete``, ``continuous``, or ``categorical``.
    :type trait_type: str
    :return: A dictionary containing pandas DataFrames with the uncorrected p-values. The keys are the values provided
        in `test`
    """
    n_loci, __ = genes.shape
    leaf_names = tree.get_leaf_names()
    edges = tree.edge_df

    # Recommended to take at least ten times the number of empirical loci for simulation
    if n_sim is None:
        n_sim = 10 * n_loci

    homoplasy_distribution, rec_genes, rec_traits = _handle_reconstruction(genes, traits, tree, trait_type,
                                                                           homoplasy_distribution, gene_reconstruction,
                                                                           trait_reconstruction, test)
    sim_genes = simulate_loci(n_sim, homoplasy_distribution, tree, rec_genes.columns, seed)
    rec_sim_genes = sim_genes

    if "simultaneous" in test or "subsequent" in test:
        rec_sim_genes, __ = _reconstruct(sim_genes, tree, sim_gene_reconstruction)

    emp_scores =  _get_scores(rec_genes, rec_traits, edges,  leaf_names, test, trait_type)
    sim_scores = _get_scores(rec_sim_genes, rec_traits, edges, leaf_names, test, trait_type)

    p_values = _get_p_values(emp_scores, sim_scores, test, p_value_by)

    return p_values


def _load_genes(path, delimiter):
    genes = pd.read_csv(path, delimiter=delimiter, index_col=0)
    return genes


def _load_traits(path, delimiter):
    genes = pd.read_csv(path, delimiter=delimiter, index_col=0)
    return genes


def _load_tree(path):
    return TreeWrapper(path, format=1)


def _load_homoplasy(path):
    return np.fromfile(path)


def _correct_p_values(p_values, p_value_correction, base_p):
    # TODO implement
    raise NotImplementedError("P-value correction is currently not supported")


def run_treewas(gene_path,
                trait_path,
                tree_path,
                out_dir,
                trait_type,
                delimiter=",",
                homoplasy_path=None,
                n_sim=None,
                test=("terminal", "simultaneous", "subsequent"),
                gene_reconstruction=None,
                sim_gene_reconstruction="parsimony",
                trait_reconstruction=None,
                filter_na=True,
                base_p=0.05,
                p_value_correction=None,
                p_value_by="count",
                seed=None,
                ):
    """Wrapper function for :func:`treewas` for command line compatibility"""

    if trait_type not in TRAIT_TYPES:
        raise ValueError(f"Invalid trait type specified. Must be one of {TRAIT_TYPES}")

    genes = _load_genes(gene_path, delimiter)
    traits = _load_traits(trait_path, delimiter)
    tree = _load_tree(tree_path)

    if filter_na:
        # Discount loci that have more than 75% NA
        threshold = int(genes.shape[1] * 0.25)
        genes = genes.dropna(thresh=threshold)

    homoplasy_dist = None
    if homoplasy_path is not None:
        homoplasy_dist = _load_homoplasy(homoplasy_path)

    p_values = treewas(genes, traits, tree, trait_type, n_sim, homoplasy_dist, test, gene_reconstruction,
                       sim_gene_reconstruction, trait_reconstruction, p_value_by, seed)

    if p_value_correction is not None:
        p_values = _correct_p_values(p_values, p_value_correction, base_p)

    for test_name, score in p_values.items():
        file_path = os.path.join(out_dir, f"{test_name}.csv")
        score.to_csv(file_path, sep=delimiter)


def main():
    fire.Fire(run_treewas)


if __name__ == '__main__':
    main()
