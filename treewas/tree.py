from ete3 import Tree
from functools import cached_property
from pandas import DataFrame


class TreeWrapper(Tree):
    @cached_property
    def edge_df(self) -> DataFrame:
        edges = DataFrame(columns=["parent", "child", "length"])
        for node in super().traverse("preorder"):
            if not node.is_root():
                edges.loc[len(edges)] = [node.up.name, node.name, node.dist]
        return edges
