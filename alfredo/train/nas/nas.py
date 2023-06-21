"""
Use https://github.com/TylerYep/torchinfo for visualization
"""
import inspect
import random
from collections import defaultdict

import networkx as nx
import torch.nn as nn
from nodes import In, Node, Out
from vis import draw_graph


class Net:
    def __init__(self, node_types=[Node]):
        self._node_types = node_types
        assert len(node_types)

        self._g = nx.DiGraph()
        self._in = In()
        self._out = Out()
        self._g.add_node(self._in)
        self._g.add_node(self._out)
        self._g.add_edge(self._in, self._out)

    def topological_sort(self):
        """
        This function guarantee that `In` is the first node and `Out` the last.
        """
        all_nodes = list(nx.topological_sort(self._g))
        all_nodes.remove(self._in)
        all_nodes.remove(self._out)
        return [self._in] + all_nodes + [self._out]

    def add_node(self):
        """
        If we add a node and it has no input or output, then it is pointless
        because it can have no effect on the remainder of the network.
        """
        all_nodes = self.topological_sort()
        index, u = random_choice(all_nodes[:-1])  # out is not a legal input node
        v = random.choice(all_nodes[index + 1 :])

        node_type = random.choice(self._node_types)
        node = node_type()
        self._g.add_node(node)
        self._g.add_edge(u, node)
        self._g.add_edge(node, v)

    def remove_node(self):
        """
        All adjacent edges connected to the removed node are also removed.
        """
        all_nodes = self.topological_sort()
        if len(all_nodes) == 2:
            return  # there are no more nodes to remove
        self._g.remove_node(random.choice(all_nodes[1:-1]))

    def get_potential_edges(self):
        """
        Create a list of all new edges that would not create cycles
        Given n nodes there are n^2 possible edges.
        """
        all_nodes = self.topological_sort()
        for i, u in enumerate(all_nodes[:-1]):
            for v in all_nodes[i + 1 :]:
                if self._g.has_edge(u, v):
                    continue
                yield (u, v)

    def add_edge(self):
        """
        We must guarantee the graph has no cycles and no parallel edges.
        It returns false if the randomly generated edge already exists
        """
        possible_edges = list(self.get_potential_edges())
        if not possible_edges:
            return
        u, v = random.choice(possible_edges)
        self._g.add_edge(u, v)

    def remove_edge(self):
        """
        We must delete all computation that is not accessible or reconnect it
        """
        edges = list(self._g.edges)
        if not edges:
            return
        u, v = random.choice(edges)
        self._g.remove_edge(u, v)

    def __str__(self):
        return self._g.__str__()

    def forward(self, x):
        inputs = defaultdict(list)
        inputs[self._in].append(x)
        for node in self.topological_sort():
            params = sum(inputs[node])
            x = node.compute(params)
            for suc in self._g.successors(node):
                inputs[suc].append(x)
        return x

    def mutate(self):
        mutation = random.choice(
            [self.add_node, self.remove_node, self.add_edge, self.remove_edge]
        )
        mutation()
        return self

    def load(self):
        pass


def random_choice(l):
    i = random.choice(range(len(l)))
    return i, l[i]


def get_module_name(layer):
    return str(layer).split("(")[0]


def extract_params(layer, params):
    return {param: getattr(layer, param) for param in params}


def get_module_params(layer):
    if isinstance(layer, nn.Conv2d):
        return extract_params(layer, ["in_channels", "out_channels", "kernel_size"])
    elif isinstance(layer, nn.Linear):
        return extract_params(layer, ["in_features", "out_features"])
    elif isinstance(layer, nn.Dropout):
        return extract_params(layer, ["p", "inplace"])
    elif isinstance(layer, nn.MaxPool2d):
        return extract_params(layer, ["kernel_size"])
    elif isinstance(layer, nn.LogSoftmax):
        return extract_params(layer, ["dim"])
    elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.Flatten)):
        return {}
    elif issubclass(type(layer), nn.Module):
        if layer.forward != nn.Module.forward:
            raise ValueError(
                f"Custom 'forward' method detected in layer: {type(layer)}"
            )
        return {}
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")


def from_pytorch(module):
    G = nx.DiGraph()
    prev_layer = None
    node_count = 0

    def process_module(module, prev_layer, node_count):
        for layer in module.children():
            layer_name = get_module_name(layer)
            layer_params = get_module_params(layer)

            G.add_node(node_count, operation=layer_name, **layer_params)

            if prev_layer is not None:
                G.add_edge(prev_layer, node_count)
            prev_layer = node_count
            node_count += 1

            if len(list(layer.children())) > 0:
                prev_layer, node_count = process_module(layer, prev_layer, node_count)
        return prev_layer, node_count

    process_module(module, prev_layer, node_count)
    return G


def to_pytorch(G):
    module = []
    for node in nx.topological_sort(G):
        operation = G.nodes[node]["operation"]
        params = {k: v for k, v in G.nodes[node].items() if k != "operation"}
        if operation == "Conv2d":
            layer = nn.Conv2d(**params)
        elif operation == "ReLU":
            layer = nn.ReLU()
        elif operation == "Tanh":
            layer = nn.Tanh()
        elif operation == "Flatten":
            layer = nn.Flatten()
        elif operation == "Linear":
            layer = nn.Linear(**params)
        elif operation == "Dropout":
            layer = nn.Dropout(**params)
        elif operation == "MaxPool2d":
            layer = nn.MaxPool2d(**params)
        elif operation == "LogSoftmax":
            layer = nn.LogSoftmax(**params)
        else:
            raise ValueError(f"Unknown operation {operation} with params{params}")
        module.append(layer)
    return nn.Sequential(*module)
