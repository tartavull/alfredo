import random
from functools import partial

import networkx as nx
import pytest
import torch
import torch.nn as nn
from evolve import evolve
from nodes import Multiply, Sum

from nas import Net, Node, from_pytorch, to_pytorch


def test_empty():
    n = Net()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_single_node():
    n = Net()
    n.add_node()
    assert len(n._g.nodes) == 3
    assert len(n._g.edges) == 2 + 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_add_many_nodes():
    n = Net()
    for _ in range(1000):
        n.add_node()
    assert len(n._g.nodes) == 1002
    assert len(n._g.edges) == 2000 + 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_add_and_remove_single():
    n = Net()
    n.add_node()
    n.remove_node()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_add_and_remove_node_multiple():
    n = Net()
    for i in range(100):
        n.add_node()
    for i in range(100):
        n.remove_node()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_remove_node_on_empty():
    n = Net()
    for i in range(10):
        n.remove_node()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_add_edge_empty():
    n = Net()
    n.add_edge()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_add_all_edges():
    n = Net()
    for _ in range(5):
        n.add_node()
    for _ in range(50):
        n.add_edge()
    assert len(n._g.nodes) == 7
    assert len(n._g.edges) == 6 + 5 + 4 + 3 + 2 + 1
    assert nx.is_directed_acyclic_graph(n._g)


def test_remove_edge_empty():
    n = Net()
    n.remove_edge()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 0
    assert nx.is_directed_acyclic_graph(n._g)


def test_remove_edge_too_many_edges():
    n = Net()
    for _ in range(100):
        n.remove_edge()
    assert len(n._g.nodes) == 2
    assert len(n._g.edges) == 0
    assert nx.is_directed_acyclic_graph(n._g)


def test_sum_node():
    n = Net(node_types=[Sum])
    n.add_node()
    assert len(n._g.nodes) == 3
    assert isinstance(n.topological_sort()[1], Sum)
    assert isinstance(n.forward(1), float), f"Expected float but got {type(obj)}"


def test_sum_evolution():
    random.seed(0)
    fitness_fn = lambda n: abs(n.forward(1) - 4.5)
    constructor = partial(Net, node_types=[Sum])
    net = evolve(constructor, fitness_fn, population_size=20, steps=100)
    assert fitness_fn(net) < 0.01


def test_load():
    G = nx.DiGraph()
    G.add_node(0, operation="conv2d", in_channels=3, out_channels=16, kernel_size=3)
    G.add_node(1, operation="relu")
    G.add_node(2, operation="conv2d", in_channels=16, out_channels=32, kernel_size=3)
    G.add_node(3, operation="tanh")
    G.add_node(
        4, operation="linear", in_features=32 * 6 * 6, out_features=10
    )  # Assuming input size is 8x8
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    n = Net()


def test_pytoch_translation_simplest():
    m1 = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(32 * 6 * 6, 10),
    )

    m2 = to_pytorch(from_pytorch(m1))
    assert str(m1) == str(m2)


def test_pytoch_translation_simple_broken():
    """
    The function from_pytorch traverses the layers as they
    are defined in the constructor of the model class. However, the forward
    method of a PyTorch model can contain additional logic that doesn't
    directly correspond to the structure defined in the constructor, including
    conditionals, loops, or operations on multiple layers at once. The function
    can't automatically infer this structure from the model alone.

    To handle these cases, you might have to manually inspect the forward method
    and adjust your function accordingly. Unfortunately, there's no general way to
    automatically extract the exact computation graph from the forward method,
    because it can contain arbitrary Python code.
    """

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    # FIXME it would be better if we could raise an exception.
    m1 = Net()
    m2 = to_pytorch(from_pytorch(m1))
    assert str(m1) != str(m2)


def test_pytorch_translation_simple_fixed():
    """
    In order for this to work we would need to rewrite the Net module in the
    following way.
    """

    Net = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1),
    )

    m1 = Net
    m2 = to_pytorch(from_pytorch(m1))
    assert str(m1) == str(m2)
