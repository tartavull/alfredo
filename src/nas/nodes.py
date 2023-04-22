import random


class Node:
    def compute(self, x):
        self._value = x

    def __repr__(self):
        return "Node"


class In(Node):
    def compute(self, x):
        return x

    def __repr__(self):
        return "In"


class Out(Node):
    def compute(self, x):
        return x

    def __repr__(self):
        return "Out"


class Sum(Node):
    def __init__(self):
        self._other = random.uniform(-1.0, 1)

    def compute(self, x):
        return x + self._other

    def __repr__(self):
        return "Sum"


class Multiply(Node):
    def __init__(self):
        self._other = random.uniform(-1.0, 1)

    def compute(self, x):
        return x * self._other

    def __repr__(self):
        return "Multiply"
