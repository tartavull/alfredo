import random
from copy import deepcopy
from typing import Callable

from nas import Net


def select_parents(population, fitness_scores):
    """
    Select the top 20% fittest individuals in the population
    """
    sorted_population = sorted(
        population,
        key=lambda individual: fitness_scores[population.index(individual)],
        reverse=False,
    )
    return sorted_population[: len(sorted_population) // 5]


def evolve(constructor, fitness_fn: Callable, population_size: int, steps: int):
    """ """
    population = [constructor() for _ in range(population_size)]
    for i in range(steps):
        fitness_scores = [fitness_fn(individual) for individual in population]
        print(fitness_scores)
        next_gen = select_parents(population, fitness_scores)
        while len(next_gen) < population_size:
            individual = deepcopy(random.choice(next_gen))
            next_gen.append(individual.mutate())
        population = next_gen

    return min(population, key=fitness_fn)
