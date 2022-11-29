import pytest


def test_gpu_availability():
    import torch

    # assert torch.cuda.is_available()


def test_gym():
    import gym

    env = gym.make("MountainCar-v0")
    observation, info = env.reset(seed=42)
    action = env.action_space.sample()
    env.close()
    assert observation[0] == pytest.approx(-0.4452088, 0.001)


def test_solid():
    from random import choice, randint, random
    from string import ascii_lowercase

    from Solid.EvolutionaryAlgorithm import EvolutionaryAlgorithm

    class Algorithm(EvolutionaryAlgorithm):
        """
        Tries to get a randomly-generated string to match string "clout"
        """

        def _initial_population(self):
            return list(
                "".join([choice(ascii_lowercase) for _ in range(5)]) for _ in range(50)
            )

        def _fitness(self, member):
            return float(sum(member[i] == "clout"[i] for i in range(5)))

        def _crossover(self, parent1, parent2):
            partition = randint(0, len(self.population[0]) - 1)
            return parent1[0:partition] + parent2[partition:]

        def _mutate(self, member):
            if self.mutation_rate >= random():
                member = list(member)
                member[randint(0, 4)] = choice(ascii_lowercase)
                member = "".join(member)
            return member

    def test_algorithm():
        algorithm = Algorithm(0.5, 0.7, 500, max_fitness=None)
        best_solution, best_objective_value = algorithm.run()

    test_algorithm()


def test_genetic_intelligence():
    import genetic_intelligence as gi

    assert hasattr(gi, "__version__")
    assert "hello" == gi.echo("hello")
