import gym
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from Solid.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from random import choice, randint, random


def train(Model):
    env = gym.make("MountainCarContinuous-v0")
    env.seed(42)
    observation, info = env.reset()
    # For some reason the first observation has shape (1,) instead of (2,)
    observation, reward, terminated, info = env.step(np.zeros((1,)))
    rewards = []
    model = Model()
    for _ in tqdm(range(10**6)):
        action = model.run(observation)
        observation, reward, terminated, info = env.step(action)
        #env.render()
        if terminated:
            rewards.append(reward)
            observation, info = env.reset()
            observation, reward, terminated, info = env.step(np.zeros((1,)))
            model = Model()

    env.close()
    return rewards

def compute_stats(rewards):
    mean =  np.mean(rewards)
    mean_std = np.std(rewards) / np.sqrt(len(rewards))
    best = np.max(rewards)
    return f"Rewards {mean} ± {mean_std}, best score={best}"


class UniformModel():
    def run(self, obseration):
        return np.random.uniform(-1, 1, (1,))


class FullyConnectedModel():
    def __init__(self):
        # generating some random features
        self.W1 = torch.randn((2, 1), requires_grad=False)
        self.B1 = torch.randn((1), requires_grad=False)

    def run(self, observation):
        features = torch.tensor(observation)
        return F.relu((features @ self.W1) + self.B1)


class FullyConnected(EvolutionaryAlgorithm):
    """
    TODO consider using https://github.com/sugarme/gotch
    """
    def _initial_population(self):
        return list(FullyConnectedModel() for _ in range(500))

    def _fitness(self, member):
        env = gym.make("MountainCarContinuous-v0")
        env.seed(42)
        observation, info = env.reset()
        observation, reward, terminated, info = env.step(np.zeros((1,)))
        while True:
            action = member.run(observation)
            observation, reward, terminated, info = env.step(action)
            if terminated:
                return reward

    def _crossover(self, parent1, parent2):
        new_model = FullyConnectedModel()
        new_model.W1 = parent2.W1
        new_model.B1 = parent1.B1
        return new_model

    def _mutate(self, member):
        if self.mutation_rate >= random():
            member.W1 += torch.randn((2, 1), requires_grad=False) * 0.1
            member.B1 += torch.randn((1), requires_grad=False) * 0.1
        return member
 

if __name__ == '__main__':
    #print("Uniform Model", compute_stats(train(UniformModel)))
    #print("Untrained Neural Network Model", compute_stats(train(FullyConnectedModel)))
    algorithm = FullyConnected(.1, .1, 500, max_fitness=None)
    print(algorithm.run())
    
