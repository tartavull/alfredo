from brax import base
from jax import numpy as jp

class Reward:
    
    def __init__(self, f, sc, ps):
        """
        :param f: A function handle (ie. function that computes this reward)
        :param sc: A float that gets multiplied to base computation provided by f
        :param ps: A dictionary of parameters required for the reward computation
        """

        self.f = f
        self.scale = sc
        self.params = ps

    def add_param(self, p_name, p_value):
        """
        Updates self.params dictionary with provided key and value
        """

        self.params[p_name] = p_value

    def compute(self):
        """
        computes reward as specified by self.f given 
        scale and general parameters are set.
        Otherwise, this errors out quite spectacularly
        """

        res = self.f(**self.params)
        res = res.at[0].multiply(self.scale) #may not be the best way to do this
        
        return res

    def __str__(self):
        """
        provides a standard string output  
        """

        return f'reward: {self.f}, scale: {self.scale}'
