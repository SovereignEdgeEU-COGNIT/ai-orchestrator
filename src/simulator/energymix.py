import numpy as np
import random

transition_matrix = [
    [0.999990, 0.000001, 0.000009],  # raining to [raining, sunny, cloudy]
    [0.000001, 0.999990, 0.000009],  # sunny to [raining, sunny, cloudy]
    [0.000009, 0.000001, 0.999990]   # cloudy to [raining, sunny, cloudy]
]
states = ['rainy', 'sunny', 'cloudy']

class EnergyMix:
    def __init__(self):
        self._state = random.choice(states)

    def change_state(self):
        self._state = np.random.choice(states, p=transition_matrix[states.index(self._state)])
    
    @property
    def state(self):
        if self._state == "rainy":
            return 0
        elif self._state == "sunny":
            return 1
        elif self._state == "cloudy":
            return 2
        else:
            return -1
    
    def calc(self):
        if self._state == "sunny":
            green_energy = np.random.beta(5, 1.5)
        elif self._state == "cloudy":
            green_energy = np.random.beta(2, 5)
        elif self._state == "rainy":
            green_energy = np.random.beta(1.1, 10)

        non_green_energy = 1 - green_energy

        return green_energy, non_green_energy
