import numpy as np
from typing import List, Optional


class SeededRandom:
    """Random number generator with consistent seeding"""
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def uniform(self, low: float, high: float) -> float:
        return self.rng.uniform(low, high)
    
    def choice(self, options: List, p: Optional[List[float]] = None) -> any:
        return self.rng.choice(options, p=p)
    
    def random(self) -> float:
        return self.rng.random()
    
    def normal(self, mean: float, std: float) -> float:
        return self.rng.normal(mean, std)
    
    def randint(self, low: int, high: int) -> int:
        return self.rng.integers(low, high)
