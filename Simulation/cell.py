"""Cell state container for lineage simulations.

Represents a single cell with age, division time, and per-gene mRNA counts.
No dynamics are implemented here; it is a mutable state used by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Cell:
    """Mutable cell state for stochastic lineage simulation."""
    cell_id: int
    parent_id: Optional[int]
    generation: int
    age: float
    division_time: float
    mrna: np.ndarray = field(repr=False)

    def step_age(self, dt: float) -> None:
        """Advance cell age by one time step."""
        self.age += dt

    def snapshot(self) -> dict:
        """Return lightweight snapshot dictionary for serialization."""
        return {
            "cell_id": self.cell_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "age": self.age,
            "mrna": self.mrna.copy(),
        }
