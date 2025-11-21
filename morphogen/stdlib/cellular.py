"""Cellular Automata operations for emergent pattern simulation.

This module provides comprehensive cellular automata capabilities for:
- Classic CA simulations (Game of Life, Brian's Brain)
- Wolfram Elementary Cellular Automata
- Multi-state cellular systems
- Pattern evolution and analysis
- Emergent behavior visualization

Supports multiple CA types:
- Game of Life (Conway's classic 2-state CA)
- Brian's Brain (3-state CA with firing neurons)
- Wolfram Elementary CA (1D rule-based systems)
- Custom rule sets (totalistic and non-totalistic)
- Larger-than-Life variants
"""

from typing import Tuple, Optional, Callable, Union
import numpy as np
from dataclasses import dataclass
from morphogen.core.operator import operator, OpCategory


class CellularField2D:
    """2D cellular automaton field with NumPy backend.

    Represents a grid of cells with discrete states.
    """

    def __init__(self, data: np.ndarray, states: int = 2):
        """Initialize cellular field.

        Args:
            data: NumPy array of cell states (shape: (height, width))
            states: Number of possible states per cell
        """
        self.data = data.astype(np.int32)
        self.states = states
        self.shape = data.shape
        self.generation = 0

    @property
    def height(self) -> int:
        """Get field height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get field width."""
        return self.shape[1]

    def copy(self) -> 'CellularField2D':
        """Create a copy of this field."""
        result = CellularField2D(self.data.copy(), self.states)
        result.generation = self.generation
        return result

    def __repr__(self) -> str:
        return f"CellularField2D(shape={self.shape}, states={self.states}, gen={self.generation})"


class CellularField1D:
    """1D cellular automaton field (for Wolfram Elementary CA).

    Represents a 1D array of cells with discrete states.
    """

    def __init__(self, data: np.ndarray, states: int = 2):
        """Initialize 1D cellular field.

        Args:
            data: NumPy array of cell states (shape: (width,))
            states: Number of possible states per cell
        """
        self.data = data.astype(np.int32)
        self.states = states
        self.shape = data.shape
        self.generation = 0

    @property
    def width(self) -> int:
        """Get field width."""
        return self.shape[0]

    def copy(self) -> 'CellularField1D':
        """Create a copy of this field."""
        result = CellularField1D(self.data.copy(), self.states)
        result.generation = self.generation
        return result

    def __repr__(self) -> str:
        return f"CellularField1D(width={self.width}, states={self.states}, gen={self.generation})"


@dataclass
class CARule:
    """Cellular automaton rule definition."""
    birth: set  # States that cause a dead cell to become alive
    survival: set  # States that keep an alive cell alive
    states: int = 2  # Number of cell states
    neighborhood: str = "moore"  # "moore" or "von_neumann"

    def __repr__(self) -> str:
        b = ''.join(str(x) for x in sorted(self.birth))
        s = ''.join(str(x) for x in sorted(self.survival))
        return f"B{b}/S{s}"


class CellularOperations:
    """Namespace for cellular automata operations (accessed as 'cellular' in DSL)."""

    # ============================================================================
    # LAYER 1: ATOMIC CA OPERATIONS
    # ============================================================================

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Union[Tuple[int, int], int], states: int, fill_value: int) -> Union[CellularField2D, CellularField1D]",
        deterministic=True,
        doc="Allocate a new cellular field"
    )
    def alloc(shape: Union[Tuple[int, int], int], states: int = 2,
              fill_value: int = 0) -> Union[CellularField2D, CellularField1D]:
        """Allocate a new cellular field.

        Args:
            shape: Field shape (height, width) for 2D or width for 1D
            states: Number of possible states per cell
            fill_value: Initial state for all cells

        Returns:
            New cellular field filled with fill_value

        Example:
            >>> ca = cellular.alloc((100, 100), states=2)
        """
        if isinstance(shape, tuple):
            data = np.full(shape, fill_value, dtype=np.int32)
            return CellularField2D(data, states)
        else:
            data = np.full((shape,), fill_value, dtype=np.int32)
            return CellularField1D(data, states)

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Union[Tuple[int, int], int], states: int, density: float, seed: Optional[int]) -> Union[CellularField2D, CellularField1D]",
        deterministic=False,
        doc="Initialize field with random states"
    )
    def random_init(shape: Union[Tuple[int, int], int], states: int = 2,
                   density: float = 0.5, seed: Optional[int] = None) -> Union[CellularField2D, CellularField1D]:
        """Initialize field with random states.

        Args:
            shape: Field shape (height, width) for 2D or width for 1D
            states: Number of possible states
            density: Probability of non-zero state
            seed: Random seed

        Returns:
            Randomly initialized field

        Example:
            >>> ca = cellular.random_init((100, 100), density=0.3)
        """
        rng = np.random.RandomState(seed)

        if isinstance(shape, tuple):
            data = (rng.random(shape) < density).astype(np.int32)
            return CellularField2D(data, states)
        else:
            data = (rng.random((shape,)) < density).astype(np.int32)
            return CellularField1D(data, states)

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.QUERY,
        signature="(field: CellularField2D, state: int) -> ndarray",
        deterministic=True,
        doc="Count neighbors using Moore neighborhood (8 neighbors)"
    )
    def count_neighbors_moore(field: CellularField2D, state: int = 1) -> np.ndarray:
        """Count neighbors using Moore neighborhood (8 neighbors).

        Args:
            field: Cellular field
            state: State to count (default: 1 for alive cells)

        Returns:
            Array of neighbor counts for each cell
        """
        h, w = field.shape
        data = field.data
        counts = np.zeros((h, w), dtype=np.int32)

        # Use binary mask for the state we're counting
        mask = (data == state).astype(np.int32)

        # Add all 8 neighbors
        counts += np.roll(mask, 1, axis=0)   # N
        counts += np.roll(mask, -1, axis=0)  # S
        counts += np.roll(mask, 1, axis=1)   # W
        counts += np.roll(mask, -1, axis=1)  # E
        counts += np.roll(np.roll(mask, 1, axis=0), 1, axis=1)    # NW
        counts += np.roll(np.roll(mask, 1, axis=0), -1, axis=1)   # NE
        counts += np.roll(np.roll(mask, -1, axis=0), 1, axis=1)   # SW
        counts += np.roll(np.roll(mask, -1, axis=0), -1, axis=1)  # SE

        return counts

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.QUERY,
        signature="(field: CellularField2D, state: int) -> ndarray",
        deterministic=True,
        doc="Count neighbors using von Neumann neighborhood (4 neighbors)"
    )
    def count_neighbors_von_neumann(field: CellularField2D, state: int = 1) -> np.ndarray:
        """Count neighbors using von Neumann neighborhood (4 neighbors).

        Args:
            field: Cellular field
            state: State to count

        Returns:
            Array of neighbor counts for each cell
        """
        data = field.data
        mask = (data == state).astype(np.int32)

        counts = np.zeros_like(mask)
        counts += np.roll(mask, 1, axis=0)   # N
        counts += np.roll(mask, -1, axis=0)  # S
        counts += np.roll(mask, 1, axis=1)   # W
        counts += np.roll(mask, -1, axis=1)  # E

        return counts

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.TRANSFORM,
        signature="(field: CellularField2D, rule: CARule) -> CellularField2D",
        deterministic=True,
        doc="Apply a cellular automaton rule to advance one generation"
    )
    def apply_rule(field: CellularField2D, rule: CARule) -> CellularField2D:
        """Apply a cellular automaton rule to advance one generation.

        Args:
            field: Current field state
            rule: CA rule to apply

        Returns:
            New field after rule application

        Example:
            >>> rule = CARule(birth={3}, survival={2, 3})  # Game of Life
            >>> next_gen = cellular.apply_rule(field, rule)
        """
        # Count neighbors
        if rule.neighborhood == "moore":
            neighbors = CellularOperations.count_neighbors_moore(field, state=1)
        elif rule.neighborhood == "von_neumann":
            neighbors = CellularOperations.count_neighbors_von_neumann(field, state=1)
        else:
            raise ValueError(f"Unknown neighborhood: {rule.neighborhood}")

        # Apply birth/survival rules
        alive = field.data == 1
        dead = field.data == 0

        # Birth: dead cells with correct neighbor count become alive
        birth_mask = dead & np.isin(neighbors, list(rule.birth))

        # Survival: alive cells with correct neighbor count stay alive
        survival_mask = alive & np.isin(neighbors, list(rule.survival))

        # Create new state
        new_data = np.zeros_like(field.data)
        new_data[birth_mask | survival_mask] = 1

        result = CellularField2D(new_data, field.states)
        result.generation = field.generation + 1
        return result

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.TRANSFORM,
        signature="(field: CellularField1D, rule_number: int) -> CellularField1D",
        deterministic=True,
        doc="Apply a Wolfram Elementary CA rule"
    )
    def apply_wolfram_rule(field: CellularField1D, rule_number: int) -> CellularField1D:
        """Apply a Wolfram Elementary CA rule.

        Args:
            field: 1D cellular field
            rule_number: Rule number (0-255)

        Returns:
            New field after rule application

        Example:
            >>> ca = cellular.random_init(100, states=2)
            >>> next_gen = cellular.apply_wolfram_rule(ca, 30)  # Rule 30
        """
        if not 0 <= rule_number <= 255:
            raise ValueError(f"Rule number must be 0-255, got {rule_number}")

        # Convert rule number to lookup table
        rule_binary = format(rule_number, '08b')
        rule_table = {i: int(rule_binary[7-i]) for i in range(8)}

        data = field.data
        width = field.width
        new_data = np.zeros(width, dtype=np.int32)

        # Apply rule to each cell
        for i in range(width):
            left = data[(i - 1) % width]
            center = data[i]
            right = data[(i + 1) % width]

            # Convert neighborhood to index (0-7)
            neighborhood = (left << 2) | (center << 1) | right
            new_data[i] = rule_table[neighborhood]

        result = CellularField1D(new_data, field.states)
        result.generation = field.generation + 1
        return result

    # ============================================================================
    # LAYER 2: COMPOSITE CA OPERATIONS
    # ============================================================================

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.TRANSFORM,
        signature="(field: Union[CellularField2D, CellularField1D], rule: Union[CARule, int]) -> Union[CellularField2D, CellularField1D]",
        deterministic=True,
        doc="Advance CA by one generation"
    )
    def step(field: Union[CellularField2D, CellularField1D],
             rule: Union[CARule, int]) -> Union[CellularField2D, CellularField1D]:
        """Advance CA by one generation.

        Args:
            field: Current field
            rule: CA rule or Wolfram rule number

        Returns:
            Field after one generation
        """
        if isinstance(field, CellularField2D) and isinstance(rule, CARule):
            return CellularOperations.apply_rule(field, rule)
        elif isinstance(field, CellularField1D) and isinstance(rule, int):
            return CellularOperations.apply_wolfram_rule(field, rule)
        else:
            raise ValueError("Invalid field/rule combination")

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.TRANSFORM,
        signature="(field: Union[CellularField2D, CellularField1D], rule: Union[CARule, int], steps: int) -> Union[CellularField2D, CellularField1D]",
        deterministic=True,
        doc="Evolve CA for multiple generations"
    )
    def evolve(field: Union[CellularField2D, CellularField1D],
              rule: Union[CARule, int],
              steps: int) -> Union[CellularField2D, CellularField1D]:
        """Evolve CA for multiple generations.

        Args:
            field: Initial field
            rule: CA rule or Wolfram rule number
            steps: Number of generations to evolve

        Returns:
            Field after evolution

        Example:
            >>> result = cellular.evolve(ca, rule, steps=100)
        """
        current = field
        for _ in range(steps):
            current = CellularOperations.step(current, rule)
        return current

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.QUERY,
        signature="(field: Union[CellularField2D, CellularField1D], rule: Union[CARule, int], steps: int) -> list",
        deterministic=True,
        doc="Generate evolution history"
    )
    def history(field: Union[CellularField2D, CellularField1D],
               rule: Union[CARule, int],
               steps: int) -> list:
        """Generate evolution history.

        Args:
            field: Initial field
            rule: CA rule or Wolfram rule number
            steps: Number of generations

        Returns:
            List of fields for each generation

        Example:
            >>> history = cellular.history(ca, rule, steps=50)
        """
        states = [field.copy()]
        current = field

        for _ in range(steps):
            current = CellularOperations.step(current, rule)
            states.append(current.copy())

        return states

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.QUERY,
        signature="(field: CellularField2D) -> dict",
        deterministic=True,
        doc="Analyze pattern properties"
    )
    def analyze_pattern(field: CellularField2D) -> dict:
        """Analyze pattern properties.

        Args:
            field: Cellular field to analyze

        Returns:
            Dictionary with pattern statistics
        """
        alive_count = np.sum(field.data == 1)
        total_cells = field.height * field.width

        return {
            'alive_count': int(alive_count),
            'dead_count': int(total_cells - alive_count),
            'density': float(alive_count / total_cells),
            'generation': field.generation,
            'shape': field.shape
        }

    # ============================================================================
    # LAYER 3: CA CONSTRUCTS (CLASSIC AUTOMATA)
    # ============================================================================

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], density: float, seed: Optional[int]) -> Tuple[CellularField2D, CARule]",
        deterministic=False,
        doc="Create Game of Life cellular automaton"
    )
    def game_of_life(shape: Tuple[int, int],
                     density: float = 0.3,
                     seed: Optional[int] = None) -> Tuple[CellularField2D, CARule]:
        """Create Game of Life cellular automaton.

        Conway's Game of Life: B3/S23
        - Birth: 3 neighbors
        - Survival: 2 or 3 neighbors

        Args:
            shape: Field dimensions
            density: Initial alive cell density
            seed: Random seed

        Returns:
            Tuple of (initial field, rule)

        Example:
            >>> field, rule = cellular.game_of_life((100, 100))
            >>> for i in range(100):
            ...     field = cellular.step(field, rule)
        """
        field = CellularOperations.random_init(shape, states=2, density=density, seed=seed)
        rule = CARule(birth={3}, survival={2, 3}, states=2, neighborhood="moore")
        return field, rule

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], density: float, seed: Optional[int]) -> CellularField2D",
        deterministic=False,
        doc="Create Brian's Brain cellular automaton"
    )
    def brians_brain(shape: Tuple[int, int],
                     density: float = 0.1,
                     seed: Optional[int] = None) -> CellularField2D:
        """Create Brian's Brain cellular automaton.

        3-state CA: 0=dead, 1=firing, 2=refractory

        Args:
            shape: Field dimensions
            density: Initial firing cell density
            seed: Random seed

        Returns:
            Initial field (use brians_brain_step for evolution)
        """
        field = CellularOperations.alloc(shape, states=3, fill_value=0)
        rng = np.random.RandomState(seed)
        mask = rng.random(shape) < density
        field.data[mask] = 1
        return field

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.TRANSFORM,
        signature="(field: CellularField2D) -> CellularField2D",
        deterministic=True,
        doc="Advance Brian's Brain by one generation"
    )
    def brians_brain_step(field: CellularField2D) -> CellularField2D:
        """Advance Brian's Brain by one generation.

        Rules:
        - Dead (0) becomes firing (1) if exactly 2 neighbors are firing
        - Firing (1) becomes refractory (2)
        - Refractory (2) becomes dead (0)

        Args:
            field: Current field

        Returns:
            Next generation
        """
        new_data = np.zeros_like(field.data)

        # Count firing neighbors
        firing_neighbors = CellularOperations.count_neighbors_moore(field, state=1)

        # Dead cells with 2 firing neighbors become firing
        new_data[(field.data == 0) & (firing_neighbors == 2)] = 1

        # Firing cells become refractory
        new_data[field.data == 1] = 2

        # Refractory cells become dead (already 0)

        result = CellularField2D(new_data, 3)
        result.generation = field.generation + 1
        return result

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], density: float, seed: Optional[int]) -> Tuple[CellularField2D, CARule]",
        deterministic=False,
        doc="Create HighLife cellular automaton"
    )
    def highlife(shape: Tuple[int, int],
                density: float = 0.3,
                seed: Optional[int] = None) -> Tuple[CellularField2D, CARule]:
        """Create HighLife cellular automaton.

        HighLife: B36/S23
        - Birth: 3 or 6 neighbors
        - Survival: 2 or 3 neighbors
        - Notable for self-replicating patterns

        Args:
            shape: Field dimensions
            density: Initial density
            seed: Random seed

        Returns:
            Tuple of (initial field, rule)
        """
        field = CellularOperations.random_init(shape, states=2, density=density, seed=seed)
        rule = CARule(birth={3, 6}, survival={2, 3}, states=2, neighborhood="moore")
        return field, rule

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(shape: Tuple[int, int], density: float, seed: Optional[int]) -> Tuple[CellularField2D, CARule]",
        deterministic=False,
        doc="Create Seeds cellular automaton"
    )
    def seeds(shape: Tuple[int, int],
             density: float = 0.1,
             seed: Optional[int] = None) -> Tuple[CellularField2D, CARule]:
        """Create Seeds cellular automaton.

        Seeds: B2/S
        - Birth: 2 neighbors
        - Survival: none (cells live only one generation)
        - Creates explosive growth patterns

        Args:
            shape: Field dimensions
            density: Initial density
            seed: Random seed

        Returns:
            Tuple of (initial field, rule)
        """
        field = CellularOperations.random_init(shape, states=2, density=density, seed=seed)
        rule = CARule(birth={2}, survival=set(), states=2, neighborhood="moore")
        return field, rule

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(width: int, rule_number: int, initial_state: Optional[ndarray]) -> Tuple[CellularField1D, int]",
        deterministic=True,
        doc="Create Wolfram Elementary Cellular Automaton"
    )
    def wolfram_ca(width: int, rule_number: int,
                  initial_state: Optional[np.ndarray] = None) -> Tuple[CellularField1D, int]:
        """Create Wolfram Elementary Cellular Automaton.

        Args:
            width: Width of 1D field
            rule_number: Wolfram rule (0-255)
            initial_state: Optional initial configuration (default: single cell)

        Returns:
            Tuple of (initial field, rule number)

        Example:
            >>> field, rule = cellular.wolfram_ca(100, 30)  # Rule 30
            >>> history = cellular.history(field, rule, steps=50)
        """
        if initial_state is not None:
            field = CellularField1D(initial_state, states=2)
        else:
            # Default: single alive cell in center
            data = np.zeros(width, dtype=np.int32)
            data[width // 2] = 1
            field = CellularField1D(data, states=2)

        return field, rule_number

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.QUERY,
        signature="(field: Union[CellularField2D, CellularField1D]) -> ndarray",
        deterministic=True,
        doc="Convert cellular field to NumPy array"
    )
    def to_array(field: Union[CellularField2D, CellularField1D]) -> np.ndarray:
        """Convert cellular field to NumPy array.

        Args:
            field: Cellular field

        Returns:
            NumPy array of cell states
        """
        return field.data.copy()

    @staticmethod
    @operator(
        domain="cellular",
        category=OpCategory.CONSTRUCT,
        signature="(data: ndarray, states: int) -> Union[CellularField2D, CellularField1D]",
        deterministic=True,
        doc="Create cellular field from NumPy array"
    )
    def from_array(data: np.ndarray, states: int = 2) -> Union[CellularField2D, CellularField1D]:
        """Create cellular field from NumPy array.

        Args:
            data: Array of cell states
            states: Number of possible states

        Returns:
            Cellular field
        """
        if len(data.shape) == 2:
            return CellularField2D(data, states)
        elif len(data.shape) == 1:
            return CellularField1D(data, states)
        else:
            raise ValueError(f"Array must be 1D or 2D, got shape {data.shape}")


# Export namespace
cellular = CellularOperations()

# Export operators for domain registry discovery
alloc = CellularOperations.alloc
from_array = CellularOperations.from_array
random_init = CellularOperations.random_init
step = CellularOperations.step
evolve = CellularOperations.evolve
count_neighbors_moore = CellularOperations.count_neighbors_moore
count_neighbors_von_neumann = CellularOperations.count_neighbors_von_neumann
apply_rule = CellularOperations.apply_rule
to_array = CellularOperations.to_array
history = CellularOperations.history
analyze_pattern = CellularOperations.analyze_pattern
game_of_life = CellularOperations.game_of_life
highlife = CellularOperations.highlife
brians_brain = CellularOperations.brians_brain
seeds = CellularOperations.seeds
brians_brain_step = CellularOperations.brians_brain_step
wolfram_ca = CellularOperations.wolfram_ca
apply_wolfram_rule = CellularOperations.apply_wolfram_rule

__all__ = ['cellular', 'CellularField2D', 'CellularField1D', 'CARule']
