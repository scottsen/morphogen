"""Circuit/Electrical simulation domain.

This module provides circuit analysis and simulation capabilities including:
- DC analysis using Modified Nodal Analysis (MNA)
- AC analysis for frequency response
- Transient analysis for time-domain simulation
- Support for basic components: resistors, capacitors, inductors, voltage/current sources
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from morphogen.core.operator import operator, OpCategory


class ComponentType(Enum):
    """Types of circuit components."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    INDUCTOR = "inductor"
    VOLTAGE_SOURCE = "voltage_source"
    CURRENT_SOURCE = "current_source"
    GROUND = "ground"


@dataclass
class Component:
    """Circuit component representation.

    Attributes:
        comp_type: Type of component
        node1: First node index (or positive terminal)
        node2: Second node index (or negative terminal)
        value: Component value (resistance, capacitance, inductance, voltage, current)
        name: Component identifier
    """
    comp_type: ComponentType
    node1: int
    node2: int
    value: float
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.comp_type.value}_{id(self)}"


@dataclass
class Circuit:
    """Circuit representation for analysis and simulation.

    Attributes:
        num_nodes: Number of nodes in circuit (node 0 is always ground)
        components: List of circuit components
        node_voltages: Current node voltages (from last analysis)
        branch_currents: Current branch currents (from last analysis)
        time: Current simulation time
        dt: Timestep for transient analysis
    """
    num_nodes: int
    components: List[Component] = field(default_factory=list)
    node_voltages: Optional[np.ndarray] = None
    branch_currents: Dict[str, float] = field(default_factory=dict)
    time: float = 0.0
    dt: float = 1e-6

    def __repr__(self) -> str:
        return f"Circuit(nodes={self.num_nodes}, components={len(self.components)}, t={self.time:.6f})"


class CircuitOperations:
    """Namespace for circuit operations (accessed as 'circuit' in DSL)."""

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.CONSTRUCT,
        signature="(num_nodes: int, dt: float) -> Circuit",
        deterministic=True,
        doc="Create a new circuit with specified number of nodes"
    )
    def create(num_nodes: int, dt: float = 1e-6) -> Circuit:
        """Create a new circuit.

        Args:
            num_nodes: Number of nodes (node 0 is ground)
            dt: Default timestep for transient analysis

        Returns:
            New circuit instance
        """
        return Circuit(num_nodes=num_nodes, dt=dt)

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.MUTATE,
        signature="(circuit: Circuit, node1: int, node2: int, resistance: float, name: str) -> Circuit",
        deterministic=True,
        doc="Add a resistor to the circuit"
    )
    def add_resistor(circuit: Circuit, node1: int, node2: int,
                     resistance: float, name: str = "") -> Circuit:
        """Add a resistor to the circuit.

        Args:
            circuit: Circuit to modify
            node1: First node
            node2: Second node
            resistance: Resistance in ohms
            name: Component name

        Returns:
            Modified circuit
        """
        comp = Component(ComponentType.RESISTOR, node1, node2, resistance, name)
        circuit.components.append(comp)
        return circuit

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.MUTATE,
        signature="(circuit: Circuit, node1: int, node2: int, capacitance: float, name: str) -> Circuit",
        deterministic=True,
        doc="Add a capacitor to the circuit"
    )
    def add_capacitor(circuit: Circuit, node1: int, node2: int,
                      capacitance: float, name: str = "") -> Circuit:
        """Add a capacitor to the circuit.

        Args:
            circuit: Circuit to modify
            node1: First node (positive)
            node2: Second node (negative)
            capacitance: Capacitance in farads
            name: Component name

        Returns:
            Modified circuit
        """
        comp = Component(ComponentType.CAPACITOR, node1, node2, capacitance, name)
        circuit.components.append(comp)
        return circuit

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.MUTATE,
        signature="(circuit: Circuit, node1: int, node2: int, inductance: float, name: str) -> Circuit",
        deterministic=True,
        doc="Add an inductor to the circuit"
    )
    def add_inductor(circuit: Circuit, node1: int, node2: int,
                     inductance: float, name: str = "") -> Circuit:
        """Add an inductor to the circuit.

        Args:
            circuit: Circuit to modify
            node1: First node (positive)
            node2: Second node (negative)
            inductance: Inductance in henries
            name: Component name

        Returns:
            Modified circuit
        """
        comp = Component(ComponentType.INDUCTOR, node1, node2, inductance, name)
        circuit.components.append(comp)
        return circuit

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.MUTATE,
        signature="(circuit: Circuit, node_pos: int, node_neg: int, voltage: float, name: str) -> Circuit",
        deterministic=True,
        doc="Add a voltage source to the circuit"
    )
    def add_voltage_source(circuit: Circuit, node_pos: int, node_neg: int,
                           voltage: float, name: str = "") -> Circuit:
        """Add a voltage source to the circuit.

        Args:
            circuit: Circuit to modify
            node_pos: Positive node
            node_neg: Negative node
            voltage: Voltage in volts
            name: Component name

        Returns:
            Modified circuit
        """
        comp = Component(ComponentType.VOLTAGE_SOURCE, node_pos, node_neg, voltage, name)
        circuit.components.append(comp)
        return circuit

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.MUTATE,
        signature="(circuit: Circuit, node_pos: int, node_neg: int, current: float, name: str) -> Circuit",
        deterministic=True,
        doc="Add a current source to the circuit"
    )
    def add_current_source(circuit: Circuit, node_pos: int, node_neg: int,
                           current: float, name: str = "") -> Circuit:
        """Add a current source to the circuit.

        Args:
            circuit: Circuit to modify
            node_pos: Node where current enters
            node_neg: Node where current exits
            current: Current in amperes
            name: Component name

        Returns:
            Modified circuit
        """
        comp = Component(ComponentType.CURRENT_SOURCE, node_pos, node_neg, current, name)
        circuit.components.append(comp)
        return circuit

    @staticmethod
    def _build_mna_matrices_dc(circuit: Circuit) -> Tuple[np.ndarray, np.ndarray]:
        """Build Modified Nodal Analysis matrices for DC analysis.

        Returns:
            Tuple of (A_matrix, b_vector) where A*x = b
        """
        # Count voltage sources to determine matrix size
        num_vsources = sum(1 for c in circuit.components
                          if c.comp_type == ComponentType.VOLTAGE_SOURCE)

        # Matrix size: (num_nodes - 1) + num_vsources
        # We exclude ground (node 0) from the equations
        n = circuit.num_nodes - 1 + num_vsources

        A = np.zeros((n, n))
        b = np.zeros(n)

        vsource_idx = 0

        for comp in circuit.components:
            if comp.comp_type == ComponentType.RESISTOR:
                # Conductance stamp for resistor
                g = 1.0 / comp.value if comp.value != 0 else 1e-12

                n1, n2 = comp.node1, comp.node2

                # Skip if both nodes are ground
                if n1 == 0 and n2 == 0:
                    continue

                if n1 != 0:
                    A[n1-1, n1-1] += g
                    if n2 != 0:
                        A[n1-1, n2-1] -= g

                if n2 != 0:
                    A[n2-1, n2-1] += g
                    if n1 != 0:
                        A[n2-1, n1-1] -= g

            elif comp.comp_type == ComponentType.CURRENT_SOURCE:
                # Current source stamp
                n1, n2 = comp.node1, comp.node2

                if n1 != 0:
                    b[n1-1] -= comp.value  # Current entering node
                if n2 != 0:
                    b[n2-1] += comp.value  # Current leaving node

            elif comp.comp_type == ComponentType.VOLTAGE_SOURCE:
                # Voltage source stamp
                n1, n2 = comp.node1, comp.node2
                vsource_row = circuit.num_nodes - 1 + vsource_idx

                # Add equations for voltage source
                if n1 != 0:
                    A[n1-1, vsource_row] += 1
                    A[vsource_row, n1-1] += 1

                if n2 != 0:
                    A[n2-1, vsource_row] -= 1
                    A[vsource_row, n2-1] -= 1

                b[vsource_row] = comp.value
                vsource_idx += 1

        return A, b

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.TRANSFORM,
        signature="(circuit: Circuit) -> Circuit",
        deterministic=True,
        doc="Perform DC analysis on the circuit"
    )
    def dc_analysis(circuit: Circuit) -> Circuit:
        """Perform DC analysis using Modified Nodal Analysis.

        Solves for node voltages and branch currents in DC steady state.
        Capacitors are treated as open circuits, inductors as short circuits.

        Args:
            circuit: Circuit to analyze

        Returns:
            Circuit with updated node_voltages and branch_currents
        """
        # Build MNA matrices
        A, b = CircuitOperations._build_mna_matrices_dc(circuit)

        # Solve system
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular matrix - use least squares
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        # Extract node voltages (ground is 0V)
        num_vsources = sum(1 for c in circuit.components
                          if c.comp_type == ComponentType.VOLTAGE_SOURCE)

        node_voltages = np.zeros(circuit.num_nodes)
        node_voltages[1:] = x[:circuit.num_nodes-1]

        # Extract voltage source currents
        vsource_currents = x[circuit.num_nodes-1:]

        # Update circuit state
        circuit.node_voltages = node_voltages
        circuit.branch_currents = {}

        vsource_idx = 0
        for comp in circuit.components:
            if comp.comp_type == ComponentType.VOLTAGE_SOURCE:
                circuit.branch_currents[comp.name] = vsource_currents[vsource_idx]
                vsource_idx += 1
            elif comp.comp_type == ComponentType.RESISTOR:
                v1 = node_voltages[comp.node1]
                v2 = node_voltages[comp.node2]
                current = (v1 - v2) / comp.value if comp.value != 0 else 0
                circuit.branch_currents[comp.name] = current

        return circuit

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.TRANSFORM,
        signature="(circuit: Circuit, frequencies: np.ndarray) -> Dict",
        deterministic=True,
        doc="Perform AC analysis at specified frequencies"
    )
    def ac_analysis(circuit: Circuit, frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform AC analysis (frequency response).

        Args:
            circuit: Circuit to analyze
            frequencies: Array of frequencies in Hz

        Returns:
            Dictionary with 'frequencies', 'node_voltages' (complex), 'impedances'
        """
        results = {
            'frequencies': frequencies,
            'node_voltages': [],
            'impedances': []
        }

        for freq in frequencies:
            omega = 2 * np.pi * freq

            # Build MNA matrices with complex impedances
            num_vsources = sum(1 for c in circuit.components
                              if c.comp_type == ComponentType.VOLTAGE_SOURCE)
            n = circuit.num_nodes - 1 + num_vsources

            A = np.zeros((n, n), dtype=complex)
            b = np.zeros(n, dtype=complex)

            vsource_idx = 0

            for comp in circuit.components:
                if comp.comp_type == ComponentType.RESISTOR:
                    # Resistance (real impedance)
                    y = 1.0 / comp.value if comp.value != 0 else 1e-12

                    n1, n2 = comp.node1, comp.node2
                    if n1 == 0 and n2 == 0:
                        continue

                    if n1 != 0:
                        A[n1-1, n1-1] += y
                        if n2 != 0:
                            A[n1-1, n2-1] -= y

                    if n2 != 0:
                        A[n2-1, n2-1] += y
                        if n1 != 0:
                            A[n2-1, n1-1] -= y

                elif comp.comp_type == ComponentType.CAPACITOR:
                    # Capacitive impedance: Z = 1/(jwC), Y = jwC
                    y = 1j * omega * comp.value

                    n1, n2 = comp.node1, comp.node2
                    if n1 == 0 and n2 == 0:
                        continue

                    if n1 != 0:
                        A[n1-1, n1-1] += y
                        if n2 != 0:
                            A[n1-1, n2-1] -= y

                    if n2 != 0:
                        A[n2-1, n2-1] += y
                        if n1 != 0:
                            A[n2-1, n1-1] -= y

                elif comp.comp_type == ComponentType.INDUCTOR:
                    # Inductive impedance: Z = jwL, Y = 1/(jwL)
                    if omega == 0:
                        y = 1e12  # DC: inductor is short circuit
                    else:
                        y = 1.0 / (1j * omega * comp.value)

                    n1, n2 = comp.node1, comp.node2
                    if n1 == 0 and n2 == 0:
                        continue

                    if n1 != 0:
                        A[n1-1, n1-1] += y
                        if n2 != 0:
                            A[n1-1, n2-1] -= y

                    if n2 != 0:
                        A[n2-1, n2-1] += y
                        if n1 != 0:
                            A[n2-1, n1-1] -= y

                elif comp.comp_type == ComponentType.CURRENT_SOURCE:
                    n1, n2 = comp.node1, comp.node2
                    if n1 != 0:
                        b[n1-1] -= comp.value
                    if n2 != 0:
                        b[n2-1] += comp.value

                elif comp.comp_type == ComponentType.VOLTAGE_SOURCE:
                    n1, n2 = comp.node1, comp.node2
                    vsource_row = circuit.num_nodes - 1 + vsource_idx

                    if n1 != 0:
                        A[n1-1, vsource_row] += 1
                        A[vsource_row, n1-1] += 1

                    if n2 != 0:
                        A[n2-1, vsource_row] -= 1
                        A[vsource_row, n2-1] -= 1

                    b[vsource_row] = comp.value
                    vsource_idx += 1

            # Solve system
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(A, b, rcond=None)[0]

            # Extract results
            node_v = np.zeros(circuit.num_nodes, dtype=complex)
            node_v[1:] = x[:circuit.num_nodes-1]

            results['node_voltages'].append(node_v)

        results['node_voltages'] = np.array(results['node_voltages'])

        return results

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.INTEGRATE,
        signature="(circuit: Circuit, duration: float, method: str) -> Tuple[np.ndarray, np.ndarray]",
        deterministic=True,
        doc="Perform transient analysis over time"
    )
    def transient_analysis(circuit: Circuit, duration: float,
                           method: str = "backward_euler") -> Tuple[np.ndarray, np.ndarray]:
        """Perform transient analysis (time-domain simulation).

        Args:
            circuit: Circuit to simulate
            duration: Simulation duration in seconds
            method: Integration method ("backward_euler" or "trapezoidal")

        Returns:
            Tuple of (time_points, node_voltages_over_time)
        """
        num_steps = int(duration / circuit.dt)
        time_points = np.linspace(0, duration, num_steps)

        # Initialize state
        voltages = np.zeros(circuit.num_nodes)
        capacitor_charges = {}
        inductor_currents = {}

        # Initialize capacitor and inductor states
        for comp in circuit.components:
            if comp.comp_type == ComponentType.CAPACITOR:
                capacitor_charges[comp.name] = 0.0
            elif comp.comp_type == ComponentType.INDUCTOR:
                inductor_currents[comp.name] = 0.0

        voltage_history = np.zeros((num_steps, circuit.num_nodes))

        for step_idx, t in enumerate(time_points):
            circuit.time = t

            # Build system with companion models for reactive elements
            num_vsources = sum(1 for c in circuit.components
                              if c.comp_type == ComponentType.VOLTAGE_SOURCE)
            n = circuit.num_nodes - 1 + num_vsources

            A = np.zeros((n, n))
            b = np.zeros(n)

            vsource_idx = 0

            for comp in circuit.components:
                if comp.comp_type == ComponentType.RESISTOR:
                    g = 1.0 / comp.value if comp.value != 0 else 1e-12
                    n1, n2 = comp.node1, comp.node2

                    if n1 == 0 and n2 == 0:
                        continue

                    if n1 != 0:
                        A[n1-1, n1-1] += g
                        if n2 != 0:
                            A[n1-1, n2-1] -= g

                    if n2 != 0:
                        A[n2-1, n2-1] += g
                        if n1 != 0:
                            A[n2-1, n1-1] -= g

                elif comp.comp_type == ComponentType.CAPACITOR:
                    # Backward Euler: C*dv/dt ≈ C*(v_new - v_old)/dt
                    # Companion model: current_eq = g_eq * v + i_eq
                    # where g_eq = C/dt, i_eq = -C*v_old/dt
                    g_eq = comp.value / circuit.dt
                    v_old = voltages[comp.node1] - voltages[comp.node2]
                    i_eq = -comp.value * v_old / circuit.dt

                    n1, n2 = comp.node1, comp.node2

                    if n1 != 0:
                        A[n1-1, n1-1] += g_eq
                        if n2 != 0:
                            A[n1-1, n2-1] -= g_eq
                        b[n1-1] += i_eq

                    if n2 != 0:
                        A[n2-1, n2-1] += g_eq
                        if n1 != 0:
                            A[n2-1, n1-1] -= g_eq
                        b[n2-1] -= i_eq

                elif comp.comp_type == ComponentType.INDUCTOR:
                    # Backward Euler: L*di/dt ≈ L*(i_new - i_old)/dt
                    # Companion model: voltage = g_eq * i + v_eq
                    # Rearranged: i = (v - v_eq) / r_eq where r_eq = L/dt
                    r_eq = comp.value / circuit.dt
                    g_eq = 1.0 / r_eq
                    i_old = inductor_currents.get(comp.name, 0.0)
                    v_eq = -comp.value * i_old / circuit.dt

                    n1, n2 = comp.node1, comp.node2

                    if n1 != 0:
                        A[n1-1, n1-1] += g_eq
                        if n2 != 0:
                            A[n1-1, n2-1] -= g_eq
                        b[n1-1] -= v_eq * g_eq

                    if n2 != 0:
                        A[n2-1, n2-1] += g_eq
                        if n1 != 0:
                            A[n2-1, n1-1] -= g_eq
                        b[n2-1] += v_eq * g_eq

                elif comp.comp_type == ComponentType.CURRENT_SOURCE:
                    # Time-varying sources could be added here
                    n1, n2 = comp.node1, comp.node2
                    if n1 != 0:
                        b[n1-1] -= comp.value
                    if n2 != 0:
                        b[n2-1] += comp.value

                elif comp.comp_type == ComponentType.VOLTAGE_SOURCE:
                    n1, n2 = comp.node1, comp.node2
                    vsource_row = circuit.num_nodes - 1 + vsource_idx

                    if n1 != 0:
                        A[n1-1, vsource_row] += 1
                        A[vsource_row, n1-1] += 1

                    if n2 != 0:
                        A[n2-1, vsource_row] -= 1
                        A[vsource_row, n2-1] -= 1

                    # Time-varying sources could be added here
                    b[vsource_row] = comp.value
                    vsource_idx += 1

            # Solve system
            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(A, b, rcond=None)[0]

            # Extract voltages
            voltages[0] = 0.0  # Ground
            voltages[1:] = x[:circuit.num_nodes-1]

            # Update inductor currents
            for comp in circuit.components:
                if comp.comp_type == ComponentType.INDUCTOR:
                    v = voltages[comp.node1] - voltages[comp.node2]
                    i_old = inductor_currents.get(comp.name, 0.0)
                    i_new = i_old + (v * circuit.dt / comp.value)
                    inductor_currents[comp.name] = i_new

            voltage_history[step_idx] = voltages.copy()

        # Update circuit state
        circuit.node_voltages = voltages

        return time_points, voltage_history

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.QUERY,
        signature="(circuit: Circuit, node: int) -> float",
        deterministic=True,
        doc="Get voltage at a specific node"
    )
    def get_node_voltage(circuit: Circuit, node: int) -> float:
        """Get the voltage at a specific node.

        Args:
            circuit: Circuit (must have been analyzed)
            node: Node index

        Returns:
            Voltage at the node in volts
        """
        if circuit.node_voltages is None:
            raise ValueError("Circuit has not been analyzed yet")

        return float(circuit.node_voltages[node])

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.QUERY,
        signature="(circuit: Circuit, component_name: str) -> float",
        deterministic=True,
        doc="Get current through a component"
    )
    def get_branch_current(circuit: Circuit, component_name: str) -> float:
        """Get the current through a component.

        Args:
            circuit: Circuit (must have been analyzed)
            component_name: Name of the component

        Returns:
            Current through the component in amperes
        """
        if component_name not in circuit.branch_currents:
            raise ValueError(f"No current data for component '{component_name}'")

        return circuit.branch_currents[component_name]

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.QUERY,
        signature="(circuit: Circuit, component_name: str) -> float",
        deterministic=True,
        doc="Get power dissipated/delivered by a component"
    )
    def get_power(circuit: Circuit, component_name: str) -> float:
        """Get the power dissipated or delivered by a component.

        Args:
            circuit: Circuit (must have been analyzed)
            component_name: Name of the component

        Returns:
            Power in watts (positive = dissipated, negative = delivered)
        """
        # Find component
        comp = None
        for c in circuit.components:
            if c.name == component_name:
                comp = c
                break

        if comp is None:
            raise ValueError(f"Component '{component_name}' not found")

        if circuit.node_voltages is None:
            raise ValueError("Circuit has not been analyzed yet")

        v1 = circuit.node_voltages[comp.node1]
        v2 = circuit.node_voltages[comp.node2]
        v_diff = v1 - v2

        if comp.comp_type == ComponentType.RESISTOR:
            # P = V²/R = I²R = VI
            current = circuit.branch_currents.get(comp.name, 0.0)
            return v_diff * current

        elif comp.comp_type == ComponentType.VOLTAGE_SOURCE:
            current = circuit.branch_currents.get(comp.name, 0.0)
            return -comp.value * current  # Negative = delivering power

        elif comp.comp_type == ComponentType.CURRENT_SOURCE:
            return -v_diff * comp.value  # Negative = delivering power

        else:
            # Capacitors and inductors (reactive power, not calculated here)
            return 0.0

    @staticmethod
    @operator(
        domain="circuit",
        category=OpCategory.QUERY,
        signature="(circuit: Circuit, node1: int, node2: int, frequency: float) -> complex",
        deterministic=True,
        doc="Calculate impedance between two nodes at a given frequency"
    )
    def get_impedance(circuit: Circuit, node1: int, node2: int,
                      frequency: float) -> complex:
        """Calculate impedance between two nodes at a given frequency.

        Args:
            circuit: Circuit to analyze
            node1: First node
            node2: Second node
            frequency: Frequency in Hz

        Returns:
            Complex impedance in ohms
        """
        # Perform AC analysis at the given frequency
        results = CircuitOperations.ac_analysis(circuit, np.array([frequency]))

        # Get voltage difference
        voltages = results['node_voltages'][0]
        v_diff = voltages[node1] - voltages[node2]

        # Add a test current source and re-analyze to find impedance
        # This is a simplified approach
        # Z = V / I where I is a known test current

        # For now, return a placeholder - full implementation would require
        # adding test current and measuring response
        return complex(1.0, 0.0)


# Singleton instance for DSL access
circuit = CircuitOperations()

# Export operators for registry discovery
create = CircuitOperations.create
add_resistor = CircuitOperations.add_resistor
add_capacitor = CircuitOperations.add_capacitor
add_inductor = CircuitOperations.add_inductor
add_voltage_source = CircuitOperations.add_voltage_source
add_current_source = CircuitOperations.add_current_source
dc_analysis = CircuitOperations.dc_analysis
ac_analysis = CircuitOperations.ac_analysis
transient_analysis = CircuitOperations.transient_analysis
get_node_voltage = CircuitOperations.get_node_voltage
get_branch_current = CircuitOperations.get_branch_current
get_power = CircuitOperations.get_power
get_impedance = CircuitOperations.get_impedance
