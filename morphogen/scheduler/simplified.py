"""
Simplified Multirate Scheduler for Morphogen GraphIR

This is a minimal viable scheduler implementing:
- ✅ Multirate (audio + control)
- ✅ GCD-based master clock
- ✅ Cross-rate resampling (linear only)
- ✅ Topological execution order
- ✅ Simple execution loop

Deferred to future versions:
- ⏸️ Event fencing (sample-accurate events)
- ⏸️ Hot reload (state snapshots)
- ⏸️ Double buffering (optimization)
- ⏸️ Multiple resampling modes (hold, cubic, sinc)
- ⏸️ Visual/sim rates (audio + control only for v1)

Based on: /home/scottsen/src/projects/morphogen/docs/specifications/scheduler.md
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math
import numpy as np
from functools import reduce


@dataclass
class RateGroup:
    """
    Operators grouped by execution rate

    Attributes:
        rate: Rate class name (e.g., "audio", "control")
        rate_hz: Rate in Hz
        operators: List of operator IDs in topological order
        multiplier: Samples per master tick (rate_hz / master_rate)
    """
    rate: str  # "audio" | "control"
    rate_hz: float
    operators: List[str]  # Node IDs in topological order
    multiplier: int  # Samples per master tick


class SimplifiedScheduler:
    """
    Minimal multirate scheduler for GraphIR v1.0

    Features:
    - Multirate execution (audio + control)
    - GCD-based master clock
    - Linear cross-rate resampling
    - Topological execution order
    - Simple hop-based execution loop

    Limitations (v1.0):
    - No event fencing
    - No hot reload
    - No double buffering
    - Linear resampling only
    - Audio + control rates only

    Example:
        graph = GraphIR.from_json("synth.morph.json")
        scheduler = SimplifiedScheduler(graph, sample_rate=48000)
        output = scheduler.execute(duration_samples=48000)  # 1 second
    """

    # Default rate mappings (Hz)
    DEFAULT_RATES = {
        "audio": 48000,
        "control": 1000,
        "visual": 60,  # Not used in v1
        "sim": 100,    # Not used in v1
    }

    def __init__(
        self,
        graph,
        sample_rate: int = 48000,
        hop_size: int = 128,
        rate_overrides: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize scheduler from GraphIR

        Args:
            graph: GraphIR instance to execute
            sample_rate: Audio sample rate in Hz (default: 48000)
            hop_size: Block size for processing (default: 128)
            rate_overrides: Optional rate overrides (e.g., {"control": 500})
        """
        self.graph = graph
        self.sample_rate = sample_rate
        self.hop_size = hop_size

        # Apply rate overrides
        self.rates = self.DEFAULT_RATES.copy()

        # Update audio rate from graph (before overrides, so overrides take precedence)
        if hasattr(graph, 'sample_rate') and graph.sample_rate:
            self.rates["audio"] = graph.sample_rate
            self.sample_rate = graph.sample_rate

        # Apply rate overrides (takes precedence over graph settings)
        if rate_overrides:
            self.rates.update(rate_overrides)
            # Update sample_rate if audio was overridden
            if "audio" in rate_overrides:
                self.sample_rate = int(rate_overrides["audio"])

        # Extract rates used in this graph
        self.active_rates = self._extract_active_rates()

        # Compute GCD master clock
        self.master_rate = self._compute_master_clock()

        # Create rate groups with multipliers
        self.rate_groups = self._create_rate_groups()

        # Operator state storage (buffers)
        self.buffers: Dict[str, np.ndarray] = {}  # port_ref -> buffer

        # Initialize operator registry and executor
        from morphogen.scheduler.operator_executor import create_audio_registry, OperatorExecutor
        self.operator_registry = create_audio_registry()
        self.operator_executor = OperatorExecutor(self.operator_registry, self.sample_rate)

    def _extract_active_rates(self) -> Dict[str, float]:
        """Extract rates actually used in the graph"""
        active = {}
        for node in self.graph.nodes:
            if node.rate in self.rates:
                active[node.rate] = self.rates[node.rate]
        return active

    def _compute_master_clock(self) -> float:
        """
        Compute GCD master clock from active rates

        The master clock is the GCD of all rates, allowing each rate
        to execute as a multiple of the master tick.

        Returns:
            Master clock rate in Hz
        """
        if not self.active_rates:
            return self.sample_rate

        # Convert to integers for GCD
        rate_values = [int(rate) for rate in self.active_rates.values()]

        # Compute GCD of all rates
        master = reduce(math.gcd, rate_values)

        return float(master)

    def _create_rate_groups(self) -> List[RateGroup]:
        """
        Partition nodes into rate groups and compute topological order

        Returns:
            List of rate groups with topologically sorted operators
        """
        # Group nodes by rate
        groups_dict: Dict[str, List[str]] = {}
        for node in self.graph.nodes:
            if node.rate not in groups_dict:
                groups_dict[node.rate] = []
            groups_dict[node.rate].append(node.id)

        # Compute topological sort for the entire graph
        from ..graph_ir.validation import GraphValidator
        validator = GraphValidator(self.graph)
        topo_order = validator.topological_sort()

        if topo_order is None:
            raise ValueError("Graph contains cycle - cannot create execution order")

        # Create rate groups with sorted operators
        rate_groups = []
        for rate_name, node_ids in groups_dict.items():
            if rate_name not in self.active_rates:
                continue  # Skip inactive rates

            # Filter topological order to just this rate's nodes
            sorted_nodes = [nid for nid in topo_order if nid in node_ids]

            rate_hz = self.active_rates[rate_name]
            multiplier = int(rate_hz / self.master_rate)

            rate_group = RateGroup(
                rate=rate_name,
                rate_hz=rate_hz,
                operators=sorted_nodes,
                multiplier=multiplier,
            )
            rate_groups.append(rate_group)

        # Sort rate groups by rate (highest first)
        rate_groups.sort(key=lambda g: g.rate_hz, reverse=True)

        return rate_groups

    def _linear_resample(
        self,
        input_buffer: np.ndarray,
        from_rate: float,
        to_rate: float,
    ) -> np.ndarray:
        """
        Linear interpolation resampling

        Resamples input_buffer from from_rate to to_rate using
        linear interpolation.

        Args:
            input_buffer: Input samples
            from_rate: Source rate in Hz
            to_rate: Target rate in Hz

        Returns:
            Resampled buffer
        """
        if from_rate == to_rate:
            return input_buffer.copy()

        input_len = len(input_buffer)
        ratio = to_rate / from_rate
        output_len = int(input_len * ratio)

        if output_len == 0:
            return np.array([])

        # Create output indices in input space
        output_indices = np.linspace(0, input_len - 1, output_len)

        # Linear interpolation
        output = np.interp(output_indices, np.arange(input_len), input_buffer)

        return output

    def _get_input_buffer(
        self,
        port_ref: str,
        num_samples: int,
        target_rate: float,
    ) -> np.ndarray:
        """
        Get input buffer with resampling if needed

        Args:
            port_ref: Port reference (e.g., "osc1:out")
            num_samples: Number of samples needed
            target_rate: Target rate in Hz

        Returns:
            Buffer with resampled data
        """
        # Check if buffer exists
        if port_ref not in self.buffers:
            # Return zeros if buffer doesn't exist yet
            return np.zeros(num_samples)

        buffer = self.buffers[port_ref]

        # Find source rate
        node_id = port_ref.split(":")[0]
        source_node = self.graph.get_node(node_id)
        if source_node is None:
            return np.zeros(num_samples)

        source_rate = self.rates.get(source_node.rate, target_rate)

        # Resample if rates don't match
        if source_rate != target_rate:
            buffer = self._linear_resample(buffer, source_rate, target_rate)

        # Trim or pad to match num_samples
        if len(buffer) > num_samples:
            return buffer[:num_samples]
        elif len(buffer) < num_samples:
            # Pad with zeros or repeat last value
            padded = np.zeros(num_samples)
            padded[:len(buffer)] = buffer
            if len(buffer) > 0:
                padded[len(buffer):] = buffer[-1]  # Hold last value
            return padded

        return buffer

    def _execute_operator(
        self,
        node_id: str,
        inputs: Dict[str, np.ndarray],
        num_samples: int,
    ) -> Dict[str, np.ndarray]:
        """
        Execute operator using real operator implementations.

        Args:
            node_id: Node ID
            inputs: Input buffers (port_name -> buffer)
            num_samples: Number of samples to generate

        Returns:
            Output buffers (port_name -> buffer)
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return {}

        # Get operator rate
        rate_hz = self.rates.get(node.rate, self.sample_rate)

        # Execute operator via operator executor
        outputs = self.operator_executor.execute(
            operator_name=node.op,
            node_id=node_id,
            params=node.params,
            inputs=inputs,
            num_samples=num_samples,
            rate_hz=rate_hz,
        )

        # Ensure all declared output ports are present
        for output_port in node.outputs:
            if output_port.name not in outputs:
                # If operator didn't provide this output, try "out" as default
                if "out" in outputs:
                    outputs[output_port.name] = outputs["out"]
                else:
                    # Last resort: zeros
                    outputs[output_port.name] = np.zeros(num_samples)

        return outputs

    def execute(
        self,
        duration_samples: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute graph for specified duration

        Args:
            duration_samples: Duration in samples (at sample_rate)
            duration_seconds: Duration in seconds

        Returns:
            Dictionary of output buffers (output_name -> buffer)
        """
        # Determine duration
        if duration_seconds is not None:
            duration_samples = int(duration_seconds * self.sample_rate)
        elif duration_samples is None:
            duration_samples = self.sample_rate  # Default: 1 second

        current_sample = 0
        output_buffers = {name: [] for name in self.graph.outputs.keys()}

        # Main execution loop
        while current_sample < duration_samples:
            # Compute hop size (may be smaller at end)
            hop = min(self.hop_size, duration_samples - current_sample)

            # Execute each rate group for this hop
            for rate_group in self.rate_groups:
                # Calculate how many samples this rate needs for this hop
                rate_samples = int((hop / self.sample_rate) * rate_group.rate_hz)
                if rate_samples == 0:
                    rate_samples = 1  # Always execute at least one sample

                # Execute operators in topological order
                for op_id in rate_group.operators:
                    node = self.graph.get_node(op_id)
                    if node is None:
                        continue

                    # Find input edges
                    input_edges = [e for e in self.graph.edges if e.to_node == op_id]

                    # Gather inputs with resampling
                    inputs = {}
                    for edge in input_edges:
                        port_name = edge.to_port_name
                        buffer = self._get_input_buffer(
                            edge.from_port,
                            rate_samples,
                            rate_group.rate_hz,
                        )
                        inputs[port_name] = buffer

                    # Execute operator (mock for now)
                    outputs = self._execute_operator(op_id, inputs, rate_samples)

                    # Store outputs in buffers
                    for port_name, buffer in outputs.items():
                        port_ref = f"{op_id}:{port_name}"
                        self.buffers[port_ref] = buffer

            # Collect graph outputs for this hop
            for output_name, port_refs in self.graph.outputs.items():
                # For now, just take the first output
                if port_refs:
                    port_ref = port_refs[0]
                    if port_ref in self.buffers:
                        # Resample to audio rate if needed
                        buffer = self.buffers[port_ref]
                        node_id = port_ref.split(":")[0]
                        source_node = self.graph.get_node(node_id)
                        if source_node:
                            source_rate = self.rates.get(source_node.rate, self.sample_rate)
                            if source_rate != self.sample_rate:
                                buffer = self._linear_resample(buffer, source_rate, self.sample_rate)

                        # Take only hop samples
                        buffer = buffer[:hop]
                        output_buffers[output_name].append(buffer)

            current_sample += hop

        # Concatenate output buffers
        final_outputs = {}
        for output_name, buffers in output_buffers.items():
            if buffers:
                final_outputs[output_name] = np.concatenate(buffers)
            else:
                final_outputs[output_name] = np.zeros(duration_samples)

        return final_outputs

    def get_info(self) -> Dict[str, Any]:
        """
        Get scheduler information

        Returns:
            Dictionary with scheduler configuration
        """
        return {
            "sample_rate": self.sample_rate,
            "hop_size": self.hop_size,
            "master_rate": self.master_rate,
            "active_rates": self.active_rates,
            "rate_groups": [
                {
                    "rate": rg.rate,
                    "rate_hz": rg.rate_hz,
                    "multiplier": rg.multiplier,
                    "num_operators": len(rg.operators),
                    "operators": rg.operators,
                }
                for rg in self.rate_groups
            ],
        }
