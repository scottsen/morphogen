"""Tests for v0.3.1 example programs."""

import pytest
from pathlib import Path
from morphogen.parser.parser import parse
from morphogen.runtime.runtime import Runtime


class TestV031Examples:
    """Test that v0.3.1 example programs parse and execute correctly."""

    def test_velocity_calculation_example(self):
        """Test velocity calculation example."""
        example_path = Path(__file__).parent.parent / "examples" / "v0_3_1_velocity_calculation.kairo"
        source = example_path.read_text()

        # Parse the program
        program = parse(source)
        assert program is not None

        # Execute the program
        runtime = Runtime()
        runtime.execute_program(program)

        # Verify state variables exist
        assert runtime.context.has_variable('total_distance')
        assert runtime.context.has_variable('time_elapsed')

        # Check that distance and time have been updated
        total_distance = runtime.context.get_variable('total_distance')
        time_elapsed = runtime.context.get_variable('time_elapsed')

        # After 10 steps: distance = 10 * 1.5 = 15, time = 10 * 0.1 = 1.0
        assert abs(total_distance - 15.0) < 0.001
        assert abs(time_elapsed - 1.0) < 0.001

    def test_lambdas_and_flow_example(self):
        """Test lambdas and flow example."""
        example_path = Path(__file__).parent.parent / "examples" / "v0_3_1_lambdas_and_flow.kairo"
        source = example_path.read_text()

        # Parse the program
        program = parse(source)
        assert program is not None

        # Execute the program
        runtime = Runtime()
        runtime.execute_program(program)

        # Verify state variables exist and have been updated
        assert runtime.context.has_variable('position')
        assert runtime.context.has_variable('velocity')

        position = runtime.context.get_variable('position')
        velocity = runtime.context.get_variable('velocity')

        # Position and velocity should have changed from initial values
        assert position != 0.0
        assert velocity != 0.0

    def test_recursive_factorial_example(self):
        """Test recursive factorial example."""
        example_path = Path(__file__).parent.parent / "examples" / "v0_3_1_recursive_factorial.kairo"
        source = example_path.read_text()

        # Parse the program
        program = parse(source)
        assert program is not None

        # Execute the program
        runtime = Runtime()
        runtime.execute_program(program)

        # Verify factorial and fibonacci results
        fact5 = runtime.context.get_variable('fact5')
        fib7 = runtime.context.get_variable('fib7')

        assert fact5 == 120.0
        assert fib7 == 13.0

        # Verify flow executed
        assert runtime.context.has_variable('counter')
        counter = runtime.context.get_variable('counter')
        assert counter == 6.0  # Started at 1, incremented 5 times

    def test_complete_demo_example(self):
        """Test complete demo example."""
        example_path = Path(__file__).parent.parent / "examples" / "v0_3_1_complete_demo.kairo"
        source = example_path.read_text()

        # Parse the program
        program = parse(source)
        assert program is not None

        # Execute the program
        runtime = Runtime()
        runtime.execute_program(program)

        # Verify all state variables exist
        assert runtime.context.has_variable('position')
        assert runtime.context.has_variable('velocity')
        assert runtime.context.has_variable('energy')

        position = runtime.context.get_variable('position')
        velocity = runtime.context.get_variable('velocity')
        energy = runtime.context.get_variable('energy')

        # All should have changed from initial values
        assert position != 0.0
        assert velocity != 5.0
        assert energy != 100.0

        # Position should be clamped between 0 and 10
        assert 0.0 <= position <= 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
