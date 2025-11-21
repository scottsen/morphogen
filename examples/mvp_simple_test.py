"""
Simple Python test of MVP functionality.
Tests field operations and visualization directly without the DSL parser.
"""

from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual

# Create random field (simulates initial heat distribution)
temp = field.random((128, 128), seed=42, low=0.0, high=1.0)
print(f"Created field: {temp}")

# Apply diffusion to smooth it out
temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=10)
print(f"After diffusion: {temp}")

# Apply boundary conditions
temp = field.boundary(temp, spec="reflect")
print(f"After boundary: {temp}")

# Visualize result
vis = visual.colorize(temp, palette="fire")
print(f"Created visual: {vis}")

visual.output(vis, path="mvp_test_output.png")
print("Test completed successfully!")
