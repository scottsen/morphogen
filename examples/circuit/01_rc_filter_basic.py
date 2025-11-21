"""Basic RC Low-Pass Filter Example

Demonstrates Phase 1 circuit domain capabilities:
- Creating circuits and adding components
- DC analysis using Modified Nodal Analysis
- AC analysis for frequency response
- Transient analysis for time-domain simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from morphogen.stdlib.circuit import (
    Circuit, CircuitOperations as circuit
)

def main():
    print("=" * 60)
    print("RC Low-Pass Filter Circuit Example")
    print("=" * 60)

    # Create circuit (3 nodes: input, output, ground)
    c = circuit.create(num_nodes=3, dt=1e-6)

    # Add components
    # R1: input (node 1) to output (node 2), 1kΩ
    circuit.add_resistor(c, node1=1, node2=2, resistance=1000.0, name="R1")

    # C1: output (node 2) to ground (node 0), 100nF
    circuit.add_capacitor(c, node1=2, node2=0, capacitance=100e-9, name="C1")

    # Voltage source: input (node 1) to ground (node 0), 5V
    circuit.add_voltage_source(c, node_pos=1, node_neg=0, voltage=5.0, name="Vin")

    print(f"\nCircuit: {c}")
    print(f"Components: {len(c.components)}")

    # ========================================
    # DC Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("DC Analysis")
    print("=" * 60)

    circuit.dc_analysis(c)

    print(f"Node voltages:")
    print(f"  Node 0 (ground): {c.node_voltages[0]:.4f} V")
    print(f"  Node 1 (input):  {c.node_voltages[1]:.4f} V")
    print(f"  Node 2 (output): {c.node_voltages[2]:.4f} V")

    print(f"\nBranch currents:")
    for name, current in c.branch_currents.items():
        print(f"  {name}: {current*1000:.4f} mA")

    # Calculate power
    power_r1 = circuit.get_power(c, "R1")
    print(f"\nPower dissipation:")
    print(f"  R1: {power_r1*1000:.4f} mW")

    # ========================================
    # AC Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("AC Analysis (Frequency Response)")
    print("=" * 60)

    # Frequency sweep: 10 Hz to 100 kHz
    frequencies = np.logspace(1, 5, 100)  # 10 Hz to 100 kHz
    results = circuit.ac_analysis(c, frequencies)

    # Calculate magnitude and phase at output node
    node_voltages = results['node_voltages']
    output_voltage = node_voltages[:, 2]  # Node 2
    input_voltage = node_voltages[:, 1]   # Node 1

    # Transfer function: Vout / Vin
    transfer_function = output_voltage / input_voltage

    # Magnitude (dB) and Phase (degrees)
    magnitude_db = 20 * np.log10(np.abs(transfer_function))
    phase_deg = np.angle(transfer_function) * 180 / np.pi

    # Find -3dB point (cutoff frequency)
    idx_3db = np.argmin(np.abs(magnitude_db - (-3.0)))
    fc_measured = frequencies[idx_3db]

    # Theoretical cutoff: fc = 1 / (2π * R * C)
    R = 1000.0  # Ω
    C = 100e-9  # F
    fc_theory = 1.0 / (2 * np.pi * R * C)

    print(f"Cutoff frequency (-3dB):")
    print(f"  Theoretical: {fc_theory:.2f} Hz")
    print(f"  Measured:    {fc_measured:.2f} Hz")
    print(f"  Error:       {abs(fc_measured - fc_theory) / fc_theory * 100:.2f}%")

    # ========================================
    # Transient Analysis
    # ========================================
    print("\n" + "=" * 60)
    print("Transient Analysis (Step Response)")
    print("=" * 60)

    # Simulate for 5 time constants
    time_constant = R * C
    duration = 5 * time_constant

    print(f"Time constant (τ = RC): {time_constant*1e6:.2f} μs")
    print(f"Simulation duration: {duration*1e6:.2f} μs")

    time_points, voltage_history = circuit.transient_analysis(c, duration=duration)

    # Final value should be ~5V (at 5τ, reaches ~99.3% of final value)
    final_voltage = voltage_history[-1, 2]
    expected_final = 5.0 * (1 - np.exp(-5))

    print(f"\nOutput voltage:")
    print(f"  Initial: {voltage_history[0, 2]:.4f} V")
    print(f"  Final:   {final_voltage:.4f} V")
    print(f"  Expected (5τ): {expected_final:.4f} V")
    print(f"  Error:   {abs(final_voltage - expected_final) / expected_final * 100:.2f}%")

    # ========================================
    # Plotting
    # ========================================
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("RC Low-Pass Filter Analysis", fontsize=16, fontweight='bold')

    # Plot 1: Magnitude response
    ax1 = axes[0, 0]
    ax1.semilogx(frequencies, magnitude_db, 'b-', linewidth=2)
    ax1.axhline(-3, color='r', linestyle='--', label='-3 dB')
    ax1.axvline(fc_measured, color='g', linestyle='--', label=f'fc = {fc_measured:.1f} Hz')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Magnitude Response')
    ax1.legend()

    # Plot 2: Phase response
    ax2 = axes[0, 1]
    ax2.semilogx(frequencies, phase_deg, 'r-', linewidth=2)
    ax2.axvline(fc_measured, color='g', linestyle='--', label=f'fc = {fc_measured:.1f} Hz')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_title('Phase Response')
    ax2.legend()

    # Plot 3: Step response
    ax3 = axes[1, 0]
    ax3.plot(time_points * 1e6, voltage_history[:, 2], 'b-', linewidth=2, label='Output')
    ax3.axhline(5.0, color='r', linestyle='--', alpha=0.5, label='Final value')
    ax3.axhline(5.0 * 0.632, color='g', linestyle='--', alpha=0.5, label='1τ (63.2%)')
    ax3.axvline(time_constant * 1e6, color='g', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title('Step Response')
    ax3.legend()

    # Plot 4: Circuit diagram (text-based)
    ax4 = axes[1, 1]
    ax4.axis('off')
    circuit_text = """
    Circuit Diagram:

         R1 (1kΩ)
    ●────/\\/\\/\\────●
    1              2
    +              |
    Vin (5V)       C1 (100nF)
    -              |
    ●──────────────●
    0 (ground)

    Components:
    • R1: 1 kΩ resistor
    • C1: 100 nF capacitor
    • Vin: 5 V voltage source

    Analysis Results:
    • DC Output: {:.3f} V
    • Cutoff (-3dB): {:.1f} Hz
    • Time Constant: {:.2f} μs
    """.format(c.node_voltages[2], fc_measured, time_constant * 1e6)

    ax4.text(0.1, 0.5, circuit_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()
    plt.savefig('rc_filter_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: rc_filter_analysis.png")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
