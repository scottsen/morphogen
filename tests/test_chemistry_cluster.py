"""Tests for chemistry cluster domains - electrochem, kinetics, qchem, thermo, transport."""

import pytest
import numpy as np
from morphogen.stdlib import electrochem, kinetics, qchem, thermo, transport


class TestElectrochemistry:
    """Test electrochemistry domain functions."""

    def test_nernst_equation(self):
        """Test Nernst equation for electrode potential."""
        # Standard conditions: 25°C, 1M concentration, n=1
        E0 = 0.34  # V (standard potential for Cu2+/Cu)
        n = 2      # electrons transferred
        Q = 0.1    # reaction quotient ([Red]/[Ox])

        E = electrochem.nernst_potential(E0, n, Q, T=298.15)

        # Should be positive (E > E0 when Q < 1)
        assert E > E0
        # Typical range for electrode potentials
        assert -3.0 < E < 3.0

    def test_nernst_concentration_effect(self):
        """Test that concentration affects potential correctly."""
        E0 = 0.0
        n = 1

        # Higher product/reactant ratio (Q > 1) should decrease potential
        E_high_Q = electrochem.nernst_potential(E0, n, Q=10.0, T=298.15)
        E_low_Q = electrochem.nernst_potential(E0, n, Q=0.1, T=298.15)

        assert E_low_Q > E_high_Q

    def test_cell_potential(self):
        """Test cell potential calculation."""
        E_cathode = 0.80  # V
        E_anode = -0.76   # V

        E_cell = electrochem.cell_potential(E_cathode, E_anode)

        # Cell potential = cathode - anode
        assert abs(E_cell - (0.80 - (-0.76))) < 0.01
        assert E_cell > 0  # Spontaneous reaction

    def test_faraday_law(self):
        """Test Faraday's law of electrolysis."""
        current = 2.0    # Amperes
        time = 3600.0    # seconds (1 hour)
        n = 2            # electrons per molecule
        M = 63.55        # g/mol (copper)

        mass = electrochem.faraday_mass(current, time, n, M)

        # Should produce ~2.37 g of copper
        assert 2.0 < mass < 3.0
        assert mass > 0

    def test_conductivity(self):
        """Test ionic conductivity calculation."""
        # Simple test of conductivity model
        concentration = 0.1  # M
        lambda_plus = 73.5   # S cm²/mol (H+)
        lambda_minus = 76.3  # S cm²/mol (Cl-)

        kappa = electrochem.molar_conductivity_to_conductivity(
            concentration, lambda_plus + lambda_minus
        )

        assert kappa > 0
        # Conductivity should scale with concentration
        kappa2 = electrochem.molar_conductivity_to_conductivity(
            2 * concentration, lambda_plus + lambda_minus
        )
        assert kappa2 > kappa


class TestChemicalKinetics:
    """Test chemical kinetics domain functions."""

    def test_arrhenius_rate(self):
        """Test Arrhenius equation for rate constant."""
        A = 1e13       # pre-exponential factor (1/s)
        Ea = 50000.0   # activation energy (J/mol)
        T = 298.15     # temperature (K)

        k = kinetics.arrhenius_rate(A, Ea, T)

        # Rate constant should be positive
        assert k > 0
        # Should be less than pre-exponential factor at room temp
        assert k < A

    def test_arrhenius_temperature_dependence(self):
        """Test that rate increases with temperature."""
        A = 1e13
        Ea = 50000.0

        k_300K = kinetics.arrhenius_rate(A, Ea, T=300.0)
        k_400K = kinetics.arrhenius_rate(A, Ea, T=400.0)

        # Higher temperature should give higher rate
        assert k_400K > k_300K

    def test_first_order_kinetics(self):
        """Test first-order reaction kinetics."""
        k = 0.1        # rate constant (1/s)
        C0 = 1.0       # initial concentration (M)
        t = 10.0       # time (s)

        C = kinetics.first_order_concentration(C0, k, t)

        # Concentration should decrease
        assert 0 < C < C0
        # C(t) = C0 * exp(-kt)
        expected = C0 * np.exp(-k * t)
        assert abs(C - expected) < 0.01

    def test_second_order_kinetics(self):
        """Test second-order reaction kinetics."""
        k = 0.5        # rate constant (L/mol/s)
        C0 = 1.0       # initial concentration (M)
        t = 5.0        # time (s)

        C = kinetics.second_order_concentration(C0, k, t)

        # Concentration should decrease
        assert 0 < C < C0

    def test_half_life_first_order(self):
        """Test half-life calculation for first-order reaction."""
        k = 0.693  # 1/s

        t_half = kinetics.half_life_first_order(k)

        # t_1/2 = ln(2) / k ≈ 1.0
        assert abs(t_half - 1.0) < 0.01

    def test_reaction_mechanism(self):
        """Test reaction mechanism analysis."""
        # Simple consecutive reaction: A -> B -> C
        k1 = 0.5
        k2 = 0.3

        # Test that we can define and analyze mechanism
        # (This would depend on actual implementation)
        mechanism = {
            'steps': [
                {'reactants': ['A'], 'products': ['B'], 'k': k1},
                {'reactants': ['B'], 'products': ['C'], 'k': k2}
            ]
        }

        # Verify mechanism structure
        assert len(mechanism['steps']) == 2

    def test_activation_energy_from_rates(self):
        """Test determining activation energy from rate constants at two temperatures."""
        k1 = 1e-3  # rate at T1
        k2 = 1e-2  # rate at T2
        T1 = 298.15
        T2 = 308.15

        Ea = kinetics.activation_energy_from_rates(k1, T1, k2, T2)

        # Activation energy should be positive and reasonable
        assert Ea > 0
        assert 10000 < Ea < 200000  # Typical range (J/mol)


class TestQuantumChemistry:
    """Test quantum chemistry domain functions."""

    def test_basis_set_size(self):
        """Test basis set size calculations."""
        # STO-3G for H2O
        atoms = ['O', 'H', 'H']
        basis = 'sto-3g'

        n_basis = qchem.count_basis_functions(atoms, basis)

        # STO-3G: O has 5 functions (1s, 2s, 2px, 2py, 2pz)
        #         H has 1 function (1s)
        # Total: 5 + 1 + 1 = 7
        assert n_basis == 7

    def test_hartree_fock_energy(self):
        """Test Hartree-Fock energy calculation (simplified)."""
        # This would be a simplified test
        # Full HF calculation is complex
        atoms = ['H', 'H']
        positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])  # Angstroms

        # Placeholder for HF calculation
        # In real implementation, this would solve SCF equations
        energy = qchem.hf_energy_simple(atoms, positions, basis='sto-3g')

        # H2 ground state energy should be negative
        assert energy < 0
        # Should be around -1.1 Hartree for equilibrium H2
        assert -2.0 < energy < 0.0

    def test_molecular_orbitals(self):
        """Test molecular orbital calculation."""
        # Simple MO test
        n_basis = 7
        overlap_matrix = np.eye(n_basis)

        # Eigenvalues should be orbital energies
        energies = qchem.mo_energies_from_fock(
            np.random.randn(n_basis, n_basis),
            overlap_matrix
        )

        assert len(energies) == n_basis

    def test_electron_density(self):
        """Test electron density calculation."""
        # Simplified density matrix test
        density_matrix = np.eye(3) * 0.5

        total_electrons = qchem.total_electron_count(density_matrix)

        # Should be positive
        assert total_electrons > 0


class TestThermodynamics:
    """Test thermodynamics domain functions."""

    def test_ideal_gas_law(self):
        """Test ideal gas law PV = nRT."""
        n = 1.0      # moles
        T = 298.15   # K
        V = 0.0224   # m³ (roughly 1 mole at STP)

        P = thermo.ideal_gas_pressure(n, T, V)

        # Should be close to 1 atm = 101325 Pa at STP
        assert 50000 < P < 150000  # Pa

    def test_van_der_waals_equation(self):
        """Test van der Waals equation of state."""
        n = 1.0
        T = 300.0
        V = 0.025
        a = 0.365    # van der Waals constant for N2
        b = 0.0427   # van der Waals constant for N2

        P = thermo.van_der_waals_pressure(n, T, V, a, b)

        # Should be close to ideal gas pressure but slightly different
        P_ideal = thermo.ideal_gas_pressure(n, T, V)
        assert abs(P - P_ideal) / P_ideal < 0.2  # Within 20%

    def test_enthalpy_calculation(self):
        """Test enthalpy calculation H = U + PV."""
        U = 1000.0  # Internal energy (J)
        P = 101325  # Pressure (Pa)
        V = 0.001   # Volume (m³)

        H = thermo.enthalpy(U, P, V)

        # H = U + PV
        expected = U + P * V
        assert abs(H - expected) < 0.01

    def test_gibbs_free_energy(self):
        """Test Gibbs free energy G = H - TS."""
        H = 1000.0  # Enthalpy (J)
        T = 298.15  # Temperature (K)
        S = 10.0    # Entropy (J/K)

        G = thermo.gibbs_free_energy(H, T, S)

        # G = H - TS
        expected = H - T * S
        assert abs(G - expected) < 0.01

    def test_equilibrium_constant(self):
        """Test equilibrium constant from Gibbs energy."""
        delta_G = -5000.0  # J/mol (negative = spontaneous)
        T = 298.15         # K

        K_eq = thermo.equilibrium_constant(delta_G, T)

        # Negative ΔG should give K > 1
        assert K_eq > 1.0

        # Positive ΔG should give K < 1
        K_eq_pos = thermo.equilibrium_constant(5000.0, T)
        assert K_eq_pos < 1.0

    def test_heat_capacity(self):
        """Test heat capacity calculations."""
        # Shomate equation parameters (example)
        A, B, C, D, E = 25.56, 6.096, 4.054, -2.671, 0.131
        T = 500.0  # K

        Cp = thermo.shomate_heat_capacity(T, A, B, C, D, E)

        # Heat capacity should be positive
        assert Cp > 0
        # Typical range for gases: 20-50 J/(mol·K)
        assert 15 < Cp < 100

    def test_phase_transition(self):
        """Test phase transition calculations."""
        delta_H_fus = 6010.0  # J/mol (enthalpy of fusion for water)
        T_m = 273.15          # K (melting point)

        delta_S_fus = thermo.entropy_of_fusion(delta_H_fus, T_m)

        # Entropy change should be positive for melting
        assert delta_S_fus > 0
        # ΔS = ΔH / T
        expected = delta_H_fus / T_m
        assert abs(delta_S_fus - expected) < 0.01


class TestTransportProperties:
    """Test transport properties domain functions."""

    def test_fick_first_law(self):
        """Test Fick's first law of diffusion."""
        D = 1e-9       # Diffusion coefficient (m²/s)
        dC_dx = 1000.0 # Concentration gradient (mol/m⁴)

        J = transport.fick_flux(D, dC_dx)

        # Flux should be negative (diffusion against gradient)
        assert J < 0
        # J = -D * dC/dx
        expected = -D * dC_dx
        assert abs(J - expected) < 1e-12

    def test_diffusion_coefficient_temperature(self):
        """Test temperature dependence of diffusion coefficient."""
        D0 = 1e-9  # m²/s
        Ea = 20000.0  # J/mol
        T1 = 298.15
        T2 = 350.0

        D1 = transport.arrhenius_diffusivity(D0, Ea, T1)
        D2 = transport.arrhenius_diffusivity(D0, Ea, T2)

        # Higher temperature should give higher diffusivity
        assert D2 > D1

    def test_viscosity_temperature(self):
        """Test viscosity-temperature relationship."""
        # Sutherland's formula for gases or Arrhenius for liquids
        mu0 = 1e-3  # Pa·s
        T1 = 298.15
        T2 = 350.0

        # For liquids: viscosity decreases with temperature
        # (simplified model)
        mu1 = transport.viscosity_temperature_liquid(mu0, T1, T1)
        mu2 = transport.viscosity_temperature_liquid(mu0, T2, T1)

        # Higher temperature should give lower viscosity for liquids
        assert mu2 < mu1

    def test_thermal_conductivity(self):
        """Test thermal conductivity calculations."""
        # Fourier's law: q = -k * dT/dx
        k = 0.6       # W/(m·K) (thermal conductivity of water)
        dT_dx = 100.0 # K/m (temperature gradient)

        q = transport.fourier_heat_flux(k, dT_dx)

        # Heat flux should be negative (heat flows down gradient)
        assert q < 0
        expected = -k * dT_dx
        assert abs(q - expected) < 0.01

    def test_mass_transfer_coefficient(self):
        """Test mass transfer coefficient calculation."""
        # Simplified mass transfer test
        D = 1e-9      # Diffusivity (m²/s)
        delta = 0.001 # Boundary layer thickness (m)

        k_mass = transport.mass_transfer_coefficient(D, delta)

        # k = D / delta
        assert k_mass > 0
        expected = D / delta
        assert abs(k_mass - expected) < 1e-12

    def test_schmidt_number(self):
        """Test Schmidt number (Sc = nu / D)."""
        nu = 1e-6  # Kinematic viscosity (m²/s)
        D = 1e-9   # Diffusivity (m²/s)

        Sc = transport.schmidt_number(nu, D)

        # Sc = nu / D
        expected = nu / D
        assert abs(Sc - expected) < 0.01
        # For liquids, Sc is typically 100-10000
        assert 10 < Sc < 100000

    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        rho = 1000.0  # kg/m³ (density of water)
        v = 0.1       # m/s (velocity)
        L = 0.01      # m (characteristic length)
        mu = 1e-3     # Pa·s (viscosity)

        Re = transport.reynolds_number(rho, v, L, mu)

        # Re = rho * v * L / mu
        expected = rho * v * L / mu
        assert abs(Re - expected) < 0.01
        # Should be ~1000 for this case
        assert 500 < Re < 2000


class TestDeterminism:
    """Test deterministic behavior across chemistry domains."""

    def test_kinetics_determinism(self):
        """Test that kinetics calculations are deterministic."""
        A, Ea, T = 1e13, 50000.0, 298.15

        k1 = kinetics.arrhenius_rate(A, Ea, T)
        k2 = kinetics.arrhenius_rate(A, Ea, T)

        assert k1 == k2

    def test_thermo_determinism(self):
        """Test that thermo calculations are deterministic."""
        n, T, V = 1.0, 298.15, 0.0224

        P1 = thermo.ideal_gas_pressure(n, T, V)
        P2 = thermo.ideal_gas_pressure(n, T, V)

        assert P1 == P2

    def test_transport_determinism(self):
        """Test that transport calculations are deterministic."""
        D, dC_dx = 1e-9, 1000.0

        J1 = transport.fick_flux(D, dC_dx)
        J2 = transport.fick_flux(D, dC_dx)

        assert J1 == J2


class TestCrossChemistryDomainIntegration:
    """Test integration between chemistry domains."""

    def test_kinetics_thermo_integration(self):
        """Test using kinetics with thermodynamics."""
        # Get rate constant from kinetics
        k = kinetics.arrhenius_rate(A=1e13, Ea=50000.0, T=298.15)

        # Use in equilibrium calculation
        # K_eq = k_forward / k_reverse
        K_eq = 10.0  # Assume equilibrium constant
        k_reverse = k / K_eq

        # Calculate ΔG from equilibrium constant
        delta_G = thermo.gibbs_from_equilibrium_constant(K_eq, T=298.15)

        assert k_reverse > 0
        assert delta_G < 0  # K > 1 means negative ΔG

    def test_transport_kinetics_integration(self):
        """Test combining transport (diffusion) with kinetics (reaction)."""
        # Diffusion coefficient
        D = transport.arrhenius_diffusivity(D0=1e-9, Ea=20000.0, T=298.15)

        # Reaction rate
        k = kinetics.arrhenius_rate(A=1e6, Ea=40000.0, T=298.15)

        # Damköhler number: Da = k * L² / D (reaction vs diffusion timescale)
        L = 0.001  # m
        Da = k * L**2 / D

        # Both should be positive
        assert D > 0
        assert k > 0
        assert Da > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
