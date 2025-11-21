"""Acoustics operations implementation using NumPy backend.

This module provides NumPy-based implementations of acoustic wave propagation,
resonance, and impedance modeling for musical instruments, architectural acoustics,
exhaust systems, and speaker design.

Phase 1 focuses on 1D waveguide acoustics (digital waveguides for pipes, tubes, strings).
"""

from typing import Callable, Optional, Dict, Any, Tuple, Union, List
import numpy as np
from dataclasses import dataclass

from morphogen.core.operator import operator, OpCategory


# Physical constants
SPEED_OF_SOUND = 343.0  # m/s at 20°C
AIR_DENSITY = 1.2  # kg/m³ at 20°C
DEFAULT_SAMPLE_RATE = 44100  # Hz


@dataclass
class PipeGeometry:
    """Represents a simple pipe geometry for acoustic modeling.

    Attributes:
        diameter: Pipe diameter in meters
        length: Pipe length in meters
        segments: List of (position, diameter) tuples for variable diameter pipes
    """
    diameter: float  # meters
    length: float  # meters
    segments: Optional[List[Tuple[float, float]]] = None  # [(position, diameter), ...]

    def __repr__(self) -> str:
        if self.segments:
            return f"PipeGeometry(length={self.length}m, {len(self.segments)} segments)"
        return f"PipeGeometry(diameter={self.diameter}m, length={self.length}m)"


@dataclass
class WaveguideNetwork:
    """Digital waveguide network for 1D acoustic propagation.

    Represents discretized acoustic system using digital waveguide algorithm.

    Attributes:
        num_segments: Number of spatial segments
        segment_length: Physical length of each segment (meters)
        diameters: Array of diameters at each segment (meters)
        sample_rate: Simulation sample rate (Hz)
        speed_of_sound: Wave propagation speed (m/s)
    """
    num_segments: int
    segment_length: float
    diameters: np.ndarray
    sample_rate: int
    speed_of_sound: float = SPEED_OF_SOUND

    @property
    def total_length(self) -> float:
        """Total physical length in meters."""
        return self.num_segments * self.segment_length

    @property
    def delay_samples(self) -> int:
        """Round-trip delay in samples."""
        return int(2 * self.total_length / self.speed_of_sound * self.sample_rate)

    def __repr__(self) -> str:
        return (f"WaveguideNetwork({self.num_segments} segments, "
                f"{self.total_length:.3f}m, {self.sample_rate}Hz)")


@dataclass
class ReflectionCoeff:
    """Reflection coefficient at acoustic discontinuity.

    Range: -1.0 (open end, pressure node) to +1.0 (closed end, pressure antinode)

    Attributes:
        position: Position along waveguide (segment index)
        coefficient: Reflection coefficient (-1.0 to +1.0)
    """
    position: int
    coefficient: float

    def __repr__(self) -> str:
        return f"ReflectionCoeff(pos={self.position}, R={self.coefficient:.3f})"


@dataclass
class FrequencyResponse:
    """Frequency response of acoustic system.

    Attributes:
        frequencies: Array of frequencies (Hz)
        magnitude: Array of magnitudes (dB)
        phase: Array of phases (radians)
    """
    frequencies: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray

    def __repr__(self) -> str:
        return f"FrequencyResponse({len(self.frequencies)} points, {self.frequencies[0]:.1f}-{self.frequencies[-1]:.1f} Hz)"


class AcousticsOperations:
    """Namespace for acoustics operations (accessed as 'acoustics' in DSL)."""

    # ========================================================================
    # WAVEGUIDE CONSTRUCTION (1D Acoustics)
    # ========================================================================

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.CONSTRUCT,
        signature="(geometry: PipeGeometry, discretization: float, sample_rate: int, speed_of_sound: float) -> WaveguideNetwork",
        deterministic=True,
        doc="Build digital waveguide from pipe geometry"
    )
    def waveguide_from_geometry(
        geometry: PipeGeometry,
        discretization: float = 0.01,  # segment length in meters
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speed_of_sound: float = SPEED_OF_SOUND
    ) -> WaveguideNetwork:
        """Build digital waveguide from pipe geometry.

        Args:
            geometry: Pipe geometry specification
            discretization: Segment length in meters (spatial resolution)
            sample_rate: Sample rate in Hz
            speed_of_sound: Wave speed in m/s (temperature-dependent)

        Returns:
            WaveguideNetwork ready for simulation

        Example:
            # Simple pipe
            pipe = PipeGeometry(diameter=0.025, length=1.0)  # 25mm x 1m
            wg = acoustics.waveguide_from_geometry(pipe, discretization=0.01)

            # Variable diameter pipe (expansion chamber)
            segments = [(0.0, 0.04), (0.3, 0.12), (0.7, 0.05)]  # inlet, belly, outlet
            chamber = PipeGeometry(diameter=0.04, length=1.0, segments=segments)
            wg = acoustics.waveguide_from_geometry(chamber)
        """
        # Calculate number of segments
        num_segments = int(np.ceil(geometry.length / discretization))
        segment_length = geometry.length / num_segments

        # Build diameter array
        if geometry.segments is None:
            # Uniform diameter
            diameters = np.full(num_segments, geometry.diameter, dtype=np.float32)
        else:
            # Variable diameter - interpolate
            positions = np.array([seg[0] for seg in geometry.segments])
            diams = np.array([seg[1] for seg in geometry.segments])

            # Create segment positions
            seg_positions = np.linspace(0, geometry.length, num_segments)

            # Interpolate diameters
            diameters = np.interp(seg_positions, positions, diams).astype(np.float32)

        return WaveguideNetwork(
            num_segments=num_segments,
            segment_length=segment_length,
            diameters=diameters,
            sample_rate=sample_rate,
            speed_of_sound=speed_of_sound
        )

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(waveguide: WaveguideNetwork, end_condition: str) -> List[ReflectionCoeff]",
        deterministic=True,
        doc="Compute reflection coefficients at area discontinuities"
    )
    def reflection_coefficients(
        waveguide: WaveguideNetwork,
        end_condition: str = "open"  # "open", "closed", or "matched"
    ) -> List[ReflectionCoeff]:
        """Compute reflection coefficients at area discontinuities.

        Reflection coefficient R = (Z2 - Z1) / (Z2 + Z1) where Z = impedance.
        For area change: R = (A1 - A2) / (A1 + A2)

        Args:
            waveguide: Waveguide network
            end_condition: Boundary condition at pipe end
                - "open": R ≈ -1.0 (pressure node)
                - "closed": R ≈ +1.0 (pressure antinode)
                - "matched": R = 0.0 (no reflection)

        Returns:
            List of reflection coefficients at discontinuities

        Example:
            reflections = acoustics.reflection_coefficients(wg, end_condition="open")
        """
        reflections = []

        # Internal reflections at area changes
        for i in range(len(waveguide.diameters) - 1):
            A1 = np.pi * (waveguide.diameters[i] / 2) ** 2
            A2 = np.pi * (waveguide.diameters[i + 1] / 2) ** 2

            # Only add reflection if area change is significant (> 5%)
            if abs(A1 - A2) / A1 > 0.05:
                R = (A1 - A2) / (A1 + A2)
                reflections.append(ReflectionCoeff(position=i, coefficient=R))

        # End boundary condition
        if end_condition == "open":
            R_end = -0.95  # Slightly less than -1.0 for numerical stability
        elif end_condition == "closed":
            R_end = 0.95  # Slightly less than +1.0
        elif end_condition == "matched":
            R_end = 0.0
        else:
            raise ValueError(f"Unknown end condition: {end_condition}")

        reflections.append(
            ReflectionCoeff(position=waveguide.num_segments - 1, coefficient=R_end)
        )

        return reflections

    # ========================================================================
    # WAVEGUIDE PROPAGATION
    # ========================================================================

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.TRANSFORM,
        signature="(pressure_forward: ndarray, pressure_backward: ndarray, waveguide: WaveguideNetwork, reflections: List[ReflectionCoeff], excitation: Optional[ndarray], excitation_pos: int) -> Tuple[ndarray, ndarray]",
        deterministic=True,
        doc="Single time step of waveguide simulation"
    )
    def waveguide_step(
        pressure_forward: np.ndarray,
        pressure_backward: np.ndarray,
        waveguide: WaveguideNetwork,
        reflections: List[ReflectionCoeff],
        excitation: Optional[np.ndarray] = None,
        excitation_pos: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single time step of waveguide simulation.

        Digital waveguide algorithm: waves travel bidirectionally, reflecting
        at impedance discontinuities.

        Args:
            pressure_forward: Right-traveling wave (Pa)
            pressure_backward: Left-traveling wave (Pa)
            waveguide: Waveguide network
            reflections: List of reflection coefficients
            excitation: Optional external excitation signal
            excitation_pos: Position to inject excitation (segment index)

        Returns:
            (pressure_forward', pressure_backward') for next time step

        Example:
            # Initialize
            p_fwd = np.zeros(wg.num_segments)
            p_bwd = np.zeros(wg.num_segments)

            # Simulate
            for t in range(num_steps):
                excitation = impulse[t] if t < len(impulse) else 0.0
                p_fwd, p_bwd = acoustics.waveguide_step(
                    p_fwd, p_bwd, wg, reflections,
                    excitation=np.array([excitation])
                )
        """
        # Create output arrays
        p_fwd_new = np.zeros_like(pressure_forward)
        p_bwd_new = np.zeros_like(pressure_backward)

        # Propagate waves (shift by one sample delay)
        # Forward wave travels right
        p_fwd_new[1:] = pressure_forward[:-1]
        # Backward wave travels left
        p_bwd_new[:-1] = pressure_backward[1:]

        # Apply reflections at discontinuities
        for refl in reflections:
            pos = refl.position
            R = refl.coefficient

            if pos < len(pressure_forward) - 1:
                # Reflection: part bounces back, part transmits
                incident_fwd = pressure_forward[pos]
                incident_bwd = pressure_backward[pos + 1] if pos + 1 < len(pressure_backward) else 0.0

                # Reflected waves
                p_bwd_new[pos] += R * incident_fwd
                p_fwd_new[pos + 1] += R * incident_bwd

                # Transmitted waves
                p_fwd_new[pos + 1] += (1 - abs(R)) * incident_fwd
                p_bwd_new[pos] += (1 - abs(R)) * incident_bwd

        # Add excitation if provided
        if excitation is not None and len(excitation) > 0:
            if excitation_pos < len(p_fwd_new):
                p_fwd_new[excitation_pos] += excitation[0]

        return p_fwd_new, p_bwd_new

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(pressure_forward: ndarray, pressure_backward: ndarray) -> ndarray",
        deterministic=True,
        doc="Compute total pressure from bidirectional waves"
    )
    def total_pressure(
        pressure_forward: np.ndarray,
        pressure_backward: np.ndarray
    ) -> np.ndarray:
        """Compute total pressure from bidirectional waves.

        Total pressure = forward wave + backward wave

        Args:
            pressure_forward: Right-traveling wave
            pressure_backward: Left-traveling wave

        Returns:
            Total pressure at each position
        """
        return pressure_forward + pressure_backward

    # ========================================================================
    # HELMHOLTZ RESONATORS
    # ========================================================================

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(volume: float, neck_length: float, neck_area: float, speed_of_sound: float) -> float",
        deterministic=True,
        doc="Compute resonant frequency of Helmholtz resonator"
    )
    def helmholtz_frequency(
        volume: float,  # m³
        neck_length: float,  # m
        neck_area: float,  # m²
        speed_of_sound: float = SPEED_OF_SOUND
    ) -> float:
        """Compute resonant frequency of Helmholtz resonator.

        Classic formula: f = (c / 2π) * sqrt(A / (V * L))

        Args:
            volume: Resonator volume (m³)
            neck_length: Neck length (m)
            neck_area: Neck cross-sectional area (m²)
            speed_of_sound: Wave speed (m/s)

        Returns:
            Resonant frequency in Hz

        Example:
            # Quarter-wave resonator for muffler
            f_res = acoustics.helmholtz_frequency(
                volume=500e-6,  # 500 cm³
                neck_length=0.05,  # 50 mm
                neck_area=20e-4  # 20 cm²
            )
            # f_res ≈ 150 Hz
        """
        f_res = (speed_of_sound / (2 * np.pi)) * np.sqrt(neck_area / (volume * neck_length))
        return f_res

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(frequency: float, volume: float, neck_length: float, neck_area: float, damping: float, speed_of_sound: float, air_density: float) -> complex",
        deterministic=True,
        doc="Compute acoustic impedance of Helmholtz resonator"
    )
    def helmholtz_impedance(
        frequency: float,  # Hz
        volume: float,  # m³
        neck_length: float,  # m
        neck_area: float,  # m²
        damping: float = 0.0,  # 0.0 to 1.0
        speed_of_sound: float = SPEED_OF_SOUND,
        air_density: float = AIR_DENSITY
    ) -> complex:
        """Compute acoustic impedance of Helmholtz resonator.

        Z = R + j*X where:
        - R = resistance (damping)
        - X = reactance (mass + compliance)

        Args:
            frequency: Frequency in Hz
            volume: Resonator volume (m³)
            neck_length: Neck length (m)
            neck_area: Neck cross-sectional area (m²)
            damping: Damping factor (0.0 = lossless, 1.0 = critically damped)
            speed_of_sound: Wave speed (m/s)
            air_density: Air density (kg/m³)

        Returns:
            Complex impedance (Pa·s/m³)
        """
        omega = 2 * np.pi * frequency

        # Acoustic mass (inertance)
        M_a = air_density * neck_length / neck_area

        # Acoustic compliance
        C_a = volume / (air_density * speed_of_sound ** 2)

        # Acoustic resistance (damping)
        R_a = damping * np.sqrt(M_a / C_a)

        # Impedance
        Z = R_a + 1j * (omega * M_a - 1 / (omega * C_a))

        return Z

    # ========================================================================
    # RADIATION IMPEDANCE
    # ========================================================================

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(diameter: float, frequency: float, speed_of_sound: float, air_density: float) -> complex",
        deterministic=True,
        doc="Radiation impedance for unflanged circular pipe"
    )
    def radiation_impedance_unflanged(
        diameter: float,  # meters
        frequency: float,  # Hz
        speed_of_sound: float = SPEED_OF_SOUND,
        air_density: float = AIR_DENSITY
    ) -> complex:
        """Radiation impedance for unflanged circular pipe.

        Low-frequency approximation using Rayleigh's formula.

        Args:
            diameter: Pipe diameter (m)
            frequency: Frequency (Hz)
            speed_of_sound: Wave speed (m/s)
            air_density: Air density (kg/m³)

        Returns:
            Complex radiation impedance (Pa·s/m³)
        """
        k = 2 * np.pi * frequency / speed_of_sound  # wavenumber
        a = diameter / 2  # radius

        # Characteristic impedance
        Z0 = air_density * speed_of_sound / (np.pi * a ** 2)

        # Radiation impedance (Rayleigh formula)
        # Real part: radiation resistance
        # Imaginary part: radiation reactance
        ka = k * a

        if ka < 2.0:  # Low frequency approximation
            R_rad = Z0 * (ka) ** 2 / 2
            X_rad = Z0 * (8 * ka) / (3 * np.pi)
        else:  # High frequency
            R_rad = Z0
            X_rad = 0.0

        return R_rad + 1j * X_rad

    # ========================================================================
    # TRANSFER FUNCTIONS & FREQUENCY ANALYSIS
    # ========================================================================

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(waveguide: WaveguideNetwork, reflections: List[ReflectionCoeff], freq_range: Tuple[float, float], resolution: float, excitation_pos: int, measurement_pos: int) -> FrequencyResponse",
        deterministic=True,
        doc="Compute acoustic transfer function (input → output)"
    )
    def transfer_function(
        waveguide: WaveguideNetwork,
        reflections: List[ReflectionCoeff],
        freq_range: Tuple[float, float] = (20.0, 5000.0),
        resolution: float = 10.0,
        excitation_pos: int = 0,
        measurement_pos: int = -1
    ) -> FrequencyResponse:
        """Compute acoustic transfer function (input → output).

        Sweeps frequencies and measures amplitude/phase response.

        Args:
            waveguide: Waveguide network
            reflections: Reflection coefficients
            freq_range: (min_freq, max_freq) in Hz
            resolution: Frequency resolution in Hz
            excitation_pos: Input position (segment index)
            measurement_pos: Output position (segment index, -1 = end)

        Returns:
            FrequencyResponse with magnitude (dB) and phase (rad)

        Example:
            response = acoustics.transfer_function(
                wg, reflections,
                freq_range=(50, 5000),
                resolution=10
            )
        """
        # Generate frequency points
        freqs = np.arange(freq_range[0], freq_range[1], resolution)
        magnitudes = np.zeros(len(freqs))
        phases = np.zeros(len(freqs))

        measurement_pos = measurement_pos if measurement_pos >= 0 else waveguide.num_segments - 1

        # For each frequency, inject sine wave and measure steady-state response
        for i, freq in enumerate(freqs):
            # Number of samples for a few cycles
            duration = 5.0 / freq  # 5 cycles
            num_samples = int(duration * waveguide.sample_rate)

            # Initialize waveguide state
            p_fwd = np.zeros(waveguide.num_segments, dtype=np.float32)
            p_bwd = np.zeros(waveguide.num_segments, dtype=np.float32)

            # Generate excitation
            t = np.arange(num_samples) / waveguide.sample_rate
            excitation_signal = np.sin(2 * np.pi * freq * t)

            # Storage for output
            output = np.zeros(num_samples)

            # Simulate
            for sample_idx in range(num_samples):
                exc = np.array([excitation_signal[sample_idx]])
                p_fwd, p_bwd = AcousticsOperations.waveguide_step(
                    p_fwd, p_bwd, waveguide, reflections,
                    excitation=exc, excitation_pos=excitation_pos
                )

                # Measure total pressure at output
                p_total = AcousticsOperations.total_pressure(p_fwd, p_bwd)
                output[sample_idx] = p_total[measurement_pos]

            # Analyze steady-state response (last 2 cycles)
            steady_start = num_samples - int(2.0 / freq * waveguide.sample_rate)
            steady_output = output[steady_start:]
            steady_input = excitation_signal[steady_start:]

            # Compute magnitude and phase via FFT
            fft_output = np.fft.rfft(steady_output)
            fft_input = np.fft.rfft(steady_input)

            # Find peak frequency bin
            freqs_fft = np.fft.rfftfreq(len(steady_output), 1 / waveguide.sample_rate)
            peak_idx = np.argmin(np.abs(freqs_fft - freq))

            # Transfer function H(f) = Output(f) / Input(f)
            H = fft_output[peak_idx] / (fft_input[peak_idx] + 1e-10)

            magnitudes[i] = 20 * np.log10(np.abs(H) + 1e-10)  # dB
            phases[i] = np.angle(H)  # radians

        return FrequencyResponse(
            frequencies=freqs,
            magnitude=magnitudes,
            phase=phases
        )

    @staticmethod
    @operator(
        domain="acoustics",
        category=OpCategory.QUERY,
        signature="(frequency_response: FrequencyResponse, threshold_db: float) -> ndarray",
        deterministic=True,
        doc="Find resonant frequencies (peaks in transfer function)"
    )
    def resonant_frequencies(
        frequency_response: FrequencyResponse,
        threshold_db: float = -3.0
    ) -> np.ndarray:
        """Find resonant frequencies (peaks in transfer function).

        Args:
            frequency_response: Frequency response from transfer_function()
            threshold_db: Magnitude threshold for peak detection (dB)

        Returns:
            Array of resonant frequencies (Hz)

        Example:
            response = acoustics.transfer_function(wg, reflections)
            resonances = acoustics.resonant_frequencies(response, threshold_db=-3.0)
        """
        # Find peaks above threshold
        mag = frequency_response.magnitude
        freqs = frequency_response.frequencies

        # Simple peak detection: local maxima above threshold
        # Handle both sharp peaks and flat plateaus
        peaks = []
        for i in range(1, len(mag) - 1):
            if mag[i] > threshold_db and mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]:
                # Only add if it's strictly greater than previous (avoid duplicates on plateaus)
                if mag[i] > mag[i - 1] or (mag[i] == mag[i - 1] and mag[i] > mag[i + 1]):
                    peaks.append(freqs[i])

        return np.array(peaks)


# Create singleton instance for DSL access
acoustics = AcousticsOperations()


# Helper function for creating simple geometries
def create_pipe(diameter: float, length: float) -> PipeGeometry:
    """Create simple uniform pipe geometry.

    Args:
        diameter: Pipe diameter in meters
        length: Pipe length in meters

    Returns:
        PipeGeometry instance
    """
    return PipeGeometry(diameter=diameter, length=length)


def create_expansion_chamber(
    inlet_diameter: float,
    belly_diameter: float,
    outlet_diameter: float,
    total_length: float,
    belly_position: float = 0.5
) -> PipeGeometry:
    """Create expansion chamber geometry (for mufflers, etc).

    Args:
        inlet_diameter: Inlet diameter (m)
        belly_diameter: Belly (widest) diameter (m)
        outlet_diameter: Outlet diameter (m)
        total_length: Total length (m)
        belly_position: Belly position as fraction of length (0.0-1.0)

    Returns:
        PipeGeometry with variable diameter
    """
    segments = [
        (0.0, inlet_diameter),
        (belly_position * total_length, belly_diameter),
        (total_length, outlet_diameter)
    ]
    return PipeGeometry(diameter=inlet_diameter, length=total_length, segments=segments)


# Export operators for domain registry discovery
waveguide_from_geometry = AcousticsOperations.waveguide_from_geometry
reflection_coefficients = AcousticsOperations.reflection_coefficients
waveguide_step = AcousticsOperations.waveguide_step
total_pressure = AcousticsOperations.total_pressure
helmholtz_frequency = AcousticsOperations.helmholtz_frequency
helmholtz_impedance = AcousticsOperations.helmholtz_impedance
radiation_impedance_unflanged = AcousticsOperations.radiation_impedance_unflanged
transfer_function = AcousticsOperations.transfer_function
resonant_frequencies = AcousticsOperations.resonant_frequencies
