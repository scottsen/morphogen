# Platform Architecture: Layers & Interfaces

## The Core Principle
**Modules should be substrate-agnostic black boxes.**

A module can be:
- Analog circuitry (op-amps, transistors, filters)
- Digital DSP (ARM, FPGA, specialized ICs)
- Analog neural inference (crossbar arrays, memristors)
- Hybrid (analog + digital compute)
- Future: biological, optical, or other exotic substrates

**The interface makes them interchangeable.**

---

## The Seven Layers

### Layer 0: Physical Essence (de Forest)
**Substrate:** Electrons, transistors, tubes, resistors, capacitors, silicon

**In Your System:**
- Discrete analog electrical behavior
- Low-noise signal paths
- Transistor/op-amp "character"
- Physical continuous-time dynamics

**Modules:** Gain, clip, buffer, saturation, tone stacks

---

### Layer 1: Stability & Refinement (Black)
**Principle:** Negative feedback and linearization

**In Your System:**
- Controlled feedback loops
- Distortion control
- Gain structure
- Predictable behavior across modules

**Modules:** Feedback blocks, dynamic shapers, compression primitives

---

### Layer 2: Functional Analog Blocks (Philbrick)
**Principle:** Modular plug-in analog computation

**In Your System:**
- Standardized module sizes and pin counts
- Composable building blocks
- Routing between blocks
- Physical signal chain composition

**Critical:** This is your revolution - standardized analog function modules that can be rearranged at will.

---

### Layer 3: Digital Control & Logic (Shannon)
**Principle:** Information, switching, and discrete signals

**In Your System:**
- Module identification (I²C/SPI)
- Digital modulation and clocking
- Routing matrices
- Presets, memory, states
- Clean analog/digital separation

**Modules:** LFO controllers, sequencers, MIDI blocks, routing matrices

---

### Layer 4: Voltage-Controlled Expression (Moog)
**Principle:** Musical modularity and human interface

**In Your System:**
- Knobs, sliders, footswitches
- Expression pedals
- Performance macro controllers
- CV interaction
- MIDI/USB integration

---

### Layer 5: Routing & Orchestration (Turing)
**Principle:** Programmable composition and scheduling

**In Your System:**
- Motherboard routing
- Multi-module coordination
- Scene/preset memory
- Topology switching
- Real-time control changes

---

### Layer 6: Experience & Feel (Neve, Moog, Buchla)
**Principle:** Aesthetic experience and musical interface

**In Your System:**
- CNC enclosures
- Modular faceplates
- Tactile workflow
- Visual feedback
- Touring reliability

---

## Power Architecture

### The 48V Decision

**Why 48V at the top tier:**
1. **Massive analog headroom** - critical for clean analog computation
2. **Lower noise floor** (better SNR)
3. **Lower current draw** for same power (less heat, thinner traces)
4. **Industry standard** (telecom, studio phantom power)
5. **Safe** (below shock hazard ~50-60V)

**Three-Tier Power Distribution:**

```
Tier 1: 48V Backbone (power distribution)
  ↓
Tier 2: ±12V Analog Rails (on-module DC-DC conversion)
  ↓
Tier 3: +5V, +3.3V Digital Rails (on-module regulators)
```

**Key Insight:** Each module generates its own operating rails from 48V, preventing cross-module noise.

**Grounding Strategy:**
- AGND (analog ground)
- DGND (digital ground)
- Separate domains with local regulation
- Prevent digital switching noise from polluting analog

---

## Module Pin Standard

### The Great Pin Count Debate: 6, 8, or 10?

**Minimum Viable (4 pins):**
- Power
- Ground
- Audio In
- Audio Out

**Smart Analog (6 pins) - RECOMMENDED:**
1. AIN (Analog In)
2. AOUT (Analog Out)
3. V+ (Power, e.g., 48V or 12V)
4. GND (Ground)
5. CV/CTRL (Control voltage or digital control)
6. ID/OPT (Module identification or extra analog I/O)

**Why 6 is the sweet spot:**
- Simple, cheap, not intimidating
- Enough for analog + digital hybrid modules
- Supports both pure analog and DSP blocks
- Future-proof without being excessive
- Easy to manufacture (JST-PH connectors)

**Awesome Mixed-Signal (10 pins) - for advanced modules:**
1. AIN
2. AOUT
3. AVDD (Analog supply, e.g., +12V)
4. DVDD (Digital supply, e.g., +5V)
5. AGND (Analog ground)
6. DGND (Digital ground)
7. CV (Control voltage)
8. CTRL (Digital control - I²C SDA or GPIO)
9. CLK (I²C SCL or shared clock)
10. ID/RESET

**When to use 10 pins:**
- Analog-neural inference chips
- High-precision analog compute
- Modules needing true analog/digital isolation
- Future "pro" tier modules

---

## ADC/DAC Strategy

### Where Converters Live: **Inside Each Module**

**NOT on the main board.** Each module handles its own domain crossing.

**Why:**
- Modules are self-contained
- Optimal ADC/DAC selection per use case
- Simplest backplane
- Best signal isolation
- Least noise
- Future-proof

### Converter Types by Use Case

| Type | Speed | Resolution | Latency | Use Case |
|------|-------|-----------|---------|----------|
| SAR | 50kS-5MS/s | 12-24 bit | ~µs | Audio→DSP, CV, sensors |
| Delta-Sigma | 20-200kHz | 16-32 bit | 1-4ms | High-fidelity audio |
| Flash | GHz | 6-8 bit | ~ns | Chaos capture, neuromorphic |
| Pipeline | 10-200MS/s | 12-16 bit | ~10ns | DSP-grade audio, FFT |

**The Domain Crossing Contract:**

All modules MUST declare:
- `AD_latency`
- `DA_latency`
- `Internal_latency`
- `Bandwidth`
- `Precision`
- `Signal_ground_domain`

This prevents analog feedback loops from being broken by unexpected digital delays.

---

## Latency Protocol

### Automatic Latency Measurement & Propagation

**Core Idea:** Each module carries a "latency fingerprint" that propagates through the signal chain.

**Implementation:**
1. **Controller embeds timestamp in every packet**
   ```
   CMD + T_issue (global microseconds)
   ```

2. **Module measures internal processing time**
   - Simple: Host measures round-trip time
   - Advanced: Module reports `t_in_local`, `t_out_local`

3. **Controller computes latency:**
   ```
   module_latency = T_response - T_issue - bus_overhead
   ```

4. **Accumulate along chain:**
   ```
   Total_latency = Σ(module_latencies) + routing_overhead
   ```

**Benefits:**
- No distributed clock sync needed
- Works with analog-only, digital-only, and hybrid modules
- Controller knows full path delay
- Can block unsafe feedback loops
- Can warn about "laggy feel"

**Latency Classes:**
- Ultra-low: < 0.5 ms (pure analog, feedback-safe)
- Live: 0.5-5 ms (playable, real-time performance)
- High: 5-20 ms (acceptable for non-critical paths)
- Offline: > 20 ms (rendering, heavy processing)

---

## Module Self-Description Protocol

### Borrowed from USB Descriptors

Every module exposes:

```c
struct ModuleDescriptor {
  uint16 vendor_id;
  uint16 product_id;
  uint8 module_type;  // ANALOG, DSP, HYBRID, NEURAL
  uint16 firmware_version;
  uint8 num_modes;
  uint8 num_interfaces;
}

struct ModeInfo {
  uint8 mode_id;           // 0=eco, 1=live, 2=hq
  float latency_us;
  uint8 quality_level;
  float power_mw;
  uint8 flags;             // CAN_FEEDBACK, REALTIME_SAFE, etc.
}

struct ModuleTimingInfo {
  float latency_nominal_us;
  float latency_measured_us;
  uint8 confidence;
  uint8 latency_type;      // CONTINUOUS_ANALOG, DSP_BLOCK, etc.
  float sample_rate_hz;
}
```

**Discovery Process:**
1. Controller scans all module addresses on bus
2. Reads descriptors from known offset (I²C/SPI)
3. Builds routing graph and capability map
4. Optionally runs calibration to measure actual latency

**Mode Switching:**
- Controller can ask modules to switch modes based on global latency budget
- Example: "We're over budget → go to lower-latency 'live' mode"
- Modules adjust: smaller buffers, simpler algorithms, bypass extras

---

## Dev Board Architecture

### The Universal Module Dev Board

**Core Component:** Cheap Cortex-M MCU ($0.10 - $1.50)
- Cortex-M0+/M3/M4 @ 48-120 MHz
- 64-256 KB flash, 16-64 KB RAM
- USB Full-Speed device
- Multiple ADCs (10-12 bit)
- Timers, PWMs, SPI/I²C/UART

**Built-in Features:**
- Power regulation: 48V → 12V → 5V → 3.3V ($0.30)
- Input/output buffer op-amps ($0.20)
- Protection (overvoltage, reverse polarity, ESD) ($0.20)
- Auto-discovery EEPROM ($0.11)
- USB-C connector with CC resistors ($0.30-0.50)
- Programming header (SWD/UART)

**Total BOM: ~$2.50 in volume, sell for $5-10**

**What This Enables:**
- Every module can be semi-smart
- Self-calibration, health reporting, preset storage
- Digital + analog modules look identical
- Kids can build boutique-quality modules with zero EE background
- Firmware becomes the differentiator

---

## Signal Flow Example

### "Crappy Guitar → Sounds Like Symphony"

```
User Input (guitar pickup)
  ↓
[Analog Buffer Block] - 6-pin, pure analog, 0.01ms
  ↓
[ML Body Model Block] - 10-pin, hybrid (analog I/O, neural inference), 3ms
  ↓
[Analog Color Block] - 6-pin, transformer saturation, 0.02ms
  ↓
[DSP Reverb Block] - 10-pin, digital, 8ms
  ↓
Main Output

Total Latency: ~11ms (acceptable for non-feedback path)
Controller knows this, tags chain as "high-latency, not for monitoring"
```

**Key Point:** Some modules analog, some digital, some neural. User doesn't care. System manages latency automatically.

---

## Key Design Principles

1. **Substrate Agnostic:** Modules can be anything that obeys the interface
2. **Deterministic Latency:** Every path has known, measurable delay
3. **Safe Composition:** System prevents unsafe feedback loops
4. **Progressive Complexity:** Start simple (analog), add smarts as needed
5. **Open Ecosystem:** Third parties can build modules easily
6. **Unified Time:** Shared clock/timestamp enables global coordination
7. **Self-Describing:** Modules advertise capabilities like USB devices

---

## Next Steps

See:
- `03-PRIMITIVES.md` - The four core operations everything builds from
- `04-MODULES.md` - Specific module types and designs
- `05-PROTOCOL.md` - Detailed bus protocol specification
- `06-KAIRO-BRIDGE.md` - How this connects to Morphogen
