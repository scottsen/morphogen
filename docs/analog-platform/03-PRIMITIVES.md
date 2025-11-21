# The Four Primitives: DNA of Computational Systems

## The Universal Truth

> "The human body has four instructions and does some pretty cool stuff."
> — referring to DNA's four bases: A, C, G, T

Complex systems are built from shockingly simple primitives. This is true across **every domain:**

| Domain | Primitives | Result |
|--------|-----------|--------|
| DNA | A, C, G, T | All life |
| Neurons | Weighted sum, nonlinearity, integrate, fire | Intelligence |
| Analog Circuits | Summation, integration, nonlinearity, events | All signal processing |
| Digital Logic | NAND gates (or AND, OR, NOT) | All computation |
| Unix | Pipes, files, processes | Operating systems |
| Moog Synth | VCO, VCF, VCA, envelopes | Music |

**Your platform follows the same pattern.**

---

## The Four Core Operations

Everything in analog and digital signal processing reduces to:

### 1. **Summation** (Linear Combination)
**What it does:** Adds, mixes, scales signals

**Analog Implementation:**
- Op-amp summer
- Resistor mixing networks
- Kirchhoff's current law at a node

**Digital Implementation:**
- Simple addition/multiplication
- Matrix operations
- Convolution (weighted sum)

**What it enables:**
- Mix multiple sources
- Feed-forward networks
- Filters (weighted sums of delays)
- Gain staging
- Wet/dry blends
- Matrix multiply (neural nets)

**Examples:**
- Audio mixer
- Feedback networks
- FIR filters
- Crossbar MAC arrays

---

### 2. **Integration / Differentiation** (Dynamics over Time)
**What it does:** Accumulates or detects change over time

**Analog Implementation:**
- RC integrator (capacitor + resistor)
- Op-amp integrator
- Analog delay lines
- OTA filters

**Digital Implementation:**
- IIR/FIR filters
- Running sums
- Delay buffers
- Leaky integrators

**What it enables:**
- Lowpass, bandpass, highpass filters
- Envelopes
- Oscillators
- Resonators
- Dynamic systems
- Physical modeling
- Control loops
- Memory

**Examples:**
- Every filter ever made
- Envelope followers
- LFOs
- Analog integrators for ODE solving
- Audio delays

**Critical Insight:** Integration is where **90% of audio magic** comes from.

---

### 3. **Nonlinearity** (Squashing, Clipping, Shaping)
**What it does:** Non-proportional transformation of signals

**Analog Implementation:**
- Diode curves
- Transistor transfer curves
- Tanh approximations via differential pairs
- Soft/hard clipping
- Saturation

**Digital Implementation:**
- Waveshapers
- Sigmoid, ReLU, tanh functions
- Lookup tables
- Polynomial approximations

**What it enables:**
- Compression
- Distortion
- Saturation
- Waveshaping
- Neural network activations
- Threshold logic
- Oscillator shaping
- Character and "warmth"

**Examples:**
- Guitar distortion
- Compressor knee
- Neural net activation functions
- Tube saturation emulation
- Analog chaos generators

**Critical Insight:** Nonlinearity is the **spice that makes everything alive and musical.**

---

### 4. **Sample/Hold/Trigger** (Discrete Events)
**What it does:** Captures moments, creates events, resets state

**Analog Implementation:**
- Sample-and-hold circuits
- Comparators (threshold detection)
- Monostable multivibrators
- Gate signals

**Digital Implementation:**
- Clocks
- Triggers
- Sampling
- Quantization
- State transitions

**What it enables:**
- Modulation sources
- Clocks and timing
- Triggers and gates
- Event sequencing
- Resets and initialization
- Discrete updates
- A/D conversion

**Examples:**
- LFO reset
- Gate/trigger signals
- Sample clocks
- ADC sample-and-hold
- Tempo clocks

---

## How These Four Build Everything

### Filters
- **Integration** (lowpass RC) + **Summation** (multiple poles)

### Oscillators
- **Integration** (phase accumulation) + **Nonlinearity** (waveshaping) + **Trigger** (reset)

### Compressors
- **Integration** (envelope follower) + **Nonlinearity** (threshold/knee) + **Summation** (gain reduction)

### Neural Networks
- **Summation** (weighted inputs) + **Nonlinearity** (activation) + **Integration** (recurrent state)

### Physical Models
- **Integration** (differential equations) + **Nonlinearity** (contact, friction) + **Summation** (forces)

### Reverb
- **Integration** (delays/resonators) + **Summation** (mix taps) + **Nonlinearity** (soft saturation)

### Chaos Generators
- **Integration** (state evolution) + **Nonlinearity** (feedback) + **Summation** (coupling)

---

## The Primitives Across Substrates

### Analog Circuits
| Primitive | Implementation |
|-----------|---------------|
| Sum | Op-amp summer, mixer |
| Integrate | RC + op-amp |
| Nonlinearity | Diodes, transistor curves |
| Event | Comparator, S&H |

### Digital DSP
| Primitive | Implementation |
|-----------|---------------|
| Sum | Multiply-add |
| Integrate | IIR filter, delay + feedback |
| Nonlinearity | Waveshaper, sigmoid |
| Event | Sample clock, gate |

### Analog Neural Chips
| Primitive | Implementation |
|-----------|---------------|
| Sum | Crossbar MAC array |
| Integrate | Capacitor accumulation |
| Nonlinearity | Transistor activation |
| Event | Spike detection |

### Morphogen (Software)
| Primitive | Implementation |
|-----------|---------------|
| Sum | Linear transform, convolution |
| Integrate | Field evolution, ODE solver |
| Nonlinearity | Activation, penalty |
| Event | Discrete timesteps, thresholds |

**All the same. Different substrates.**

---

## Module Taxonomy Based on Primitives

Your platform should offer modules that implement these primitives cleanly:

### Essential Module Set (4-6 blocks)

1. **SumBlock**
   - Analog: Op-amp summer, mixer
   - Digital: Adder, combiner
   - Use: Mix sources, feed-forward, gain staging

2. **IntegratorBlock**
   - Analog: RC integrator, OTA filter
   - Digital: IIR/FIR filter
   - Use: Filters, envelopes, oscillators, resonators

3. **NonlinearBlock**
   - Analog: Diode clipper, transistor saturation
   - Digital: Waveshaper, sigmoid
   - Use: Distortion, compression, activation functions

4. **TriggerBlock**
   - Analog: Comparator, S&H
   - Digital: Clock, gate generator
   - Use: Modulation, timing, events, resets

5. **ControlBlock** (Optional)
   - CV input/output
   - Parameter routing
   - Macro controls

6. **UtilityBlock** (Optional)
   - Buffers
   - Inverters
   - Splitters
   - Level shifters

---

## Why This Approach Works

### 1. **Keeps the System Simple**
Don't build 50 module types. Build 4-6 primitives.
Everything else is composition.

### 2. **Enables Emergent Complexity**
Like DNA: simple alphabet → infinite combinations.

### 3. **Substrate Agnostic**
Primitives work whether implemented in:
- Analog circuits
- DSP chips
- Neural accelerators
- Hybrid systems

### 4. **Educational Power**
Users learn the fundamentals by composing primitives.
They understand WHY things work, not just WHAT they do.

### 5. **Future-Proof**
New exotic substrates (optical, biological, quantum) will still implement these same operations.

---

## The Anti-Pattern: Don't Overbuild

**DON'T create:**
- "J-45 Body Model Block"
- "Tube Preamp Block"
- "Vintage Tape Saturator Block"
- "Specific Plugin Emulation Block"

**DO create:**
- Primitives that COMPOSE into those behaviors

**Why:**
- Fewer modules to maintain
- More creative possibilities
- Users build their own "J-45" from primitives
- Ecosystem stays lean and hackable
- Third parties innovate within your framework

---

## Comparison to Other Systems

| System | Primitives | Composable? | Result |
|--------|-----------|-------------|--------|
| DNA | 4 bases | ✅ | All life |
| Unix | Pipes + files | ✅ | Infinite tools |
| Moog | VCO/VCF/VCA/ENV | ✅ | Modular synthesis |
| Eurorack | 1000s of modules | ❌ | Confusion & redundancy |
| Your Platform | 4-6 primitives | ✅ | **Emergent complexity** |

**Eurorack failed to define primitives.**
They have hundreds of modules that often do redundant things.
Users are overwhelmed.

**Your platform succeeds by staying focused on primitives.**

---

## Implementation Strategy

### Phase 1: Core Primitives (MVP)
Build 4 modules:
1. SumBlock
2. IntegratorBlock
3. NonlinearBlock
4. TriggerBlock

Prove the concept. Show composition works.

### Phase 2: Control Layer
Add:
5. ControlBlock (CV routing, macro knobs)
6. UtilityBlock (buffers, splitters)

Enable richer modulation and routing.

### Phase 3: Advanced Implementations
Offer multiple versions of each primitive:
- IntegratorBlock_Analog (op-amp RC)
- IntegratorBlock_DSP (IIR filters)
- IntegratorBlock_Neural (analog capacitor network)

Users pick the implementation that fits their needs.

### Phase 4: Composite Modules (User-Defined)
Once primitives are established, users create composite modules:
- "MyCustomFilter" = IntegratorBlock + NonlinearBlock + SumBlock
- Save as preset or share as community module

---

## The Deep Insight

> "We don't need 100 module types. We need a small, powerful alphabet."

This is:
- How biology works (DNA)
- How brains work (neurons)
- How music works (notes)
- How language works (letters)
- How math works (axioms)

**Your platform is no different.**

Four operations.
Infinite emergence.

---

## Next: Read These

- `04-MODULES.md` - Specific module implementations
- `05-PROTOCOL.md` - How modules communicate
- `06-KAIRO-BRIDGE.md` - Mapping primitives between hardware and Morphogen
