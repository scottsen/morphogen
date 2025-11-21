# Cellular Automata Examples

This directory contains demonstrations of Kairo's cellular automata capabilities, showcasing classic CA systems and their emergent behaviors.

## Examples

### 01_game_of_life.py
Classic Conway's Game of Life and variants.

**Features:**
- Conway's Game of Life (B3/S23)
- Classic patterns: gliders, oscillators, Gosper glider gun
- HighLife variant (B36/S23) with self-replication
- Seeds variant (B2/S) with explosive growth
- Beautiful visualizations with color palettes
- Pattern analysis and statistics

**Run:**
```bash
python 01_game_of_life.py
```

**Generates:**
- `output_game_of_life_step*.png` - Evolution frames
- `output_game_of_life_patterns_step*.png` - Classic patterns
- `output_highlife_step*.png` - HighLife variant
- `output_seeds_step*.png` - Seeds variant

### 02_wolfram_ca.py
Wolfram Elementary Cellular Automata (1D, 256 rules).

**Features:**
- Famous rules: 30 (chaotic), 90 (fractal), 110 (Turing complete), 184 (traffic)
- Classification examples from all 4 Wolfram classes
- Random initial conditions
- Rule gallery visualization
- Behavioral analysis and classification

**Run:**
```bash
python 02_wolfram_ca.py
```

**Generates:**
- `output_wolfram_rule*.png` - Individual rule spacetime diagrams
- `output_wolfram_gallery_*_rules.png` - Multi-rule gallery

**Key Rules:**
- **Rule 30**: Chaotic, used in Mathematica's random number generator
- **Rule 90**: Produces Sierpinski triangle pattern
- **Rule 110**: Turing complete, supports universal computation
- **Rule 184**: Models traffic flow

### 03_brians_brain.py
Brian's Brain - Beautiful 3-state cellular automaton.

**Features:**
- 3-state system: dead/firing/refractory
- Wave propagation patterns
- Spiral patterns and rotating waves
- Custom color mapping for states
- Dynamics analysis
- Density comparison studies

**Run:**
```bash
python 03_brians_brain.py
```

**Generates:**
- `output_brians_brain_step*.png` - Random initialization
- `output_brians_brain_patterns_step*.png` - Structured patterns
- `output_brians_brain_spirals_step*.png` - Rotating spirals
- `output_brians_brain_density_comparison.png` - Density comparison

**Visual Appearance:**
- Creates organic wave-like patterns
- Beautiful spiral and rotating structures
- Mesmerizing color transitions

## Domain Integration

These examples demonstrate cross-domain capabilities:

- **Cellular**: Core CA simulation
- **Palette**: Color mapping and gradients
- **Image**: Output and composition
- **Visual**: Rendering and display

## Cellular Automata Theory

### 2D Cellular Automata

Characterized by:
- **States**: Number of possible cell states (typically 2)
- **Neighborhood**: Moore (8 neighbors) or von Neumann (4 neighbors)
- **Rules**: Birth/survival conditions

**Notation**: B{birth}/S{survival}
- **B3/S23** (Game of Life): Birth with 3 neighbors, survive with 2-3
- **B36/S23** (HighLife): Birth with 3 or 6, survive with 2-3
- **B2/S** (Seeds): Birth with 2, no survival

### 1D Cellular Automata (Wolfram)

- **Rule number**: 0-255 (8-bit lookup table)
- **Neighborhood**: Left neighbor, self, right neighbor
- **Update**: Synchronous, deterministic

**Wolfram Classes:**
1. **Class 1**: Evolution leads to uniform state
2. **Class 2**: Evolution leads to simple periodic structures
3. **Class 3**: Chaotic, random-looking patterns
4. **Class 4**: Complex, localized structures (capable of computation)

## Pattern Types

### Game of Life

- **Still lifes**: Block, beehive, loaf, boat
- **Oscillators**: Blinker, toad, pulsar, pentadecathlon
- **Spaceships**: Glider, lightweight/middleweight/heavyweight spaceship
- **Guns**: Gosper glider gun (period 30)
- **Methuselahs**: R-pentomino, acorn

### Brian's Brain

- **Waves**: Propagating activity fronts
- **Spirals**: Rotating wave patterns
- **Turbulence**: Chaotic interactions

## Performance Notes

- Field sizes up to 500x500 run smoothly
- 300-500 generations typical for interesting evolution
- Use lower densities (0.1-0.3) for better patterns
- Random seeds provide reproducible results

## Further Exploration

Try experimenting with:
- Custom CA rules (modify birth/survival sets)
- Different initial patterns
- Larger/smaller grid sizes
- Alternative color schemes
- Combining multiple CA fields
- Cross-domain demos (CA + audio, CA + optimization)

## References

- Conway, J. (1970). "The Game of Life"
- Wolfram, S. (2002). "A New Kind of Science"
- Silverman, B. (1987). "Brian's Brain" (originally "Brian Silverman's Brain")

## Credits

Part of the Kairo temporal programming language ecosystem.
See the main README for more examples and documentation.
