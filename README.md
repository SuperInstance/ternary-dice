# ternary-dice: Stochastic exploration with configurable randomness for {-1, 0, +1} systems

## Why This Exists

Deterministic ternary systems are useful, but sometimes you need controlled randomness — exploring the possibility space, breaking symmetry, generating novel combinations, or simulating chance-based outcomes. This crate provides a deterministic PRNG-based dice system over the ternary domain, with weighted distributions, statistical analysis, probability rebalancing, and a D&D-style FatesTable for narrative outcomes. The "stochastic flavor exploration engine" Casey described.

## Core Concepts

**Balanced ternary**: Three values: -1 (Neg), 0 (Zero), +1 (Pos). The domain of all random outcomes.

**Prng**: A deterministic xorshift32 pseudo-random number generator. Same seed always produces the same sequence. No external RNG dependency.

**Dice**: A ternary dice with configurable weights for [Neg, Zero, Pos]. Default weights [1,1,1] give uniform distribution. Bias toward any value by adjusting weights.

**DiceSet**: Multiple Dice with independent seeds and distributions. Roll all at once.

**DiceRoller**: Generates combinations of rolls — multiple dice rolled multiple times — for exploration.

**DiceStatistics**: Counts occurrences of each Trit value, computes frequencies, identifies the mode, and checks if a distribution is balanced within tolerance.

**DiceRebalance**: Given observed statistics and a target distribution, computes new weights to steer future rolls toward the target.

**FatesTable**: A lookup table mapping roll sums to narrative outcomes with severity levels (CriticalFail through CriticalSuccess). Default table covers sums -3 through +3.

## Quick Start

```toml
[dependencies]
ternary-dice = "0.1"
```

```rust
use ternary_dice::*;

let mut dice = Dice::new(42);
let rolls = dice.roll_n(10);

let stats = DiceStatistics::from_rolls(&rolls);
println!("Pos frequency: {:.2}", stats.frequency(Trit::Pos));

let table = FatesTable::new();
let roll = vec![Trit::Pos, Trit::Pos, Trit::Neg]; // sum = 1
let fate = table.evaluate(&roll).unwrap();
println!("Outcome: {} ({:?})", fate.outcome, fate.severity);
```

## API Overview

| Type | Description |
|------|-------------|
| `Trit` | Balanced ternary value: Neg, Zero, or Pos |
| `Prng` | Deterministic xorshift32 PRNG |
| `Dice` | Weighted ternary dice with embedded PRNG |
| `DiceSet` | Collection of independent Dice |
| `DiceRoller` | Generates roll combinations (N dice × M rolls) |
| `DiceStatistics` | Tallies and analyzes roll distributions |
| `DiceRebalance` | Computes adjusted weights from statistics |
| `FatesTable` | Maps roll sums to narrative outcomes |
| `FatesEntry` | One outcome: roll value, description, severity |
| `Severity` | Five levels: CriticalFail, Fail, Neutral, Success, CriticalSuccess |

## How It Works

The Prng uses xorshift32: three bitwise shift-xor operations on a 32-bit state. Seed of 0 is converted to 1 to avoid the degenerate all-zero state. This is fast, deterministic, and adequate for simulation — not for cryptography.

Dice roll by generating a u32 in [0, total_weight) and checking which weight bucket it falls into. Weights [neg, zero, pos] partition the range: [0, neg) → Neg, [neg, neg+zero) → Zero, [neg+zero, total) → Pos.

DiceRebalance computes adjusted weights by taking the target frequency for each outcome and dividing by the observed frequency (with a minimum of 1 to avoid zero weights). This is a simple inverse-probability weighting scheme.

The FatesTable sums the Trit values of a roll (e.g., [Pos, Neg, Pos] = 1 + -1 + 1 = 1) and looks up the matching entry.

## Known Limitations

- xorshift32 has known statistical weaknesses (correlated bits, fails some TestU01 batteries). Adequate for simulations, not for serious statistical work.
- No seed saving/restoration. You can clone the Prng state, but there's no serialization.
- FatesTable lookup is by exact sum. If your roll sums don't match any entry, you get None.
- DiceStatistics with 0 total returns 0.0 frequency and claims "balanced" — a degenerate case.
- Rebalance weights can grow unbounded if observed frequency is very low. The implementation clamps to 3× scale as a rough safeguard.
- All rolls are synchronous and single-threaded. No parallel roll generation.

## Use Cases

- **Strategy exploration**: Roll random ternary strategies to test against opponents in `ternary-arena`.
- **Flavor generation**: Use FatesTable to generate narrative outcomes for game events.
- **Symmetry breaking**: Seed-dependent randomness to break ties in deterministic systems.
- **Fairness testing**: Roll many times, collect statistics, verify distribution balance.
- **Adaptive difficulty**: Use DiceRebalance to steer probabilities toward harder or easier outcomes.

## Ecosystem Context

Part of the SuperInstance ternary ecosystem. Feeds stochastic strategies into `ternary-arena` and `ternary-agent`. Complements `ternary-chaos` (which studies chaotic dynamics) by providing controlled randomness. Could use `ternary-statistics` for deeper distribution analysis.

## License

MIT

## See Also
- **ternary-games** — related
- **ternary-random** — related
- **ternary-auction** — related
- **ternary-scoring** — related
- **ternary-market** — related

