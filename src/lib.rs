#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Stochastic exploration with configurable randomness for balanced ternary systems.
//!
//! Provides dice-based random generation over the ternary domain {-1, 0, +1},
//! with configurable distributions, statistics tracking, probability rebalancing,
//! and a FatesTable for D&D-style lookup outcomes. All randomness is deterministic
//! based on a provided seed/state — no external RNG dependency needed.
//!
//! # When to use this crate
//!
//! - You have a small {-1, 0, +1} outcome space and want **deterministic**,
//!   seed-reproducible randomness (e.g. fuzzing a ternary decision procedure,
//!   breaking ties, generating test fixtures, or running a narrative table).
//! - You want **weighted** distributions over those three outcomes and the
//!   ability to **rebalance** them in response to observed statistics.
//! - You want to map roll **sums** to human-readable narrative outcomes
//!   (the [`FatesTable`]).
//!
//! When *not* to use this crate: xorshift32 is unsuitable for cryptography or
//! statistically rigorous work (see [`Prng`] caveats). Reach for `rand` instead.
//!
//! # Quick start
//!
//! ```
//! use ternary_dice::{Dice, DiceStatistics, FatesTable, Trit};
//!
//! let mut dice = Dice::new(42);
//! let rolls = dice.roll_n(10);
//!
//! let stats = DiceStatistics::from_rolls(&rolls);
//! println!("Pos frequency: {:.2}", stats.frequency(Trit::Pos));
//!
//! let table = FatesTable::new();
//! let roll = vec![Trit::Pos, Trit::Pos, Trit::Neg]; // sum = 1
//! let fate = table.evaluate(&roll).unwrap();
//! println!("Outcome: {} ({:?})", fate.outcome, fate.severity);
//! ```

// No external dependencies needed.

/// A single balanced ternary value drawn from the set {-1, 0, +1}.
///
/// Variants follow the standard balanced-ternary convention: [`Trit::Neg`]
/// represents -1, [`Trit::Zero`] represents 0, and [`Trit::Pos`] represents
/// +1. Use [`Trit::value`] to obtain the numeric value, or [`Trit::from_i8`]
/// to construct one from an integer (returning `None` outside the domain).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trit {
    /// The negative balanced-ternary digit, with numeric value -1.
    Neg,
    /// The zero balanced-ternary digit, with numeric value 0.
    Zero,
    /// The positive balanced-ternary digit, with numeric value +1.
    Pos,
}

impl Trit {
    /// Returns the signed numeric value of this trit (`-1`, `0`, or `+1`).
    pub fn value(self) -> i8 {
        match self {
            Trit::Neg => -1,
            Trit::Zero => 0,
            Trit::Pos => 1,
        }
    }

    /// Constructs a `Trit` from an `i8`, returning `None` for any value
    /// outside `{-1, 0, 1}`.
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }
}

/// A simple deterministic PRNG implementing the xorshift32 algorithm.
///
/// Each call advances a 32-bit state through three bitwise shift-xor steps.
/// The same seed always produces the same output sequence, which makes rolls
/// reproducible — useful for tests, fixtures, and replay debugging.
///
/// # Determinism guarantees
///
/// A seed of `0` is silently replaced with `1`, because xorshift32 has a
/// degenerate all-zero fixed point (it would emit `0` forever).
///
/// # Cryptographic / statistical caveat
///
/// xorshift32 fails several TestU01 batteries and has correlated bits. Use it
/// for simulations and gameplay, **not** for security or rigorous statistics.
#[derive(Clone, Debug)]
pub struct Prng {
    state: u32,
}

impl Prng {
    /// Creates a new PRNG with the given seed. A seed of `0` is replaced
    /// with `1` to avoid xorshift32's degenerate all-zero fixed point.
    pub fn new(seed: u32) -> Self {
        // Ensure non-zero state
        Prng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Advances the PRNG state and returns the next `u32` in the sequence.
    ///
    /// The full sequence is deterministic for a given seed.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Returns a value uniformly distributed in `[0, max)` via modular reduction.
    ///
    /// # Panics
    ///
    /// Panics if `max == 0` — the half-open range `[0, 0)` is empty and has no
    /// valid return value. Callers that already special-case a zero total (as
    /// [`Dice::roll`] does) never hit this path.
    ///
    /// # Numerical caveat
    ///
    /// Uses naive `% max` reduction, so outputs are very slightly biased when
    /// `max` does not evenly divide `2^32`. Adequate for simulation; not for
    /// cryptographic or statistically rigorous work.
    pub fn next_range(&mut self, max: u32) -> u32 {
        assert!(
            max > 0,
            "next_range requires max > 0; the range [0, 0) is empty"
        );
        self.next_u32() % max
    }
}

/// A configurable ternary dice with weighted probabilities over `Trit`.
///
/// Each roll draws a `u32` from the embedded [`Prng`] and partitions
/// `[0, total_weight)` into three contiguous buckets — one per outcome —
/// sized by `weights`. With the default weights `[1, 1, 1]` each outcome is
/// equally likely. All-zero weights cause [`Dice::roll`] to deterministically
/// return [`Trit::Zero`].
#[derive(Clone, Debug)]
pub struct Dice {
    /// Probability weights in the order `[neg_weight, zero_weight, pos_weight]`.
    /// Only the relative magnitudes matter; the default is `[1, 1, 1]`.
    pub weights: [u32; 3],
    prng: Prng,
}

impl Dice {
    /// Creates a uniformly-weighted dice (`weights = [1, 1, 1]`) driven by a
    /// PRNG seeded with `seed`.
    pub fn new(seed: u32) -> Self {
        Dice {
            weights: [1, 1, 1],
            prng: Prng::new(seed),
        }
    }

    /// Creates a dice with custom probability weights. The weights need not
    /// sum to any particular value; only their relative magnitudes affect the
    /// distribution. All-zero weights cause [`Dice::roll`] to return
    /// [`Trit::Zero`].
    pub fn with_weights(seed: u32, weights: [u32; 3]) -> Self {
        Dice {
            weights,
            prng: Prng::new(seed),
        }
    }

    /// Rolls the dice once and returns the resulting [`Trit`].
    ///
    /// Bucket partition: with weights `[neg, zero, pos]` and total `t =
    /// neg + zero + pos`, a uniform draw `r ∈ [0, t)` maps to:
    /// - [`Trit::Neg`] if `r < neg`,
    /// - [`Trit::Zero`] if `neg ≤ r < neg + zero`,
    /// - [`Trit::Pos`] otherwise.
    ///
    /// If `t == 0` (all weights zero) this returns [`Trit::Zero`]. Weight
    /// summation uses saturating arithmetic so adversarial inputs near
    /// `u32::MAX` cannot trigger an arithmetic-overflow panic.
    pub fn roll(&mut self) -> Trit {
        // Sum with saturating arithmetic so adversarial weight magnitudes
        // (e.g. `[u32::MAX, u32::MAX, 1]`) cannot trigger arithmetic overflow
        // panics. Realistic inputs are unaffected; only pathological inputs
        // near `u32::MAX` saturate and lose precision in proportion to how
        // badly they overflow.
        let total = self.weights[0]
            .saturating_add(self.weights[1])
            .saturating_add(self.weights[2]);
        if total == 0 {
            return Trit::Zero;
        }
        let r = self.prng.next_range(total);
        let neg_bound = self.weights[0];
        let zero_bound = neg_bound.saturating_add(self.weights[1]);
        if r < neg_bound {
            Trit::Neg
        } else if r < zero_bound {
            Trit::Zero
        } else {
            Trit::Pos
        }
    }

    /// Rolls the dice `count` times, collecting the results into a `Vec`.
    /// Equivalent to calling [`Dice::roll`] `count` times in sequence, so the
    /// PRNG state advances by exactly `count` steps.
    pub fn roll_n(&mut self, count: usize) -> Vec<Trit> {
        (0..count).map(|_| self.roll()).collect()
    }

    /// Replaces the weights with a preset that mildly favours `target`:
    /// weight 3 for `target`, 1 for the other two outcomes.
    pub fn bias_toward(&mut self, target: Trit) {
        self.weights = match target {
            Trit::Neg => [3, 1, 1],
            Trit::Zero => [1, 3, 1],
            Trit::Pos => [1, 1, 3],
        };
    }
}

/// A collection of independent [`Dice`] that can be rolled together.
///
/// Each die retains its own seed/state and weight distribution, so adding
/// multiple dice with different seeds gives you independent streams of
/// reproducible rolls.
#[derive(Clone, Debug)]
pub struct DiceSet {
    dice: Vec<Dice>,
}

impl DiceSet {
    /// Creates an empty `DiceSet`. Use [`DiceSet::add`] to populate it.
    pub fn new() -> Self {
        DiceSet { dice: Vec::new() }
    }

    /// Adds a [`Dice`] to the set, returning the index it was assigned.
    pub fn add(&mut self, dice: Dice) -> usize {
        let idx = self.dice.len();
        self.dice.push(dice);
        idx
    }

    /// Rolls every die in the set in insertion order, returning one
    /// [`Trit`] per die. The returned `Vec` is empty if the set is empty.
    pub fn roll_all(&mut self) -> Vec<Trit> {
        self.dice.iter_mut().map(|d| d.roll()).collect()
    }

    /// Rolls a single die by index, returning `None` if the index is out
    /// of range.
    pub fn roll_one(&mut self, index: usize) -> Option<Trit> {
        self.dice.get_mut(index).map(|d| d.roll())
    }

    /// Returns the number of dice currently in the set.
    pub fn len(&self) -> usize {
        self.dice.len()
    }

    /// Returns `true` if the set contains no dice.
    pub fn is_empty(&self) -> bool {
        self.dice.is_empty()
    }
}

impl Default for DiceSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates combinations of dice rolls (`dice_count` dice × `rolls_per_die`
/// rounds) from a single root seed.
///
/// The roller seeds each die deterministically from the root seed (die `i`
/// in round `r` gets `seed.wrapping_add(r).wrapping_add(i * 7919)`), so the
/// same `seed` always produces the same combination matrix. Useful for
/// exploration / fuzzing over a ternary strategy space.
#[derive(Clone, Debug)]
pub struct DiceRoller {
    /// Number of dice rolled in each round.
    pub dice_count: usize,
    /// Number of rounds (i.e. separate combinations) produced by [`DiceRoller::generate`].
    pub rolls_per_die: usize,
}

impl DiceRoller {
    /// Creates a roller that will produce `rolls_per_die` combinations, each
    /// consisting of `dice_count` dice rolls.
    pub fn new(dice_count: usize, rolls_per_die: usize) -> Self {
        DiceRoller {
            dice_count,
            rolls_per_die,
        }
    }

    /// Generates `rolls_per_die` combinations by constructing and rolling a
    /// fresh [`DiceSet`] each round. Each combination has exactly
    /// `dice_count` entries. Setting either dimension to `0` yields an
    /// empty result for that axis (no rounds, or empty combinations).
    pub fn generate(&self, seed: u32) -> Vec<Vec<Trit>> {
        let mut results = Vec::new();
        let mut base_seed = seed;
        for _ in 0..self.rolls_per_die {
            let mut set = DiceSet::new();
            for i in 0..self.dice_count {
                set.add(Dice::new(base_seed.wrapping_add(i as u32 * 7919)));
            }
            results.push(set.roll_all());
            base_seed = base_seed.wrapping_add(1);
        }
        results
    }

    /// Generates a single combination (one round, all `dice_count` dice
    /// rolled once). Cheaper than [`DiceRoller::generate`] when you only
    /// need one sample.
    pub fn roll_once(&self, seed: u32) -> Vec<Trit> {
        let mut set = DiceSet::new();
        for i in 0..self.dice_count {
            set.add(Dice::new(seed.wrapping_add(i as u32 * 7919)));
        }
        set.roll_all()
    }
}

/// Tallies occurrences of each [`Trit`] value across a sequence of rolls and
/// answers basic statistical questions about them (frequency, mode, balance,
/// signed sum).
///
/// Build one incrementally with [`DiceStatistics::record`] or all at once
/// with [`DiceStatistics::from_rolls`].
#[derive(Clone, Debug)]
pub struct DiceStatistics {
    /// Number of [`Trit::Neg`] observations recorded so far.
    pub neg_count: usize,
    /// Number of [`Trit::Zero`] observations recorded so far.
    pub zero_count: usize,
    /// Number of [`Trit::Pos`] observations recorded so far.
    pub pos_count: usize,
    /// Total number of observations recorded so far (`neg + zero + pos`).
    pub total: usize,
}

impl DiceStatistics {
    /// Creates an empty statistics container (all counts zero).
    pub fn new() -> Self {
        DiceStatistics {
            neg_count: 0,
            zero_count: 0,
            pos_count: 0,
            total: 0,
        }
    }

    /// Builds statistics by tallying an entire slice of rolls in one call.
    pub fn from_rolls(rolls: &[Trit]) -> Self {
        let mut stats = DiceStatistics::new();
        for &trit in rolls {
            stats.record(trit);
        }
        stats
    }

    /// Records a single roll, incrementing the appropriate counter (and `total`).
    pub fn record(&mut self, trit: Trit) {
        self.total += 1;
        match trit {
            Trit::Neg => self.neg_count += 1,
            Trit::Zero => self.zero_count += 1,
            Trit::Pos => self.pos_count += 1,
        }
    }

    /// Returns the empirical frequency of `trit` as a fraction in `[0.0, 1.0]`.
    /// Returns `0.0` for every variant when no rolls have been recorded
    /// (the documented degenerate case).
    pub fn frequency(&self, trit: Trit) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let count = match trit {
            Trit::Neg => self.neg_count,
            Trit::Zero => self.zero_count,
            Trit::Pos => self.pos_count,
        };
        count as f64 / self.total as f64
    }

    /// Returns `true` if every outcome's frequency is within `tolerance`
    /// (absolute error) of the uniform value `1/3`.
    ///
    /// The empty case (`total == 0`) is considered vacuously balanced and
    /// returns `true`. Behaviour with `tolerance = NaN` follows IEEE-754
    /// comparison rules (every comparison against `NaN` is `false`, so the
    /// function returns `true`).
    pub fn is_balanced(&self, tolerance: f64) -> bool {
        if self.total == 0 {
            return true;
        }
        let expected = 1.0 / 3.0;
        for trit in [Trit::Neg, Trit::Zero, Trit::Pos] {
            if (self.frequency(trit) - expected).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Returns the most-frequent [`Trit`], or `None` when no rolls have been
    /// recorded.
    ///
    /// Tie-break: when two or more outcomes share the maximum count, the
    /// *last* one in iteration order `Neg → Zero → Pos` wins. This mirrors
    /// [`Iterator::max_by_key`]'s contract.
    pub fn mode(&self) -> Option<Trit> {
        if self.total == 0 {
            return None;
        }
        let counts = [
            (Trit::Neg, self.neg_count),
            (Trit::Zero, self.zero_count),
            (Trit::Pos, self.pos_count),
        ];
        let max = counts.iter().max_by_key(|&&(_, c)| c).unwrap();
        Some(max.0)
    }

    /// Returns the signed sum of all recorded trit values (`-1` per Neg, `0`
    /// per Zero, `+1` per Pos).
    ///
    /// Despite the `_i8` suffix (which refers to the constituent [`Trit`]
    /// value type), the return type is `i32` to avoid overflow on large
    /// roll counts: a sequence of `n` Pos rolls sums to `n`, which overflows
    /// `i8` for `n > 127`.
    pub fn sum_i8(&self) -> i32 {
        -(self.neg_count as i32) + (self.pos_count as i32)
    }
}

impl Default for DiceStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes new dice weights that nudge future rolls toward a target
/// distribution, given observed [`DiceStatistics`].
///
/// Uses a simple inverse-probability weighting scheme (see
/// [`DiceRebalance::compute_weights`]). Useful for adaptive difficulty,
/// fairness correction, or forcing exploration of under-represented outcomes.
#[derive(Clone, Debug)]
pub struct DiceRebalance {
    /// Target frequencies in `[neg, zero, pos]` order. Values need not sum to
    /// 1 — only their ratios are used.
    pub target_distribution: [f64; 3],
}

impl DiceRebalance {
    /// Targets the uniform distribution `[1/3, 1/3, 1/3]`.
    pub fn balanced() -> Self {
        DiceRebalance {
            target_distribution: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        }
    }

    /// Targets a custom distribution. The values need not be normalised;
    /// only their ratios affect the resulting weights.
    pub fn custom(neg: f64, zero: f64, pos: f64) -> Self {
        DiceRebalance {
            target_distribution: [neg, zero, pos],
        }
    }

    /// Compute new weights from current statistics to achieve the target distribution.
    ///
    /// Implements the inverse-probability weighting scheme described in the
    /// crate-level docs: `weight_i ∝ target_i / observed_i`. The result is
    /// scaled by 1000 for integer precision and clamped to `[1, 3000]` per
    /// outcome so a single under-represented value cannot dominate the next
    /// roll (the README's "3× scale safeguard"). When an outcome has not been
    /// observed at all we cannot divide; we fall back to the scaled target.
    ///
    /// Returns `[neg_weight, zero_weight, pos_weight]` as `u32`.
    pub fn compute_weights(&self, stats: &DiceStatistics) -> [u32; 3] {
        if stats.total == 0 {
            return [1, 1, 1];
        }

        // Integer-precision scale factor. The clamp at `scale * 3.0` matches the
        // "3× scale safeguard" documented in the README.
        let scale = 1000.0;

        let current = [
            stats.frequency(Trit::Neg),
            stats.frequency(Trit::Zero),
            stats.frequency(Trit::Pos),
        ];

        let adjusted: Vec<u32> = (0..3)
            .map(|i| {
                let w = if current[i] > 0.001 {
                    // Inverse-probability: weight ∝ target / observed.
                    (self.target_distribution[i] / current[i] * scale).min(scale * 3.0)
                } else {
                    // Outcome unobserved — fall back to scaled target.
                    self.target_distribution[i] * scale
                };
                (w as u32).max(1)
            })
            .collect();

        [adjusted[0], adjusted[1], adjusted[2]]
    }

    /// Convenience wrapper: computes weights via [`DiceRebalance::compute_weights`]
    /// and writes them straight onto `dice.weights`.
    pub fn rebalance(&self, stats: &DiceStatistics, dice: &mut Dice) {
        dice.weights = self.compute_weights(stats);
    }
}

/// A D&D-style lookup table that maps the integer sum of a roll (a sequence
/// of [`Trit`]s) to a narrative outcome.
///
/// The default table installed by [`FatesTable::new`] covers sums `-3..=+3`,
/// which is the full range achievable by three balanced-ternary dice. Rolls
/// whose sum falls outside the table return `None` from [`FatesTable::lookup`]
/// and [`FatesTable::evaluate`] (see the README's "Known Limitations").
#[derive(Clone, Debug)]
pub struct FatesTable {
    entries: Vec<FatesEntry>,
}

/// One row of a [`FatesTable`]: the `roll_value` (roll sum) that triggers it,
/// the human-readable `outcome` text, and a coarse [`Severity`].
#[derive(Clone, Debug, PartialEq)]
pub struct FatesEntry {
    /// The integer sum of trit values that this entry matches.
    pub roll_value: i8,
    /// Human-readable narrative description of the outcome.
    pub outcome: String,
    /// Coarse severity bucket for downstream branching / display logic.
    pub severity: Severity,
}

/// Coarse outcome severity used by [`FatesEntry`].
///
/// Variants are listed from worst to best. The default [`FatesTable`]
/// installation uses [`Severity::CriticalFail`] for sum `-3`,
/// [`Severity::Fail`] for `-2`/`-1`, [`Severity::Neutral`] for `0`,
/// [`Severity::Success`] for `+1`/`+2`, and [`Severity::CriticalSuccess`]
/// for `+3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Severity {
    /// Worst outcome (e.g. sum -3 on the default table).
    CriticalFail,
    /// Unfavourable outcome (e.g. sums -2 or -1).
    Fail,
    /// Neither favourable nor unfavourable (e.g. sum 0).
    Neutral,
    /// Favourable outcome (e.g. sums +1 or +2).
    Success,
    /// Best outcome (e.g. sum +3).
    CriticalSuccess,
}

impl FatesTable {
    /// Creates a table pre-populated with seven default entries covering
    /// roll sums `-3..=+3`, with severities ranging from
    /// [`Severity::CriticalFail`] to [`Severity::CriticalSuccess`].
    pub fn new() -> Self {
        let mut table = FatesTable {
            entries: Vec::new(),
        };
        // Default D&D-style outcomes based on sum of trits
        table.add(FatesEntry {
            roll_value: -3,
            outcome: "Catastrophic failure".into(),
            severity: Severity::CriticalFail,
        });
        table.add(FatesEntry {
            roll_value: -2,
            outcome: "Major setback".into(),
            severity: Severity::Fail,
        });
        table.add(FatesEntry {
            roll_value: -1,
            outcome: "Minor setback".into(),
            severity: Severity::Fail,
        });
        table.add(FatesEntry {
            roll_value: 0,
            outcome: "Status quo".into(),
            severity: Severity::Neutral,
        });
        table.add(FatesEntry {
            roll_value: 1,
            outcome: "Minor breakthrough".into(),
            severity: Severity::Success,
        });
        table.add(FatesEntry {
            roll_value: 2,
            outcome: "Major success".into(),
            severity: Severity::Success,
        });
        table.add(FatesEntry {
            roll_value: 3,
            outcome: "Extraordinary triumph".into(),
            severity: Severity::CriticalSuccess,
        });
        table
    }

    /// Appends a new [`FatesEntry`] to the table.
    pub fn add(&mut self, entry: FatesEntry) {
        self.entries.push(entry);
    }

    /// Look up an outcome by the exact integer sum of a roll.
    ///
    /// Per the README's "Known Limitations": lookup is by exact sum — if no
    /// entry has `roll_value == roll_sum`, this returns `None`. (The comment
    /// here previously said "Find closest match", which described behaviour
    /// the code never actually implemented.)
    pub fn lookup(&self, roll_sum: i8) -> Option<&FatesEntry> {
        self.entries.iter().find(|e| e.roll_value == roll_sum)
    }

    /// Evaluates a roll (a sequence of [`Trit`]s) by summing its values and
    /// looking up the result. An empty roll sums to `0`. Returns `None` when
    /// the sum is outside the installed range.
    pub fn evaluate(&self, roll: &[Trit]) -> Option<&FatesEntry> {
        let sum: i8 = roll.iter().map(|t| t.value()).sum();
        self.lookup(sum)
    }

    /// Returns the number of entries currently installed in the table.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for FatesTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prng_deterministic() {
        let mut a = Prng::new(42);
        let mut b = Prng::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn prng_nonzero_seed() {
        let mut prng = Prng::new(0);
        assert_ne!(prng.next_u32(), 0);
    }

    #[test]
    fn prng_different_seeds() {
        let mut a = Prng::new(1);
        let mut b = Prng::new(2);
        assert_ne!(a.next_u32(), b.next_u32());
    }

    #[test]
    fn dice_roll_produces_trit() {
        let mut dice = Dice::new(42);
        for _ in 0..100 {
            let t = dice.roll();
            assert!(t == Trit::Neg || t == Trit::Zero || t == Trit::Pos);
        }
    }

    #[test]
    fn dice_weighted_bias() {
        let mut dice = Dice::with_weights(42, [0, 0, 100]); // always Pos
        for _ in 0..50 {
            assert_eq!(dice.roll(), Trit::Pos);
        }
    }

    #[test]
    fn dice_weighted_neg_only() {
        let mut dice = Dice::with_weights(42, [100, 0, 0]);
        for _ in 0..50 {
            assert_eq!(dice.roll(), Trit::Neg);
        }
    }

    #[test]
    fn dice_zero_weights() {
        let mut dice = Dice::with_weights(42, [0, 0, 0]);
        assert_eq!(dice.roll(), Trit::Zero);
    }

    #[test]
    fn dice_roll_n() {
        let mut dice = Dice::new(42);
        let rolls = dice.roll_n(10);
        assert_eq!(rolls.len(), 10);
    }

    #[test]
    fn dice_bias_toward() {
        let mut dice = Dice::new(42);
        dice.bias_toward(Trit::Pos);
        assert_eq!(dice.weights, [1, 1, 3]);
    }

    #[test]
    fn dice_set_add_and_roll() {
        let mut set = DiceSet::new();
        set.add(Dice::new(1));
        set.add(Dice::new(2));
        assert_eq!(set.len(), 2);
        let rolls = set.roll_all();
        assert_eq!(rolls.len(), 2);
    }

    #[test]
    fn dice_set_roll_one() {
        let mut set = DiceSet::new();
        set.add(Dice::new(1));
        set.add(Dice::new(2));
        let result = set.roll_one(0);
        assert!(result.is_some());
        let missing = set.roll_one(99);
        assert!(missing.is_none());
    }

    #[test]
    fn dice_roller_generate() {
        let roller = DiceRoller::new(3, 5);
        let results = roller.generate(42);
        assert_eq!(results.len(), 5);
        for combo in &results {
            assert_eq!(combo.len(), 3);
        }
    }

    #[test]
    fn dice_roller_roll_once() {
        let roller = DiceRoller::new(4, 1);
        let combo = roller.roll_once(42);
        assert_eq!(combo.len(), 4);
    }

    #[test]
    fn dice_statistics_from_rolls() {
        let rolls = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos];
        let stats = DiceStatistics::from_rolls(&rolls);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.neg_count, 1);
        assert_eq!(stats.zero_count, 1);
        assert_eq!(stats.pos_count, 2);
    }

    #[test]
    fn dice_statistics_frequency() {
        let rolls = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Neg];
        let stats = DiceStatistics::from_rolls(&rolls);
        assert!((stats.frequency(Trit::Pos) - 0.75).abs() < 0.001);
        assert!((stats.frequency(Trit::Neg) - 0.25).abs() < 0.001);
    }

    #[test]
    fn dice_statistics_empty() {
        let stats = DiceStatistics::new();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.frequency(Trit::Pos), 0.0);
        assert!(stats.is_balanced(0.1));
        assert_eq!(stats.mode(), None);
    }

    #[test]
    fn dice_statistics_mode() {
        let rolls = vec![Trit::Pos, Trit::Pos, Trit::Neg];
        let stats = DiceStatistics::from_rolls(&rolls);
        assert_eq!(stats.mode(), Some(Trit::Pos));
    }

    #[test]
    fn dice_statistics_sum() {
        let rolls = vec![Trit::Pos, Trit::Neg, Trit::Pos, Trit::Neg];
        let stats = DiceStatistics::from_rolls(&rolls);
        assert_eq!(stats.sum_i8(), 0);
    }

    #[test]
    fn dice_rebalance_balanced() {
        // target = [1/3, 1/3, 1/3], observed = [Neg, Zero, Pos] -> current = [1/3, 1/3, 1/3].
        // Inverse-probability ratio is 1.0 for each outcome, so each weight is
        // exactly `scale * 1.0 = 1000`. (The pre-fix test only asserted > 0,
        // which passed even when the formula was mathematically wrong.)
        let rebalance = DiceRebalance::balanced();
        let stats = DiceStatistics::from_rolls(&[Trit::Pos, Trit::Neg, Trit::Zero]);
        let weights = rebalance.compute_weights(&stats);
        assert_eq!(weights, [1000, 1000, 1000]);
    }

    #[test]
    fn dice_rebalance_custom() {
        // target = [0.5, 0.25, 0.25], observed = [Pos; 10] -> current = [0, 0, 1].
        // Neg and Zero are unobserved -> fall back to scaled target: 500 and 250.
        // Pos: (0.25 / 1.0 * 1000).min(3000) = 250.
        let rebalance = DiceRebalance::custom(0.5, 0.25, 0.25);
        let stats = DiceStatistics::from_rolls(&[Trit::Pos; 10]);
        let weights = rebalance.compute_weights(&stats);
        assert_eq!(weights, [500, 250, 250]);
    }

    #[test]
    fn dice_rebalance_apply() {
        // target = [0.8, 0.1, 0.1], observed = [Pos; 10] -> current = [0, 0, 1].
        // Neg: 0.8 * 1000 = 800 (unobserved fallback). Zero: 0.1 * 1000 = 100.
        // Pos: (0.1 / 1.0 * 1000).min(3000) = 100.
        let rebalance = DiceRebalance::custom(0.8, 0.1, 0.1);
        let stats = DiceStatistics::from_rolls(&[Trit::Pos; 10]);
        let mut dice = Dice::new(42);
        rebalance.rebalance(&stats, &mut dice);
        assert_eq!(dice.weights, [800, 100, 100]);
    }

    #[test]
    fn dice_rebalance_mixed_observed_inverse_probability() {
        // Hand derivation:
        //   target = [0.5, 0.25, 0.25]
        //   rolls   = [Pos, Pos, Pos, Pos, Pos, Neg, Neg]  (total=7, neg=2, zero=0, pos=5)
        //   current = [2/7, 0, 5/7]
        //   i=0 (Neg): (0.5 / (2/7) * 1000) = 0.5 * 7/2 * 1000 = 1750
        //   i=1 (Zero): unobserved -> 0.25 * 1000 = 250
        //   i=2 (Pos): (0.25 / (5/7) * 1000) = 0.25 * 7/5 * 1000 = 350
        // Ratio Neg:Pos should be 1750:350 = 5:1.
        // (Pre-fix code computed target^2/current, which gave 8:2:1 here.)
        let rebalance = DiceRebalance::custom(0.5, 0.25, 0.25);
        let mut rolls = vec![Trit::Pos; 5];
        rolls.extend([Trit::Neg, Trit::Neg]);
        let stats = DiceStatistics::from_rolls(&rolls);
        let weights = rebalance.compute_weights(&stats);
        assert_eq!(weights, [1750, 250, 350]);
        // Cross-check the inverse-probability ratio directly.
        assert_eq!(weights[0] / weights[2], 5);
    }

    #[test]
    fn dice_rebalance_clamps_at_three_x_scale() {
        // target = [1.0, 0.0, 0.0], observed = [Pos; 100] -> current Pos = 1.0.
        // Neg is unobserved -> 1.0 * 1000 = 1000. Pos: (0.0 / 1.0 * 1000) = 0 -> max(1) = 1.
        // Zero: 0 * 1000 = 0 -> max(1) = 1.
        let rebalance = DiceRebalance::custom(1.0, 0.0, 0.0);
        let stats = DiceStatistics::from_rolls(&[Trit::Pos; 100]);
        let weights = rebalance.compute_weights(&stats);
        assert_eq!(weights, [1000, 1, 1]);
    }

    #[test]
    fn dice_rebalance_empty_stats_returns_unit_weights() {
        let rebalance = DiceRebalance::custom(0.7, 0.2, 0.1);
        let stats = DiceStatistics::new();
        assert_eq!(rebalance.compute_weights(&stats), [1, 1, 1]);
    }

    #[test]
    fn fates_table_lookup() {
        let table = FatesTable::new();
        let entry = table.lookup(0).unwrap();
        assert_eq!(entry.severity, Severity::Neutral);
    }

    #[test]
    fn fates_table_evaluate() {
        let table = FatesTable::new();
        let roll = vec![Trit::Pos, Trit::Pos, Trit::Pos]; // sum = 3
        let entry = table.evaluate(&roll).unwrap();
        assert_eq!(entry.severity, Severity::CriticalSuccess);
    }

    #[test]
    fn fates_table_critical_fail() {
        let table = FatesTable::new();
        let roll = vec![Trit::Neg, Trit::Neg, Trit::Neg]; // sum = -3
        let entry = table.evaluate(&roll).unwrap();
        assert_eq!(entry.severity, Severity::CriticalFail);
    }

    #[test]
    fn fates_table_default_entries() {
        let table = FatesTable::new();
        assert_eq!(table.entry_count(), 7);
    }

    #[test]
    fn fates_table_missing_sum_returns_none() {
        // A 4-die all-Pos roll sums to +4, but the default table only covers
        // -3..=+3. Per the README's "Known Limitations", unmatched sums give None.
        let table = FatesTable::new();
        let roll = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos]; // sum = 4
        assert!(table.evaluate(&roll).is_none());
    }

    #[test]
    fn fates_table_custom_entry_is_found() {
        let mut table = FatesTable::new();
        table.add(FatesEntry {
            roll_value: 4,
            outcome: "Beyond extraordinary".into(),
            severity: Severity::CriticalSuccess,
        });
        let roll = vec![Trit::Pos, Trit::Pos, Trit::Pos, Trit::Pos];
        let entry = table.evaluate(&roll).expect("custom +4 entry should match");
        assert_eq!(entry.outcome, "Beyond extraordinary");
    }

    #[test]
    fn fates_table_empty_roll_sums_to_zero() {
        let table = FatesTable::new();
        let entry = table.evaluate(&[]).expect("empty roll sums to 0");
        assert_eq!(entry.severity, Severity::Neutral);
    }

    #[test]
    fn prng_zero_seed_aliases_seed_one() {
        // Documented behaviour: seed 0 is replaced with 1 to dodge the
        // all-zero xorshift32 fixed point.
        let mut a = Prng::new(0);
        let mut b = Prng::new(1);
        for _ in 0..16 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    #[should_panic(expected = "next_range requires max > 0")]
    fn prng_next_range_zero_panics() {
        let mut prng = Prng::new(42);
        let _ = prng.next_range(0);
    }

    #[test]
    fn prng_next_range_respects_bounds() {
        let mut prng = Prng::new(7);
        for _ in 0..1000 {
            let v = prng.next_range(6);
            assert!(v < 6, "next_range(6) returned {v} >= 6");
        }
    }

    #[test]
    fn trit_value_roundtrip() {
        for v in -1..=1i8 {
            let trit = Trit::from_i8(v).expect("valid trit value");
            assert_eq!(trit.value(), v);
        }
    }

    #[test]
    fn trit_from_i8_rejects_out_of_domain() {
        assert_eq!(Trit::from_i8(-2), None);
        assert_eq!(Trit::from_i8(2), None);
        assert_eq!(Trit::from_i8(i8::MIN), None);
        assert_eq!(Trit::from_i8(i8::MAX), None);
    }

    #[test]
    fn dice_extreme_weights_dont_panic() {
        // saturating_add path — would have panicked under plain `+` in debug.
        let mut dice = Dice::with_weights(42, [u32::MAX, u32::MAX, 1]);
        for _ in 0..32 {
            let _ = dice.roll();
        }
    }

    #[test]
    fn dice_max_weight_single_outcome_dominates() {
        // [u32::MAX, 0, 0] -> total = u32::MAX, r in [0, u32::MAX) always < u32::MAX.
        let mut dice = Dice::with_weights(42, [u32::MAX, 0, 0]);
        for _ in 0..50 {
            assert_eq!(dice.roll(), Trit::Neg);
        }
    }

    #[test]
    fn dice_statistics_is_balanced_exact() {
        // [Neg, Zero, Pos] -> frequencies exactly 1/3 each.
        let stats = DiceStatistics::from_rolls(&[Trit::Neg, Trit::Zero, Trit::Pos]);
        assert!(stats.is_balanced(0.0));
    }

    #[test]
    fn dice_statistics_is_balanced_rejects_skew() {
        // [Pos, Pos, Pos] -> frequencies (0, 0, 1), each off by >= 2/3 from 1/3.
        let stats = DiceStatistics::from_rolls(&[Trit::Pos, Trit::Pos, Trit::Pos]);
        assert!(!stats.is_balanced(0.0));
        // ...but tolerant enough to pass a loose threshold.
        assert!(stats.is_balanced(0.9));
    }

    #[test]
    fn dice_statistics_mode_tie_returns_pos_last() {
        // [Pos, Neg] -> counts [(Neg,1), (Zero,0), (Pos,1)]. Iterator::max_by_key
        // returns the LAST maximum, so on a Neg/Pos tie the winner is Pos.
        // Documents the tie-break contract.
        let stats = DiceStatistics::from_rolls(&[Trit::Pos, Trit::Neg]);
        assert_eq!(stats.mode(), Some(Trit::Pos));
    }

    #[test]
    fn dice_statistics_sum_extremes() {
        let all_pos = DiceStatistics::from_rolls(&[Trit::Pos; 5]);
        assert_eq!(all_pos.sum_i8(), 5);
        let all_neg = DiceStatistics::from_rolls(&[Trit::Neg; 5]);
        assert_eq!(all_neg.sum_i8(), -5);
    }

    #[test]
    fn dice_set_empty_handling() {
        let mut set = DiceSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.roll_all(), Vec::<Trit>::new());
        assert!(set.roll_one(0).is_none());
    }

    #[test]
    fn dice_set_default_matches_new() {
        let set = DiceSet::default();
        assert!(set.is_empty());
    }

    #[test]
    fn dice_statistics_default_matches_new() {
        let stats = DiceStatistics::default();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn fates_table_default_matches_new() {
        assert_eq!(
            FatesTable::default().entry_count(),
            FatesTable::new().entry_count()
        );
    }

    #[test]
    fn dice_roller_zero_dice_yields_empty_combos() {
        let roller = DiceRoller::new(0, 3);
        let results = roller.generate(42);
        assert_eq!(results.len(), 3);
        for combo in &results {
            assert!(combo.is_empty());
        }
    }

    #[test]
    fn dice_roller_zero_rolls_yields_empty_vec() {
        let roller = DiceRoller::new(3, 0);
        let results = roller.generate(42);
        assert!(results.is_empty());
    }

    #[test]
    fn prng_next_range_uniform_distribution() {
        // Loose sanity check: over many samples, next_range(3) should hit each
        // bucket roughly 1/3 of the time. xorshift32 has known biases so we
        // use a generous tolerance (±3 percentage points).
        let mut prng = Prng::new(12345);
        let n = 60_000u32;
        let mut counts = [0u32; 3];
        for _ in 0..n {
            counts[prng.next_range(3) as usize] += 1;
        }
        let expected = n as f64 / 3.0;
        for c in counts {
            let deviation = (c as f64 - expected).abs() / expected;
            assert!(
                deviation < 0.03,
                "count {c} deviates {deviation:.4} > 0.03 from 1/3"
            );
        }
    }
}
