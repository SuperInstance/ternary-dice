#![forbid(unsafe_code)]

//! Stochastic exploration with configurable randomness for balanced ternary systems.
//!
//! Provides dice-based random generation over the ternary domain {-1, 0, +1},
//! with configurable distributions, statistics tracking, probability rebalancing,
//! and a FatesTable for D&D-style lookup outcomes. All randomness is deterministic
//! based on a provided seed/state — no external RNG dependency needed.

// No external dependencies needed.

/// A single balanced ternary value: -1, 0, or +1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trit {
    Neg,
    Zero,
    Pos,
}

impl Trit {
    pub fn value(self) -> i8 {
        match self {
            Trit::Neg => -1,
            Trit::Zero => 0,
            Trit::Pos => 1,
        }
    }

    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::Neg),
            0 => Some(Trit::Zero),
            1 => Some(Trit::Pos),
            _ => None,
        }
    }
}

/// Simple deterministic PRNG (xorshift32) for reproducible randomness.
#[derive(Clone, Debug)]
pub struct Prng {
    state: u32,
}

impl Prng {
    pub fn new(seed: u32) -> Self {
        // Ensure non-zero state
        Prng { state: if seed == 0 { 1 } else { seed } }
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Returns a value in [0, max) using modular reduction.
    pub fn next_range(&mut self, max: u32) -> u32 {
        self.next_u32() % max
    }
}

/// A configurable ternary dice with weighted probabilities.
#[derive(Clone, Debug)]
pub struct Dice {
    /// Probability weights: [neg_weight, zero_weight, pos_weight]. Default: [1, 1, 1].
    pub weights: [u32; 3],
    prng: Prng,
}

impl Dice {
    pub fn new(seed: u32) -> Self {
        Dice { weights: [1, 1, 1], prng: Prng::new(seed) }
    }

    pub fn with_weights(seed: u32, weights: [u32; 3]) -> Self {
        Dice { weights, prng: Prng::new(seed) }
    }

    pub fn roll(&mut self) -> Trit {
        let total = self.weights[0] + self.weights[1] + self.weights[2];
        if total == 0 {
            return Trit::Zero;
        }
        let r = self.prng.next_range(total);
        if r < self.weights[0] {
            Trit::Neg
        } else if r < self.weights[0] + self.weights[1] {
            Trit::Zero
        } else {
            Trit::Pos
        }
    }

    pub fn roll_n(&mut self, count: usize) -> Vec<Trit> {
        (0..count).map(|_| self.roll()).collect()
    }

    /// Set weights to favor a specific trit (weight 3 for target, 1 for others).
    pub fn bias_toward(&mut self, target: Trit) {
        self.weights = match target {
            Trit::Neg => [3, 1, 1],
            Trit::Zero => [1, 3, 1],
            Trit::Pos => [1, 1, 3],
        };
    }
}

/// A set of multiple dice with different distributions.
#[derive(Clone, Debug)]
pub struct DiceSet {
    dice: Vec<Dice>,
}

impl DiceSet {
    pub fn new() -> Self {
        DiceSet { dice: Vec::new() }
    }

    pub fn add(&mut self, dice: Dice) -> usize {
        let idx = self.dice.len();
        self.dice.push(dice);
        idx
    }

    /// Roll all dice, returning one result per die.
    pub fn roll_all(&mut self) -> Vec<Trit> {
        self.dice.iter_mut().map(|d| d.roll()).collect()
    }

    /// Roll a specific die by index.
    pub fn roll_one(&mut self, index: usize) -> Option<Trit> {
        self.dice.get_mut(index).map(|d| d.roll())
    }

    pub fn len(&self) -> usize {
        self.dice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dice.is_empty()
    }
}

impl Default for DiceSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates combinations from multiple dice rolls.
#[derive(Clone, Debug)]
pub struct DiceRoller {
    pub dice_count: usize,
    pub rolls_per_die: usize,
}

impl DiceRoller {
    pub fn new(dice_count: usize, rolls_per_die: usize) -> Self {
        DiceRoller { dice_count, rolls_per_die }
    }

    /// Generate all combinations by rolling the dice set multiple times.
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

    /// Generate a single combination (all dice rolled once).
    pub fn roll_once(&self, seed: u32) -> Vec<Trit> {
        let mut set = DiceSet::new();
        for i in 0..self.dice_count {
            set.add(Dice::new(seed.wrapping_add(i as u32 * 7919)));
        }
        set.roll_all()
    }
}

/// Analyze roll distributions.
#[derive(Clone, Debug)]
pub struct DiceStatistics {
    pub neg_count: usize,
    pub zero_count: usize,
    pub pos_count: usize,
    pub total: usize,
}

impl DiceStatistics {
    pub fn new() -> Self {
        DiceStatistics { neg_count: 0, zero_count: 0, pos_count: 0, total: 0 }
    }

    pub fn from_rolls(rolls: &[Trit]) -> Self {
        let mut stats = DiceStatistics::new();
        for &trit in rolls {
            stats.record(trit);
        }
        stats
    }

    pub fn record(&mut self, trit: Trit) {
        self.total += 1;
        match trit {
            Trit::Neg => self.neg_count += 1,
            Trit::Zero => self.zero_count += 1,
            Trit::Pos => self.pos_count += 1,
        }
    }

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

    pub fn mode(&self) -> Option<Trit> {
        if self.total == 0 {
            return None;
        }
        let counts = [(Trit::Neg, self.neg_count), (Trit::Zero, self.zero_count), (Trit::Pos, self.pos_count)];
        let max = counts.iter().max_by_key(|&&(_, c)| c).unwrap();
        Some(max.0)
    }

    pub fn sum_i8(&self) -> i32 {
        (self.neg_count as i32 * -1) + (self.pos_count as i32)
    }
}

impl Default for DiceStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Adjusts dice probabilities for fairness or novelty.
#[derive(Clone, Debug)]
pub struct DiceRebalance {
    pub target_distribution: [f64; 3], // [neg, zero, pos] target frequencies
}

impl DiceRebalance {
    pub fn balanced() -> Self {
        DiceRebalance { target_distribution: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0] }
    }

    pub fn custom(neg: f64, zero: f64, pos: f64) -> Self {
        DiceRebalance { target_distribution: [neg, zero, pos] }
    }

    /// Compute new weights from current statistics to achieve target distribution.
    /// Returns [neg_weight, zero_weight, pos_weight] as u32.
    pub fn compute_weights(&self, stats: &DiceStatistics) -> [u32; 3] {
        if stats.total == 0 {
            return [1, 1, 1];
        }

        // Scale target distribution to integer weights (multiply by 1000 for precision)
        let scale = 1000.0;
        let w: Vec<u32> = self.target_distribution
            .iter()
            .map(|&t| (t * scale) as u32)
            .collect();

        // If current distribution deviates, adjust by inverse ratio
        let current = [
            stats.frequency(Trit::Neg),
            stats.frequency(Trit::Zero),
            stats.frequency(Trit::Pos),
        ];

        let adjusted: Vec<u32> = (0..3)
            .map(|i| {
                if current[i] > 0.001 {
                    let ratio = self.target_distribution[i] / current[i];
                    ((w[i] as f64 * ratio).min(scale * 3.0)) as u32
                } else {
                    w[i]
                }
            })
            .collect();

        [adjusted[0].max(1), adjusted[1].max(1), adjusted[2].max(1)]
    }

    /// Apply rebalanced weights to a dice.
    pub fn rebalance(&self, stats: &DiceStatistics, dice: &mut Dice) {
        dice.weights = self.compute_weights(stats);
    }
}

/// A D&D-style lookup table mapping roll outcomes to narrative results.
#[derive(Clone, Debug)]
pub struct FatesTable {
    entries: Vec<FatesEntry>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FatesEntry {
    pub roll_value: i8,    // sum of trit values for the roll
    pub outcome: String,
    pub severity: Severity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Severity {
    CriticalFail,
    Fail,
    Neutral,
    Success,
    CriticalSuccess,
}

impl FatesTable {
    pub fn new() -> Self {
        let mut table = FatesTable { entries: Vec::new() };
        // Default D&D-style outcomes based on sum of trits
        table.add(FatesEntry { roll_value: -3, outcome: "Catastrophic failure".into(), severity: Severity::CriticalFail });
        table.add(FatesEntry { roll_value: -2, outcome: "Major setback".into(), severity: Severity::Fail });
        table.add(FatesEntry { roll_value: -1, outcome: "Minor setback".into(), severity: Severity::Fail });
        table.add(FatesEntry { roll_value: 0, outcome: "Status quo".into(), severity: Severity::Neutral });
        table.add(FatesEntry { roll_value: 1, outcome: "Minor breakthrough".into(), severity: Severity::Success });
        table.add(FatesEntry { roll_value: 2, outcome: "Major success".into(), severity: Severity::Success });
        table.add(FatesEntry { roll_value: 3, outcome: "Extraordinary triumph".into(), severity: Severity::CriticalSuccess });
        table
    }

    pub fn add(&mut self, entry: FatesEntry) {
        self.entries.push(entry);
    }

    /// Look up an outcome by the sum of a roll.
    pub fn lookup(&self, roll_sum: i8) -> Option<&FatesEntry> {
        // Find closest match
        self.entries
            .iter()
            .filter(|e| e.roll_value == roll_sum)
            .min_by_key(|e| (e.roll_value - roll_sum).abs())
    }

    /// Evaluate a roll (sequence of trits) against the table.
    pub fn evaluate(&self, roll: &[Trit]) -> Option<&FatesEntry> {
        let sum: i8 = roll.iter().map(|t| t.value()).sum();
        self.lookup(sum)
    }

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
        let rebalance = DiceRebalance::balanced();
        let stats = DiceStatistics::from_rolls(&[Trit::Pos, Trit::Neg, Trit::Zero]);
        let weights = rebalance.compute_weights(&stats);
        // Already balanced, weights should be roughly equal
        assert!(weights[0] > 0);
        assert!(weights[1] > 0);
        assert!(weights[2] > 0);
    }

    #[test]
    fn dice_rebalance_custom() {
        let rebalance = DiceRebalance::custom(0.5, 0.25, 0.25);
        let stats = DiceStatistics::from_rolls(&[Trit::Pos; 10]);
        let weights = rebalance.compute_weights(&stats);
        // Should bias toward Neg to compensate
        assert!(weights[0] > weights[2]);
    }

    #[test]
    fn dice_rebalance_apply() {
        let rebalance = DiceRebalance::custom(0.8, 0.1, 0.1);
        let stats = DiceStatistics::from_rolls(&[Trit::Pos; 10]);
        let mut dice = Dice::new(42);
        rebalance.rebalance(&stats, &mut dice);
        assert!(dice.weights[0] > dice.weights[2]);
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
}
