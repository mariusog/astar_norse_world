"""Settlement data model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.constants import (
    BASE_FOOD_PRODUCTION,
    COLLAPSE_RAID_DAMAGE_MULTIPLIER,
    FOOD_CONSUMPTION_RATE,
    FOOD_PER_FOREST,
    GROWTH_FOOD_THRESHOLD_MULTIPLIER,
    GROWTH_RATE,
    INITIAL_DEFENSE,
    INITIAL_FOOD,
    INITIAL_POPULATION,
    INITIAL_TECH_LEVEL,
    INITIAL_WEALTH,
    STARVATION_COLLAPSE_POP,
    TECH_FOOD_BONUS,
)

DESPERATE_FOOD_THRESHOLD = 30
STRENGTH_TECH_MULTIPLIER = 5
RAID_POP_DIVISOR = 3
RAID_FOOD_DIVISOR = 2


@dataclass
class Settlement:
    """A Norse settlement on the map."""

    x: int
    y: int
    owner_id: int
    population: int = INITIAL_POPULATION
    food: int = INITIAL_FOOD
    wealth: int = INITIAL_WEALTH
    defense: int = INITIAL_DEFENSE
    tech_level: int = INITIAL_TECH_LEVEL
    is_port: bool = False
    has_longship: bool = False
    alive: bool = True

    # Track raiding damage this year (reset each year)
    raid_damage: int = field(default=0, repr=False)

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def is_desperate(self) -> bool:
        """Low food triggers aggressive raiding."""
        return self.food < DESPERATE_FOOD_THRESHOLD

    @property
    def strength(self) -> int:
        """Combat strength for raiding."""
        return self.population + self.defense + self.tech_level * STRENGTH_TECH_MULTIPLIER

    def produce_food(self, adjacent_forest_count: int) -> None:
        """Generate food from adjacent forests and base production."""
        base = BASE_FOOD_PRODUCTION + self.tech_level * TECH_FOOD_BONUS
        forest_bonus = adjacent_forest_count * FOOD_PER_FOREST
        self.food += base + forest_bonus

    def consume_food(self, winter_severity: float) -> None:
        """Consume food during winter. Severity in [0.5, 1.5]."""
        consumption = int(self.population * FOOD_CONSUMPTION_RATE * winter_severity)
        self.food -= consumption

    def grow(self) -> None:
        """Population growth when food is sufficient."""
        if self.food > self.population * GROWTH_FOOD_THRESHOLD_MULTIPLIER:
            growth = max(1, int(self.population * GROWTH_RATE))
            self.population += growth
            self.defense += 1

    def take_raid_damage(self, damage: int) -> None:
        """Accumulate raid damage for this year."""
        self.raid_damage += damage
        self.population = max(0, self.population - damage // RAID_POP_DIVISOR)
        self.food = max(0, self.food - damage // RAID_FOOD_DIVISOR)

    def should_collapse(self) -> bool:
        """Check if settlement should become a ruin."""
        if self.population <= 0:
            return True
        if self.food <= 0 and self.population < STARVATION_COLLAPSE_POP:
            return True
        return self.raid_damage > self.strength * COLLAPSE_RAID_DAMAGE_MULTIPLIER

    def reset_yearly(self) -> None:
        """Reset per-year accumulators."""
        self.raid_damage = 0

    def copy(self) -> Settlement:
        """Return an independent copy of this settlement."""
        return replace(self)
