"""Settlement data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Settlement:
    """A Norse settlement on the map."""

    x: int
    y: int
    owner_id: int
    population: int = 50
    food: int = 100
    wealth: int = 0
    defense: int = 10
    tech_level: int = 1
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
        return self.food < 30

    @property
    def strength(self) -> int:
        """Combat strength for raiding."""
        return self.population + self.defense + self.tech_level * 5

    def produce_food(self, adjacent_forest_count: int) -> None:
        """Generate food from adjacent forests and base production."""
        base = 10 + self.tech_level * 2
        forest_bonus = adjacent_forest_count * 15
        self.food += base + forest_bonus

    def consume_food(self, winter_severity: float) -> None:
        """Consume food during winter. Severity in [0.5, 1.5]."""
        consumption = int(self.population * 0.4 * winter_severity)
        self.food -= consumption

    def grow(self) -> None:
        """Population growth when food is sufficient."""
        if self.food > self.population * 2:
            growth = max(1, self.population // 10)
            self.population += growth
            self.defense += 1

    def take_raid_damage(self, damage: int) -> None:
        """Accumulate raid damage for this year."""
        self.raid_damage += damage
        self.population = max(0, self.population - damage // 3)
        self.food = max(0, self.food - damage // 2)

    def should_collapse(self) -> bool:
        """Check if settlement should become a ruin."""
        if self.population <= 0:
            return True
        if self.food <= 0 and self.population < 10:
            return True
        return self.raid_damage > self.strength * 2

    def reset_yearly(self) -> None:
        """Reset per-year accumulators."""
        self.raid_damage = 0
