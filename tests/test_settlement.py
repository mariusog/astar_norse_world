"""Tests for Settlement data model."""

from src.settlement import Settlement


class TestSettlementProperties:
    def test_pos_returns_tuple(self) -> None:
        s = Settlement(x=3, y=7, owner_id=0)
        assert s.pos == (3, 7)

    def test_is_desperate_when_low_food(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, food=10)
        assert s.is_desperate is True

    def test_not_desperate_when_food_ok(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, food=100)
        assert s.is_desperate is False

    def test_strength_includes_population_defense_tech(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, population=50, defense=10, tech_level=2)
        assert s.strength == 50 + 10 + 2 * 5


class TestSettlementFood:
    def test_produce_food_with_forests(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, food=0, tech_level=1)
        s.produce_food(adjacent_forest_count=2)
        # base=10 + tech*2=2 + forests*15=30 = 42
        assert s.food == 42

    def test_consume_food_normal_winter(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, population=100, food=200)
        s.consume_food(winter_severity=1.0)
        # consumption = int(100 * 0.4 * 1.0) = 40
        assert s.food == 160


class TestSettlementCombat:
    def test_take_raid_damage_reduces_pop_and_food(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, population=50, food=100)
        s.take_raid_damage(30)
        assert s.population == 40  # 50 - 30//3
        assert s.food == 85  # 100 - 30//2

    def test_should_collapse_when_no_population(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, population=0, food=50)
        assert s.should_collapse() is True

    def test_should_collapse_when_starving_small(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, population=5, food=0)
        assert s.should_collapse() is True

    def test_should_not_collapse_healthy(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0, population=50, food=100)
        assert s.should_collapse() is False


class TestGrow:
    def test_grows_when_food_exceeds_threshold(self) -> None:
        # food > population * GROWTH_FOOD_THRESHOLD_MULTIPLIER (2)
        s = Settlement(x=0, y=0, owner_id=0, population=50, food=200)
        initial_pop = s.population
        initial_defense = s.defense
        s.grow()
        # growth = max(1, int(50 * 0.1)) = 5
        assert s.population == initial_pop + 5
        assert s.defense == initial_defense + 1

    def test_no_growth_when_food_insufficient(self) -> None:
        # food <= population * 2 -> no growth
        s = Settlement(x=0, y=0, owner_id=0, population=50, food=50)
        initial_pop = s.population
        initial_defense = s.defense
        s.grow()
        assert s.population == initial_pop
        assert s.defense == initial_defense

    def test_no_growth_at_exact_threshold(self) -> None:
        # food == population * 2 -> not greater, so no growth
        s = Settlement(x=0, y=0, owner_id=0, population=50, food=100)
        initial_pop = s.population
        s.grow()
        assert s.population == initial_pop

    def test_minimum_growth_is_one(self) -> None:
        # population=5, growth = max(1, int(5 * 0.1)) = max(1, 0) = 1
        s = Settlement(x=0, y=0, owner_id=0, population=5, food=100)
        initial_pop = s.population
        s.grow()
        assert s.population == initial_pop + 1


class TestResetYearly:
    def test_resets_raid_damage(self) -> None:
        s = Settlement(x=0, y=0, owner_id=0)
        s.raid_damage = 50
        s.reset_yearly()
        assert s.raid_damage == 0
