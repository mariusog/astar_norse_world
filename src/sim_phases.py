"""Simulation phase functions -- growth, conflict, trade, winter, environment."""

from __future__ import annotations

import numpy as np

from src.constants import (
    CONQUEST_PROB,
    DESPERATE_RAID_BONUS,
    EXPANSION_NEW_POPULATION,
    EXPANSION_POPULATION_THRESHOLD,
    LONGSHIP_BUILD_THRESHOLD,
    LONGSHIP_RAID_RANGE,
    PORT_DEVELOPMENT_THRESHOLD,
    RAID_DAMAGE_BASE,
    RAID_LOOT_FRACTION,
    RAID_RANGE,
    REBUILD_INHERIT_TECH_FRACTION,
    REBUILD_POPULATION,
    REFUGEE_FRACTION,
    REFUGEE_RANGE,
    RUIN_REBUILD_PROB,
    RUIN_REBUILD_RANGE,
    RUIN_RECLAIM_AS_FOREST_PROB,
    RUIN_TO_PLAINS_PROB,
    TECH_DIFFUSION_PROB,
    TRADE_FOOD_EXCHANGE,
    TRADE_RANGE,
    TRADE_WEALTH_GAIN,
    WINTER_SEVERITY_RANGE,
)
from src.settlement import Settlement
from src.terrain import InternalTerrain, neighbors_4, neighbors_8


def _reset_yearly(settlements: list[Settlement]) -> None:
    for s in settlements:
        if s.alive:
            s.reset_yearly()


def _alive(settlements: list[Settlement]) -> list[Settlement]:
    return [s for s in settlements if s.alive]


def _distance(a: Settlement, b: Settlement) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def _phase_growth(
    grid: np.ndarray,
    settlements: list[Settlement],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Produce food, grow population, develop ports, build ships, expand."""
    for s in _alive(settlements):
        # Count adjacent forests
        adj_forests = sum(
            1
            for nx, ny in neighbors_8(s.x, s.y, width, height)
            if grid[ny, nx] == InternalTerrain.FOREST
        )
        s.produce_food(adj_forests)
        s.grow()

        # Port development: coastal settlement with enough population
        if not s.is_port and s.population >= PORT_DEVELOPMENT_THRESHOLD:
            is_coastal = any(
                grid[ny, nx] == InternalTerrain.OCEAN
                for nx, ny in neighbors_4(s.x, s.y, width, height)
            )
            if is_coastal:
                s.is_port = True
                grid[s.y, s.x] = InternalTerrain.PORT

        # Longship construction
        if s.is_port and not s.has_longship and s.population >= LONGSHIP_BUILD_THRESHOLD:
            s.has_longship = True

        # Expansion: found new settlement on nearby land
        if s.population >= EXPANSION_POPULATION_THRESHOLD:
            _try_expand(grid, settlements, s, width, height, rng)


def _try_expand(
    grid: np.ndarray,
    settlements: list[Settlement],
    parent: Settlement,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Attempt to found a new settlement on nearby plains."""
    # Search within range 2-3 for empty plains
    candidates = []
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx == 0 and dy == 0:
                continue
            nx, ny = parent.x + dx, parent.y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == InternalTerrain.PLAINS:
                # Not occupied by another settlement
                occupied = any(s.x == nx and s.y == ny for s in _alive(settlements))
                if not occupied:
                    candidates.append((nx, ny))

    if not candidates:
        return

    idx = rng.integers(0, len(candidates))
    nx, ny = candidates[idx]

    # Check if coastal
    is_coastal = any(
        grid[cy, cx] == InternalTerrain.OCEAN for cx, cy in neighbors_4(nx, ny, width, height)
    )

    new_settlement = Settlement(
        x=nx,
        y=ny,
        owner_id=parent.owner_id,
        population=EXPANSION_NEW_POPULATION,
        food=parent.food // 4,
        tech_level=max(1, parent.tech_level - 1),
        is_port=is_coastal,
    )
    settlements.append(new_settlement)
    grid[ny, nx] = InternalTerrain.PORT if is_coastal else InternalTerrain.SETTLEMENT
    parent.population -= EXPANSION_NEW_POPULATION


# Phase 2: Conflict
def _phase_conflict(
    grid: np.ndarray,
    settlements: list[Settlement],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Settlements raid each other."""
    alive = _alive(settlements)
    for attacker in alive:
        if not attacker.alive:
            continue
        raid_range = LONGSHIP_RAID_RANGE if attacker.has_longship else RAID_RANGE

        # Find targets within range (different faction)
        targets = [
            t
            for t in alive
            if t.alive and t.owner_id != attacker.owner_id and _distance(attacker, t) <= raid_range
        ]
        if not targets:
            continue

        # Desperate settlements always raid; others raid probabilistically
        raid_prob = 0.7 if attacker.is_desperate else 0.3
        if rng.random() > raid_prob:
            continue

        # Pick target (prefer weaker)
        target = min(targets, key=lambda t: t.strength)

        # Calculate damage
        damage = RAID_DAMAGE_BASE + attacker.tech_level * 3
        if attacker.is_desperate:
            damage = int(damage * DESPERATE_RAID_BONUS)

        # Attacker wins if stronger
        if attacker.strength > target.strength:
            target.take_raid_damage(damage)
            # Loot
            loot_food = int(target.food * RAID_LOOT_FRACTION)
            loot_wealth = int(target.wealth * RAID_LOOT_FRACTION)
            attacker.food += loot_food
            attacker.wealth += loot_wealth
            target.food -= loot_food
            target.wealth -= loot_wealth

            # Possible conquest
            if rng.random() < CONQUEST_PROB:
                target.owner_id = attacker.owner_id
        else:
            # Failed raid -- attacker takes some damage
            attacker.take_raid_damage(damage // 3)


# Phase 3: Trade
def _phase_trade(
    settlements: list[Settlement],
    rng: np.random.Generator,
) -> None:
    """Ports within range trade if not at war (different factions don't trade)."""
    ports = [s for s in _alive(settlements) if s.is_port]

    for i, a in enumerate(ports):
        for b in ports[i + 1 :]:
            if a.owner_id != b.owner_id:
                continue  # at war
            if _distance(a, b) > TRADE_RANGE:
                continue

            # Exchange food and gain wealth
            a.food += TRADE_FOOD_EXCHANGE
            b.food += TRADE_FOOD_EXCHANGE
            a.wealth += TRADE_WEALTH_GAIN
            b.wealth += TRADE_WEALTH_GAIN

            # Tech diffusion
            if a.tech_level != b.tech_level and rng.random() < TECH_DIFFUSION_PROB:
                higher = max(a.tech_level, b.tech_level)
                a.tech_level = higher
                b.tech_level = higher


# Phase 4: Winter
def _phase_winter(
    grid: np.ndarray,
    settlements: list[Settlement],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Winter food consumption and settlement collapse."""
    severity = rng.uniform(WINTER_SEVERITY_RANGE[0], WINTER_SEVERITY_RANGE[1])

    alive = _alive(settlements)
    collapsed: list[Settlement] = []

    for s in alive:
        s.consume_food(severity)

        if s.should_collapse():
            s.alive = False
            grid[s.y, s.x] = InternalTerrain.RUIN
            collapsed.append(s)

    # Refugees flee to nearby friendly settlements
    for dead in collapsed:
        refugees = int(dead.population * REFUGEE_FRACTION)
        if refugees <= 0:
            continue
        nearby_friendly = [
            s
            for s in _alive(settlements)
            if s.owner_id == dead.owner_id and _distance(dead, s) <= REFUGEE_RANGE
        ]
        if nearby_friendly:
            dest = nearby_friendly[rng.integers(0, len(nearby_friendly))]
            dest.population += refugees


# Phase 5: Environment
def _phase_environment(
    grid: np.ndarray,
    settlements: list[Settlement],
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Ruins decay into forest/plains or get rebuilt by nearby settlements."""
    ruin_positions = list(zip(*np.where(grid == InternalTerrain.RUIN), strict=True))

    for ry, rx in ruin_positions:
        # Check if a thriving settlement can rebuild
        rebuilt = False
        nearby = [
            s
            for s in _alive(settlements)
            if abs(s.x - rx) + abs(s.y - ry) <= RUIN_REBUILD_RANGE
            and s.population >= EXPANSION_POPULATION_THRESHOLD // 2
        ]
        if nearby and rng.random() < RUIN_REBUILD_PROB:
            parent = nearby[rng.integers(0, len(nearby))]
            is_coastal = any(
                grid[ny, nx] == InternalTerrain.OCEAN
                for nx, ny in neighbors_4(rx, ry, width, height)
            )
            new_s = Settlement(
                x=rx,
                y=ry,
                owner_id=parent.owner_id,
                population=REBUILD_POPULATION,
                food=parent.food // 5,
                tech_level=max(1, int(parent.tech_level * REBUILD_INHERIT_TECH_FRACTION)),
                is_port=is_coastal,
            )
            settlements.append(new_s)
            grid[ry, rx] = InternalTerrain.PORT if is_coastal else InternalTerrain.SETTLEMENT
            rebuilt = True

        if not rebuilt:
            # Natural reclamation
            roll = rng.random()
            if roll < RUIN_RECLAIM_AS_FOREST_PROB:
                grid[ry, rx] = InternalTerrain.FOREST
            elif roll < RUIN_RECLAIM_AS_FOREST_PROB + RUIN_TO_PLAINS_PROB:
                grid[ry, rx] = InternalTerrain.PLAINS
