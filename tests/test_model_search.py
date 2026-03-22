"""Tests for web.model_search strategy generation."""

from __future__ import annotations

from web.model_search import SearchConfig, generate_search_strategies


class TestGenerateSearchStrategies:
    def test_returns_nonempty_list(self) -> None:
        configs = generate_search_strategies()
        assert len(configs) > 0

    def test_generates_at_least_50_strategies(self) -> None:
        configs = generate_search_strategies()
        assert len(configs) >= 50, f"Expected 50+, got {len(configs)}"

    def test_all_have_unique_names(self) -> None:
        configs = generate_search_strategies()
        names = [c.name for c in configs]
        assert len(names) == len(set(names))

    def test_all_configs_are_search_config(self) -> None:
        configs = generate_search_strategies()
        for c in configs:
            assert isinstance(c, SearchConfig)

    def test_all_have_xgboost_base(self) -> None:
        configs = generate_search_strategies()
        for c in configs:
            assert c.base == "xgboost"

    def test_all_have_transforms(self) -> None:
        configs = generate_search_strategies()
        for c in configs:
            assert len(c.transforms) >= 1
            for name, params in c.transforms:
                assert isinstance(name, str)
                assert isinstance(params, dict)

    def test_includes_temperature_variants(self) -> None:
        configs = generate_search_strategies()
        temp_names = [c.name for c in configs if c.name.startswith("temperature_")]
        assert len(temp_names) >= 5

    def test_includes_power_variants(self) -> None:
        configs = generate_search_strategies()
        power_names = [c.name for c in configs if c.name.startswith("power_")]
        assert len(power_names) >= 5

    def test_includes_combo_strategies(self) -> None:
        configs = generate_search_strategies()
        combos = [c for c in configs if len(c.transforms) >= 2]
        assert len(combos) >= 5
