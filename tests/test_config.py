"""Tests for SwapCheck configuration management."""

from swapcheck.config import ModelConfig, Settings


def test_model_config_initialization() -> None:
    """Test ModelConfig validation and fields."""
    mc = ModelConfig(
        name="premium",
        model="vendor/sample-model",
        temperature=0.7,
        max_tokens=1000,
        cost_per_input_token=0.000003,
        cost_per_output_token=0.000015,
        extra_params={"top_p": 0.9},
    )

    assert mc.name == "premium"
    assert mc.model == "vendor/sample-model"
    assert mc.temperature == 0.7
    assert mc.max_tokens == 1000
    assert mc.cost_per_input_token == 0.000003
    assert mc.cost_per_output_token == 0.000015
    assert mc.extra_params["top_p"] == 0.9


def test_settings_defaults() -> None:
    """Test default values of application Settings."""
    s = Settings()
    assert s.cache_enabled is True
    assert s.default_bootstrap_resamples == 2000
    assert s.default_overall_margin == 0.05
    assert s.default_category_margin == 0.10
    assert s.min_scenarios_overall == 50
    assert s.min_scenarios_per_category == 8
    assert s.default_runs_per_scenario == 3

