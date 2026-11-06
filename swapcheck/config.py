"""Configuration management for SwapCheck."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Specification for an LLM provider/model configuration.

    Model identifiers and vendor configs live exclusively in external YAML configs.
    """

    provider: str = Field(default="mock", description="Provider backend type (mock, litellm, etc.)")
    name: str = Field(
        ..., description="Human-readable configuration name (e.g. 'baseline', 'candidate')"
    )
    model: str = Field(
        ..., description="Provider model identifier passed to LiteLLM or mock identifier"
    )
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, ge=1, description="Max tokens for completion")
    api_key_env_var: str | None = Field(
        None, description="Environment variable name holding the API key"
    )
    cost_per_input_token: float | None = Field(
        None, ge=0.0, description="Optional manual input token cost override in USD"
    )
    cost_per_output_token: float | None = Field(
        None, ge=0.0, description="Optional manual output token cost override in USD"
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ModelConfig:
        """Load an external YAML configuration file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Model config '{config_path}' must contain a mapping at the top level."
            )
        return cls.model_validate(raw)

    @classmethod
    def from_yaml_text(cls, text: str) -> ModelConfig:
        """Load a model config from YAML text."""
        raw = yaml.safe_load(text) or {}
        if not isinstance(raw, dict):
            raise ValueError("Model config text must contain a mapping at the top level.")
        return cls.model_validate(raw)


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_prefix="SWAPCHECK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="sqlite:///swapcheck.db",
        description="Database connection string (SQLite or PostgreSQL)",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Whether LLM response caching is enabled",
    )
    cache_db_path: Path = Field(
        default=Path(".swapcheck_cache.db"),
        description="SQLite file path used for LLM response caching",
    )
    default_bootstrap_resamples: int = Field(
        default=2000,
        ge=100,
        description="Number of bootstrap resamples for confidence interval estimation",
    )
    default_overall_margin: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Default non-inferiority margin epsilon for overall pass rate (5%)",
    )
    default_category_margin: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Default non-inferiority margin epsilon for per-category pass rate (10%)",
    )
    min_scenarios_overall: int = Field(
        default=50,
        ge=5,
        description="Minimum scenarios required for a conclusive overall verdict",
    )
    min_scenarios_per_category: int = Field(
        default=8,
        ge=3,
        description="Minimum scenarios per category required before category is powered",
    )
    default_runs_per_scenario: int = Field(
        default=3,
        ge=1,
        description="Default number of repeated runs per scenario to capture non-determinism",
    )
    log_level: str = Field(
        default="INFO",
        description="Standard log level (DEBUG, INFO, WARNING, ERROR)",
    )


settings = Settings()


def load_model_config(path: str | Path) -> ModelConfig:
    """Convenience loader for external model configuration files."""
    return ModelConfig.from_yaml_file(path)
