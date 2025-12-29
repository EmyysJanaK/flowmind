"""
Configuration module for Sentinel.
"""

import os
from typing import Optional


class Config:
    """Base configuration class."""

    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = "/api/v1"

    # LLM Configuration
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    LOG_LEVEL = "WARNING"


def get_config() -> Config:
    """
    Get configuration based on environment.

    Returns:
        Configuration object.
    """
    env = os.getenv("ENVIRONMENT", "development")
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
