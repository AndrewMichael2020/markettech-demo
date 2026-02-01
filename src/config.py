"""Configuration management for MarketTech application.

This module handles all environment variables and configuration settings,
providing a single source of truth for application configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: Optional[str]

    # GCP Configuration
    gcp_project_id: Optional[str]
    gcp_region: Optional[str]
    artifact_repo: Optional[str]
    cloud_run_service: Optional[str]

    # Application Configuration
    port: int
    environment: str

    @classmethod
    def from_env(cls) -> Config:
        """Load configuration from environment variables.

        Returns:
            Config: Configuration object with values from environment.
        """
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gcp_project_id=os.getenv("GCP_PROJECT_ID"),
            gcp_region=os.getenv("GCP_REGION", "us-central1"),
            artifact_repo=os.getenv("ARTIFACT_REPO"),
            cloud_run_service=os.getenv("CLOUD_RUN_SERVICE"),
            port=int(os.getenv("PORT", "8080")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )

    def is_openai_enabled(self) -> bool:
        """Check if OpenAI API key is configured.

        Returns:
            bool: True if OpenAI API key is set and non-empty.
        """
        return bool(self.openai_api_key and self.openai_api_key.strip())


# Global configuration instance
config = Config.from_env()
