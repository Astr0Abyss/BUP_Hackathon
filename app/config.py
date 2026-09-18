"""Environment-backed application configuration."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, ge=1, le=65535, validation_alias="PORT")

    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENAI_API_KEY"
    )
    openai_model: str = Field(
        default="gpt-5.6-terra", validation_alias="OPENAI_MODEL"
    )
    codecraft_api_key: SecretStr | None = Field(
        default=None, validation_alias="CODECRAFT_API_KEY"
    )
    codecraft_base_url: str = Field(
        default="https://codecraftapi.com/v1",
        validation_alias="CODECRAFT_BASE_URL",
    )
    codecraft_model: str = Field(
        default="claude-fable-5", validation_alias="CODECRAFT_MODEL"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

