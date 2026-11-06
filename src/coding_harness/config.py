from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_", env_file=".env", extra="ignore", populate_by_name=True
    )

    model: str = "gpt-4.1-mini"
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    tracing: bool = True
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL"
    )
    max_model_calls: int = Field(default=20, ge=1, le=1000)
    max_tool_calls: int = Field(default=40, ge=1, le=1000)
    model_timeout_seconds: float = Field(default=60, gt=0, le=600)
    test_timeout_seconds: float = Field(default=60, gt=0, le=600)

    def validate_credentials(self) -> None:
        if not self.openai_api_key or not self.openai_api_key.get_secret_value().strip():
            raise ValueError("Set OPENAI_API_KEY in .env or your environment.")
        if self.tracing and (
            not self.langfuse_public_key
            or not self.langfuse_secret_key
            or not self.langfuse_secret_key.get_secret_value().strip()
        ):
            raise ValueError(
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, "
                "or use --no-tracing for local runs."
            )
