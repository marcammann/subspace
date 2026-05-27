from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseModel):
    type: str = "echo"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    extra: dict[str, Any] = {}


class MiddlewareSettings(BaseModel):
    type: str
    params: dict[str, Any] = {}


class ModelSettings(BaseModel):
    name: str
    middlewares: list[MiddlewareSettings] = []
    backend: BackendSettings = BackendSettings()


class SubspaceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUBSPACE_",
        env_nested_delimiter="__",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    models: list[ModelSettings] = [
        ModelSettings(name="echo", backend=BackendSettings(type="echo")),
    ]
