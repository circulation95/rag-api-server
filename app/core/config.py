from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # ⭐ 여기: 알 수 없는 키는 무시
        case_sensitive=False,
    )

    app_name: str = "FastAPI Starter"
    env: str = "dev"
    debug: bool = False


settings = Settings()
