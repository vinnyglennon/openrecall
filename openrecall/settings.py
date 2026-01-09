import json
import os
from dataclasses import dataclass, asdict
from typing import List

from openrecall.config import appdata_folder


@dataclass
class Settings:
    startup_enabled: bool = False
    retention: str = "3m"
    incognito_block: bool = True
    whitelist: List[str] = None  # type: ignore[assignment]

    def to_dict(self):
        data = asdict(self)
        # normalize None whitelist to empty list
        data["whitelist"] = self.whitelist or []
        return data


settings_path = os.path.join(appdata_folder, "settings.json")


def load_settings() -> Settings:
    if not os.path.exists(settings_path):
        return Settings(whitelist=[])
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return Settings(
            startup_enabled=bool(raw.get("startup_enabled", False)),
            retention=str(raw.get("retention", "3m")),
            incognito_block=bool(raw.get("incognito_block", True)),
            whitelist=list(raw.get("whitelist", [])),
        )
    except Exception:
        return Settings(whitelist=[])


def save_settings(settings: Settings) -> None:
    os.makedirs(appdata_folder, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2)
