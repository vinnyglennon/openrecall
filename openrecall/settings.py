import json
import os
from dataclasses import dataclass, asdict
from typing import List

from openrecall.config import appdata_folder


SENSITIVE_DEFAULTS: List[str] = [
    "bank",
    "stripe",
    "paypal",
    "checkout",
    "payment",
    "billing",
    "invoice",
    "receipt",
    "card",
    "credit",
    "debit",
    "statement",
    "account",
    "ssn",
    "social security",
    "passport",
    "driver",
    "license",
    "id",
    "tax",
    "irs",
    "loan",
    "mortgage",
    "wire",
    "transfer",
    "ach",
    "venmo",
    "cash app",
    "zelle",
    "quickbooks",
    "xero",
    "mint",
    "finance",
    "salary",
    "payroll",
    "benefits",
    "insurance",
    "health",
    "medical",
    "patient",
    "doctor",
    "lab",
    "results",
    "2fa",
    "otp",
    "code",
    "verify",
    "auth",
]

EXCLUDED_DOMAIN_DEFAULTS: List[str] = [
    "bank",
    "paypal.com",
    "stripe.com",
    "checkout",
    "billing",
    "payment",
    "pay.google.com",
    "apple.com/apple-card",
    "chase.com",
    "bankofamerica.com",
    "wellsfargo.com",
    "capitalone.com",
    "americanexpress.com",
    "citi.com",
    "hsbc.com",
    "paypalobjects.com",
    "pay.amazon.com",
    "venmo.com",
    "cash.app",
    "zellepay.com",
    "plaid.com",
    "quickbooks",
    "xero.com",
    "mint.intuit.com",
    "adp.com",
    "gusto.com",
    "workday",
]

HIGH_RISK_OCR_DEFAULTS: List[str] = [
    "cvv",
    "cvc",
    "security code",
    "card number",
    "credit card",
    "debit card",
    "expiry",
    "exp date",
    "sort code",
    "iban",
    "account number",
    "routing number",
    "wire instructions",
    "seed phrase",
    "mnemonic",
    "private key",
    "recovery phrase",
    "backup phrase",
    "wallet key",
    "2fa",
    "otp",
    "verification code",
    "sms code",
    "pin",
    "passcode",
    "password",
    "passphrase",
]


@dataclass
class Settings:
    startup_enabled: bool = False
    retention: str = "3m"
    incognito_block: bool = True
    remind_when_paused: bool = False
    show_in_dock: bool = False
    whitelist: List[str] = None  # type: ignore[assignment]
    sensitive_patterns: List[str] = None  # type: ignore[assignment]
    excluded_domains: List[str] = None  # type: ignore[assignment]
    high_risk_ocr_triggers: List[str] = None  # type: ignore[assignment]

    def to_dict(self):
        data = asdict(self)
        # normalize None whitelist to empty list
        data["whitelist"] = self.whitelist or []
        data["sensitive_patterns"] = self.sensitive_patterns or SENSITIVE_DEFAULTS
        data["excluded_domains"] = self.excluded_domains or EXCLUDED_DOMAIN_DEFAULTS
        data["high_risk_ocr_triggers"] = (
            self.high_risk_ocr_triggers or HIGH_RISK_OCR_DEFAULTS
        )
        return data


settings_path = os.path.join(appdata_folder, "settings.json")


def load_settings() -> Settings:
    if not os.path.exists(settings_path):
        return Settings(
            whitelist=[],
            sensitive_patterns=SENSITIVE_DEFAULTS.copy(),
            excluded_domains=EXCLUDED_DOMAIN_DEFAULTS.copy(),
            high_risk_ocr_triggers=HIGH_RISK_OCR_DEFAULTS.copy(),
        )
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return Settings(
            startup_enabled=bool(raw.get("startup_enabled", False)),
            retention=str(raw.get("retention", "3m")),
            incognito_block=bool(raw.get("incognito_block", True)),
            remind_when_paused=bool(raw.get("remind_when_paused", False)),
            show_in_dock=bool(raw.get("show_in_dock", False)),
            whitelist=list(raw.get("whitelist", [])),
            sensitive_patterns=list(
                raw.get("sensitive_patterns", SENSITIVE_DEFAULTS.copy())
            ),
            excluded_domains=list(
                raw.get("excluded_domains", EXCLUDED_DOMAIN_DEFAULTS.copy())
            ),
            high_risk_ocr_triggers=list(
                raw.get("high_risk_ocr_triggers", HIGH_RISK_OCR_DEFAULTS.copy())
            ),
        )
    except Exception:
        return Settings(
            whitelist=[],
            sensitive_patterns=SENSITIVE_DEFAULTS.copy(),
            excluded_domains=EXCLUDED_DOMAIN_DEFAULTS.copy(),
            high_risk_ocr_triggers=HIGH_RISK_OCR_DEFAULTS.copy(),
        )


def save_settings(settings: Settings) -> None:
    os.makedirs(appdata_folder, exist_ok=True)
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2)
