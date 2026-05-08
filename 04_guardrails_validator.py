"""Step 4: Custom Guardrails validators for PII and JSON output."""

from __future__ import annotations

import json
import re
from typing import Any

from config import EVIDENCE_DIR, ensure_directories

ensure_directories()

try:
    from guardrails import Guard
except ImportError as exc:
    raise RuntimeError("guardrails-ai is required. Install with: pip install -r requirements.txt") from exc

try:
    from guardrails import OnFailAction, Validator, register_validator
except ImportError:
    from guardrails.validator_base import OnFailAction, Validator, register_validator

try:
    from guardrails.validators import FailResult, PassResult
except ImportError:
    from guardrails.validator_base import FailResult, PassResult


def _pass(value: str | None = None):
    if value is None:
        return PassResult()
    return PassResult(value_override=value)


def _fail(message: str, fix_value: str):
    try:
        return FailResult(error_message=message, fix_value=fix_value)
    except TypeError:
        return FailResult(errorMessage=message, fixValue=fix_value)


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detect and redact email addresses, US phone numbers, SSNs, and cards."""

    PII_PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "PHONE": re.compile(
            r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]\d{4}(?!\d)"
        ),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    }

    def validate(self, value: str, metadata: dict[str, Any]):
        redacted = value
        found: list[str] = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(redacted)
            if matches:
                found.extend([pii_type] * len(matches))
                redacted = pattern.sub("[REDACTED]", redacted)

        if found:
            details = ", ".join(sorted(set(found)))
            return _fail(f"Detected PII: {details}", redacted)
        return _pass(value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validate JSON and repair common LLM formatting mistakes."""

    @staticmethod
    def _repair(text: str) -> str:
        repaired = text.strip()
        repaired = re.sub(r"^```(?:json)?\s*", "", repaired, flags=re.IGNORECASE)
        repaired = re.sub(r"\s*```$", "", repaired)
        repaired = repaired.strip()
        repaired = repaired.replace("'", '"')
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return repaired

    def validate(self, value: str, metadata: dict[str, Any]):
        try:
            parsed = json.loads(value)
            return _pass(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        repaired_text = self._repair(value)
        try:
            parsed = json.loads(repaired_text)
            return _pass(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as exc:
            fallback = json.dumps({"error": "invalid_json", "raw": value})
            return _fail(f"Invalid JSON after repair attempt: {exc}", fallback)


def _outcome_status(outcome) -> str:
    return "Pass" if getattr(outcome, "validation_passed", False) else "Fixed/Failed"


def _outcome_output(outcome) -> str:
    return str(getattr(outcome, "validated_output", outcome))


def _validator_output(validator: Validator, text: str) -> str:
    result = validator.validate(text, {})
    fix_value = getattr(result, "fix_value", None)
    if fix_value is not None:
        return str(fix_value)

    value_override = getattr(result, "value_override", None)
    sentinel = getattr(PassResult, "ValueOverrideSentinel", None)
    if value_override is not None and value_override is not sentinel:
        return str(value_override)
    return text


def _emit(lines: list[str], line: str = "") -> None:
    print(line)
    lines.append(line)


def demo_pii_guard() -> None:
    lines: list[str] = []
    validator = PIIDetector(on_fail=OnFailAction.FIX)
    guard = Guard().use(validator)
    test_cases = [
        ("Clean", "No sensitive information in this text."),
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email alice@example.com or call 555-123-4567."),
    ]

    _emit(lines, "=" * 55)
    _emit(lines, "PII Detection Demo")
    _emit(lines, "=" * 55)
    for label, text in test_cases:
        outcome = guard.validate(text)
        _emit(lines, f"\n[{label}] {_outcome_status(outcome)}")
        _emit(lines, f"Input : {text}")
        _emit(lines, f"Output: {_validator_output(validator, text)}")

    path = EVIDENCE_DIR / "04_pii_demo_log.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved PII demo log to {path}")


def demo_json_guard() -> None:
    lines: list[str] = []
    validator = JSONFormatter(on_fail=OnFailAction.FIX)
    guard = Guard().use(validator)
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Broken", "This is not JSON at all: ??? {]"),
    ]

    _emit(lines, "=" * 55)
    _emit(lines, "JSON Formatting Demo")
    _emit(lines, "=" * 55)
    for label, text in test_cases:
        outcome = guard.validate(text)
        _emit(lines, f"\n[{label}] {_outcome_status(outcome)}")
        _emit(lines, f"Input : {text}")
        _emit(lines, f"Output: {_validator_output(validator, text)}")

    path = EVIDENCE_DIR / "04_json_demo_log.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved JSON demo log to {path}")


def main() -> None:
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)
    demo_pii_guard()
    print()
    demo_json_guard()
    print("\nStep 4 complete")


if __name__ == "__main__":
    main()
