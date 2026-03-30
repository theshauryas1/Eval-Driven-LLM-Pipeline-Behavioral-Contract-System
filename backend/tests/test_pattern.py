"""
Unit tests for the pattern evaluator.
Run: pytest tests/test_pattern.py -v
"""
import pytest
from app.evaluators.pattern import PatternEvaluator


@pytest.fixture
def evaluator():
    return PatternEvaluator()


EMAIL_PATTERN = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"(\+?\d[\d\s\-().]{7,}\d)"


class TestNoPIIEmail:
    def test_passes_clean_response(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_email",
            config={"pattern": EMAIL_PATTERN, "must_not_match": True},
            output="The company was founded in 2010 and has 500 employees.",
        )
        assert result.passed is True

    def test_fails_when_email_present(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_email",
            config={"pattern": EMAIL_PATTERN, "must_not_match": True},
            output="Please contact john.doe@example.com for more information.",
        )
        assert result.passed is False

    def test_fails_with_multiple_emails(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_email",
            config={"pattern": EMAIL_PATTERN, "must_not_match": True},
            output="Contact alice@foo.com or bob@bar.org.",
        )
        assert result.passed is False

    def test_case_insensitive(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_email",
            config={"pattern": EMAIL_PATTERN, "must_not_match": True},
            output="Email ADMIN@COMPANY.COM to proceed.",
        )
        assert result.passed is False

    def test_partial_email_not_flagged(self, evaluator):
        """A string like '@something' without user part should not match."""
        result = evaluator.evaluate(
            contract_id="no_pii_email",
            config={"pattern": EMAIL_PATTERN, "must_not_match": True},
            output="Follow us @example on Twitter.",
        )
        assert result.passed is True


class TestNoPIIPhone:
    def test_passes_clean_response(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_phone",
            config={"pattern": PHONE_PATTERN, "must_not_match": True},
            output="The project deadline is March 15, 2025.",
        )
        assert result.passed is True

    def test_fails_with_us_phone(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_phone",
            config={"pattern": PHONE_PATTERN, "must_not_match": True},
            output="Call us at +1 (555) 123-4567 for support.",
        )
        assert result.passed is False

    def test_fails_with_international_phone(self, evaluator):
        result = evaluator.evaluate(
            contract_id="no_pii_phone",
            config={"pattern": PHONE_PATTERN, "must_not_match": True},
            output="International: +44 20 7946 0958",
        )
        assert result.passed is False


class TestMustMatch:
    def test_passes_when_pattern_found(self, evaluator):
        result = evaluator.evaluate(
            contract_id="must_have_disclaimer",
            config={"pattern": r"not financial advice", "must_match": True},
            output="Bitcoin has been volatile. This is not financial advice.",
        )
        assert result.passed is True

    def test_fails_when_pattern_absent(self, evaluator):
        result = evaluator.evaluate(
            contract_id="must_have_disclaimer",
            config={"pattern": r"not financial advice", "must_match": True},
            output="Buy Bitcoin immediately.",
        )
        assert result.passed is False


class TestNoPattern:
    def test_no_pattern_configured_passes(self, evaluator):
        result = evaluator.evaluate(
            contract_id="empty",
            config={},
            output="Anything",
        )
        assert result.passed is True

    def test_invalid_regex_fails_gracefully(self, evaluator):
        result = evaluator.evaluate(
            contract_id="bad_regex",
            config={"pattern": "[invalid(regex", "must_not_match": True},
            output="Some response",
        )
        assert result.passed is False
        assert "Invalid regex" in result.explanation
