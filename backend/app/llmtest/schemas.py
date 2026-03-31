from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ExpectedType = Literal["contains", "equals", "regex"]
ConstraintType = Literal[
    "max_length",
    "min_length",
    "contains_citation",
    "no_hallucination",
    "no_pii_email",
    "no_pii_phone",
    "regex",
    "custom",
]
ProviderType = Literal["mock", "echo", "openai_compatible"]


class ExpectedOutput(BaseModel):
    type: ExpectedType
    value: Any
    case_sensitive: bool = False


class ConstraintSpec(BaseModel):
    type: ConstraintType
    value: Any | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class PromptSpec(BaseModel):
    id: str = "default_prompt"
    version: str = "v1"
    template: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    test_name: str
    input: str
    context: str = ""
    expected: ExpectedOutput | None = None
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    id: str
    provider: ProviderType = "mock"
    model_name: str | None = None
    prompt_override: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    responses: dict[str, str] = Field(default_factory=dict)
    default_response: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Model id cannot be empty")
        return cleaned


class SuiteFile(BaseModel):
    suite_name: str
    description: str = ""
    prompt: PromptSpec
    tests: list[TestCase] = Field(default_factory=list)
    models: list[ModelSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tests(self) -> "SuiteFile":
        if not self.tests:
            raise ValueError("Suite must contain at least one test")
        return self


class ModelCatalog(BaseModel):
    models: list[ModelSpec]

    @model_validator(mode="after")
    def validate_models(self) -> "ModelCatalog":
        if not self.models:
            raise ValueError("At least one model configuration is required")
        return self
