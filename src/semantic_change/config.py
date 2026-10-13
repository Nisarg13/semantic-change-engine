"""Runtime settings.

An environment variable with the prefix ``SCD_`` and the field name in upper case
overrides each setting, for example ``SCD_LLM_MODEL=claude-sonnet-5-5``. The Anthropic
SDK reads the API key from ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Settings of the pipeline. An instance cannot be changed after it is created."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Stage 1: clause segmentation
    spacy_model: str = Field(default="en_core_web_sm", min_length=1)
    # Stage 2: sentence embeddings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", min_length=1)
    # Stage 3: minimum cosine similarity to match a unit of text A with a unit of text B
    align_threshold: float = Field(default=0.45, ge=0, le=1)
    # Stage 4: evaluation agent
    llm_model: str = Field(default="claude-opus-5-5", min_length=1)
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"  # thinking depth
    llm_max_tokens: int = Field(default=16_000, gt=0)  # output limit of one agent turn
    agent_max_turns: int = Field(default=10, gt=0)  # the verdict must come within this limit
    # Maximum length of each input text, in characters
    max_text_chars: int = Field(default=20_000, gt=0)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read the settings from the ``SCD_*`` variables of ``env`` (default: ``os.environ``).

        Raises:
            pydantic.ValidationError: A variable has a value that is not valid.
        """
        env = os.environ if env is None else env
        values = {
            name: env[key] for name in cls.model_fields if (key := "SCD_" + name.upper()) in env
        }
        return cls.model_validate(values)
