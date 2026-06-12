"""Pydantic models for registered SenseBench prompts."""

from __future__ import annotations

from enum import StrEnum
from re import Pattern
from re import compile as compile_regex
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROMPT_ID_PATTERN: str = r"^p[0-9]{3,}$"
TEMPLATE_VARIABLE_PATTERN: Pattern[str] = compile_regex(
    r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
)
TEMPLATE_VARIABLE_CANDIDATE_SENSES: str = "candidate_senses"
TEMPLATE_VARIABLE_CONTEXT: str = "context"
TEMPLATE_VARIABLE_ITEM_ID: str = "item_id"
TEMPLATE_VARIABLE_TARGET_LEMMA: str = "target_lemma"
TEMPLATE_VARIABLE_TARGET_POS: str = "target_pos"
TEMPLATE_VARIABLE_TARGET_TEXT: str = "target_text"
KNOWN_TEMPLATE_VARIABLES: frozenset[str] = frozenset(
    {
        TEMPLATE_VARIABLE_CANDIDATE_SENSES,
        TEMPLATE_VARIABLE_CONTEXT,
        TEMPLATE_VARIABLE_ITEM_ID,
        TEMPLATE_VARIABLE_TARGET_LEMMA,
        TEMPLATE_VARIABLE_TARGET_POS,
        TEMPLATE_VARIABLE_TARGET_TEXT,
    }
)
SENSE_INDEX_FIELD: str = "sense_index"
TEMPLATE_KIND_FIELD: Final[str] = "template_type"

type PromptID = str


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TemplateKind(StrEnum):
    CHAT_MESSAGES = "chat_messages"


class TargetMarker(StrEnum):
    NONE = "none"
    XML_T = "xml_t"
    XML_WSD = "xml_wsd"
    XML_TARGET = "xml_target"
    SQUARE_BRACKETS = "square_brackets"
    DOUBLE_SQUARE_BRACKETS = "double_square_brackets"
    DOUBLE_ASTERISK = "double_asterisk"


class SenseOrder(StrEnum):
    FREQUENCY = "frequency"
    RANDOM_FIXED = "random_fixed"
    LEXICOGRAPHIC = "lexicographic"
    DATASET = "dataset"


class CandidateFormat(StrEnum):
    SENSEBENCH_MULTILINE = "sensebench_multiline"
    COMPACT_LABELED_INLINE = "compact_labeled_inline"


class WordNetIdKind(StrEnum):
    NONE = "none"
    SENSE_KEY = "sense_key"
    SYNSET_ID = "synset_id"


class OutputMode(StrEnum):
    JSON_SENSE_INDEX = "json_sense_index"
    PLAIN_SENSE_INDEX = "plain_sense_index"


class StrictPromptModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class PromptMessage(StrictPromptModel):
    role: MessageRole
    content: str = Field(min_length=1)


class PromptTemplate(StrictPromptModel):
    messages: list[PromptMessage] = Field(min_length=1)


class PromptParams(StrictPromptModel):
    previous_sentences: int = Field(ge=0)
    next_sentences: int = Field(ge=0)
    target_marker: TargetMarker
    sense_order: SenseOrder
    candidate_format: CandidateFormat
    include_wordnet_id: bool
    wordnet_id_kind: WordNetIdKind
    include_definition: bool
    include_examples: bool
    examples_max_per_sense: int = Field(ge=0)
    include_pos: bool
    include_synonyms: bool
    synonyms_max_per_sense: int = Field(ge=0)
    detokenize: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> PromptParams:
        if not self.include_wordnet_id and self.wordnet_id_kind != WordNetIdKind.NONE:
            raise ValueError('wordnet_id_kind must be "none" when include_wordnet_id is false')
        if self.include_wordnet_id and self.wordnet_id_kind == WordNetIdKind.NONE:
            raise ValueError('wordnet_id_kind must not be "none" when include_wordnet_id is true')
        if not self.include_examples and self.examples_max_per_sense != 0:
            raise ValueError("examples_max_per_sense must be 0 when include_examples is false")
        if not self.include_synonyms and self.synonyms_max_per_sense != 0:
            raise ValueError("synonyms_max_per_sense must be 0 when include_synonyms is false")
        return self


class PromptOutput(StrictPromptModel):
    mode: OutputMode


class PromptDefinition(StrictPromptModel):
    schema_version: Literal["sensebench-prompt-v1"]
    id: PromptID = Field(pattern=PROMPT_ID_PATTERN)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supersedes: PromptID | None = Field(default=None, pattern=PROMPT_ID_PATTERN)
    notes: str | None = Field(default=None, min_length=1)
    template_kind: Annotated[TemplateKind, Field(alias=TEMPLATE_KIND_FIELD)]
    template: PromptTemplate
    params: PromptParams
    output: PromptOutput

    @model_validator(mode="after")
    def validate_template_semantics(self) -> PromptDefinition:
        contents: list[str] = [message.content for message in self.template.messages]
        unknown_variables: set[str] = set()
        for content in contents:
            for variable_name in TEMPLATE_VARIABLE_PATTERN.findall(content):
                if variable_name not in KNOWN_TEMPLATE_VARIABLES:
                    unknown_variables.add(variable_name)
        if len(unknown_variables) > 0:
            unknown_text = ", ".join(sorted(unknown_variables))
            raise ValueError(f"unknown template variable(s): {unknown_text}")

        joined_contents = "\n".join(contents)
        json_mode_without_field = (
            self.output.mode == OutputMode.JSON_SENSE_INDEX
            and SENSE_INDEX_FIELD not in joined_contents
        )
        if json_mode_without_field:
            raise ValueError(
                f"{OutputMode.JSON_SENSE_INDEX} prompts must mention {SENSE_INDEX_FIELD}"
            )
        plain_mode_with_field = (
            self.output.mode == OutputMode.PLAIN_SENSE_INDEX
            and SENSE_INDEX_FIELD in joined_contents
        )
        if plain_mode_with_field:
            raise ValueError(
                f"{OutputMode.PLAIN_SENSE_INDEX} prompts must not request JSON {SENSE_INDEX_FIELD}"
            )
        return self
