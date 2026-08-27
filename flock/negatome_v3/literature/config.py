# The settled configuration of each stage that calls a model, and the
# content-hashed id of one run of it.
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from flock.negatome_v3.literature.packaging import get_level
from flock.negatome_v3.literature.packaging import INPUT_ABSTRACT
from flock.negatome_v3.literature.packaging import INPUT_CUE_WINDOW
from flock.negatome_v3.literature.packaging import InputLevel
from flock.negatome_v3.literature.schemas import AccessionPick
from flock.negatome_v3.literature.schemas import CurationResult
from flock.negatome_v3.literature.schemas import ScreenResult
from flock.negatome_v3.literature.vocabulary import ACCESSIONS_STAGE
from flock.negatome_v3.literature.vocabulary import CURATION_STAGE

STAGE = 'screen'

# Prompts live in the package root and are append-only: once a run has used a
# version it is frozen, and changes become the next version.
PROMPTS_DIR = Path(__file__).resolve().parent

# Generous enough that a paper reporting many pairs cannot truncate. Covers
# adaptive thinking as well as the response.
MAX_TOKENS = 16000

# The configuration TERN-2297 settled over four experiments. Do not tune this.
#
# PARAMS sets output_config.effort and deliberately omits 'thinking', which on
# Sonnet 5 means adaptive - that is the configuration that was measured, and adding
# an explicit thinking field would change the fingerprint and de-link this config
# from its measurement. Sonnet 5 rejects temperature, so run-to-run variance can be
# measured but not suppressed: three byte-identical replicates scored 39, 41 and 44
# of 52 recall-gold rows. Model and effort are both closed - Haiku 4.5 reached about
# half the recall, and xhigh's hint of +2 rows sits inside that replicate spread
# while costing several times as much at corpus scale.
MODEL = 'claude-sonnet-5'
PARAMS: dict = {'output_config': {'effort': 'low'}}
DEFAULT_PROMPT_VERSION = 'screen_v1'

# One run covers the whole corpus, and a paper is packaged by what text it has: the
# full-text level where the corpus stage retrieved usable full text, the abstract
# level otherwise. Both are named in the run_id and both are fingerprinted, so a run's
# identity still moves if either packaging changes.
DEFAULT_INPUT_LEVEL = INPUT_CUE_WINDOW
DEFAULT_ABSTRACT_LEVEL = INPUT_ABSTRACT

# The response contract. Must expose a 'pairs' list whose items carry an 'excerpts'
# list of verbatim strings: that shape is what the runner grounds against the text
# it sent.
RESULT_MODEL = ScreenResult
PICK_RESULT_MODEL = AccessionPick


def run_id_from(fingerprint: dict, parts: list[str]) -> str:
    """Build a run's id: a readable slug plus six hex characters of its hash.

    Args:
        fingerprint: Everything that makes the run's answers what they are.
        parts: The readable slug, already ordered.

    Returns:
        The run id, e.g. 'screen__v1__cue-abs__a3f81c'.
    """
    blob = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:6]
    return '__'.join([*parts, digest])


@dataclass(frozen=True)
class StageConfig:
    """One screening run: which prompt, which packagings, and the run_id they hash to.

    Args:
        prompt_version: Prompt stem, e.g. 'screen_v1'.
        input_level: Packaging for papers with usable full text, a key of
            packaging.INPUT_LEVELS.
        abstract_level: Packaging for papers without it.
        label: Extra slug in the run_id, e.g. 'pilot'.
    """

    prompt_version: str = DEFAULT_PROMPT_VERSION
    input_level: str = DEFAULT_INPUT_LEVEL
    abstract_level: str = DEFAULT_ABSTRACT_LEVEL
    label: str = ''

    def __post_init__(self) -> None:
        # Resolved here so an unknown level or a missing prompt fails before a
        # batch is built rather than after one has been submitted and billed.
        get_level(self.input_level)
        get_level(self.abstract_level)
        if not self.prompt_path.exists():
            raise FileNotFoundError(f'No prompt file at {self.prompt_path}')

    @property
    def prompt_path(self) -> Path:
        return PROMPTS_DIR / f'{self.prompt_version}.md'

    # level is the full-text packaging specifically, not "this run's packaging" -
    # there are two. Anything deciding what to send for a given paper wants
    # level_for(), which is the only place the choice is made.
    @property
    def level(self) -> InputLevel:
        return get_level(self.input_level)

    @property
    def abstract(self) -> InputLevel:
        return get_level(self.abstract_level)

    def level_for(self, usable_fulltext: bool) -> InputLevel:
        """The packaging one paper gets, from whether it has usable full text."""
        return self.level if usable_fulltext else self.abstract

    # What to send with a request, read through the config rather than off the
    # module constants directly, so a second stage sharing the batch runner sends
    # its own settings instead of inheriting the screen's. These three are the
    # screen's and are closed; see the comment on PARAMS above.
    @property
    def model(self) -> str:
        return MODEL

    @property
    def params(self) -> dict:
        return PARAMS

    @property
    def max_tokens(self) -> int:
        return MAX_TOKENS

    # Both are read or built once per config and then shared by every request the
    # run sends. Per request they cost a file read and a schema rebuild each, and
    # gave each of a chunk's 20,000 requests its own copy of the same nested dict.
    @cached_property
    def prompt_text(self) -> str:
        return self.prompt_path.read_text()

    # The screen's prompt is instructions only, with the paper as the user
    # message, so the whole of it is the system prompt.
    @property
    def system_prompt(self) -> str:
        return self.prompt_text

    @cached_property
    def output_json_schema(self) -> dict:
        return RESULT_MODEL.model_json_schema()

    def fingerprint(self) -> dict:
        """Everything that makes this run's numbers what they are.

        Hashes the prompt *text*, the packaging's resolved parameters and the schema
        JSON actually sent - never their names, each of which is a pointer to
        something that can be edited without the name moving. Editing screen_v1.md
        therefore produces a different run_id rather than silently overwriting the
        old run's results, which is the failure this ledger exists to prevent.

        'model' here is always the primary MODEL constant, even once a retry has
        answered some of the run's requests under a fallback model (batch.retry
        takes model as a call-scoped parameter, not part of this config, precisely
        so it cannot move run_id). config.json therefore records the run's settled
        identity, not what every request actually used - that lives per-chunk in
        manifest.json and per-record in records/*.jsonl.
        """
        return {
            'stage': STAGE,
            'prompt_text': self.prompt_text,
            'model': self.model,
            'params': self.params,
            'max_tokens': self.max_tokens,
            'input_level': self.input_level,
            'input_level_fingerprint': self.level.fingerprint(),
            'abstract_level': self.abstract_level,
            'abstract_level_fingerprint': self.abstract.fingerprint(),
            'schema': self.output_json_schema,
        }

    @property
    def run_id(self) -> str:
        """Readable slug plus six hex characters, e.g. 'screen__v1__cue-abs__a3f81c'."""
        parts = [
            STAGE,
            self.prompt_version.removeprefix(f'{STAGE}_'),
            f'{self.level.slug}-{self.abstract.slug}',
        ]
        if self.label:
            parts.append(self.label)
        return run_id_from(self.fingerprint(), parts)


# The configuration TERN-2301 settled over an input-level and model sweep on
# 119-146 gold sides. Do not tune this without re-measuring. Paragraph is the
# operating point: excerpt to paragraph gained 11.5 points of accuracy for 1.29x
# the tokens, paragraph to full text a further 3.4 for 11.6x. Haiku 4.5 cannot do
# this below full text, at 41.2% accuracy and 47.1% abstention on the paragraph
# arm. As with the screen above, PARAMS omits 'thinking', which on Sonnet 5 means
# adaptive; that is what was measured.
PICK_MODEL = 'claude-sonnet-5'
PICK_PARAMS: dict = {'output_config': {'effort': 'low'}}

# The answer is three short fields, so this is headroom for adaptive thinking
# rather than for output. max_tokens caps thinking and response together.
PICK_MAX_TOKENS = 4000

# v2 renders one user message holding instructions, name, paragraph and
# candidates, which is the shape the sweep measured; v1 split the instructions
# into a system prompt, a structure nothing was measured on.
PICK_PROMPT_VERSION = 'pick_v2'

# How a request's text is built - which text the model is shown and how the
# candidates are rendered beside it - is code rather than configuration, so this
# names the version of that code for the fingerprint. Bump it when
# render_candidates or build_request_text changes, or a run will resolve to a
# run_id that already has results under it.
PICK_INPUT_VERSION = 'para-v2'


@dataclass(frozen=True)
class PickConfig:
    """One accession-pick run: which prompt and settings, and their run_id.

    Args:
        prompt_version: Prompt stem, e.g. 'pick_v2'.
        label: Extra slug in the run_id, e.g. 'pilot'.
    """

    prompt_version: str = PICK_PROMPT_VERSION
    label: str = ''

    def __post_init__(self) -> None:
        # Both checks run before a batch is built rather than after one has been
        # submitted and billed. The placeholders matter as much as the file:
        # str.format leaves an unknown one untouched, so a prompt missing them
        # would send instructions with no paper text and no candidates, and every
        # answer would be an abstention nobody could explain.
        if not self.prompt_path.exists():
            raise FileNotFoundError(f'No prompt file at {self.prompt_path}')
        missing = [
            field for field in ('name', 'text', 'candidates')
            if '{' + field + '}' not in self.prompt_text
        ]
        if missing:
            raise ValueError(
                f'{self.prompt_path} has no {", ".join(missing)} placeholder. '
                f'The pick prompt is a template rendered per side.',
            )

    @property
    def prompt_path(self) -> Path:
        return PROMPTS_DIR / f'{self.prompt_version}.md'

    @property
    def model(self) -> str:
        return PICK_MODEL

    @property
    def params(self) -> dict:
        return PICK_PARAMS

    @property
    def max_tokens(self) -> int:
        return PICK_MAX_TOKENS

    @cached_property
    def prompt_text(self) -> str:
        return self.prompt_path.read_text()

    # Nothing goes in the system slot: the instructions travel in the user
    # message with the paragraph and the candidates, which is the shape the
    # input-level sweep measured. prompt_text stays the template, so the run
    # still freezes and fingerprints it.
    @property
    def system_prompt(self) -> str:
        return ''

    @cached_property
    def output_json_schema(self) -> dict:
        return PICK_RESULT_MODEL.model_json_schema()

    def fingerprint(self) -> dict:
        """Everything that makes this run's answers what they are.

        The input table is not hashed here - it is recorded in the manifest
        instead, so that rebuilding it is caught as a resume against different
        inputs rather than silently forking the run_id.
        """
        return {
            'stage': ACCESSIONS_STAGE,
            'prompt_text': self.prompt_text,
            'model': self.model,
            'params': self.params,
            'max_tokens': self.max_tokens,
            'input_version': PICK_INPUT_VERSION,
            'schema': self.output_json_schema,
        }

    @property
    def run_id(self) -> str:
        """Readable slug plus six hex characters."""
        parts = [
            ACCESSIONS_STAGE,
            self.prompt_version.replace('_', '-'),
            PICK_INPUT_VERSION,
        ]
        if self.label:
            parts.append(self.label)
        return run_id_from(self.fingerprint(), parts)


# As with the two configurations above, PARAMS omits 'thinking', which on Sonnet
# 5 means adaptive. Sonnet 5 rejects temperature. Effort is what makes this stage
# work: below xhigh the model answers in one turn without calling a tool.
CURATION_MODEL = 'claude-sonnet-5'
CURATION_PARAMS: dict = {'output_config': {'effort': 'xhigh'}}

# Thinking counts against max_tokens, so this is headroom rather than a target.
# It is also why every request streams: the SDK refuses a non-streaming call
# whose max_tokens could run past ten minutes. A pair-dense paper can still spend
# the whole budget on thinking and return no answer, which is a failure the run
# records; the ceiling on Sonnet 5 is 128,000, so such a paper can be re-answered
# under a larger value. That moves the run_id, which is why it is a config field.
CURATION_MAX_TOKENS = 24000

# The rubric a run answers under. A superseded version can be deleted rather than
# kept: write_run_config freezes a copy of the prompt into every run directory, so
# an earlier run's records stay interpretable from the run directory alone.
CURATION_PROMPT_VERSION = 'curate_v2'

# How a packet is built - which pairs, which cards, how much of the paper, and
# what the tools return - is code rather than configuration, so this names the
# version of that code for the fingerprint. Bump it when build_packet, the tool
# definitions or the tool handlers change; it moves the run_id, so a run under the
# previous value is no longer resumed past.
CURATION_INPUT_VERSION = 'packet-v2'

CURATION_RESULT_MODEL = CurationResult

# The two tools the loop serves locally. Their text is fingerprinted with
# everything else: a tool description is prompt, and the run that answered with
# one is not the run that would answer with another.
CURATION_TOOLS: list[dict] = [
    {
        'name': 'uniprot_lookup',
        'description': (
            'Fetch the current UniProt entry for one or more accessions: entry '
            'name, protein and gene names, organism, length and review status. '
            'Call this when a card in the packet leaves you unsure whether an '
            'accession is the protein the paper means, or when you suspect the '
            'organism is wrong.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'accessions': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'UniProt accessions, e.g. ["P18111", "P43241"].',
                },
            },
            'required': ['accessions'],
        },
    },
    {
        'name': 'paper_methods',
        'description': (
            "Return this paper's methods section, which the packet omits. Call "
            'this when the assay, the construct or the organism matters to a '
            'verdict and the body text does not settle it.'
        ),
        'input_schema': {'type': 'object', 'properties': {}},
    },
]


@dataclass(frozen=True)
class CurationConfig:
    """One curation run: which rubric and settings, and their run_id.

    Args:
        prompt_version: Prompt stem, e.g. 'curate_v2'.
        label: Extra slug in the run_id, e.g. 'pilot'.
        max_tokens: Output ceiling per turn, thinking included. Raising it is a
            different configuration and so a different run_id, which is what
            keeps a paper answered under a larger budget distinguishable from
            one answered under the settled value.
    """

    prompt_version: str = CURATION_PROMPT_VERSION
    label: str = ''
    max_tokens: int = CURATION_MAX_TOKENS

    def __post_init__(self) -> None:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f'No prompt file at {self.prompt_path}')

    @property
    def prompt_path(self) -> Path:
        return PROMPTS_DIR / f'{self.prompt_version}.md'

    @property
    def model(self) -> str:
        return CURATION_MODEL

    # Copied out, not handed over: both are nested mutable module state, and a
    # caller that edited one in place would move this run's fingerprint for the
    # rest of the process without touching a line of configuration.
    @property
    def params(self) -> dict:
        return copy.deepcopy(CURATION_PARAMS)

    @property
    def tools(self) -> list[dict]:
        return copy.deepcopy(CURATION_TOOLS)

    @cached_property
    def prompt_text(self) -> str:
        return self.prompt_path.read_text()

    # The rubric is instructions only, with the packet as the user message, so
    # the whole of it is the system prompt.
    @property
    def system_prompt(self) -> str:
        return self.prompt_text

    @cached_property
    def output_json_schema(self) -> dict:
        return CURATION_RESULT_MODEL.model_json_schema()

    def fingerprint(self) -> dict:
        """Everything that makes this run's verdicts what they are.

        The input table is not hashed here, as in PickConfig: it is recorded in
        the manifest instead, so rebuilding it is caught as a resume against
        different inputs rather than silently forking the run_id.
        """
        return {
            'stage': CURATION_STAGE,
            'prompt_text': self.prompt_text,
            'model': self.model,
            'params': self.params,
            'max_tokens': self.max_tokens,
            'input_version': CURATION_INPUT_VERSION,
            'tools': self.tools,
            'schema': self.output_json_schema,
        }

    @property
    def run_id(self) -> str:
        """Readable slug plus six hex characters."""
        parts = [
            CURATION_STAGE,
            self.prompt_version.replace('_', '-'),
            CURATION_INPUT_VERSION,
        ]
        if self.label:
            parts.append(self.label)
        return run_id_from(self.fingerprint(), parts)


# What the shared batch runner accepts. The two stages have nothing in common
# beyond a prompt, a schema, request settings and an identity - the screen
# packages whole papers by what text the corpus retrieved, the pick sends rows
# built into a table before any request exists - so this is a union of the two
# rather than a base class either has to inherit.
RequestConfig = StageConfig | PickConfig

# What the run directory accepts, which is wider: curation writes its config,
# frozen prompt, manifest and records the same way but sends its requests itself,
# a multi-turn tool loop being a shape the Batch API has no equivalent of.
# Deliberately a second alias rather than a widened RequestConfig - putting
# CurationConfig in the union above would tell the batch runner it can submit a
# configuration it cannot.
RunConfig = RequestConfig | CurationConfig
