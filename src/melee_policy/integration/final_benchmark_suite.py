"""Frozen schedule and artifact checks for the final-winner benchmark suite.

This module contains no launcher side effects.  The gameplay orchestrator and
the deferred media worker share it so that a completed game can only be counted
or rendered under one identical, immutable match contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

BENCHMARK_VARIANT = os.environ.get(
    "MELEE_FINAL_BENCHMARK_VARIANT",
    "pretraining",
).strip().lower()
if BENCHMARK_VARIANT not in {"pretraining", "posttraining", "postrl"}:
    raise RuntimeError(
        "MELEE_FINAL_BENCHMARK_VARIANT must be 'pretraining' or 'posttraining'"
    )
POSTTRAINING_BENCHMARK = BENCHMARK_VARIANT in {"posttraining", "postrl"}
POST_RL_BENCHMARK = BENCHMARK_VARIANT == "postrl"
CURRENT_PHASE = "postrl" if POST_RL_BENCHMARK else "posttraining" if POSTTRAINING_BENCHMARK else "pretraining"

SUITE_ID = (
    "posttraining-winners-expanded-step195248-step127214-v1"
    if POSTTRAINING_BENCHMARK
    else "final-winners-expanded-step122064-step86016-v5"
)
SUITE_SCHEMA_VERSION = (
    "integration.posttraining_winner_benchmark_suite.v1"
    if POSTTRAINING_BENCHMARK
    else "integration.final_winner_benchmark_suite.v5"
)
STATE_SCHEMA_VERSION = (
    "integration.posttraining_winner_benchmark_state.v1"
    if POSTTRAINING_BENCHMARK
    else "integration.final_winner_benchmark_state.v5"
)
GAME_COUNT = 329 if POSTTRAINING_BENCHMARK else 309
STAGE = "FINAL_DESTINATION"
FRISSON_CHARACTER = "FOX"
TEN_M_STEP = 195_248 if POSTTRAINING_BENCHMARK else 122_064
TEN_M_FRAMES = 12_795_772_928 if POSTTRAINING_BENCHMARK else 7_999_586_304
TEN_M_VALIDATION_NLL = 0.7574608703491693 if POSTTRAINING_BENCHMARK else 0.796817
TEN_M_PROFILE_SLUG = (
    "10m-posttrained-step195248" if POSTTRAINING_BENCHMARK else "10m-step122064"
)
SEVENTY_FIVE_M_STEP = 127_214 if POSTTRAINING_BENCHMARK else 86_016
SEVENTY_FIVE_M_FRAMES = (
    8_337_096_704 if POSTTRAINING_BENCHMARK else 5_637_144_576
)
SEVENTY_FIVE_M_VALIDATION_NLL = (
    0.7248941140799456 if POSTTRAINING_BENCHMARK else 0.764821
)
SEVENTY_FIVE_M_PROFILE_SLUG = (
    "75m-posttrained-step127214" if POSTTRAINING_BENCHMARK else "75m-step86016"
)
V1_SUITE_ID = "final-winners-step122064-step86016-v1"
V4_SUITE_ID = "final-winners-expanded-step122064-step86016-v4"
V4_SUITE_SCHEMA_VERSION = "integration.final_winner_benchmark_suite.v4"
V4_STATE_SCHEMA_VERSION = "integration.final_winner_benchmark_state.v4"
V4_SCHEDULE_SHA256 = "7e8913cea5141e53d5f060d9172d494d320a2eaa4e208090809b186f7e988d92"
V4_MANIFEST_SHA256 = "abb516dffa5ad589e0b426453284592c97a7e3f3d4d41bd7fe3e843a838eaf91"
V4_MANIFEST_BYTE_LENGTH = 194_290
V4_STATE_SHA256 = "de9830741e90b1caf9bb1b93adf635d4543f6435ab11259bf1865424612b75f4"
V4_STATE_BYTE_LENGTH = 367_786
V4_STATE_UPDATED_AT_UTC = "2026-09-03T23:42:17.286556+00:00"
V4_GAME_COUNT = 309
UNCHANGED_V4_GAME_COUNT = 177
REPLACED_V4_GAME_COUNT = 132
REPLACED_V4_MIMIC_GAME_COUNT = 124
REPLACED_V4_SLIPPI_GAME_COUNT = 8
PRESERVED_V4_COMPLETED_GAME_COUNT = 58
V3_SUITE_ID = "final-winners-expanded-step122064-step86016-v3"
V3_SUITE_SCHEMA_VERSION = "integration.final_winner_benchmark_suite.v3"
V3_STATE_SCHEMA_VERSION = "integration.final_winner_benchmark_state.v3"
V3_SCHEDULE_SHA256 = "e389d05a370ee20375cfdff4b8b8761a1d20154c5a801836e961ed801c888e6f"
V3_GAME_COUNT = 285
UNCHANGED_V3_PREFIX_GAME_COUNT = 117

if POST_RL_BENCHMARK:
    SUITE_ID = "postrl-p21-step1318-step222-v1"
    SUITE_SCHEMA_VERSION = "integration.postrl_benchmark_suite.v1"
    STATE_SCHEMA_VERSION = "integration.postrl_benchmark_state.v1"
    TEN_M_STEP, SEVENTY_FIVE_M_STEP = 1318, 222
    TEN_M_FRAMES = SEVENTY_FIVE_M_FRAMES = None
    TEN_M_VALIDATION_NLL = SEVENTY_FIVE_M_VALIDATION_NLL = None
    TEN_M_PROFILE_SLUG = "10m-postrl-step1318"
    SEVENTY_FIVE_M_PROFILE_SLUG = "75m-postrl-step222"

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_OUTPUT = ROOT / "artifacts" / "integration" / "frisson_ai"
SUITE_DIRECTORY = ARTIFACT_OUTPUT / SUITE_ID
MANIFEST_PATH = SUITE_DIRECTORY / "manifest.json"
STATE_PATH = SUITE_DIRECTORY / "state.json"
LOCK_PATH = SUITE_DIRECTORY / ".suite.lock"
GLOBAL_LOCK_PATH = ARTIFACT_OUTPUT / ".final-winner-benchmark.lock"
V3_SUITE_DIRECTORY = ARTIFACT_OUTPUT / V3_SUITE_ID
V3_MANIFEST_PATH = V3_SUITE_DIRECTORY / "manifest.json"
V3_STATE_PATH = V3_SUITE_DIRECTORY / "state.json"
V3_LOCK_PATH = V3_SUITE_DIRECTORY / ".suite.lock"
V4_SUITE_DIRECTORY = ARTIFACT_OUTPUT / V4_SUITE_ID
V4_MANIFEST_PATH = V4_SUITE_DIRECTORY / "manifest.json"
V4_STATE_PATH = V4_SUITE_DIRECTORY / "state.json"
V4_LOCK_PATH = V4_SUITE_DIRECTORY / ".suite.lock"

TEN_M_CHECKPOINT = (
    ROOT
    / (
        ".e013-cache/posttraining-winners/frisson-melee-10m-posttrained-best-val.pt"
        if POSTTRAINING_BENCHMARK
        else ".e011-cache/final-winners/remote-verification/10m-step-122064.pt"
    )
)
SEVENTY_FIVE_M_CHECKPOINT = (
    ROOT
    / (
        ".e013-cache/posttraining-winners/frisson-melee-75m-posttrained-best-val.pt"
        if POSTTRAINING_BENCHMARK
        else ".e011-cache/final-winners/remote-verification/75m-step-86016.pt"
    )
)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: Path
    sha256: str
    byte_length: int


TEN_M_IDENTITY = FileIdentity(
    path=TEN_M_CHECKPOINT,
    sha256=(
        "63b5ff05ef30476c4f41590c478f4a7218f3b72a8b5ef24a0e336eeb2b7c287b"
        if POSTTRAINING_BENCHMARK
        else "5a54ea4ecfa150198180dff4d06ac4fe6ecec5d9433803cc13b527e41f41d14e"
    ),
    byte_length=96_452_075 if POSTTRAINING_BENCHMARK else 96_451_691,
)
SEVENTY_FIVE_M_IDENTITY = FileIdentity(
    path=SEVENTY_FIVE_M_CHECKPOINT,
    sha256=(
        "8211f1832198646e9f4e3bacde26f181614f93320e68bd326dd5942c4e0f4077"
        if POSTTRAINING_BENCHMARK
        else "8662de0c4c0deaae2879548d6373def792dbb52adec44f2b474adb7fe155c648"
    ),
    byte_length=644_291_987,
)
if POST_RL_BENCHMARK:
    from melee_policy.integration.post_rl_checkpoints import CHECKPOINTS as POST_RL_CHECKPOINTS
    TEN_M_CHECKPOINT = ROOT / POST_RL_CHECKPOINTS["10m"]["relative_path"]
    SEVENTY_FIVE_M_CHECKPOINT = ROOT / POST_RL_CHECKPOINTS["75m"]["relative_path"]
    TEN_M_IDENTITY = FileIdentity(TEN_M_CHECKPOINT, POST_RL_CHECKPOINTS["10m"]["sha256"], POST_RL_CHECKPOINTS["10m"]["byte_length"])
    SEVENTY_FIVE_M_IDENTITY = FileIdentity(SEVENTY_FIVE_M_CHECKPOINT, POST_RL_CHECKPOINTS["75m"]["sha256"], POST_RL_CHECKPOINTS["75m"]["byte_length"])

TEN_M_ANCESTOR_IDENTITY = FileIdentity(
    path=(
        ROOT
        / ".e011-cache/final-winners/remote-verification/10m-step-122064.pt"
    ),
    sha256="5a54ea4ecfa150198180dff4d06ac4fe6ecec5d9433803cc13b527e41f41d14e",
    byte_length=96_451_691,
)
SEVENTY_FIVE_M_ANCESTOR_IDENTITY = FileIdentity(
    path=(
        ROOT
        / ".e011-cache/final-winners/remote-verification/75m-step-86016.pt"
    ),
    sha256="8662de0c4c0deaae2879548d6373def792dbb52adec44f2b474adb7fe155c648",
    byte_length=644_291_987,
)
SLIPPI_MEDIUM_V2_IDENTITY = FileIdentity(
    path=ROOT / ".e001-cache" / "slippi-ai" / "models" / "medium-v2",
    sha256="48dcfd87c52fde9899fb37b0293ce2a597fb2384b7b40baf7fb49c760152a96c",
    byte_length=95_559_707,
)


@dataclass(frozen=True, slots=True)
class SlippiSpecialistRelease:
    character: str
    slug: str
    release_name: str
    display_name: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    variable_count: int
    parameter_count: int
    policy_delay_frames: int
    checkpoint_tag: str
    checkpoint_config_version: int
    official_url: str

    @property
    def checkpoint(self) -> Path:
        return ROOT / ".e001-cache" / "slippi-ai" / "models" / self.release_name


SLIPPI_SPECIALIST_RELEASES = (
    SlippiSpecialistRelease(
        character="DK",
        slug="dk",
        release_name="dk_d18_imitation_v2",
        display_name="vladfi1 Slippi-AI dk_d18_imitation_v2 native DK imitation",
        checkpoint_sha256=(
            "e4e3f07d32812154b4401e40d6d83b741cbf153887dcd14417fa50c35eeeae4d"
        ),
        checkpoint_byte_length=42_078_949,
        variable_count=137,
        parameter_count=10_517_970,
        policy_delay_frames=18,
        checkpoint_tag="dk_delay_18_v2",
        checkpoint_config_version=3,
        official_url=(
            "https://www.dropbox.com/scl/fo/mg916t9exid4stqmx2bjf/"
            "AGBmOvmlKjo7iGjm4CEvyf4/dk_d18_imitation_v2"
            "?rlkey=baqxnfxg2uytvcz62w9o8mwzt&dl=1"
        ),
    ),
    SlippiSpecialistRelease(
        character="DOC",
        slug="doc",
        release_name="doc_d18_imitation_v3",
        display_name=(
            "vladfi1 Slippi-AI doc_d18_imitation_v3 native Dr. Mario imitation"
        ),
        checkpoint_sha256=(
            "ade88cc5c04974df5123da1a0539ab2324d9b76dc41c5561f8e94a54c6d2ac14"
        ),
        checkpoint_byte_length=42_078_998,
        variable_count=137,
        parameter_count=10_517_970,
        policy_delay_frames=18,
        checkpoint_tag="doc_d18_as16_imitation_v3",
        checkpoint_config_version=3,
        official_url=(
            "https://www.dropbox.com/scl/fo/mg916t9exid4stqmx2bjf/"
            "AOigWSZfGvUkNT5MB8VOI7o/doc_d18_imitation_v3"
            "?rlkey=baqxnfxg2uytvcz62w9o8mwzt&dl=1"
        ),
    ),
)
SLIPPI_SPECIALIST_BY_CHARACTER = {
    release.character: release for release in SLIPPI_SPECIALIST_RELEASES
}


def _slippi_specialist_release_contract(
    release: SlippiSpecialistRelease,
) -> dict[str, Any]:
    """Return the exact upstream metadata emitted by the live release registry."""

    return {
        "schema_version": "melee_policy.slippi_ai.release.v1",
        "key": release.release_name,
        "display_name": release.display_name,
        "checkpoint_sha256": release.checkpoint_sha256,
        "checkpoint_bytes": release.checkpoint_byte_length,
        "variable_count": release.variable_count,
        "parameter_count": release.parameter_count,
        "policy_delay_frames": release.policy_delay_frames,
        "supported_characters": [release.character],
        "allowed_opponents": "all",
        "checkpoint_kind": "imitation",
        "checkpoint_tag": release.checkpoint_tag,
        "checkpoint_config_version": release.checkpoint_config_version,
        "checkpoint_step": None,
        "rl_trained_names": [],
        "official_url": release.official_url,
    }

MIMIC_SOURCE_REVISION = "01eb974962c8338147518dd360ddbf4b9d4c48e3"
MIMIC_SOURCE_REPOSITORY = "https://github.com/erickfm/MIMIC.git"
MIMIC_SOURCE_DIRECTORY = ".e012-cache/mimic-source-current"
MIMIC_V4_RELEASE_REVISION = "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07"
MIMIC_RELEASE_REVISION = "0629eb174bb7548038469f538d98b3691ea647be"
MIMIC_NATIVE_EXTENSION_REVISION = MIMIC_RELEASE_REVISION
MIMIC_LEGACY_CACHE_DIRECTORY = ".e001-cache/mimic"
MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY = ".e012-cache/mimic-native-0629eb17"
SLIPPI_SOURCE_REVISION = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
SLIPPI_PLAYER_NAME = "Master Player"
SLIPPI_TEMPERATURE = 1.0
SLIPPI_POLICY_DELAY_FRAMES = 21
MIMIC_OOD_TRANSFER_SCHEMA_VERSION = (
    "integration.frisson_vs_mimic.mimic_ood_character_transfer.v1"
)
MIMIC_OOD_TRANSFER_MODE = "released-checkpoint-on-ood-physical-character"
MIMIC_OOD_PROVENANCE_CHECKS = (
    "explicit_opt_in",
    "physical_character_launchable",
    "physical_character_outside_released_mimic_roster",
    "explicit_checkpoint_and_assets_supplied",
    "requested_checkpoint_loaded_exactly",
    "requested_assets_loaded_exactly",
    "checkpoint_sha256_bound_to_released_bundle",
    "released_bundle_semantics_validated",
    "checkpoint_character_matches_bundle",
    "checkpoint_character_has_released_mimic_bundle",
    "checkpoint_character_separate_from_physical_character",
    "state_dictionary_loaded_strictly",
)
DATASET_MANIFEST_ID = "e018460bb8b2d5398b9005217a8cc0ba1ff0f93fb3fe8fb26b7e5f18e03d4ee3"
DATASET_SLIPPI_COMMIT = SLIPPI_SOURCE_REVISION
FRISSON_ROSTER = (
    "MARIO",
    "FOX",
    "CPTFALCON",
    "DK",
    "KIRBY",
    "BOWSER",
    "LINK",
    "SHEIK",
    "NESS",
    "PEACH",
    "POPO",
    "PIKACHU",
    "SAMUS",
    "YOSHI",
    "JIGGLYPUFF",
    "MEWTWO",
    "LUIGI",
    "MARTH",
    "ZELDA",
    "YLINK",
    "DOC",
    "FALCO",
    "PICHU",
    "GAMEANDWATCH",
    "GANONDORF",
    "ROY",
)


@dataclass(frozen=True, slots=True)
class MimicRelease:
    character: str
    slug: str
    directory: str
    bundle_name: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    revision: str = MIMIC_RELEASE_REVISION
    cache_directory: str = MIMIC_LEGACY_CACHE_DIRECTORY

    @property
    def checkpoint(self) -> Path:
        return ROOT / self.cache_directory / self.directory / "model.pt"

    @property
    def assets(self) -> Path:
        return self.checkpoint.parent


MIMIC_CURRENT_CORE_RELEASES = (
    MimicRelease(
        "FOX",
        "fox",
        "fox",
        "fox",
        "8941df4c19437162f1acc2302d900895608481742ea649a7597d3ae6ed3b6e29",
        265_251_232,
        revision=MIMIC_RELEASE_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "FALCO",
        "falco",
        "falco",
        "falco",
        "5083b30df7bf1516fe536d5b95262d90c85d4483e7b8fe124aeddc135af11196",
        265_257_515,
        revision=MIMIC_RELEASE_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "MARTH",
        "marth",
        "marth",
        "marth",
        "df0170bd7f9c66fc21ec404d6bf8b1f0c6253ea84791c28979cb6684d94e258d",
        345_254_299,
        revision=MIMIC_RELEASE_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "SHEIK",
        "sheik",
        "sheik",
        "sheik",
        "93d6eaec8dda0251da564c928e55e170cfac4648764d98b486fbd9189e03d090",
        345_254_299,
        revision=MIMIC_RELEASE_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "CPTFALCON",
        "cptfalcon",
        "cptfalcon",
        "cptfalcon",
        "14fc5eacd1b00ea258f6ed2ad7ddaea2e96e3f2e8c86306092b7727d75635187",
        265_259_503,
        revision=MIMIC_RELEASE_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "LUIGI",
        "luigi",
        "luigi",
        "luigi",
        "eeaf96faa9c385cbdd034fbe5b509a5e3d6f5ae41d9d27549c8930ade3a41df2",
        265_254_759,
        revision=MIMIC_RELEASE_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
)
MIMIC_NATIVE_EXTENSION_RELEASES = (
    MimicRelease(
        "BOWSER",
        "bowser",
        "bowser",
        "bowser",
        "870dfd38d6365fecbde17761310b5801ec73e0e74fb75527eb4ecfe77ed259fb",
        265_256_104,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "DK",
        "dk",
        "dk",
        "dk",
        "c0b080d8652abce3684322d40ca81d25a53430c9d14e172832e8eeb6b6055e4e",
        345_252_561,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "DOC",
        "doc",
        "doc",
        "doc",
        "459e3e2b02017d8d4c547dd4609f51970eef6ee26566cca3dbdada72b2ce03a0",
        345_253_183,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "GAMEANDWATCH",
        "gameandwatch",
        "gameandwatch",
        "gameandwatch",
        "b2692fcec81979c645fe99225f86ee11af9f964ee8fe70b4a80f0a86429f81ed",
        265_259_054,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "GANONDORF",
        "ganondorf",
        "ganondorf",
        "ganondorf",
        "ffd9c76c6576357df4ba8b1b84c3e0ad1558472c215da821607aa0ff2c24730c",
        345_257_043,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "POPO",
        "popo",
        "ice_climbers",
        "ice_climbers",
        "4c5037f48868500e8730fb9f462f6a1d9455e073c9f82d2b037837c6fbf3b04f",
        345_260_061,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "LINK",
        "link",
        "link",
        "link",
        "bd227f2a37f51ee6c2a5fa33449bea4bb531f237b60c543952a520961929a524",
        265_254_310,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "MARIO",
        "mario",
        "mario",
        "mario",
        "19740b53a7c41858859db1369b0405c23a36758a2bd6bf4d46b12f0653084a60",
        345_254_299,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "MEWTWO",
        "mewtwo",
        "mewtwo",
        "mewtwo",
        "806a4b4342ba59e49c8ee89322d108c999f84315f79f45011dcc0fbf199e1496",
        265_258_156,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "NESS",
        "ness",
        "ness",
        "ness",
        "453d2135da5cd14761b237d8a63ad5304b174e327aa2ef6fbd7ff58b3886ebcf",
        265_257_066,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "PEACH",
        "peach",
        "peach",
        "peach",
        "d72c65d535fb60d0211c65bbabad7e008ee33942c9abee75ffec67fb88bdf476",
        265_257_515,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "PIKACHU",
        "pikachu",
        "pikachu",
        "pikachu",
        "2d168b4d815e7a36466a29b3eae910c723c38348014d0c2015bfc813cae5d35f",
        265_256_553,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "JIGGLYPUFF",
        "jigglypuff",
        "puff",
        "puff",
        "3d716c89674dfdfb689c099999f4580186127ecbc87150ed6b30604c2136f917",
        345_253_741,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "ROY",
        "roy",
        "roy",
        "roy",
        "19aa05716e566aa2650d738f5e0015407cfa24860e7e0bfd3f623ea3455593e0",
        345_253_183,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "SAMUS",
        "samus",
        "samus",
        "samus",
        "551e70ca9d4eaa2243c317e148247a673ad4376fde2de04ab0eb7afde05485fb",
        345_254_299,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "YLINK",
        "ylink",
        "ylink",
        "ylink",
        "06d6699010fa746c6beccab221ff461e96a85a4f74face33df3b0d1201699a2e",
        265_257_515,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
    MimicRelease(
        "YOSHI",
        "yoshi",
        "yoshi",
        "yoshi",
        "3f9464c0c08a3f2a9b538625b6a7298a1b8743ccd26956a661b15b958a5bce59",
        265_254_759,
        revision=MIMIC_NATIVE_EXTENSION_REVISION,
        cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
    ),
)
MIMIC_CURRENT_MASTER_FOX_RELEASE = MimicRelease(
    "FOX",
    "master-fox",
    "fox-master",
    "fox-master",
    "af020445bd5ac4ad00c368c9e530c21c0a762348b33c1393a56e42478206a1a2",
    345_239_517,
    revision=MIMIC_RELEASE_REVISION,
    cache_directory=MIMIC_NATIVE_EXTENSION_CACHE_DIRECTORY,
)
MIMIC_NATIVE_RELEASES = (
    MIMIC_CURRENT_MASTER_FOX_RELEASE,
    *MIMIC_CURRENT_CORE_RELEASES[1:],
    *MIMIC_NATIVE_EXTENSION_RELEASES,
)
MIMIC_RELEASES = MIMIC_NATIVE_RELEASES
MIMIC_BY_CHARACTER = {release.character: release for release in MIMIC_NATIVE_RELEASES}
MIMIC_BY_DIRECTORY = {release.directory: release for release in MIMIC_RELEASES}

SLIPPI_CHARACTERS = (
    ("FOX", "fox"),
    ("FALCO", "falco"),
    ("MARTH", "marth"),
    ("SHEIK", "sheik"),
    ("JIGGLYPUFF", "jigglypuff"),
    ("CPTFALCON", "cptfalcon"),
    ("PEACH", "peach"),
    ("YOSHI", "yoshi"),
    ("POPO", "popo"),
    ("LUIGI", "luigi"),
    ("PIKACHU", "pikachu"),
    ("SAMUS", "samus"),
)

HEAD_TO_HEAD_SEEDS = (101, 211, 307, 401, 503)
TEN_GAME_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)
CHARACTER_SWEEP_SEEDS = (1201, 1301)
EXTENSION_SEEDS = (1401, 1501)
REMAINING_FRISSON_CHARACTERS = (
    ("MARIO", "mario"),
    ("DK", "dk"),
    ("KIRBY", "kirby"),
    ("BOWSER", "bowser"),
    ("LINK", "link"),
    ("NESS", "ness"),
    ("MEWTWO", "mewtwo"),
    ("ZELDA", "zelda"),
    ("YLINK", "ylink"),
    ("DOC", "doc"),
    ("PICHU", "pichu"),
    ("GAMEANDWATCH", "gameandwatch"),
    ("GANONDORF", "ganondorf"),
    ("ROY", "roy"),
)
MIMIC_ONLY_EXTENSION_CHARACTERS = (
    ("JIGGLYPUFF", "jigglypuff"),
    ("PEACH", "peach"),
    ("YOSHI", "yoshi"),
    ("POPO", "popo"),
    ("PIKACHU", "pikachu"),
    ("SAMUS", "samus"),
)
MIMIC_EXTENSION_CHARACTERS = (
    *REMAINING_FRISSON_CHARACTERS,
    *MIMIC_ONLY_EXTENSION_CHARACTERS,
)
MIMIC_OOD_CHARACTERS = ("KIRBY", "PICHU", "ZELDA")

MatchKind = Literal["head-to-head", "ancestor", "mimic", "cpu9", "slippi-ai"]
FrissonProfile = Literal["10m", "75m"]
MatchupClass = Literal["head-to-head", "mirror", "cross-character"]
OpponentEvaluationMode = Literal["supported", "forced-ood-character-transfer"]


@dataclass(frozen=True, slots=True)
class GameSpec:
    ordinal: int
    group: str
    kind: MatchKind
    label: str
    seed: int
    frisson_profile: FrissonProfile
    matchup_class: MatchupClass
    player_1_character: str
    opponent_character: str
    opponent_slug: str
    opponent_evaluation_mode: OpponentEvaluationMode = "supported"
    mimic_directory: str | None = None
    head_to_head_swap_ports: bool = False

    @property
    def artifact_root(self) -> Path:
        return ARTIFACT_OUTPUT / self.label

    @property
    def summary_path(self) -> Path:
        return self.artifact_root / "summary.json"


class SuiteArtifactError(RuntimeError):
    """A present artifact cannot be accepted under the frozen suite contract."""


def _mimic_release_for_spec(spec: GameSpec) -> MimicRelease:
    """Resolve a MIMIC bundle independently from the physical CSS character."""

    if spec.kind != "mimic" or spec.mimic_directory is None:
        raise SuiteArtifactError(f"{spec.label} does not identify a MIMIC release bundle")
    try:
        return MIMIC_BY_DIRECTORY[spec.mimic_directory]
    except KeyError as error:
        raise SuiteArtifactError(
            f"{spec.label} uses unknown MIMIC release directory {spec.mimic_directory!r}"
        ) from error


def _slippi_specialist_for_spec(spec: GameSpec) -> SlippiSpecialistRelease | None:
    if spec.kind != "slippi-ai" or "slippi-native-specialist" not in spec.group:
        return None
    try:
        return SLIPPI_SPECIALIST_BY_CHARACTER[spec.opponent_character]
    except KeyError as error:
        raise SuiteArtifactError(
            f"{spec.label} uses an unknown Slippi-AI native specialist"
        ) from error


def build_schedule() -> tuple[GameSpec, ...]:
    games: list[GameSpec] = []

    def add(
        *,
        group: str,
        kind: MatchKind,
        label: str,
        seed: int,
        frisson_profile: FrissonProfile,
        matchup_class: MatchupClass,
        player_1_character: str,
        opponent_character: str,
        opponent_slug: str,
        opponent_evaluation_mode: OpponentEvaluationMode = "supported",
        mimic_directory: str | None = None,
        head_to_head_swap_ports: bool = False,
    ) -> None:
        games.append(
            GameSpec(
                ordinal=len(games) + 1,
                group=group,
                kind=kind,
                label=label,
                seed=seed,
                frisson_profile=frisson_profile,
                matchup_class=matchup_class,
                player_1_character=player_1_character,
                opponent_character=opponent_character,
                opponent_slug=opponent_slug,
                opponent_evaluation_mode=opponent_evaluation_mode,
                mimic_directory=mimic_directory,
                head_to_head_swap_ports=head_to_head_swap_ports,
            )
        )

    def add_mimic_extension(
        *,
        profile: FrissonProfile,
        profile_slug: str,
        character: str,
        slug: str,
        seed: int,
    ) -> None:
        release = MIMIC_BY_CHARACTER.get(character)
        if release is not None:
            add(
                group=f"{profile}-mimic-native-extension-character-mirrors-2",
                kind="mimic",
                label=(
                    f"final-{profile_slug}-{slug}-v-mimic-native-{slug}-s{seed}"
                ),
                seed=seed,
                frisson_profile=profile,
                matchup_class="mirror",
                player_1_character=character,
                opponent_character=character,
                opponent_slug=slug,
                mimic_directory=release.directory,
            )
            return
        add(
            group=f"{profile}-mimic-current-fox-master-ood-character-mirrors-2",
            kind="mimic",
            label=(
                f"final-{profile_slug}-{slug}-v-mimic-current-fox-master-ood-"
                f"{slug}-s{seed}"
            ),
            seed=seed,
            frisson_profile=profile,
            matchup_class="mirror",
            player_1_character=character,
            opponent_character=character,
            opponent_slug=slug,
            opponent_evaluation_mode="forced-ood-character-transfer",
            mimic_directory="fox-master",
        )

    def add_slippi_extension(
        *,
        profile: FrissonProfile,
        profile_slug: str,
        character: str,
        slug: str,
        seed: int,
    ) -> None:
        specialist = SLIPPI_SPECIALIST_BY_CHARACTER.get(character)
        if specialist is not None:
            add(
                group=f"{profile}-slippi-native-specialist-character-mirrors-2",
                kind="slippi-ai",
                label=(
                    f"final-{profile_slug}-{slug}-v-slippi-native-{slug}-s{seed}"
                ),
                seed=seed,
                frisson_profile=profile,
                matchup_class="mirror",
                player_1_character=character,
                opponent_character=character,
                opponent_slug=slug,
            )
            return
        add(
            group=f"{profile}-slippi-medium-v2-ood-character-mirrors-2",
            kind="slippi-ai",
            label=(
                f"final-{profile_slug}-{slug}-v-slippi-medium-v2-ood-"
                f"{slug}-s{seed}"
            ),
            seed=seed,
            frisson_profile=profile,
            matchup_class="mirror",
            player_1_character=character,
            opponent_character=character,
            opponent_slug=slug,
            opponent_evaluation_mode="forced-ood-character-transfer",
        )

    for index, seed in enumerate(HEAD_TO_HEAD_SEEDS):
        add(
            group="head-to-head-5",
            kind="head-to-head",
            label=(
                f"final-{TEN_M_PROFILE_SLUG}-v-"
                f"{SEVENTY_FIVE_M_PROFILE_SLUG}-s{seed}"
            ),
            seed=seed,
            frisson_profile="75m" if index % 2 == 1 else "10m",
            matchup_class="head-to-head",
            player_1_character="FOX",
            opponent_character="FOX",
            opponent_slug="fox",
            head_to_head_swap_ports=index % 2 == 1,
        )
    profiles: tuple[tuple[FrissonProfile, str], ...] = (
        ("75m", SEVENTY_FIVE_M_PROFILE_SLUG),
        ("10m", TEN_M_PROFILE_SLUG),
    )
    for profile, profile_slug in profiles:
        for seed in TEN_GAME_SEEDS:
            add(
                group=f"{profile}-mimic-current-master-fox-10",
                kind="mimic",
                label=(
                    f"final-{SEVENTY_FIVE_M_PROFILE_SLUG}-v-mimic-current-master-fox-s{seed}"
                    if profile == "75m"
                    else f"final-{profile_slug}-fox-v-mimic-current-master-fox-s{seed}"
                ),
                seed=seed,
                frisson_profile=profile,
                matchup_class="mirror",
                player_1_character="FOX",
                opponent_character="FOX",
                opponent_slug="master-fox",
                mimic_directory="fox-master",
            )
        for seed in TEN_GAME_SEEDS:
            add(
                group=f"{profile}-cpu9-fox-10",
                kind="cpu9",
                label=(
                    f"final-{SEVENTY_FIVE_M_PROFILE_SLUG}-v-cpu9-fox-s{seed}"
                    if profile == "75m"
                    else f"final-{profile_slug}-fox-v-cpu9-fox-s{seed}"
                ),
                seed=seed,
                frisson_profile=profile,
                matchup_class="mirror",
                player_1_character="FOX",
                opponent_character="FOX",
                opponent_slug="fox",
            )
        for core_release in MIMIC_CURRENT_CORE_RELEASES:
            release = MIMIC_BY_CHARACTER[core_release.character]
            for seed in CHARACTER_SWEEP_SEEDS:
                add(
                    group=f"{profile}-mimic-current-native-core-character-mirrors-2",
                    kind="mimic",
                    label=(
                        f"final-{profile_slug}-fox-v-mimic-current-master-fox-s{seed}"
                        if release.character == "FOX"
                        else (
                            f"final-{profile_slug}-{release.slug}-v-mimic-current-"
                            f"{release.slug}-s{seed}"
                        )
                    ),
                    seed=seed,
                    frisson_profile=profile,
                    matchup_class="mirror",
                    player_1_character=release.character,
                    opponent_character=release.character,
                    opponent_slug=core_release.slug,
                    mimic_directory=release.directory,
                )
        for character, slug in SLIPPI_CHARACTERS:
            for seed in CHARACTER_SWEEP_SEEDS:
                add(
                    group=f"{profile}-slippi-medium-v2-all-character-mirrors-2",
                    kind="slippi-ai",
                    label=(
                        f"final-{SEVENTY_FIVE_M_PROFILE_SLUG}-v-slippi-medium-v2-fox-s{seed}"
                        if profile == "75m" and character == "FOX"
                        else f"final-{profile_slug}-{slug}-v-slippi-medium-v2-{slug}-s{seed}"
                    ),
                    seed=seed,
                    frisson_profile=profile,
                    matchup_class="mirror",
                    player_1_character=character,
                    opponent_character=character,
                    opponent_slug=slug,
                )

    for profile, profile_slug in profiles:
        for character, slug in REMAINING_FRISSON_CHARACTERS:
            for seed in EXTENSION_SEEDS:
                add(
                    group=f"{profile}-remaining-cpu9-character-mirrors-2",
                    kind="cpu9",
                    label=f"final-{profile_slug}-{slug}-v-cpu9-{slug}-ext-s{seed}",
                    seed=seed,
                    frisson_profile=profile,
                    matchup_class="mirror",
                    player_1_character=character,
                    opponent_character=character,
                    opponent_slug=slug,
                )
                add_mimic_extension(
                    profile=profile,
                    profile_slug=profile_slug,
                    character=character,
                    slug=slug,
                    seed=seed,
                )
                add_slippi_extension(
                    profile=profile,
                    profile_slug=profile_slug,
                    character=character,
                    slug=slug,
                    seed=seed,
                )

        # These six mirrors complete the same physical roster coverage while
        # retaining the v4 ordinal and seed layout.
        for character, slug in MIMIC_ONLY_EXTENSION_CHARACTERS:
            for seed in EXTENSION_SEEDS:
                add_mimic_extension(
                    profile=profile,
                    profile_slug=profile_slug,
                    character=character,
                    slug=slug,
                    seed=seed,
                )

    if POSTTRAINING_BENCHMARK:
        for profile, post_step, ancestor_step in (
            ("75m", SEVENTY_FIVE_M_STEP, 86_016),
            ("10m", TEN_M_STEP, 122_064),
        ):
            for index, seed in enumerate(TEN_GAME_SEEDS):
                add(
                    group=f"{profile}-{CURRENT_PHASE}-v-pretraining-ancestor-10",
                    kind="ancestor",
                    label=(
                        f"final-{profile}-{'postrl' if POST_RL_BENCHMARK else 'posttrained'}-step{post_step}-v-"
                        f"{profile}-ancestor-step{ancestor_step}-s{seed}"
                    ),
                    seed=seed,
                    frisson_profile=profile,
                    matchup_class="head-to-head",
                    player_1_character="FOX",
                    opponent_character="FOX",
                    opponent_slug="ancestor",
                    head_to_head_swap_ports=index % 2 == 1,
                )

    schedule = tuple(games)
    if len(schedule) != GAME_COUNT:
        raise AssertionError(f"suite schedule contains {len(schedule)} games")
    labels = [game.label for game in schedule]
    if len(labels) != len(set(labels)):
        raise AssertionError("suite schedule contains duplicate labels")
    for game in schedule:
        if game.opponent_evaluation_mode == "supported":
            continue
        if (
            game.opponent_evaluation_mode != "forced-ood-character-transfer"
            or game.kind not in {"mimic", "slippi-ai"}
            or game.matchup_class != "mirror"
            or game.player_1_character != game.opponent_character
        ):
            raise AssertionError(f"invalid OOD character-transfer game: {game.label}")
        if game.kind == "mimic":
            release = _mimic_release_for_spec(game)
            if release.directory != "fox-master" or game.opponent_character in MIMIC_BY_CHARACTER:
                raise AssertionError(f"invalid MIMIC OOD character-transfer game: {game.label}")
        elif game.opponent_character in {
            *{character for character, _ in SLIPPI_CHARACTERS},
            *SLIPPI_SPECIALIST_BY_CHARACTER,
        }:
            raise AssertionError(f"invalid Slippi-AI OOD character-transfer game: {game.label}")
    return schedule


SCHEDULE = build_schedule()
SCHEDULE_BY_LABEL = {game.label: game for game in SCHEDULE}


def _legacy_v1_label(spec: GameSpec) -> str | None:
    if spec.kind == "head-to-head":
        return f"final-10m-step122064-v-75m-step86016-s{spec.seed}"
    if spec.frisson_profile != "75m" or spec.player_1_character != "FOX":
        return None
    if spec.group.endswith("mimic-master-fox-10"):
        return f"final-75m-step86016-v-mimic-master-fox-s{spec.seed}"
    if spec.kind == "cpu9":
        return f"final-75m-step86016-v-cpu9-fox-s{spec.seed}"
    if spec.kind == "mimic" and spec.opponent_character == "FOX":
        return f"final-75m-step86016-v-mimic-fox-s{spec.seed}"
    if spec.kind == "slippi-ai" and spec.opponent_character == "FOX":
        return f"final-75m-step86016-v-slippi-medium-v2-fox-s{spec.seed}"
    return None


def _rewrite_artifact_paths(value: Any, old_label: str, new_label: str) -> Any:
    if isinstance(value, str):
        return value.replace(f"/{old_label}/", f"/{new_label}/")
    if isinstance(value, list):
        return [_rewrite_artifact_paths(item, old_label, new_label) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _rewrite_artifact_paths(item, old_label, new_label)
            for key, item in value.items()
        }
    return value


def import_semantically_valid_v1_artifacts() -> list[dict[str, Any]]:
    """Copy validated v1 raw artifacts into current labels without changing source SLPs."""

    if POSTTRAINING_BENCHMARK:
        return []
    imports: list[dict[str, Any]] = []
    ARTIFACT_OUTPUT.mkdir(parents=True, exist_ok=True)
    for spec in SCHEDULE:
        legacy_label = _legacy_v1_label(spec)
        if legacy_label is None:
            continue
        legacy_spec = replace(spec, label=legacy_label)
        if not legacy_spec.summary_path.is_file():
            continue
        try:
            legacy_result = validate_completed_game(legacy_spec)
        except (FileNotFoundError, SuiteArtifactError):
            continue
        if legacy_label == spec.label:
            imports.append(
                {
                    "mode": "in-place-validated",
                    "source_label": legacy_label,
                    "target_label": spec.label,
                    "winner": legacy_result["winner"],
                    "replay_sha256": legacy_result["replay"]["sha256"],
                }
            )
            continue
        if spec.artifact_root.exists():
            continue
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{spec.label}.import-", dir=ARTIFACT_OUTPUT)
        )
        try:
            for source in legacy_spec.artifact_root.iterdir():
                if source.name == "game" or source.name.startswith(".game-finalize"):
                    continue
                destination = temporary / source.name
                if source.is_dir():
                    shutil.copytree(source, destination, copy_function=shutil.copy2)
                elif source.is_file():
                    shutil.copy2(source, destination)
            imported_summary = read_json(temporary / "summary.json", "imported v1 summary")
            rewritten = require_mapping(
                _rewrite_artifact_paths(imported_summary, legacy_label, spec.label),
                "rewritten imported summary",
            )
            rewritten["suite_import"] = {
                "schema_version": "integration.final_winner_benchmark_import.v1",
                "source_suite_id": V1_SUITE_ID,
                "source_label": legacy_label,
                "source_summary_sha256": sha256_file(legacy_spec.summary_path),
                "source_replay": legacy_result["replay"],
                "raw_media_copied": False,
                "gameplay_rerun": False,
            }
            atomic_json(temporary / "summary.json", rewritten)
            temporary.replace(spec.artifact_root)
            result = validate_completed_game(spec)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            if spec.artifact_root.exists():
                shutil.rmtree(spec.artifact_root)
            raise
        imports.append(
            {
                "mode": "copied-raw-artifact",
                "source_label": legacy_label,
                "target_label": spec.label,
                "winner": result["winner"],
                "replay_sha256": result["replay"]["sha256"],
            }
        )
    return imports


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise SuiteArtifactError(f"file changed while hashing: {path}")
    current = path.stat()
    if any(getattr(current, name) != getattr(after, name) for name in stable):
        raise SuiteArtifactError(f"file path changed while hashing: {path}")
    return digest.hexdigest()


def verify_file_identity(identity: FileIdentity, label: str) -> dict[str, Any]:
    path = identity.path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed_bytes = path.stat().st_size
    if observed_bytes != identity.byte_length:
        raise SuiteArtifactError(
            f"{label} byte length differs: {observed_bytes} != {identity.byte_length}"
        )
    observed_sha = sha256_file(path)
    if observed_sha != identity.sha256:
        raise SuiteArtifactError(
            f"{label} SHA-256 differs: {observed_sha} != {identity.sha256}"
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": observed_sha,
        "byte_length": observed_bytes,
    }


def _checkpoint_for_profile(profile: FrissonProfile) -> tuple[FileIdentity, int, int]:
    if profile == "10m":
        return TEN_M_IDENTITY, TEN_M_STEP, TEN_M_FRAMES
    return SEVENTY_FIVE_M_IDENTITY, SEVENTY_FIVE_M_STEP, SEVENTY_FIVE_M_FRAMES


def _checkpoint_capability(path: Path) -> dict[str, Any]:
    import torch

    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    expected_unsafe_globals = ["builtins.frozenset"] if POSTTRAINING_BENCHMARK else []
    if unsafe_globals != expected_unsafe_globals:
        raise SuiteArtifactError(
            f"{path.name} serialized global set differs: {unsafe_globals!r}"
        )
    safe_context = (
        torch.serialization.safe_globals([frozenset])
        if POSTTRAINING_BENCHMARK
        else nullcontext()
    )
    with safe_context:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping):
        raise SuiteArtifactError(f"checkpoint payload is malformed: {path}")
    model = require_mapping(payload.get("model"), "checkpoint.model")
    if POSTTRAINING_BENCHMARK:
        resolved = require_mapping(
            payload.get("resolved_config"),
            "checkpoint.resolved_config",
        )
        dataset = require_mapping(
            resolved.get("dataset"),
            "checkpoint.resolved_config.dataset",
        )
    else:
        metadata = require_mapping(payload.get("metadata"), "checkpoint.metadata")
        dataset = require_mapping(metadata.get("dataset"), "checkpoint.metadata.dataset")
    character = model.get("encoder.character_embedding.weight")
    character_action = model.get("encoder.character_action_embedding.weight")
    if not isinstance(character, torch.Tensor) or tuple(character.shape) != (33, 128):
        raise SuiteArtifactError(f"{path.name} character embedding is not 33x128")
    if not isinstance(character_action, torch.Tensor) or tuple(character_action.shape) != (
        33 * 399,
        128,
    ):
        raise SuiteArtifactError(f"{path.name} character-action embedding is not 33*399x128")
    grouped = character_action.reshape(33, 399, 128)
    physical_nonzero = [
        index for index in range(27) if int(torch.count_nonzero(grouped[index]).item()) > 0
    ]
    non_roster_zero = [
        index for index in range(27, 33) if int(torch.count_nonzero(grouped[index]).item()) == 0
    ]
    character_nonzero = [
        index for index in range(33) if int(torch.count_nonzero(character[index]).item()) > 0
    ]
    checks = {
        "dataset_manifest_exact": dataset.get("dataset_manifest_id") == DATASET_MANIFEST_ID,
        "dataset_slippi_commit_exact": dataset.get("slippi_ai_commit") == DATASET_SLIPPI_COMMIT,
        "character_embedding_shape_33x128": tuple(character.shape) == (33, 128),
        "character_embedding_all_rows_nonzero": character_nonzero == list(range(33)),
        "character_action_embedding_shape_33x399x128": tuple(grouped.shape) == (33, 399, 128),
        "physical_character_action_rows_0_through_26_nonzero": physical_nonzero == list(range(27)),
        "non_roster_character_action_rows_27_through_32_zero": non_roster_zero == list(range(27, 33)),
    }
    if not all(checks.values()):
        raise SuiteArtifactError(f"{path.name} full-roster capability gate failed: {checks}")
    return {
        "dataset_manifest_id": dataset["dataset_manifest_id"],
        "slippi_ai_commit": dataset["slippi_ai_commit"],
        "character_embedding_shape": list(character.shape),
        "character_embedding_nonzero_rows": character_nonzero,
        "character_action_embedding_shape": list(grouped.shape),
        "character_action_nonzero_physical_ids": physical_nonzero,
        "character_action_zero_non_roster_ids": non_roster_zero,
        "checks": checks,
    }


def verify_frozen_inputs(
    *,
    include_opponents: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    selected_config = (
        ROOT / "configs" / "integration.toml"
        if config_path is None
        else config_path.expanduser().resolve()
    )
    with selected_config.open("rb") as stream:
        config = tomllib.load(stream)
    configured_roster = require_mapping(config.get("frisson_ai"), "config.frisson_ai").get(
        "allowed_characters"
    )
    if configured_roster != list(FRISSON_ROSTER):
        raise SuiteArtifactError(
            "configured Frisson roster differs from the canonical physical-ID order"
        )
    ten_key = f"10m_step{TEN_M_STEP}"
    seventy_five_key = f"75m_step{SEVENTY_FIVE_M_STEP}"
    records: dict[str, Any] = {
        ten_key: verify_file_identity(TEN_M_IDENTITY, "10M final winner"),
        seventy_five_key: verify_file_identity(
            SEVENTY_FIVE_M_IDENTITY, "75M final winner"
        ),
    }
    if POSTTRAINING_BENCHMARK:
        records["10m_pretraining_ancestor_step122064"] = verify_file_identity(
            TEN_M_ANCESTOR_IDENTITY,
            "10M pretraining ancestor",
        )
        records["75m_pretraining_ancestor_step86016"] = verify_file_identity(
            SEVENTY_FIVE_M_ANCESTOR_IDENTITY,
            "75M pretraining ancestor",
        )
    records["frisson_full_roster"] = {
        "config_path": str(selected_config.relative_to(ROOT)),
        "configured_roster_exact": True,
        "launchable_characters": list(FRISSON_ROSTER),
        "launchable_character_count": len(FRISSON_ROSTER),
        ten_key: _checkpoint_capability(TEN_M_IDENTITY.path),
        seventy_five_key: _checkpoint_capability(SEVENTY_FIVE_M_IDENTITY.path),
    }
    if len(FRISSON_ROSTER) != 26 or len(set(FRISSON_ROSTER)) != 26:
        raise SuiteArtifactError("Frisson launchable roster must contain 26 unique characters")
    if include_opponents:
        records["slippi_medium_v2"] = verify_file_identity(
            SLIPPI_MEDIUM_V2_IDENTITY, "Slippi-AI medium-v2"
        )
        records["slippi_specialists"] = {
            release.character: verify_file_identity(
                FileIdentity(
                    release.checkpoint,
                    release.checkpoint_sha256,
                    release.checkpoint_byte_length,
                ),
                f"Slippi-AI {release.release_name}",
            )
            for release in SLIPPI_SPECIALIST_RELEASES
        }
        records["mimic"] = {
            release.slug: verify_file_identity(
                FileIdentity(
                    release.checkpoint,
                    release.checkpoint_sha256,
                    release.checkpoint_byte_length,
                ),
                f"MIMIC {release.character}",
            )
            for release in MIMIC_RELEASES
        }
    return records


def schedule_manifest() -> dict[str, Any]:
    games = []
    for spec in SCHEDULE:
        record = asdict(spec)
        record["artifact_root"] = str(spec.artifact_root.relative_to(ROOT))
        games.append(record)
    body = {
        "suite_id": SUITE_ID,
        "game_count": GAME_COUNT,
        "stage": STAGE,
        "frisson_roster": list(FRISSON_ROSTER),
        "opponent_bundles": {
            "mimic": [
                {
                    "character": release.character,
                    "directory": release.directory,
                    "bundle_name": release.bundle_name,
                    "revision": release.revision,
                    "cache_directory": release.cache_directory,
                    "source_repository": MIMIC_SOURCE_REPOSITORY,
                    "source_revision": MIMIC_SOURCE_REVISION,
                    "source_directory": MIMIC_SOURCE_DIRECTORY,
                    "checkpoint_sha256": release.checkpoint_sha256,
                    "checkpoint_byte_length": release.checkpoint_byte_length,
                }
                for release in MIMIC_RELEASES
            ],
            "slippi_ai": {
                "medium_v2": {
                    "source_revision": SLIPPI_SOURCE_REVISION,
                    "checkpoint_sha256": SLIPPI_MEDIUM_V2_IDENTITY.sha256,
                    "checkpoint_byte_length": SLIPPI_MEDIUM_V2_IDENTITY.byte_length,
                    "policy_delay_frames": SLIPPI_POLICY_DELAY_FRAMES,
                },
                "native_specialists": [
                    {
                        "character": release.character,
                        "release_name": release.release_name,
                        "checkpoint_sha256": release.checkpoint_sha256,
                        "checkpoint_byte_length": release.checkpoint_byte_length,
                        "policy_delay_frames": release.policy_delay_frames,
                        "release_contract": _slippi_specialist_release_contract(release),
                    }
                    for release in SLIPPI_SPECIALIST_RELEASES
                ],
            },
        },
        "final_winners": {
            "10m_postrl" if POST_RL_BENCHMARK else "10m_posttrained" if POSTTRAINING_BENCHMARK else "10m_v2": {
                "step": TEN_M_STEP,
                "processed_target_frames": TEN_M_FRAMES,
                "validation_nll": TEN_M_VALIDATION_NLL,
                "path": str(TEN_M_IDENTITY.path.relative_to(ROOT)),
                "sha256": TEN_M_IDENTITY.sha256,
                "byte_length": TEN_M_IDENTITY.byte_length,
            },
            "75m_postrl" if POST_RL_BENCHMARK else "75m_posttrained" if POSTTRAINING_BENCHMARK else "75m": {
                "step": SEVENTY_FIVE_M_STEP,
                "processed_target_frames": SEVENTY_FIVE_M_FRAMES,
                "validation_nll": SEVENTY_FIVE_M_VALIDATION_NLL,
                "path": str(SEVENTY_FIVE_M_IDENTITY.path.relative_to(ROOT)),
                "sha256": SEVENTY_FIVE_M_IDENTITY.sha256,
                "byte_length": SEVENTY_FIVE_M_IDENTITY.byte_length,
            },
        },
        "games": games,
    }
    if POST_RL_BENCHMARK:
        body["post_rl_protocol"] = {
            "save_slp": True, "save_video": False,
            "checkpoint_selection": "user-selected frozen P21 exports",
            "native_training_context_mode": "prefix",
            "evaluation_context_mode": "ring",
            "comparison": "same established benchmark runtime, checkpoints changed",
            "processed_target_frames": "unknown for RL; inherited BC counters excluded",
        }
    if POSTTRAINING_BENCHMARK:
        body["pretraining_ancestors"] = {
            "10m": {
                "step": 122_064,
                "processed_target_frames": 7_999_586_304,
                "path": str(TEN_M_ANCESTOR_IDENTITY.path.relative_to(ROOT)),
                "sha256": TEN_M_ANCESTOR_IDENTITY.sha256,
                "byte_length": TEN_M_ANCESTOR_IDENTITY.byte_length,
            },
            "75m": {
                "step": 86_016,
                "processed_target_frames": 5_637_144_576,
                "path": str(
                    SEVENTY_FIVE_M_ANCESTOR_IDENTITY.path.relative_to(ROOT)
                ),
                "sha256": SEVENTY_FIVE_M_ANCESTOR_IDENTITY.sha256,
                "byte_length": SEVENTY_FIVE_M_ANCESTOR_IDENTITY.byte_length,
            },
        }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "schedule_sha256": hashlib.sha256(canonical).hexdigest(),
        **body,
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteArtifactError(f"{label} is unreadable: {error}") from error
    return require_mapping(value, label)


def require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SuiteArtifactError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SuiteArtifactError(f"{label} must be an array")
    return value


def resolve_artifact_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise SuiteArtifactError("artifact path must be a nonempty string")
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise SuiteArtifactError(f"artifact path escapes the project: {resolved}")
    return resolved


def _selected_replay(
    summary: Mapping[str, Any], spec: GameSpec
) -> tuple[Path, dict[str, Any]]:
    artifacts = require_mapping(summary.get("artifacts"), "summary.artifacts")
    replay_records = [
        require_mapping(record, f"summary.artifacts.replays[{index}]")
        for index, record in enumerate(
            require_sequence(artifacts.get("replays"), "summary.artifacts.replays")
        )
    ]
    selected = [
        record for record in replay_records if record.get("tournament_result_replay") is True
    ]
    if len(selected) != 1:
        raise SuiteArtifactError(
            f"{spec.label} has {len(selected)} selected scoring replays"
        )
    record = selected[0]
    replay_path = resolve_artifact_path(record.get("path"))
    if not replay_path.is_relative_to(spec.artifact_root.resolve()):
        raise SuiteArtifactError(f"{spec.label} scoring replay escapes its artifact root")
    if not replay_path.is_file() or replay_path.stat().st_size <= 0:
        raise SuiteArtifactError(f"{spec.label} scoring replay is missing or empty")
    observed_bytes = replay_path.stat().st_size
    if record.get("byte_length") != observed_bytes:
        raise SuiteArtifactError(f"{spec.label} scoring replay byte length changed")
    observed_sha = sha256_file(replay_path)
    if record.get("sha256") != observed_sha:
        raise SuiteArtifactError(f"{spec.label} scoring replay SHA-256 changed")
    validation = require_mapping(record.get("validation"), "scoring replay validation")
    coverage = require_mapping(
        validation.get("required_trace_coverage"), "scoring replay trace coverage"
    )
    first = validation.get("first_parsed_frame")
    last = validation.get("last_parsed_frame")
    if (
        validation.get("libmelee_parseable") is not True
        or validation.get("parsed_frames_strictly_consecutive") is not True
        or coverage.get("complete") is not True
        or isinstance(first, bool)
        or not isinstance(first, int)
        or isinstance(last, bool)
        or not isinstance(last, int)
        or last < first
    ):
        raise SuiteArtifactError(f"{spec.label} scoring replay failed completeness checks")
    return replay_path, record


def _require_checkpoint(
    value: object,
    identity: FileIdentity,
    *,
    step: int,
    frames: int,
    label: str,
) -> dict[str, Any]:
    record = require_mapping(value, label)
    expected = {
        "sha256": identity.sha256,
        "byte_length": identity.byte_length,
        "step": step,
        "processed_target_frames": frames,
    }
    differences = {
        key: (record.get(key), wanted)
        for key, wanted in expected.items()
        if record.get(key) != wanted
    }
    if differences:
        raise SuiteArtifactError(f"{label} identity differs: {differences}")
    path = record.get("path")
    if not isinstance(path, str) or Path(path).expanduser().resolve() != identity.path.resolve():
        raise SuiteArtifactError(f"{label} path differs from the frozen checkpoint")
    return record


def _strict_json_equal(observed: object, expected: object) -> bool:
    """Compare provenance as typed JSON, including exact nested object keys."""

    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            return False
        observed_mapping = {str(key): value for key, value in observed.items()}
        expected_mapping = {str(key): value for key, value in expected.items()}
        return set(observed_mapping) == set(expected_mapping) and all(
            _strict_json_equal(observed_mapping[key], value)
            for key, value in expected_mapping.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _strict_json_equal(observed_value, expected_value)
                for observed_value, expected_value in zip(observed, expected, strict=True)
            )
        )
    return type(observed) is type(expected) and observed == expected


def _require_exact_json_mapping(
    value: object,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    record = require_mapping(value, label)
    if not _strict_json_equal(record, expected):
        raise SuiteArtifactError(f"{label} differs from the frozen provenance contract")
    return record


def _require_exact_json_fields(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    differences = {
        key: (record.get(key), wanted)
        for key, wanted in expected.items()
        if not _strict_json_equal(record.get(key), wanted)
    }
    if differences:
        raise SuiteArtifactError(f"{label} differs: {differences}")


def _mimic_ood_native_inference_contract() -> dict[str, Any]:
    return {
        "live_policy_class": "MimicLivePolicy",
        "observation_builder": "tools.inference_utils.build_frame_p2",
        "model_forward": "MimicRuntime.model",
        "sampler_decoder": "tools.inference_utils.decode_and_press",
        "decode_strategy": "categorical-sampling",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.0,
        "online_delay_frames": 0,
        "inference_mode": "synchronous-concurrent",
        "current_frame_barrier_before_controller_transaction": True,
    }


def _dual_player_assignments(
    spec: GameSpec,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the exact physical-port and scoring-role contract for a dual match."""

    current_names = {
        "10m": (
            "Frisson-AI 10M post-trained step-195248"
            if POSTTRAINING_BENCHMARK
            else "Frisson-AI 10M v2 step-122064"
        ),
        "75m": (
            "Frisson-AI 75M post-trained step-127214"
            if POSTTRAINING_BENCHMARK
            else "Frisson-AI 75M step-86016"
        ),
    }
    if POST_RL_BENCHMARK:
        current_names = {"10m": "Frisson-AI 10M post-RL step-1318", "75m": "Frisson-AI 75M post-RL step-222"}
    if spec.kind == "head-to-head":
        ten = {"model": current_names["10m"], "profile": "10m", "role": "10m"}
        seventy_five = {
            "model": current_names["75m"],
            "profile": "75m",
            "role": "75m",
        }
        return (
            (seventy_five, ten)
            if spec.head_to_head_swap_ports
            else (ten, seventy_five)
        )
    if spec.kind == "ancestor":
        profile = spec.frisson_profile
        posttrained = {
            "model": current_names[profile],
            "profile": profile,
            "role": f"{profile}-{CURRENT_PHASE}",
        }
        ancestor = {
            "model": (
                "Frisson-AI 10M v2 step-122064"
                if profile == "10m"
                else "Frisson-AI 75M step-86016"
            ),
            "profile": profile,
            "role": f"{profile}-pretraining",
        }
        return (
            (ancestor, posttrained)
            if spec.head_to_head_swap_ports
            else (posttrained, ancestor)
        )
    raise SuiteArtifactError(f"{spec.label} is not a dual-Frisson match")


def _validate_players(summary: Mapping[str, Any], spec: GameSpec) -> None:
    configuration = require_mapping(summary.get("configuration"), "summary.configuration")
    if (
        configuration.get("seed") != spec.seed
        or configuration.get("stage") != STAGE
        or configuration.get("require_natural_end") is not True
    ):
        raise SuiteArtifactError(f"{spec.label} seed, stage, or natural-end contract differs")
    p1 = require_mapping(configuration.get("player_1"), "configuration.player_1")
    p2 = require_mapping(configuration.get("player_2"), "configuration.player_2")
    if p1.get("port") != 1 or p2.get("port") != 2:
        raise SuiteArtifactError(f"{spec.label} physical ports differ")
    if spec.kind in {"head-to-head", "ancestor"}:
        if spec.matchup_class != "head-to-head":
            raise SuiteArtifactError(f"{spec.label} checkpoint-pair matchup class differs")
        expected_characters = ("FOX", "FOX")
        assignments = _dual_player_assignments(spec)
        for port, (player, assignment) in enumerate(
            zip((p1, p2), assignments, strict=True),
            start=1,
        ):
            _require_exact_json_fields(
                player,
                {
                    "model": assignment["model"],
                    "profile": assignment["profile"],
                    "character": "FOX",
                    "port": port,
                },
                label=f"{spec.label} P{port} dual-player assignment",
            )
        role_to_port = {
            assignment["role"]: port
            for port, assignment in enumerate(assignments, start=1)
        }
        _require_exact_json_mapping(
            configuration.get("model_to_port"),
            role_to_port,
            label=f"{spec.label} scoring-role port assignment",
        )
        _require_exact_json_mapping(
            configuration.get("policy_evaluation_seeds"),
            {role: spec.seed for role in role_to_port},
            label=f"{spec.label} policy seed assignment",
        )
        if p1.get("profile") != spec.frisson_profile:
            raise SuiteArtifactError(f"{spec.label} P1 Frisson profile differs")
    else:
        expected_characters = (spec.player_1_character, spec.opponent_character)
        if p1.get("model") != "frisson-ai":
            raise SuiteArtifactError(f"{spec.label} P1 is not Frisson-AI")
        expected_p2_model = {
            "mimic": "mimic",
            "cpu9": "cpu",
            "slippi-ai": "slippi-ai",
        }[spec.kind]
        if p2.get("model") != expected_p2_model:
            raise SuiteArtifactError(f"{spec.label} P2 model differs")
    if (p1.get("character"), p2.get("character")) != expected_characters:
        raise SuiteArtifactError(f"{spec.label} character assignment differs")
    if spec.matchup_class == "mirror" and spec.player_1_character != spec.opponent_character:
        raise SuiteArtifactError(f"{spec.label} is not a same-character mirror")
    if spec.matchup_class == "cross-character" and (
        spec.kind not in {"mimic", "slippi-ai"}
        or spec.player_1_character == spec.opponent_character
    ):
        raise SuiteArtifactError(f"{spec.label} cross-character contract differs")
    if spec.opponent_evaluation_mode == "forced-ood-character-transfer":
        if (
            spec.kind not in {"mimic", "slippi-ai"}
            or spec.matchup_class != "mirror"
            or spec.player_1_character != spec.opponent_character
        ):
            raise SuiteArtifactError(f"{spec.label} OOD character-transfer contract differs")
    elif spec.opponent_evaluation_mode != "supported":
        raise SuiteArtifactError(f"{spec.label} opponent evaluation mode differs")
    if spec.kind == "cpu9" and (
        p2.get("cpu_level") != 9 or p2.get("gameplay_owner") != "melee-native-cpu"
    ):
        raise SuiteArtifactError(f"{spec.label} CPU level or implementation differs")


def _validate_model_identity(summary: Mapping[str, Any], spec: GameSpec) -> None:
    if spec.kind in {"head-to-head", "ancestor"}:
        checkpoints = require_mapping(summary.get("checkpoints"), "summary.checkpoints")
        p1 = require_mapping(checkpoints.get("p1"), "checkpoints.p1")
        p2 = require_mapping(checkpoints.get("p2"), "checkpoints.p2")
        if spec.kind == "head-to-head":
            wanted = (
                (SEVENTY_FIVE_M_IDENTITY, SEVENTY_FIVE_M_STEP, SEVENTY_FIVE_M_FRAMES),
                (TEN_M_IDENTITY, TEN_M_STEP, TEN_M_FRAMES),
            ) if spec.head_to_head_swap_ports else (
                (TEN_M_IDENTITY, TEN_M_STEP, TEN_M_FRAMES),
                (SEVENTY_FIVE_M_IDENTITY, SEVENTY_FIVE_M_STEP, SEVENTY_FIVE_M_FRAMES),
            )
        else:
            posttraining = _checkpoint_for_profile(spec.frisson_profile)
            ancestor = (
                (TEN_M_ANCESTOR_IDENTITY, 122_064, 7_999_586_304)
                if spec.frisson_profile == "10m"
                else (
                    SEVENTY_FIVE_M_ANCESTOR_IDENTITY,
                    86_016,
                    5_637_144_576,
                )
            )
            wanted = (
                (ancestor, posttraining)
                if spec.head_to_head_swap_ports
                else (posttraining, ancestor)
            )
        _require_checkpoint(
            p1,
            wanted[0][0],
            step=wanted[0][1],
            frames=wanted[0][2],
            label="head-to-head P1 checkpoint",
        )
        _require_checkpoint(
            p2,
            wanted[1][0],
            step=wanted[1][1],
            frames=wanted[1][2],
            label="head-to-head P2 checkpoint",
        )
        return

    contract = require_mapping(summary.get("contract"), "summary.contract")
    selected_value: object
    if spec.kind == "slippi-ai" and isinstance(contract.get("frisson"), Mapping):
        selected_value = require_mapping(contract.get("frisson"), "contract.frisson").get(
            "selected_checkpoint"
        )
    else:
        selected_value = contract.get("selected_checkpoint")
    identity, step, frames = _checkpoint_for_profile(spec.frisson_profile)
    _require_checkpoint(
        selected_value,
        identity,
        step=step,
        frames=frames,
        label=f"{spec.frisson_profile} final checkpoint",
    )

    if spec.kind == "mimic":
        release = _mimic_release_for_spec(spec)
        mimic = require_mapping(summary.get("mimic"), "summary.mimic")
        bundle = require_mapping(mimic.get("bundle_identity"), "mimic.bundle_identity")
        if (
            mimic.get("checkpoint_sha256") != release.checkpoint_sha256
            or mimic.get("controlled_character") != release.character
            or bundle.get("checkpoint_sha256") != release.checkpoint_sha256
            or bundle.get("name") != release.bundle_name
            or bundle.get("character") != release.character
            or bundle.get("revision") != release.revision
            or bundle.get("source_repository") != MIMIC_SOURCE_REPOSITORY
            or bundle.get("source_revision") != MIMIC_SOURCE_REVISION
            or bundle.get("source_directory") != MIMIC_SOURCE_DIRECTORY
        ):
            raise SuiteArtifactError(f"{spec.label} MIMIC release identity differs")
        sources = require_mapping(summary.get("sources"), "summary.sources")
        source = require_mapping(sources.get("mimic"), "summary.sources.mimic")
        _require_exact_json_fields(
            source,
            {
                "repository_url": MIMIC_SOURCE_REPOSITORY,
                "revision": MIMIC_SOURCE_REVISION,
                "tracked_tree_clean": True,
                "directory": str((ROOT / MIMIC_SOURCE_DIRECTORY).resolve()),
                "bundle_character": release.character,
                "bundle_checkpoint_sha256": release.checkpoint_sha256,
            },
            label=f"{spec.label} MIMIC source identity",
        )
        _require_exact_json_mapping(
            source.get("checks"),
            {
                "repository_url_exact": True,
                "revision_exact": True,
                "tracked_tree_clean": True,
            },
            label=f"{spec.label} MIMIC source checks",
        )
        if spec.opponent_evaluation_mode == "forced-ood-character-transfer":
            _require_exact_json_fields(
                contract,
                {
                    "mimic_bundle_selection": (
                        "explicit-released-bundle-ood-character-transfer"
                    ),
                    "mimic_checkpoint_character": release.character,
                    "mimic_physical_character": spec.opponent_character,
                    "mimic_requested_character_covered_by_released_checkpoint": False,
                    "mimic_forced_ood_character_transfer": True,
                    "allow_player_2_ood_character": True,
                },
                label=f"{spec.label} MIMIC OOD launch contract",
            )
            configuration = require_mapping(
                summary.get("configuration"),
                "summary.configuration",
            )
            _require_exact_json_fields(
                configuration,
                {"allow_player_2_ood_character": True},
                label=f"{spec.label} MIMIC OOD launch configuration",
            )
            configured_p2 = require_mapping(
                configuration.get("player_2"),
                "configuration.player_2",
            )
            _require_exact_json_fields(
                configured_p2,
                {
                    "physical_character": spec.opponent_character,
                    "checkpoint_character": release.character,
                    "character_transfer_mode": MIMIC_OOD_TRANSFER_MODE,
                },
                label=f"{spec.label} MIMIC OOD physical player provenance",
            )
            transfer = require_mapping(
                contract.get("mimic_ood_character_transfer"),
                "contract.mimic_ood_character_transfer",
            )
            expected_transfer = {
                "schema_version": MIMIC_OOD_TRANSFER_SCHEMA_VERSION,
                "mode": MIMIC_OOD_TRANSFER_MODE,
                "explicit_opt_in_field": "allow_player_2_ood_character",
                "physical_port": 2,
                "checkpoint_character": release.character,
                "physical_character": spec.opponent_character,
                "physical_character_covered_by_released_checkpoint": False,
                "forced_ood_character_transfer": True,
                "checkpoint_roster_unchanged": True,
            }
            _require_exact_json_fields(
                transfer,
                expected_transfer,
                label=f"{spec.label} MIMIC OOD character-transfer identity",
            )
            checkpoint_provenance = require_mapping(
                transfer.get("checkpoint"),
                "contract.mimic_ood_character_transfer.checkpoint",
            )
            _require_exact_json_fields(
                checkpoint_provenance,
                {"sha256": release.checkpoint_sha256},
                label=f"{spec.label} MIMIC OOD checkpoint provenance",
            )
            asset_provenance = require_mapping(
                transfer.get("assets"),
                "contract.mimic_ood_character_transfer.assets",
            )
            _require_exact_json_fields(
                asset_provenance,
                {"released_bundle_name": release.bundle_name},
                label=f"{spec.label} MIMIC OOD asset provenance",
            )
            _require_exact_json_mapping(
                transfer.get("native_inference"),
                _mimic_ood_native_inference_contract(),
                label=f"{spec.label} MIMIC OOD native inference provenance",
            )
            _require_exact_json_mapping(
                transfer.get("checks"),
                {name: True for name in MIMIC_OOD_PROVENANCE_CHECKS},
                label=f"{spec.label} MIMIC OOD provenance checks",
            )
            if not _strict_json_equal(mimic.get("ood_character_transfer"), transfer):
                raise SuiteArtifactError(
                    f"{spec.label} MIMIC OOD duplicated provenance differs from its contract"
                )
    elif spec.kind == "cpu9":
        opponent = require_mapping(contract.get("opponent"), "contract.opponent")
        if opponent.get("implementation") != "melee-native-cpu" or opponent.get("cpu_level") != 9:
            raise SuiteArtifactError(f"{spec.label} CPU9 contract differs")
    elif spec.kind == "slippi-ai":
        slippi = require_mapping(contract.get("slippi_ai"), "contract.slippi_ai")
        specialist = _slippi_specialist_for_spec(spec)
        checkpoint_sha256 = (
            specialist.checkpoint_sha256
            if specialist is not None
            else SLIPPI_MEDIUM_V2_IDENTITY.sha256
        )
        checkpoint_byte_length = (
            specialist.checkpoint_byte_length
            if specialist is not None
            else SLIPPI_MEDIUM_V2_IDENTITY.byte_length
        )
        policy_delay_frames = (
            specialist.policy_delay_frames
            if specialist is not None
            else SLIPPI_POLICY_DELAY_FRAMES
        )
        expected = {
            "source_revision": SLIPPI_SOURCE_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_byte_length": checkpoint_byte_length,
            "player_name": SLIPPI_PLAYER_NAME,
            "sample_temperature": SLIPPI_TEMPERATURE,
            "checkpoint_policy_delay_frames": policy_delay_frames,
        }
        differences = {
            key: (slippi.get(key), value)
            for key, value in expected.items()
            if slippi.get(key) != value
        }
        if differences:
            raise SuiteArtifactError(f"{spec.label} Slippi-AI identity differs: {differences}")
        if specialist is not None:
            release_contract = _slippi_specialist_release_contract(specialist)
            _require_exact_json_fields(
                slippi,
                {
                    "release": specialist.release_name,
                    "parameter_count": specialist.parameter_count,
                    "policy_delay_frames": specialist.policy_delay_frames,
                    "console_delay_frames": 0,
                    "effective_policy_delay_frames": specialist.policy_delay_frames,
                    "async_inference": True,
                    "compile": True,
                    "native_parser_observation_filter_recurrent_state_and_decoder": True,
                    "requested_character": specialist.character,
                    "checkpoint_declared_characters": [specialist.character],
                    "requested_character_covered_by_training": True,
                    "forced_ood_character_transfer": False,
                },
                label=f"{spec.label} Slippi-AI native specialist identity",
            )
            _require_exact_json_mapping(
                slippi.get("release_contract"),
                release_contract,
                label=f"{spec.label} Slippi-AI native specialist release contract",
            )
            _require_exact_json_mapping(
                slippi.get("configured_mixed_runtime"),
                {
                    "exact_mode_only": True,
                    "inference_mode": "synchronous-concurrent",
                    "checkpoint_delay_frames": specialist.policy_delay_frames,
                    "mixed_runtime_console_delay_frames": 0,
                    "effective_policy_delay_frames": specialist.policy_delay_frames,
                    "release_contract": release_contract,
                },
                label=f"{spec.label} Slippi-AI native specialist configured runtime",
            )
        if spec.opponent_evaluation_mode == "forced-ood-character-transfer":
            declared_characters = [character for character, _ in SLIPPI_CHARACTERS]
            _require_exact_json_fields(
                slippi,
                {
                    "requested_character": spec.opponent_character,
                    "checkpoint_declared_characters": declared_characters,
                    "requested_character_covered_by_training": False,
                    "forced_ood_character_transfer": True,
                    "native_parser_observation_filter_recurrent_state_and_decoder": True,
                },
                label=f"{spec.label} Slippi-AI character-transfer identity",
            )
            _require_exact_json_mapping(
                slippi.get("ood_provenance"),
                {
                    "explicit_opt_in": "allow_player_2_ood_character",
                    "physical_port": 2,
                    "checkpoint_roster_unchanged": True,
                    "playing_strength_interpretation": "out-of-training-roster transfer",
                },
                label=f"{spec.label} Slippi-AI OOD opt-in provenance",
            )
            policies = require_mapping(summary.get("policies"), "summary.policies")
            p2_policy = require_mapping(policies.get("p2"), "policies.p2")
            p2_metadata = require_mapping(p2_policy.get("metadata"), "policies.p2.metadata")
            _require_exact_json_mapping(
                p2_metadata.get("character_transfer"),
                {
                    "requested_character": spec.opponent_character,
                    "checkpoint_declared_characters": declared_characters,
                    "requested_character_covered_by_training": False,
                    "forced_ood_character_transfer": True,
                    "checkpoint_bytes_unchanged": True,
                    "native_inference_path_unchanged": True,
                },
                label=f"{spec.label} Slippi-AI native character-transfer provenance",
            )


def validate_completed_summary(
    summary: Mapping[str, Any],
    spec: GameSpec,
    *,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Validate an in-memory summary against one exact frozen game contract."""

    expected_schema = {
        "head-to-head": "integration.frisson_vs_frisson.v2",
        "ancestor": "integration.frisson_vs_frisson.v2",
        "mimic": "integration.frisson_vs_mimic.v1",
        "cpu9": "integration.frisson_vs_cpu9.v1",
        "slippi-ai": "integration.frisson_vs_slippi_ai.v1",
    }[spec.kind]
    if summary.get("schema_version") != expected_schema:
        raise SuiteArtifactError(
            f"{spec.label} schema differs: {summary.get('schema_version')!r}"
        )
    _validate_players(summary, spec)
    _validate_model_identity(summary, spec)
    execution = require_mapping(summary.get("execution"), "summary.execution")
    if (
        execution.get("game_end_observed") is not True
        or execution.get("termination") != "natural-game-end"
    ):
        raise SuiteArtifactError(f"{spec.label} lacks a natural game end")
    gate = require_mapping(summary.get("gate"), "summary.gate")
    if gate.get("decision") != "pass":
        raise SuiteArtifactError(f"{spec.label} failed its postgame fidelity gate")
    replay_path, replay_record = _selected_replay(summary, spec)
    stocks = execution.get("last_stocks")
    if not isinstance(stocks, Mapping) or len(stocks) != 2:
        raise SuiteArtifactError(f"{spec.label} stock result is missing")
    stock_values = list(stocks.values())
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in stock_values):
        raise SuiteArtifactError(f"{spec.label} stock result is malformed")
    winner = execution.get("winner")
    if not isinstance(winner, str) or not winner:
        raise SuiteArtifactError(f"{spec.label} winner is missing")
    if spec.kind in {"head-to-head", "ancestor"}:
        assignments = _dual_player_assignments(spec)
        expected_roles = {assignment["role"] for assignment in assignments}
        if set(stocks) != expected_roles:
            raise SuiteArtifactError(f"{spec.label} scoring stock roles differ")
        p1_stocks = stocks[assignments[0]["role"]]
        p2_stocks = stocks[assignments[1]["role"]]
        if p1_stocks == p2_stocks:
            raise SuiteArtifactError(f"{spec.label} dual-player stock result is tied")
        expected_winner = (
            assignments[0]["model"]
            if p1_stocks > p2_stocks
            else assignments[1]["model"]
        )
        if winner != expected_winner:
            raise SuiteArtifactError(f"{spec.label} winner disagrees with scoring stocks")
    return {
        "label": spec.label,
        "ordinal": spec.ordinal,
        "group": spec.group,
        "kind": spec.kind,
        "seed": spec.seed,
        "frisson_profile": spec.frisson_profile,
        "matchup_class": spec.matchup_class,
        "opponent_evaluation_mode": spec.opponent_evaluation_mode,
        "player_1_character": spec.player_1_character,
        "opponent_character": spec.opponent_character,
        "winner": winner,
        "stocks": dict(stocks),
        "summary": str((summary_path or spec.summary_path).resolve().relative_to(ROOT)),
        "replay": {
            "path": str(replay_path.relative_to(ROOT)),
            "sha256": replay_record["sha256"],
            "byte_length": replay_record["byte_length"],
        },
        "gate": "pass",
    }


def validate_completed_game(spec: GameSpec) -> dict[str, Any]:
    """Read, validate, and summarize one exact natural-end scoring artifact."""

    summary = read_json(spec.summary_path, f"{spec.label} summary")
    return validate_completed_summary(summary, spec, summary_path=spec.summary_path)


def expected_video_players(spec: GameSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return explicit port labels that are burned into a deferred render."""

    if spec.kind == "head-to-head":
        ten = {
            "model": _profile_display_name("10m"),
            "character": "FOX",
        }
        seventy_five = {
            "model": _profile_display_name("75m"),
            "character": "FOX",
        }
        ordered = (seventy_five, ten) if spec.head_to_head_swap_ports else (ten, seventy_five)
    elif spec.kind == "ancestor":
        posttrained = {
            "model": _profile_display_name(spec.frisson_profile),
            "character": "FOX",
        }
        ancestor = {
            "model": (
                "Frisson-AI 10M pretraining ancestor step-122064"
                if spec.frisson_profile == "10m"
                else "Frisson-AI 75M pretraining ancestor step-86016"
            ),
            "character": "FOX",
        }
        ordered = (
            (ancestor, posttrained)
            if spec.head_to_head_swap_ports
            else (posttrained, ancestor)
        )
    elif spec.kind == "mimic":
        release = _mimic_release_for_spec(spec)
        ordered = (
            {
                "model": _profile_display_name(spec.frisson_profile),
                "character": spec.player_1_character,
            },
            {
                "model": (
                    f"MIMIC released {release.directory} (OOD character transfer)"
                    if spec.opponent_evaluation_mode == "forced-ood-character-transfer"
                    else f"MIMIC released {release.directory}"
                ),
                "character": spec.opponent_character,
            },
        )
    elif spec.kind == "cpu9":
        ordered = (
            {
                "model": _profile_display_name(spec.frisson_profile),
                "character": spec.player_1_character,
            },
            {
                "model": "Melee native CPU level 9",
                "character": spec.opponent_character,
            },
        )
    else:
        specialist = _slippi_specialist_for_spec(spec)
        ordered = (
            {
                "model": _profile_display_name(spec.frisson_profile),
                "character": spec.player_1_character,
            },
            {
                "model": (
                    f"vladfi1/slippi-ai {specialist.release_name} "
                    f"(native {specialist.character} specialist)"
                    if specialist is not None
                    else "vladfi1/slippi-ai medium-v2 (Master Player)"
                ),
                "character": spec.opponent_character,
            },
        )
    return tuple(
        {"port": index, **player} for index, player in enumerate(ordered, start=1)
    )  # type: ignore[return-value]


def _profile_display_name(profile: FrissonProfile) -> str:
    if POST_RL_BENCHMARK:
        return f"Frisson-AI {profile.upper()} post-RL step-{TEN_M_STEP if profile == '10m' else SEVENTY_FIVE_M_STEP}"
    if profile == "10m":
        return (
            "Frisson-AI 10M post-trained final step-195248"
            if POSTTRAINING_BENCHMARK
            else "Frisson-AI 10M v2 final step-122064"
        )
    return (
        "Frisson-AI 75M post-trained final step-127214"
        if POSTTRAINING_BENCHMARK
        else "Frisson-AI 75M final step-86016"
    )


def command_for_game(spec: GameSpec, python: str) -> list[str]:
    """Build one raw-replay-first launcher command for a frozen game."""

    if spec.kind == "head-to-head":
        p1 = SEVENTY_FIVE_M_CHECKPOINT if spec.head_to_head_swap_ports else TEN_M_CHECKPOINT
        p2 = TEN_M_CHECKPOINT if spec.head_to_head_swap_ports else SEVENTY_FIVE_M_CHECKPOINT
        phase = CURRENT_PHASE
        p1_contract = (
            f"75m-{phase}" if spec.head_to_head_swap_ports else f"10m-{phase}"
        )
        p2_contract = (
            f"10m-{phase}" if spec.head_to_head_swap_ports else f"75m-{phase}"
        )
        return [
            python,
            str(ROOT / "scripts" / "run_frisson_dual_oneoff.py"),
            "--p1-checkpoint",
            str(p1),
            "--p2-checkpoint",
            str(p2),
            "--p1-contract",
            p1_contract,
            "--p2-contract",
            p2_contract,
            "--seed",
            str(spec.seed),
            "--artifact-label",
            spec.label,
        ]

    if spec.kind == "ancestor":
        profile = spec.frisson_profile
        posttraining = TEN_M_CHECKPOINT if profile == "10m" else SEVENTY_FIVE_M_CHECKPOINT
        ancestor = (
            TEN_M_ANCESTOR_IDENTITY.path
            if profile == "10m"
            else SEVENTY_FIVE_M_ANCESTOR_IDENTITY.path
        )
        posttraining_contract = f"{profile}-{CURRENT_PHASE}"
        ancestor_contract = f"{profile}-pretraining"
        p1, p2 = (
            (ancestor, posttraining)
            if spec.head_to_head_swap_ports
            else (posttraining, ancestor)
        )
        p1_contract, p2_contract = (
            (ancestor_contract, posttraining_contract)
            if spec.head_to_head_swap_ports
            else (posttraining_contract, ancestor_contract)
        )
        return [
            python,
            str(ROOT / "scripts" / "run_frisson_dual_oneoff.py"),
            "--p1-checkpoint",
            str(p1),
            "--p2-checkpoint",
            str(p2),
            "--p1-contract",
            p1_contract,
            "--p2-contract",
            p2_contract,
            "--seed",
            str(spec.seed),
            "--artifact-label",
            spec.label,
        ]

    checkpoint, _, _ = _checkpoint_for_profile(spec.frisson_profile)
    command = [
        python,
        "-m",
        "melee_policy.integration.play",
        "--p1",
        "frisson-ai",
        "--p2",
        {"mimic": "mimic", "cpu9": "cpu", "slippi-ai": "slippi-ai"}[spec.kind],
        "--p1-checkpoint",
        str(checkpoint.path),
        "--p1-character",
        spec.player_1_character,
        "--p2-character",
        spec.opponent_character,
        "--stage",
        STAGE,
        "--seed",
        str(spec.seed),
        "--require-natural-end",
        "--artifact-label",
        spec.label,
    ]
    if spec.kind == "mimic":
        release = _mimic_release_for_spec(spec)
        command.extend(
            [
                "--p2-checkpoint",
                str(release.checkpoint),
                "--p2-assets",
                str(release.assets),
            ]
        )
    elif spec.kind == "cpu9":
        command.extend(["--cpu-level", "9"])
    else:
        specialist = _slippi_specialist_for_spec(spec)
        command.extend(
            [
                "--p2-name",
                SLIPPI_PLAYER_NAME,
                "--p2-temperature",
                "1.0",
            ]
        )
        if specialist is not None:
            command.extend(["--p2-slippi-release", specialist.release_name])
    if spec.opponent_evaluation_mode == "forced-ood-character-transfer":
        command.append("--allow-p2-ood-character")
    if POST_RL_BENCHMARK:
        command.extend(["--save-slp", "--no-save-video"])
    return command


def initial_state() -> dict[str, Any]:
    manifest = schedule_manifest()
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "schedule_sha256": manifest["schedule_sha256"],
        "status": "pending",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "current_label": None,
        "games": {
            spec.label: {
                "ordinal": spec.ordinal,
                "group": spec.group,
                "frisson_profile": spec.frisson_profile,
                "matchup_class": spec.matchup_class,
                "player_1_character": spec.player_1_character,
                "opponent_character": spec.opponent_character,
                "opponent_evaluation_mode": spec.opponent_evaluation_mode,
                "status": "pending",
                "attempts_started": 0,
                "result": None,
            }
            for spec in SCHEDULE
        },
    }


def _read_frozen_v4_json(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_byte_length: int,
) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        raise
    except OSError as error:
        raise SuiteArtifactError(f"{label} is unreadable: {error}") from error
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_byte_length or observed_sha256 != expected_sha256:
        raise SuiteArtifactError(
            f"{label} differs from the frozen migration source: "
            f"sha256={observed_sha256} bytes={len(payload)}"
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SuiteArtifactError(f"{label} is unreadable: {error}") from error
    return require_mapping(value, label), payload


def _v4_schedule_migration(
    legacy_games: Sequence[object],
    current_games: Sequence[object],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    if len(legacy_games) != V4_GAME_COUNT or len(current_games) != GAME_COUNT:
        raise SuiteArtifactError("v4 and v5 schedule lengths differ from their frozen contracts")
    unchanged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    replacements: list[dict[str, Any]] = []
    ignored_change_fields = {
        "artifact_root",
        "group",
        "label",
        "mimic_directory",
        "opponent_evaluation_mode",
    }
    for index, (legacy_value, current_value) in enumerate(
        zip(legacy_games, current_games, strict=True),
        start=1,
    ):
        legacy = require_mapping(legacy_value, f"v4 manifest game {index}")
        current = require_mapping(current_value, f"v5 manifest game {index}")
        if legacy.get("ordinal") != index or current.get("ordinal") != index:
            raise SuiteArtifactError(f"schedule ordinal {index} is not stable across v4 and v5")
        if legacy == current:
            unchanged.append((legacy, current))
            continue
        legacy_common = {
            key: value for key, value in legacy.items() if key not in ignored_change_fields
        }
        current_common = {
            key: value for key, value in current.items() if key not in ignored_change_fields
        }
        if legacy_common != current_common:
            raise SuiteArtifactError(
                f"v5 replacement at ordinal {index} changed a gameplay field"
            )
        kind = current.get("kind")
        character = current.get("opponent_character")
        if kind == "mimic":
            profile = current["frisson_profile"]
            legacy_group = legacy.get("group")
            if legacy_group == f"{profile}-mimic-master-fox-10":
                expected_current_group = f"{profile}-mimic-current-master-fox-10"
                expected_legacy_mode = "supported"
                expected_current_mode = "supported"
                expected_current_directory = "fox-master"
                replacement_class = "mimic-current-master-fox"
            elif legacy_group == f"{profile}-mimic-all-character-mirrors-2":
                if character not in {
                    release.character for release in MIMIC_CURRENT_CORE_RELEASES
                }:
                    raise SuiteArtifactError(
                        f"ordinal {index} has an unexpected MIMIC core character"
                    )
                expected_current_group = (
                    f"{profile}-mimic-current-native-core-character-mirrors-2"
                )
                expected_legacy_mode = "supported"
                expected_current_mode = "supported"
                expected_current_directory = MIMIC_BY_CHARACTER[str(character)].directory
                replacement_class = "mimic-current-native-core"
            elif legacy_group == (
                f"{profile}-mimic-fox-master-ood-character-mirrors-2"
            ):
                expected_legacy_mode = "forced-ood-character-transfer"
                release = MIMIC_BY_CHARACTER.get(str(character))
                if release is None:
                    if character not in MIMIC_OOD_CHARACTERS:
                        raise SuiteArtifactError(
                            f"ordinal {index} has an unexpected MIMIC OOD character"
                        )
                    expected_current_group = (
                        f"{profile}-mimic-current-fox-master-ood-character-mirrors-2"
                    )
                    expected_current_mode = "forced-ood-character-transfer"
                    expected_current_directory = "fox-master"
                    replacement_class = "mimic-current-fox-master-ood"
                else:
                    expected_current_group = (
                        f"{profile}-mimic-native-extension-character-mirrors-2"
                    )
                    expected_current_mode = "supported"
                    expected_current_directory = release.directory
                    replacement_class = "mimic-native-extension"
            else:
                raise SuiteArtifactError(
                    f"ordinal {index} is not an exact current-snapshot MIMIC replacement"
                )
            expected = {
                "legacy_group": legacy_group,
                "legacy_mode": expected_legacy_mode,
                "legacy_directory": "fox-master"
                if "fox-master-ood" in str(legacy_group)
                or str(legacy_group).endswith("mimic-master-fox-10")
                or character == "FOX"
                else str(character).lower(),
                "current_group": expected_current_group,
                "current_mode": expected_current_mode,
                "current_directory": expected_current_directory,
            }
            observed = {
                "legacy_group": legacy.get("group"),
                "legacy_mode": legacy.get("opponent_evaluation_mode"),
                "legacy_directory": legacy.get("mimic_directory"),
                "current_group": current.get("group"),
                "current_mode": current.get("opponent_evaluation_mode"),
                "current_directory": current.get("mimic_directory"),
            }
            if observed != expected:
                raise SuiteArtifactError(
                    f"ordinal {index} is not an exact current-snapshot MIMIC replacement"
                )
        elif kind == "slippi-ai":
            expected = {
                "legacy_group": f"{current['frisson_profile']}-slippi-medium-v2-ood-character-mirrors-2",
                "legacy_mode": "forced-ood-character-transfer",
                "current_group": f"{current['frisson_profile']}-slippi-native-specialist-character-mirrors-2",
                "current_mode": "supported",
            }
            observed = {
                "legacy_group": legacy.get("group"),
                "legacy_mode": legacy.get("opponent_evaluation_mode"),
                "current_group": current.get("group"),
                "current_mode": current.get("opponent_evaluation_mode"),
            }
            if character not in SLIPPI_SPECIALIST_BY_CHARACTER or observed != expected:
                raise SuiteArtifactError(
                    f"ordinal {index} is not an exact Slippi-AI native-specialist replacement"
                )
            replacement_class = "slippi-native-specialist"
        else:
            raise SuiteArtifactError(f"unexpected v5 schedule replacement at ordinal {index}")
        replacements.append(
            {
                "ordinal": index,
                "kind": kind,
                "character": character,
                "seed": current.get("seed"),
                "frisson_profile": current.get("frisson_profile"),
                "replacement_class": replacement_class,
                "source_label": legacy.get("label"),
                "target_label": current.get("label"),
            }
        )
    mimic_replacements = [
        item
        for item in replacements
        if str(item["replacement_class"]).startswith("mimic-")
    ]
    slippi_replacements = [
        item for item in replacements if item["replacement_class"] == "slippi-native-specialist"
    ]
    if (
        len(unchanged) != UNCHANGED_V4_GAME_COUNT
        or len(replacements) != REPLACED_V4_GAME_COUNT
        or len(mimic_replacements) != REPLACED_V4_MIMIC_GAME_COUNT
        or len(slippi_replacements) != REPLACED_V4_SLIPPI_GAME_COUNT
    ):
        raise SuiteArtifactError(
            "v4 to v5 migration classification differs from the frozen replacement counts"
        )
    return unchanged, replacements


def _migrate_v4_state() -> dict[str, Any] | None:
    """Import only byte-authenticated, semantically unchanged v4 games."""

    if not V4_STATE_PATH.is_file() and not V4_MANIFEST_PATH.is_file():
        return None
    if not V4_STATE_PATH.is_file() or not V4_MANIFEST_PATH.is_file():
        raise SuiteArtifactError("v4 migration requires both frozen manifest and state files")
    legacy_manifest, legacy_manifest_bytes = _read_frozen_v4_json(
        V4_MANIFEST_PATH,
        label="frozen v4 suite manifest",
        expected_sha256=V4_MANIFEST_SHA256,
        expected_byte_length=V4_MANIFEST_BYTE_LENGTH,
    )
    expected_manifest_identity = {
        "schema_version": V4_SUITE_SCHEMA_VERSION,
        "suite_id": V4_SUITE_ID,
        "schedule_sha256": V4_SCHEDULE_SHA256,
        "game_count": V4_GAME_COUNT,
    }
    manifest_differences = {
        key: (legacy_manifest.get(key), expected)
        for key, expected in expected_manifest_identity.items()
        if legacy_manifest.get(key) != expected
    }
    if manifest_differences:
        raise SuiteArtifactError(f"frozen v4 manifest identity differs: {manifest_differences}")
    legacy_manifest_games = require_sequence(
        legacy_manifest.get("games"), "frozen v4 manifest games"
    )
    current_manifest_games = require_sequence(
        schedule_manifest().get("games"), "v5 manifest games"
    )
    unchanged, replacements = _v4_schedule_migration(
        legacy_manifest_games,
        current_manifest_games,
    )

    legacy_state, legacy_state_bytes = _read_frozen_v4_json(
        V4_STATE_PATH,
        label="frozen v4 suite state",
        expected_sha256=V4_STATE_SHA256,
        expected_byte_length=V4_STATE_BYTE_LENGTH,
    )
    expected_state_identity = {
        "schema_version": V4_STATE_SCHEMA_VERSION,
        "suite_id": V4_SUITE_ID,
        "schedule_sha256": V4_SCHEDULE_SHA256,
        "updated_at_utc": V4_STATE_UPDATED_AT_UTC,
    }
    state_differences = {
        key: (legacy_state.get(key), expected)
        for key, expected in expected_state_identity.items()
        if legacy_state.get(key) != expected
    }
    if state_differences:
        raise SuiteArtifactError(f"frozen v4 state identity differs: {state_differences}")
    legacy_state_games = require_mapping(legacy_state.get("games"), "frozen v4 state games")
    legacy_manifest_by_label = {
        str(require_mapping(value, "v4 manifest game")["label"]): require_mapping(
            value, "v4 manifest game"
        )
        for value in legacy_manifest_games
    }
    if set(legacy_state_games) != set(legacy_manifest_by_label):
        raise SuiteArtifactError("frozen v4 state labels differ from its manifest")

    migrated = initial_state()
    migrated_games = require_mapping(migrated.get("games"), "new v5 state games")
    copied_fields = (
        "attempts_started",
        "started_at_utc",
        "launcher_finished_at_utc",
        "launcher_returncode",
        "completed_at_utc",
        "command",
        "artifact_recovery",
        "authorized_retry_history",
    )
    migrated_labels: list[str] = []
    published_summary_labels: list[str] = []
    for _legacy_game, current_game in unchanged:
        label = str(current_game["label"])
        source = require_mapping(legacy_state_games.get(label), f"v4 game {label}")
        target = require_mapping(migrated_games.get(label), f"v5 game {label}")
        for field in copied_fields:
            if field in source:
                target[field] = source[field]
        target["status"] = "pending"
        target["result"] = None
        migrated_games[label] = target
        migrated_labels.append(label)
        if (ARTIFACT_OUTPUT / label / "summary.json").is_file():
            published_summary_labels.append(label)
    if len(published_summary_labels) != PRESERVED_V4_COMPLETED_GAME_COUNT:
        raise SuiteArtifactError(
            "frozen v4 published-summary count differs from the v5 preservation contract"
        )

    source_replacement_records = []
    for replacement in replacements:
        source_label = str(replacement["source_label"])
        source = require_mapping(
            legacy_state_games.get(source_label), f"replaced v4 game {source_label}"
        )
        source_replacement_records.append(
            {
                **replacement,
                "source_status": source.get("status"),
                "source_result": source.get("result"),
            }
        )
    migrated["games"] = migrated_games
    migrated["migration"] = {
        "schema_version": "integration.final_winner_benchmark_v4_to_v5_migration.v1",
        "source_suite_id": V4_SUITE_ID,
        "source_schedule_sha256": V4_SCHEDULE_SHA256,
        "source_manifest": {
            "path": str(
                V4_MANIFEST_PATH.relative_to(ROOT)
                if V4_MANIFEST_PATH.is_relative_to(ROOT)
                else V4_MANIFEST_PATH
            ),
            "sha256": V4_MANIFEST_SHA256,
            "byte_length": len(legacy_manifest_bytes),
        },
        "source_state": {
            "path": str(
                V4_STATE_PATH.relative_to(ROOT)
                if V4_STATE_PATH.is_relative_to(ROOT)
                else V4_STATE_PATH
            ),
            "sha256": hashlib.sha256(legacy_state_bytes).hexdigest(),
            "byte_length": len(legacy_state_bytes),
            "status": legacy_state.get("status"),
            "updated_at_utc": legacy_state.get("updated_at_utc"),
        },
        "semantically_unchanged_game_count": len(unchanged),
        "migrated_label_count": len(migrated_labels),
        "published_summary_count_at_migration": len(published_summary_labels),
        "published_summary_labels": published_summary_labels,
        "replacement_game_count": len(replacements),
        "mimic_native_replacement_game_count": len(
            [
                item
                for item in replacements
                if str(item["replacement_class"]).startswith("mimic-")
            ]
        ),
        "slippi_native_replacement_game_count": len(
            [
                item
                for item in replacements
                if item["replacement_class"] == "slippi-native-specialist"
            ]
        ),
        "replacement_entries_initialized_pending": True,
        "replaced_source_games": source_replacement_records,
        "legacy_deferred_postgame_validation_backlog": legacy_state.get(
            "deferred_postgame_validation_backlog", {}
        ),
        "legacy_deferred_backlog_imported_into_active_state": False,
        "source_artifacts_mutated": False,
        "gameplay_reexecuted": False,
        "migrated_at_utc": utc_now(),
    }
    return migrated


def _migrate_v3_state() -> dict[str, Any] | None:
    """Fallback import of the unchanged 117-game prefix from v3."""

    if not V3_STATE_PATH.is_file() and not V3_MANIFEST_PATH.is_file():
        return None
    if not V3_STATE_PATH.is_file() or not V3_MANIFEST_PATH.is_file():
        raise SuiteArtifactError("legacy v3 migration requires both manifest and state files")

    legacy_manifest = read_json(V3_MANIFEST_PATH, "legacy v3 suite manifest")
    expected_manifest_identity = {
        "schema_version": V3_SUITE_SCHEMA_VERSION,
        "suite_id": V3_SUITE_ID,
        "schedule_sha256": V3_SCHEDULE_SHA256,
        "game_count": V3_GAME_COUNT,
    }
    manifest_differences = {
        key: (legacy_manifest.get(key), expected)
        for key, expected in expected_manifest_identity.items()
        if legacy_manifest.get(key) != expected
    }
    if manifest_differences:
        raise SuiteArtifactError(
            f"legacy v3 suite manifest identity differs: {manifest_differences}"
        )

    legacy_games_value = legacy_manifest.get("games")
    if not isinstance(legacy_games_value, list) or len(legacy_games_value) != V3_GAME_COUNT:
        raise SuiteArtifactError("legacy v3 suite manifest games differ")
    current_prefix = schedule_manifest()["games"][:UNCHANGED_V3_PREFIX_GAME_COUNT]
    normalized_current_prefix = []
    for record in current_prefix:
        normalized = dict(require_mapping(record, "v4 prefix game"))
        normalized.pop("opponent_evaluation_mode", None)
        normalized_current_prefix.append(normalized)
    if legacy_games_value[:UNCHANGED_V3_PREFIX_GAME_COUNT] != normalized_current_prefix:
        raise SuiteArtifactError("legacy v3 and v4 first-117 schedule contracts differ")

    legacy_state_bytes = V3_STATE_PATH.read_bytes()
    try:
        legacy_state_value = json.loads(legacy_state_bytes)
    except json.JSONDecodeError as error:
        raise SuiteArtifactError(f"legacy v3 suite state is unreadable: {error}") from error
    legacy_state = require_mapping(legacy_state_value, "legacy v3 suite state")
    expected_state_identity = {
        "schema_version": V3_STATE_SCHEMA_VERSION,
        "suite_id": V3_SUITE_ID,
        "schedule_sha256": V3_SCHEDULE_SHA256,
    }
    state_differences = {
        key: (legacy_state.get(key), expected)
        for key, expected in expected_state_identity.items()
        if legacy_state.get(key) != expected
    }
    if state_differences:
        raise SuiteArtifactError(f"legacy v3 suite state identity differs: {state_differences}")
    legacy_games = require_mapping(legacy_state.get("games"), "legacy v3 state games")

    migrated = initial_state()
    migrated_games = require_mapping(migrated.get("games"), "new v5 state games")
    copied_fields = (
        "attempts_started",
        "started_at_utc",
        "launcher_finished_at_utc",
        "launcher_returncode",
        "completed_at_utc",
        "command",
    )
    migrated_labels: list[str] = []
    published_summary_labels: list[str] = []
    for spec in SCHEDULE[:UNCHANGED_V3_PREFIX_GAME_COUNT]:
        source = require_mapping(legacy_games.get(spec.label), f"legacy v3 game {spec.label}")
        target = require_mapping(migrated_games.get(spec.label), f"new v4 game {spec.label}")
        for field in copied_fields:
            if field in source:
                target[field] = source[field]
        # Artifact reconciliation is authoritative for completion.  Any legacy
        # blocked/running status without a published summary resumes as pending.
        target["status"] = "pending"
        target["result"] = None
        migrated_games[spec.label] = target
        migrated_labels.append(spec.label)
        if spec.summary_path.is_file():
            published_summary_labels.append(spec.label)

    migrated["games"] = migrated_games
    migrated["migration"] = {
        "schema_version": "integration.final_winner_benchmark_v3_to_v5_migration.v1",
        "source_suite_id": V3_SUITE_ID,
        "source_schedule_sha256": V3_SCHEDULE_SHA256,
        "source_manifest": {
            "path": str(
                V3_MANIFEST_PATH.relative_to(ROOT)
                if V3_MANIFEST_PATH.is_relative_to(ROOT)
                else V3_MANIFEST_PATH
            ),
            "sha256": sha256_file(V3_MANIFEST_PATH),
            "byte_length": V3_MANIFEST_PATH.stat().st_size,
        },
        "source_state": {
            "path": str(
                V3_STATE_PATH.relative_to(ROOT)
                if V3_STATE_PATH.is_relative_to(ROOT)
                else V3_STATE_PATH
            ),
            "sha256": hashlib.sha256(legacy_state_bytes).hexdigest(),
            "byte_length": len(legacy_state_bytes),
            "status": legacy_state.get("status"),
            "updated_at_utc": legacy_state.get("updated_at_utc"),
        },
        "unchanged_prefix_game_count": UNCHANGED_V3_PREFIX_GAME_COUNT,
        "migrated_label_count": len(migrated_labels),
        "published_summary_count_at_migration": len(published_summary_labels),
        "published_summary_labels": published_summary_labels,
        "legacy_deferred_postgame_validation_backlog": legacy_state.get(
            "deferred_postgame_validation_backlog",
            {},
        ),
        "obsolete_v3_extension_games_imported": False,
        "gameplay_reexecuted": False,
        "migrated_at_utc": utc_now(),
    }
    return migrated


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        if POSTTRAINING_BENCHMARK:
            return initial_state()
        migrated = _migrate_v4_state()
        if migrated is None:
            migrated = _migrate_v3_state()
        return initial_state() if migrated is None else migrated
    state = read_json(STATE_PATH, "suite state")
    manifest = schedule_manifest()
    if (
        state.get("schema_version") != STATE_SCHEMA_VERSION
        or state.get("suite_id") != SUITE_ID
        or state.get("schedule_sha256") != manifest["schedule_sha256"]
    ):
        raise SuiteArtifactError("suite state is bound to a different schedule")
    games = require_mapping(state.get("games"), "suite state games")
    if set(games) != set(SCHEDULE_BY_LABEL):
        raise SuiteArtifactError("suite state game labels differ from the frozen schedule")
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    atomic_json(STATE_PATH, state)


__all__ = [
    "ARTIFACT_OUTPUT",
    "CHARACTER_SWEEP_SEEDS",
    "FRISSON_ROSTER",
    "GAME_COUNT",
    "LOCK_PATH",
    "MANIFEST_PATH",
    "REMAINING_FRISSON_CHARACTERS",
    "ROOT",
    "SCHEDULE",
    "SCHEDULE_BY_LABEL",
    "STATE_PATH",
    "SUITE_DIRECTORY",
    "SUITE_ID",
    "GameSpec",
    "SuiteArtifactError",
    "atomic_json",
    "command_for_game",
    "expected_video_players",
    "import_semantically_valid_v1_artifacts",
    "initial_state",
    "load_state",
    "read_json",
    "require_mapping",
    "require_sequence",
    "resolve_artifact_path",
    "save_state",
    "schedule_manifest",
    "sha256_file",
    "utc_now",
    "validate_completed_game",
    "validate_completed_summary",
    "verify_frozen_inputs",
]
