"""Faynt evaluation cloud entry points.

Match execution functions retain the remote research implementation.
Runtime images, emulators, game files, upstream sources and checkpoints
are separately provided. See README.md and SOURCE_MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeVar

import modal

MODAL_ENVIRONMENT: Final[str] = os.environ.get("FAYNT_MODAL_ENVIRONMENT", "main")
APP_NAME: Final[str] = "faynt-evaluation"
VOLUME_NAME: Final[str] = os.environ.get("FAYNT_RUNS_VOLUME", "faynt-evaluation-runs")
VOLUME_MOUNT: Final[str] = "/runs"
WANDB_SECRET_NAME: Final[str] = os.environ.get("FAYNT_WANDB_SECRET", "faynt-wandb")
WANDB_SECRET_KEY: Final[str] = "WANDB_API_KEY"
ISO_VOLUME_NAME: Final[str] = os.environ.get("FAYNT_GAME_VOLUME", "faynt-local-game")
"""The user-supplied Volume containing their Melee NTSC 1.02 image as ``melee.iso``."""
ISO_MOUNT: Final[str] = "/iso"
ISO_PATH: Final[str] = f"{ISO_MOUNT}/melee.iso"
"""Equals ``melee_rl.env.dolphin.DEFAULT_ISO_PATH`` (not imported here: that module imports torch)."""

LOCAL_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""The local Faynt evaluation checkout. Its internal container path is :data:`REMOTE_ROOT`."""
REMOTE_ROOT: Final[str] = "/root/melee-policy-frisson-ai"
CONFIG_DIR: Final[Path] = LOCAL_ROOT / "configs" / "rl"
"""Equals ``melee_rl.config.CONFIG_DIR`` (not imported here: that module imports torch)."""
REMOTE_WANDB_DIR: Final[str] = "/tmp/wandb"
REMOTE_MYPY_CACHE: Final[str] = "/tmp/mypy-cache"

PYTHON_VERSION: Final[str] = "3.12"
TORCH_PIN: Final[str] = "torch==2.11.0"
TORCH_CPU_INDEX: Final[str] = "https://download.pytorch.org/whl/cpu"
TORCH_GPU_INDEX: Final[str] = "https://download.pytorch.org/whl/cu128"
"""The CUDA 12.8 wheel index (the last cu128 build is 2.11.0, ENV.md §2) for the GPU Dolphin image (P6a)."""
NUMPY_PIN: Final[str] = "numpy>=1.26,<2.2"
"""The core pin of ``pyproject.toml`` since the 22 Aug 2026 merge of Ali's ``main``.

Named because a second pip layer repeats it (:data:`SLIPPI_AI_PIP_PACKAGES`): a resolver that satisfied
TensorFlow by moving numpy would move it under torch and Ali's preprocessing at the same time."""
IMAGE_PACKAGES: Final[tuple[str, ...]] = (
    "PyYAML>=6",
    NUMPY_PIN,
    "wandb==0.28.2",
    "pytest>=8",
    "ruff>=0.12",
    "mypy>=1.18",
)
LIBMELEE_PIN: Final[str] = "melee==0.47.3"
"""libmelee in the Dolphin image (``melee_rl.env.libmelee_frames.MELEE_VERSION``; ENV.md §6.2)."""
IGNORE_PATTERNS: Final[tuple[str, ...]] = (
    ".venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    "docs",
    ".git",
    "**/.git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
    "**/*.egg-info",
    "runs",
    "wandb",
    "reports",
    "*.pt",
    "**/*.pt",
    "*.pth",
    "**/*.pth",
    ".env",
    "*.iso",
    "**/*.iso",
    "*.AppImage",
    "**/*.AppImage",
)
"""dockerignore-style patterns for the package mount (relative to :data:`LOCAL_ROOT`).  No
negations, so Modal prunes the ignored directories without walking them (the 3 GB ``.venv``);
disc images and AppImages never travel with the code."""
PYTEST_BASE_ARGS: Final[tuple[str, ...]] = ("tests", "-q", "-p", "no:cacheprovider")
DESELECTED_TESTS: Final[tuple[str, ...]] = (
    "tests/test_preprocessing.py::test_mds_partition_retry_reuses_deterministic_published_partition",
    "tests/test_preprocessing.py::test_canary_finalizer_expands_compact_worker_manifest_without_validation",
    "tests/test_preprocessing.py::test_component_finalizer_preserves_shared_root_index_and_namespaces_manifest",
    "tests/test_preprocessing.py::test_finalizer_retry_after_partial_split_merge_is_idempotent",
    "tests/test_preprocessing.py::test_strict_component_union_round_trips_real_mosaic_partitions",
)
"""Ali's tests that import ``streaming`` (``mosaicml-streaming``, his ``data`` extra) inside the test body
(``tests/test_preprocessing.py``, merged 22 Aug 2026).  The extra pulls torchvision, transformers and three
cloud SDKs, so neither the image nor the dev venv installs it; the gate deselects exactly these tests
(ENV.md §3)."""
GIT_DEPENDENT_TESTS: Final[tuple[str, ...]] = (
    "tests/test_modal_app.py::test_source_snapshot_uses_hash_checked_image_identity_without_git",
)
"""Ali's tests that need a git checkout: ``source_snapshot()`` shells out to ``git status`` / ``git diff`` and
falls back to a fixed "dirty" identity without git, which this test compares against a with-git identity.  The
container has neither ``git`` (debian_slim) nor ``.git`` (:data:`IGNORE_PATTERNS`), so ``run_tests`` deselects
them there (P6a run (a), 22 Aug 2026: the only remote failure); the local gate still runs them."""
ALI_MODULES: Final[tuple[str, ...]] = ("model.py", "controller_codec.py", "tensor_batch.py")


# ---------------------------------------------------------------------------
# Dolphin builds (ENV.md §6.1; decided 22 Aug 2026)
# ---------------------------------------------------------------------------

DOLPHIN_ROOT: Final[str] = "/opt/dolphin"
"""Where the image extracts the AppImages: ``/opt/dolphin/<build>/squashfs-root/usr/bin/dolphin-emu``."""


@dataclass(frozen=True)
class DolphinBuild:
    """One ExiAI Dolphin release: the AppImage URL, its sha256 and where the image extracts it."""

    name: str
    url: str
    sha256: str
    mainline: bool
    note: str
    kind: str = "appimage"
    """``"appimage"`` or ``"zip"`` -- the Linux playback build ships as a zip *around* an AppImage."""

    @property
    def asset(self) -> str:
        return self.url.rsplit("/", 1)[1]

    @property
    def appimage(self) -> str:
        """The downloaded archive inside the image (removed after extraction)."""

        suffix = "AppImage" if self.kind == "appimage" else "zip"
        return f"{DOLPHIN_ROOT}/{self.name}.{suffix}"

    @property
    def exe(self) -> str:
        return f"{DOLPHIN_ROOT}/{self.name}/squashfs-root/usr/bin/dolphin-emu"


DEFAULT_DOLPHIN_BUILD: Final[str] = "mainline-exiai-14.1-noleak"
DOLPHIN_BUILDS: Final[Mapping[str, DolphinBuild]] = MappingProxyType(
    {
        "mainline-exiai-14.1-noleak": DolphinBuild(
            name="mainline-exiai-14.1-noleak",
            url=(
                "https://github.com/vladfi1/dolphin/releases/download/4.0.0-mainline-beta.14.1-ExiAI/"
                "Slippi_Netplay_Mainline_ExiAI_NoLeak-x86_64.AppImage"
            ),
            sha256="3e8c26a76286d4d101b6ac252812c5c8cf9b681fdd5ce578bc4b65ccbbdd9bf0",
            mainline=True,
            note="slippi-ai's RL build since 15 Feb 2026 (35a76aa); 4 Mar 2026 'Fixed memory leak in "
            "infinite time mode'",
        ),
        "ishiiruka-exiai-0.2.0": DolphinBuild(
            name="ishiiruka-exiai-0.2.0",
            url=(
                "https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.2.0/"
                "Slippi_Online-x86_64-ExiAI.AppImage"
            ),
            sha256="87e9ef6d80ed03354a1647d0616016dbc91399aa9e86a69ae5a398edd0a0c2bd",
            mainline=False,
            note="slippi-ai Sept 2024 - Feb 2026; what MIMIC and HAL train on",
        ),
    }
)
"""The two headless ExiAI builds the image carries (``DEFAULT_DOLPHIN_BUILD`` is what the run files use)."""
DOLPHIN_APT_PACKAGES: Final[tuple[str, ...]] = (
    "ca-certificates",
    "curl",
    "libegl1",
    "libgl1",
    "libx11-6",
    "libxrandr2",
    "libxi6",
    "libevdev2",
    "libasound2",
    "libudev1",
    "libbz2-1.0",
    "liblzma5",
    "zlib1g",
    "libusb-1.0-0",
)
"""Union of the system libraries both AppImages need (``ldd``, ENV.md §6.1) plus the download tools."""
DOLPHIN_VIDEO_APT_PACKAGES: Final[tuple[str, ...]] = (
    "ffmpeg",
    "fonts-dejavu-core",
    "xvfb",
    "libegl-mesa0",
    "libgl1-mesa-dri",
    "mesa-vulkan-drivers",
    "libopengl0",
    "libgles2",
    "libxkbcommon-x11-0",
    "libxcb-icccm4",
    "libxcb-image0",
    "libxcb-keysyms1",
    "libxcb-randr0",
    "libxcb-render-util0",
    "libxcb-shape0",
    "libxcb-xinerama0",
    "libxcb-xkb1",
    "libxi6",
    "libsm6",
    "libice6",
    "libpulse0",
    "libsystemd0",
    "mesa-utils",
)
"""On top of the Dolphin image for P9: the mux, the caption font and every software / Xvfb GL path a
rendering Dolphin might pick (which of them initialises here is what ``video_check`` answers)."""
VIDEO_DOLPHIN_BUILDS: Final[Mapping[str, DolphinBuild]] = MappingProxyType(
    {
        "ishiiruka-playback-3.5.2": DolphinBuild(
            name="ishiiruka-playback-3.5.2",
            url=(
                "https://github.com/project-slippi/Ishiiruka-Playback/releases/download/v3.5.2/"
                "playback-3.5.2-Linux.zip"
            ),
            sha256="c7fcba0b6114eeab5aa20367d89a669daeebd639ddc43aa9969883b18c49cf1c",
            mainline=False,
            note="the Slippi playback build (26 Aug 2026 probe: the only binary that accepts -i and "
            "actually dumped frames; needs Xvfb and does not exit when the replay ends)",
            kind="zip",
        ),
    }
)
"""Extra builds the **video** image carries (``dolphin_video_image`` only, 77 MB): the renderer."""
DEFAULT_RENDER_BUILD: Final[str] = "ishiiruka-playback-3.5.2"
RENDER_DOLPHIN_PATH: Final[str] = VIDEO_DOLPHIN_BUILDS[DEFAULT_RENDER_BUILD].exe

PLAYBACK_RELEASES_URL: Final[str] = "https://api.github.com/repos/project-slippi/Ishiiruka/releases"
"""Where ``video_check`` looks for a ``Slippi_Playback-x86_64.AppImage`` if no candidate renders."""
PLAYBACK_ASSET_MARKER: Final[str] = "Playback"
PLAYBACK_CANDIDATES: Final[tuple[Mapping[str, str], ...]] = (
    {
        "name": "ishiiruka-playback-3.5.2",
        "url": (
            "https://github.com/project-slippi/Ishiiruka-Playback/releases/download/v3.5.2/"
            "playback-3.5.2-Linux.zip"
        ),
    },
    {
        "name": "mainline-netplay-beta.19",
        "url": (
            "https://github.com/project-slippi/dolphin/releases/download/v4.0.0-mainline-beta.19/"
            "Slippi_Netplay_Mainline-x86_64.AppImage"
        ),
    },
    {
        "name": "ishiiruka-online-3.6.4",
        "url": (
            "https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4/"
            "Slippi_Online-x86_64.AppImage"
        ),
    },
)
"""The Slippi builds ``video_check`` falls back to (26 Aug 2026).  Our ExiAI build (60.9 MB, stripped,
no GUI) lists neither ``-i`` nor ``-b`` in ``--help`` and bundles no ``libav*``, so it can neither play
a replay back nor dump a frame.  The dedicated **playback** build lives in its own repository
(``project-slippi/Ishiiruka-Playback``, a Linux *zip*, not the ``Slippi_Playback-x86_64.AppImage`` the
plan guessed); the two netplay builds are kept after it as controls -- both bundle the full ffmpeg
stack, and the mainline one already rejected ``-i``."""


def dolphin_build(name: str) -> DolphinBuild:
    try:
        return DOLPHIN_BUILDS[name]
    except KeyError as error:
        raise ValueError(f"unknown Dolphin build {name!r}; known: {sorted(DOLPHIN_BUILDS)}") from error


def dolphin_image_commands(builds: Mapping[str, DolphinBuild] | None = None) -> list[str]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


MIMIC_ROOT: Final[str] = "/opt/mimic"
"""Where the MIMIC image layer puts the checkout and the released bundle (``melee_rl.mimic``)."""
MIMIC_SOURCE_DIR: Final[str] = f"{MIMIC_ROOT}/source"
MIMIC_ASSET_DIR: Final[str] = f"{MIMIC_ROOT}/fox-master"
MIMIC_REPO_URL: Final[str] = "https://github.com/erickfm/MIMIC.git"
MIMIC_HF_REPO: Final[str] = "erickfm/MIMIC"
MIMIC_SOURCE_REVISION: Final[str] = "70c925b7675202d853472c4e54332790ad76efe7"
MIMIC_ASSET_REVISION: Final[str] = "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07"
MIMIC_ASSET_SUBDIR: Final[str] = "fox-master"
"""The pins of ``melee-policy/configs/integration_mimic_vs_hal.toml`` -- the source/weights pair E001-M1
validated.  Duplicated here on purpose: this module must stay import-light (no torch), and
``tests/rl/test_modal_app.py`` asserts they still equal :mod:`melee_rl.mimic`'s."""
MIMIC_APT_PACKAGES: Final[tuple[str, ...]] = ("git",)
MIMIC_PIP_PACKAGES: Final[tuple[str, ...]] = ("pandas>=2,<3", "huggingface_hub>=0.24")
"""``mimic.features`` imports pandas; ``huggingface_hub`` fetches the bundle at build time."""


def mimic_image_commands() -> list[str]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


SMASHBOT_ROOT: Final[str] = "/opt/smashbot"
"""Where the SmashBot image layer puts the checkout (``melee_rl.smashbot``)."""
SMASHBOT_SOURCE_DIR: Final[str] = f"{SMASHBOT_ROOT}/source"
SMASHBOT_REPO_URL: Final[str] = "https://github.com/altf4/SmashBot.git"
SMASHBOT_REVISION: Final[str] = "67c412092e4ef1576e1357a79461df5faedad3c2"
SMASHBOT_LICENSE: Final[str] = "GPL-3.0"
"""altf4/SmashBot's HEAD and last commit (``Fix stage API bugs with libmelee 0.41.0``, 15 May 2024).
Duplicated from :mod:`melee_rl.smashbot` on purpose: this module must stay import-light (no torch), and
``tests/rl/test_modal_app.py`` asserts they still agree. Users supply this GPL-3.0 source separately;
the adapter imports it in process. See ``THIRD_PARTY_NOTICES.md`` for provenance and review scope."""
SMASHBOT_APT_PACKAGES: Final[tuple[str, ...]] = ("git",)


def smashbot_image_commands() -> list[str]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


PHILLIP_ROOT: Final[str] = "/opt/phillip"
"""Where the Phillip image layer puts the checkout and the sidecar interpreter (``melee_rl.phillip``)."""
PHILLIP_SOURCE_DIR: Final[str] = f"{PHILLIP_ROOT}/source"
PHILLIP_VENV: Final[str] = f"{PHILLIP_ROOT}/venv"
PHILLIP_PYTHON: Final[str] = f"{PHILLIP_VENV}/bin/python"
PHILLIP_SYSTEM_PYTHON: Final[str] = "/usr/bin/python3.11"
"""Debian 12's ``python3`` package: the last minor TensorFlow 2.13 has wheels for (upstream's pin)."""
PHILLIP_REPO_URL: Final[str] = "https://github.com/vladfi1/phillip.git"
PHILLIP_REVISION: Final[str] = "114aea4c446b31359c5637b271db52b200440caa"
PHILLIP_LICENSE: Final[str] = "GPL-3.0"
PHILLIP_TENSORFLOW_PIN: Final[str] = "tensorflow-cpu==2.13.1"
"""vladfi1/phillip's HEAD (``Remove git-lfs tracked agents, stop using git-lfs``, 6 Jul 2026) and its
TensorFlow pin (``fcf75d3``: "Actually need tf 2.13, not 2.14").  Duplicated from :mod:`melee_rl.phillip`
on purpose: this module must stay import-light (no torch), and ``tests/rl/test_modal_app.py`` asserts
they still agree. Users supply this GPL-3.0 source and its sidecar interpreter separately.
The separate interpreter solves runtime compatibility. It does not establish license clearance;
see ``THIRD_PARTY_NOTICES.md`` for provenance and review scope."""
PHILLIP_APT_PACKAGES: Final[tuple[str, ...]] = ("git", "python3", "python3-venv")
"""``python3`` on the ``debian_slim`` (bookworm) base is 3.11 -- a second interpreter beside the image's
3.12, only ever run by the sidecar."""
PHILLIP_PIP_PACKAGES: Final[tuple[str, ...]] = (PHILLIP_TENSORFLOW_PIN, "numpy<1.25", "attrs>=23")
"""The sidecar venv: upstream's three runtime imports (TensorFlow, numpy, attrs).  ``pyzmq`` and the
Dolphin-side modules are never imported by ``phillip.agent``.  Nothing here touches the image's own
Python, so torch, libmelee and every other opponent keep their pins."""
PHILLIP_DELAY0_AGENTS: Final[tuple[str, ...]] = (
    "delay0/FoxFD",
    "delay0/FalcoFD",
    "FoxFD0",
    "MarthFD0",
    "PeachFD",
    "SheikFD",
    "FalconFalconBF",
)
"""The seven delay-0 agents whose weights the build checks for (``melee_rl.phillip.DELAY0_AGENTS``)."""


def phillip_image_commands() -> list[str]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


SLIPPI_AI_ROOT: Final[str] = "/opt/slippi-ai"
"""Where the slippi-ai image layer puts the checkout (``melee_rl.slippi_ai_agent``)."""
SLIPPI_AI_SOURCE_DIR: Final[str] = f"{SLIPPI_AI_ROOT}/source"
SLIPPI_AI_REPO_URL: Final[str] = "https://github.com/vladfi1/slippi-ai.git"
SLIPPI_AI_REVISION: Final[str] = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
SLIPPI_AI_LICENSE: Final[str] = "MIT"
"""The repository's own pin (``third_party/VERSIONS.txt``) and the revision E010 gated ``medium-v2`` at.
Duplicated from :mod:`melee_rl.slippi_ai_agent` on purpose: this module must stay import-light (no
torch), and ``tests/rl/test_modal_app.py`` asserts they still agree. This MIT source is supplied
separately and imported in process; attribution is retained in ``THIRD_PARTY_NOTICES.md``."""
SLIPPI_AI_MODEL_DIR: Final[str] = f"{VOLUME_MOUNT}/external/slippi-ai"
"""Where the released checkpoints are staged on the Volume (``[slippi_ai] model_dir``)."""
SLIPPI_AI_D0_REVISION: Final[str] = "9eca7479a9555f4ee4b7e6800c656a7f5bb55902"
SLIPPI_AI_D0_SOURCE_DIR: Final[str] = f"{SLIPPI_AI_ROOT}/source-{SLIPPI_AI_D0_REVISION[:12]}"
"""The second checkout (22 Sep 2026): the March-2026 main snapshot the shared delay-0 Fox
(``fox_d0_tx_like_3x512``) was trained on -- the pin cannot restore it (``melee_rl.slippi_ai_agent.
SLIPPI_AI_D0_REVISION`` has the story).  Duplicated like the pin; ``checkout_dir`` is the rule and the
tests keep the two equal.  Every release the registry names at a third revision would need its own."""
SLIPPI_AI_APT_PACKAGES: Final[tuple[str, ...]] = ("git",)
SLIPPI_AI_PIP_PACKAGES: Final[tuple[str, ...]] = (
    NUMPY_PIN,
    "tf-nightly==2.21.0.dev20260203",
    "tfp-nightly==0.26.0.dev20260820",
    # tensorflow_probability validates its environment by importing the *standalone* Keras 2 package
    # (``tensorflow_probability/__init__.py:87``), which tf-nightly does not pull.  The E010 lock has it
    # too; its exact nightly has since been yanked from PyPI, so this is the one dated the same day as
    # the TensorFlow pin.  Without it ``slippi_ai.tf.saving`` raises ModuleNotFoundError -- measured
    # 10 Sep 2026, one Dolphin boot into the first smoke.
    "tf-keras-nightly==2.21.0.dev2026020310",
    "dm-sonnet==2.0.2",
    "dm-tree==0.1.10",
    "pyarrow==23.0.1",
    "pandas==2.3.0",
    "fancyflags==1.2",
    "absl-py==2.5.0",
    "fsspec==2026.7.0",
    "portpicker==1.6.0",
    "peppi-py-vladfi==0.9.2",
    "py7zr==1.1.3",
    "dnspython==2.8.0",
    "parameterized==0.9.0",
)
"""slippi-ai's runtime dependencies at the versions ``melee-policy/requirements-e010.lock`` proved.

The TensorFlow trio is their ``[tf]`` extra; the rest is ``install_requires`` **minus** ``melee``,
``tqdm`` and ``wandb``.  :data:`NUMPY_PIN` is repeated from :data:`IMAGE_PACKAGES` so the resolver
cannot quietly move numpy out from under torch and Ali's preprocessing while satisfying TensorFlow
(E010 ran this TF build against numpy 2.2.6, outside that pin); if the two are unsatisfiable the
*build* fails, which is the cheapest place to find out.

``melee`` is deliberately excluded: :data:`LIBMELEE_PIN` is already installed and a resolver that moved
it would change every opponent in the repository at once (the P26 lesson cost 1.9 stocks a game).

``slippi-ai`` itself is *not* pip-installed -- the checkout goes on ``sys.path``
(``melee_rl.slippi_ai_agent.load_slippi_ai_api``), so its own ``melee>=0.47.1`` and ``peppi-py-vladfi``
requirements cannot re-resolve our pins."""
SLIPPI_AI_JAX_PIP_PACKAGES: Final[tuple[str, ...]] = (
    NUMPY_PIN,
    "jax==0.11.1",
    "jaxlib==0.11.1",
    "flax==0.12.9",
)
"""The JAX half of upstream (``jax-requirements.txt``: ``flax>=0.12``), for the shared delay-0 Fox.

22 Sep 2026: ``fox_d0_tx_like_3x512`` is ``platform: 'jax'`` (flax nnx), which upstream's
``saving.load_policy_from_state`` routes to ``slippi_ai.jax.saving`` -- ``jax``, ``flax`` and nothing
else (no ``tfp-nightly``, no ``qpax``: those serve the nash learners).  The versions are the ones
current at the pinned revision's date (jax 0.11.1 / flax 0.12.9, 17-18 Aug 2026); ``jax`` from PyPI
without an extra is the **CPU** wheel, which is what the recorder wants (its L4 belongs to our policy;
``SlippiAIPlayer`` also sets ``JAX_PLATFORMS=cpu``).  Its own layer so a change here leaves the
TensorFlow layer's cache alone; :data:`NUMPY_PIN` repeated for the reason given above (jax 0.11 needs
numpy >= 2.1, which the pin admits)."""


def slippi_ai_image_commands() -> list[str]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


def version_probe(exe: str | Path, *, timeout_s: float = 60.0) -> dict[str, Any]:
    """``<exe> --version`` as libmelee runs it: raw return code, stdout and stderr (never raises)."""

    command = [str(exe), "--version"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "error": repr(error)}
    return {
        "command": command,
        "returncode": done.returncode,
        "stdout": done.stdout[-2000:],
        "stderr": done.stderr[-2000:],
    }


# ---------------------------------------------------------------------------
# resources, prices, estimates (ENV.md §5.2-5.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resources:
    """What a function requests (Modal bills per second on ``max(request, usage)``)."""

    cpu: float
    memory_mib: int
    timeout_s: int
    gpu: str | None = None

    def describe(self) -> str:
        cores = f"{self.cpu:g} cores / {self.memory_mib / 1024:g} GiB"
        return cores if self.gpu is None else f"{self.gpu} + {cores}"


TESTS_RESOURCES: Final[Resources] = Resources(cpu=4, memory_mib=8192, timeout_s=1800)
TESTS_GPU_RESOURCES: Final[Resources] = Resources(cpu=4, memory_mib=16384, timeout_s=1800, gpu="L4")
"""``run_tests_gpu`` (5 Sep 2026, ``/rl-speedup``): the CUDA-gated tests (CUDA graphs, pinned transfers, the
prefix placement) on an L4 -- the dev container's GPU comes and goes (GOTCHAS)."""
TRAIN_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=3600)
DOLPHIN_CHECK_RESOURCES: Final[Resources] = Resources(cpu=2, memory_mib=4096, timeout_s=900)
DOLPHIN_TRAIN_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=86400)
"""``dolphin_train``: the 24 h Modal maximum, so a detached multi-day run (``modal run --detach``, one launch
per day resuming from ``latest.pt``) is bounded by the run file's ``max_runtime_s``, not by the function."""
DOLPHIN_TRAIN_GPU_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=86400, gpu="L4")
"""``dolphin_train_gpu`` (P6a): the learner on an L4, the Dolphins on the 8 cores next to it."""
DUMMY_BENCH_GPU_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=32768, timeout_s=86400, gpu="L4")
"""The dummy-env bench of the 75m paths (5 Sep 2026, /rl-speedup): ``dolphin_train_gpu`` with 32 GiB instead
of 16 -- no Dolphins, but the 75m policy + teacher + value net, 8 trajectories with their host-side prefixes
(2.7 GiB) and the profiler's trace buffers; applied through ``Function.with_options`` from the planned row."""
DOLPHIN_TRAIN_GPU_WIDE_RESOURCES: Final[Resources] = Resources(
    cpu=32, memory_mib=65536, timeout_s=86400, gpu="L4"
)
"""``dolphin_train_gpu_wide`` (P6b, 22 Aug 2026): the same L4 with 32 cores / 64 GiB for 64 Dolphins (+ the
eval Dolphins during an evaluation): the per-frame policy forward is latency-bound on the GPU (about 45 ms per
batched frame for 20m at batch 8, 10 % utilisation), so more Dolphins per container is the cheap throughput
lever (Modal caps a container at 64 cores / 336 GB)."""

ENV_BENCH_RESOURCES: Final[Resources] = Resources(cpu=32, memory_mib=65536, timeout_s=7200, gpu="L4")
"""``env_bench`` (5 Sep 2026, ``/rl-dolphin-speedup`` lever 0): the Dolphin-only frame-step bench on the
production box class -- the wide L4 box, although no policy runs, because Modal's CPU-only hosts are not the
GPU hosts and CPU-bound numbers differ by host class (GOTCHAS S45: about 30 % between L4 hosts alone)."""
DOLPHIN_TRAIN_GPU_XWIDE_RESOURCES: Final[Resources] = Resources(
    cpu=32, memory_mib=98304, timeout_s=86400, gpu="L4"
)
"""The wide box with 96 GiB (4 Sep 2026, the 75m run): ``context_mode = "prefix"`` parks the 16 x 168 fp32
prefixes of one update in host memory -- 3.3 GiB at 10m, 10.8 GiB at 75m (11 layers x 3 kv heads x 64 x 256
slots) -- on top of the ~43-47 GiB the 96 + 30 Dolphins and their workers already use on the 64 GiB box.
Not a function of its own: the ``train`` entrypoint takes a planned row's resources to the static function
through ``Function.with_options`` (:func:`resource_overrides`)."""
"""A 48-hour variant was tried on 4 Sep 2026 (user decision: one 48-h launch of the 75m day file, to see
whether Modal interrupts it): ``with_options(timeout=172800)`` is refused by the client before anything
runs -- ``InvalidError: Timeout must be between 10s and 86400s (inclusive)`` -- so a run longer than 24 h is
several 22-h launches of the same command, as before (``[runtime] max_total_runtime_s`` caps the total)."""

MIMIC_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=32768, timeout_s=10800, gpu="L4")
"""``record_mimic`` (P12): MIMIC has no KV cache -- every frame re-runs its whole 60-frame window
(``d_model`` 1024 x 4 layers, ~96 GFLOP at batch 16), which is seconds per frame on a CPU and ~10 ms on
an L4.  The 8 cores carry the 16 Dolphins next to it; 32 GiB covers them plus both models."""

VIDEO_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=10800)
SMASHBOT_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=10800)
"""``external_match_cpu`` (P25): SmashBot is an expert system in pure Python, so a ``smashbot vs cpu9``
calibration arm carries no torch model and needs no GPU -- only the cores its Dolphins run on."""
PHILLIP_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=10800)
"""``external_match_phillip_cpu`` (P32): a ``phillip vs cpu9`` arm runs one small TensorFlow graph per Dolphin
in a single-threaded sidecar session and no torch model; the cores are the Dolphins'."""
PHILLIP_CHECK_RESOURCES: Final[Resources] = Resources(cpu=4, memory_mib=8192, timeout_s=1800)
"""``phillip_check`` (P32): the sidecar preflight of every agent -- no Dolphin, no GPU, a few minutes."""
SLIPPI_CHECK_RESOURCES: Final[Resources] = Resources(cpu=4, memory_mib=8192, timeout_s=1800)
"""``slippi_check`` (22 Sep 2026): a released slippi-ai agent built and stepped on synthetic frames on the
slippi-ai image -- no Dolphin, no GPU (TensorFlow off it, JAX on its CPU wheel), a few minutes."""
"""``record_video`` (P9): playing the matchups and rendering them are both CPU work -- at batch 2 the
policy forward is latency-bound, where the L4 measured *worse* than the CPU (ENV.md §5.3).

The 3 h ceiling is worst-case insurance, not an estimate (a job is planned at 45 min): until the
render rate is measured, each of the six fresh clips may run to its own ``[video.render]`` bound of
30 min, and at ``workers = 2`` that is three rounds = 1.5 h for stage 2 alone.  Modal bills per
second of actual use, so a high ceiling costs nothing; being killed mid-job would cost the run."""
VIDEO_CHECK_RESOURCES: Final[Resources] = Resources(cpu=4, memory_mib=8192, timeout_s=1800)
"""``video_check`` (P9 §6): the one probe that answers whether any video backend initialises here."""
RENDER_BENCH_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=10800)
"""``render_bench``: the render rate and how it scales with parallel workers, on the video box."""
RENDER_BENCH_GPU_RESOURCES: Final[Resources] = Resources(cpu=8, memory_mib=16384, timeout_s=10800, gpu="L4")
"""``render_bench_gpu``: the same with an L4 attached -- does hardware GL engage at all under Xvfb?"""

PRICES_DATE: Final[str] = "21 Aug 2026"
PRICES_PER_SECOND: Final[Mapping[str, float]] = MappingProxyType(
    {
        "cpu_core": 0.0000131,
        "memory_gib": 0.00000222,
        "T4": 0.000164,
        "L4": 0.000222,
        "A10": 0.000306,
        "L40S": 0.000542,
        "A100-40GB": 0.000583,
        "A100-80GB": 0.000694,
        "H100": 0.001097,
    }
)
"""modal.com/pricing as read on :data:`PRICES_DATE` (ENV.md §5.2); USD per second."""


def cost_estimate(resources: Resources, seconds: float) -> float:
    """USD for ``resources`` held for ``seconds`` at :data:`PRICES_PER_SECOND`."""

    if seconds < 0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    prices = PRICES_PER_SECOND
    cost = seconds * (resources.cpu * prices["cpu_core"] + resources.memory_mib / 1024 * prices["memory_gib"])
    if resources.gpu is not None:
        if resources.gpu not in prices:
            priced = sorted(name for name in prices if name not in ("cpu_core", "memory_gib"))
            raise ValueError(f"no price for gpu {resources.gpu!r}; priced: {priced}")
        cost += seconds * prices[resources.gpu]
    return cost


def format_usd(cost: float) -> str:
    return f"${cost:.3f}"


@dataclass(frozen=True)
class PlannedRun:
    """One row of ENV.md §5.3: the name, the resources and the planned duration."""

    name: str
    resources: Resources
    seconds: int
    note: str

    @property
    def cost(self) -> float:
        return cost_estimate(self.resources, self.seconds)


PLANNED_RUNS: Final[tuple[PlannedRun, ...]] = (
    PlannedRun("run_tests", TESTS_RESOURCES, 240, "pytest tests -q, incl. cold start"),
    PlannedRun(
        "run_tests_gpu", TESTS_GPU_RESOURCES, 600, "the CUDA-gated tests on an L4 (tests --gpu --k ...)"
    ),
    PlannedRun("smoke_train smoke_3m", TRAIN_RESOURCES, 600, "2 steps of the 3m profile; creates the volume"),
    PlannedRun("smoke_train learn_tiny", TRAIN_RESOURCES, 180, "20 updates; learning curve in W&B"),
    PlannedRun(
        "dolphin_check",
        DOLPHIN_CHECK_RESOURCES,
        300,
        "boot smoke per build: --version, boot, fps, idle (needs the ISO)",
    ),
    PlannedRun(
        "dolphin_train dolphin_cpu",
        DOLPHIN_TRAIN_RESOURCES,
        900,
        "1-2 steps of 3m vs CPU level 9 in 2 Dolphins",
    ),
    PlannedRun(
        "dolphin_train dolphin_self",
        DOLPHIN_TRAIN_RESOURCES,
        900,
        "1-2 steps of 3m in self-play in 2 Dolphins",
    ),
    PlannedRun(
        "dolphin_train dolphin_cpu8",
        DOLPHIN_TRAIN_RESOURCES,
        720,
        "P6a: 8 Dolphins vs CPU 9, T = C = 128, reward terms on, threaded env; about 9 min + boots",
    ),
    PlannedRun(
        "dolphin_train_gpu dolphin_self8",
        DOLPHIN_TRAIN_GPU_RESOURCES,
        720,
        "P6a: two-port self-play in 8 Dolphins, the learner on the L4 (cu128 image); about 9 min + boots",
    ),
    PlannedRun(
        "dolphin_train dolphin_other8",
        DOLPHIN_TRAIN_RESOURCES,
        360,
        "P6a: the other opponent = the cpu8 checkpoint on port 2; about 4 min + boots",
    ),
    PlannedRun(
        "dolphin_train dolphin_cpu8_eval",
        DOLPHIN_TRAIN_RESOURCES,
        600,
        "P6b item 0: cpu8 + the in-loop evaluator (cpu9 + best every 150 s); about 5.5 min + boots + evals",
    ),
    PlannedRun(
        "dolphin_train dolphin_cpu8_day",
        DOLPHIN_TRAIN_RESOURCES,
        84600,
        "P6b: 3m from scratch vs CPU 9, hourly evals, one detached 23.5 h launch per day (resumes)",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_wide_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "P6b: the 20m day run on 64 Dolphins in ring mode, L4 + 32 cores / 64 GiB; resumes p6b/20m_day",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_self_wide_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "session 13: two-port self-play from scratch, 64 Dolphins x 2 ports = 128 rows, lr 3e-4",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_self_wide_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        1200,
        "session 13: 20-min shake-out of the self-play run (128 rows on the L4, 3-opponent eval)",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_self_wide_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "session 13 take two: self-play from scratch on the 5m profile, 64 Dolphins x 2 ports",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_self_wide_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        1200,
        "session 13 take two: 20-min shake-out that MEASURES the actor KL before the day launch",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_bc_self_wide_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        1500,
        "session 15 (24 Aug 2026): 20-min shake-out of the BC-init run + the end eval after the cap",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_bc_self_wide_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "session 15: self-play from Ali's 5m BC checkpoint, BC teacher KL 0.1, 64 Dolphins x 2 ports",
    ),
    PlannedRun(
        "dolphin_train_gpu dolphin_20m_day",
        DOLPHIN_TRAIN_GPU_RESOURCES,
        84600,
        "P6b: Ali's 20m from scratch vs CPU 9 on the L4, hourly evals, one detached 23.5 h launch/day",
    ),
    PlannedRun(
        "dolphin_train dolphin_cpu8_mp",
        DOLPHIN_TRAIN_RESOURCES,
        360,
        "P7a canary: cpu8 in 4 spawned env worker processes on gVisor; about 5 min + boots",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_bc_self_wide_mp_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        960,
        "P7a bench: the BC wide shape in 8 env workers, eval off; read timing/env_s vs the 7.3 s baseline",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_bc_self_wide_fast_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        960,
        "P7b bench: fast policy step + 16 env workers, eval off; read timing/policy_s vs 15.0 s",
    ),
    PlannedRun(
        "dolphin_train dolphin_cpu8_delay",
        DOLPHIN_TRAIN_RESOURCES,
        360,
        "P8 canary: the delay pipeline (D=5, b=2, k=2) on 8 Dolphins in 2 workers; correctness only",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_delay_b4k3_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        900,
        "P8 bench 1: from-scratch delayed self-play at D=5, b=4, k=3; success step <= ~6 s / >= ~1300 fps",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_delay_b2k4_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        900,
        "P8 bench 2: the env-favouring point b=2, k=4 at D=5; same success bar as bench 1",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_delay_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "P8: the first delayed day run -- 5m from scratch at D=5, self-play, the winning (b, k); resumes",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_delay_nb16_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        1800,
        "matched-batch shake-out: 96 envs, num_batches 16; measures the per-update KL and 96-env fps",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_delay_e128_probe",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        600,
        "env-count probe: 128 Dolphins (8 per worker) on the wide box; fps, container RAM, L4 memory",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_5m_delay_e160_probe",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        600,
        "env-count probe: 160 Dolphins (10 per worker); expected to reach the 64 GiB / 23 GiB wall",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_bc_self_wide_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        3600,
        "20m BC-init shake-out: measures the per-update KL at lr 4e-4, the 20m step time and GPU peak",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_bc_self_wide_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "the 20m BC-init day run: Ali's c3 checkpoint, teacher KL 0.1, D=0, nb16 x 96 envs; resumes",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_bc_stages_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        3600,
        "10m BC-init shake-out on the six legal stages: 40-min cap, start/end evals incl. MIMIC; measures KL",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_bc_stages_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "10m BC-init self-play on six stages, cpu9 / MIMIC / anchor evals; 22-h launch of a 36-h total",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_10m_bc_stages_prefix_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        2400,
        "10m smoke in context_mode prefix (patch #1): 25-min cap, one lean end eval; KL-pre, step time, RAM",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_bc_stages_prefix_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "10m day run continued in context_mode prefix with the prefix teacher; resumes selfplay/10m_stages",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_75m_bc_stages_smoke",
        DOLPHIN_TRAIN_GPU_XWIDE_RESOURCES,
        5400,
        "75m BC-init shake-out on six stages, prefix mode, 96 GiB: 70-min cap, start/end evals; KL, step",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_75m_bc_stages_day",
        DOLPHIN_TRAIN_GPU_XWIDE_RESOURCES,
        81600,
        "75m BC-init self-play on six stages, prefix mode, 96 GiB, cpu9 / MIMIC / anchor evals; 22 h each",
    ),
    PlannedRun(
        "dolphin_train_gpu dummy_75m_bench",
        DUMMY_BENCH_GPU_RESOURCES,
        1500,
        "/rl-speedup bench: the 75m policy / learner paths at production shape on the L4, no Dolphin",
    ),
    PlannedRun(
        "dolphin_train_gpu dummy_75m_bench_base",
        DUMMY_BENCH_GPU_RESOURCES,
        1500,
        "/rl-speedup A/B baseline: the bench with profiling off, metrics.jsonl on",
    ),
    PlannedRun(
        "dolphin_train_gpu dummy_75m_bench_mb24",
        DUMMY_BENCH_GPU_RESOURCES,
        1500,
        "/rl-speedup A/B: microbatch_envs 24 (half the learner passes)",
    ),
    PlannedRun(
        "dolphin_train_gpu dummy_75m_bench_nockpt",
        DUMMY_BENCH_GPU_RESOURCES,
        1500,
        "/rl-speedup A/B: gradient checkpointing off (no recompute forward; memory)",
    ),
    PlannedRun(
        "dolphin_train_gpu dummy_75m_bench_tf32",
        DUMMY_BENCH_GPU_RESOURCES,
        1500,
        "/rl-speedup A/B: TF32 GEMMs for the policy's learner passes; the precision cost is measured",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_75m_profile_smoke",
        DOLPHIN_TRAIN_GPU_XWIDE_RESOURCES,
        3600,
        "/rl-speedup profile smoke: the 75m six-stage loop, two profiled PPO steps, an end eval; 96 GiB",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_pt_stages_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "10m post-trained (cur-6-kd) init, the 10m prefix day recipe, best.pt on MIMIC; 22-h launches, 72 h",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_leash_kl0p03_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "strength gate 1: step 1318 forked, slippi-ai's recipe, teacher KL 0.03 fwd + rev; 22-h launches",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_leash_kl0p01_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "strength gate 1: step 1318 forked, slippi-ai's recipe, teacher KL 0.01 fwd + rev; 22-h launches",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_leash_kl0p003_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "strength gate 1: step 1318 forked, slippi-ai's recipe, teacher KL 0.003 fwd + rev; 22-h launches",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_delay_distill_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        4500,
        "delay fine-tuning: the Stage-1 distillation at D = 18; 55-min cap (it counts the 22-min start eval)",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_delay_distill_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        32400,
        "delay fine-tuning, Stage 1: kl0p01's final distilled to D = 18 by its hindsight teacher; an 8-h cap",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_delay_ppo_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "delay fine-tuning, Stage 2: kl0p01's PPO recipe at D = 18, anchored to the Stage-1 student; 22 h",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_delay_ema_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "delay fine-tuning, hop 2: the Stage-2 recipe with the EMA anchor, the entropy floor and the "
        "KL-targeted lr from hop 1's best; a hard 22-h directory cap",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_10m_delay_league_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        4500,
        "the league smoke (LEAGUE-PLAN.md §3.5): the day file for an hour with a fox_d18-only start eval",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_delay_league_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        81600,
        "the league run, hop 1: hop 2's recipe from its step 760 against self / gm x 12 / fox_d18 / our "
        "delay-0 best / the Falco through the slippi-ai port; a hard 22-h directory cap",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_75m_pt_stages_day",
        DOLPHIN_TRAIN_GPU_XWIDE_RESOURCES,
        81600,
        "75m post-trained (cur-3) init, the 75m day recipe on the 96 GiB box, best.pt on MIMIC; 22 h each",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_bc_mix_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        2400,
        "P10 shake-out: 32 self + 64 cpu9 Dolphins from the 20m BC run's best.pt; measures KL + step time",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_bc_mix_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "the P10 mixed-curriculum day run: 50/50 rows self vs cpu9 from best.pt, teacher = the BC file",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide dolphin_20m_bc_mix_dmg005_day",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        84600,
        "the KO-weighted mix day run: the same curriculum at damage_ratio 0.005, from the mix latest.pt",
    ),
    PlannedRun(
        "render_bench",
        RENDER_BENCH_RESOURCES,
        1200,
        "P9 stage 2 bench: render slowdown vs real time and its scaling with workers, CPU box",
    ),
    PlannedRun(
        "render_bench_gpu",
        RENDER_BENCH_GPU_RESOURCES,
        1200,
        "P9 stage 2 bench with an L4: does hardware GL engage under Xvfb, and does it pay?",
    ),
    PlannedRun(
        "video_check",
        VIDEO_CHECK_RESOURCES,
        600,
        "P9 probe: --help / ldd / bundled libav, 20 s of play -> a .slp, then the backend ladder",
    ),
    PlannedRun(
        "record_video video_20m_bc (shake-out)",
        VIDEO_RESOURCES,
        300,
        "P9 shake-out: one matchup, 20 s clips, rendered end to end",
    ),
    PlannedRun(
        "record_video video_20m_bc",
        VIDEO_RESOURCES,
        2700,
        "P9 acceptance: 10 clips x 2 min for the 20m BC run's latest.pt (6 with the baseline cached)",
    ),
    PlannedRun(
        "record_mimic video_mimic",
        MIMIC_RESOURCES,
        3600,
        "P12: 16 four-stock matches of one checkpoint against the released MIMIC Fox, .slp only",
    ),
    PlannedRun(
        "record_smashbot video_smashbot",
        MIMIC_RESOURCES,
        2700,
        "P25: 16 four-stock matches of one checkpoint against altf4/SmashBot, .slp only",
    ),
    PlannedRun(
        "external_match_cpu smashbot_cpu9",
        SMASHBOT_RESOURCES,
        2700,
        "P25 calibration: 16 matches SmashBot vs Dolphin CPU 9 at one input delay; no GPU, no model",
    ),
    PlannedRun(
        "external_match smashbot_mimic",
        MIMIC_RESOURCES,
        3600,
        "P25 calibration: 16 matches SmashBot vs the released MIMIC Fox -- the outside anchor",
    ),
    PlannedRun(
        "record_slippi_ai video_slippi_ai",
        MIMIC_RESOURCES,
        3000,
        "P27: 16 four-stock matches of one checkpoint against one released slippi-ai agent, .slp only",
    ),
    PlannedRun(
        "slippi_parity slippi_parity",
        MIMIC_RESOURCES,
        1200,
        "LEAGUE-PLAN.md §3.4: the PyTorch port shadowing slippi-ai's TensorFlow agent in real games, "
        "three releases",
    ),
    PlannedRun(
        "external_match_slippi slippi_ai_cpu9",
        MIMIC_RESOURCES,
        3000,
        "P27 calibration: 16 matches one slippi-ai release vs CPU 9, MIMIC or SmashBot -- the gate",
    ),
    PlannedRun(
        "phillip_check phillip_cpu9",
        PHILLIP_CHECK_RESOURCES,
        600,
        "P32: the sidecar preflight of the seven delay-0 Phillip agents (restore, timing); no Dolphin",
    ),
    PlannedRun(
        "slippi_check slippi_ai_cpu9",
        SLIPPI_CHECK_RESOURCES,
        900,
        "22 Sep 2026: one released slippi-ai agent built and stepped on synthetic frames; no Dolphin",
    ),
    PlannedRun(
        "external_match_phillip_cpu phillip_cpu9",
        PHILLIP_RESOURCES,
        1800,
        "P32 calibration: 8 matches one Phillip agent vs Dolphin CPU 9; no GPU, no torch model",
    ),
    PlannedRun(
        "external_match_phillip phillip_mimic",
        MIMIC_RESOURCES,
        3600,
        "P32 calibration: 16 matches one Phillip agent vs the released MIMIC Fox -- the ladder rung",
    ),
    PlannedRun(
        "record_phillip video_phillip",
        MIMIC_RESOURCES,
        2700,
        "P32: 16 four-stock matches of one checkpoint against one Phillip agent, .slp only",
    ),
    PlannedRun(
        "env_bench sweep",
        ENV_BENCH_RESOURCES,
        2400,
        "lever 0 (5 Sep 2026): the Dolphin-only frame-step bench -- a pool / worker / chunk sweep, no policy",
    ),
    PlannedRun(
        "dolphin_train_gpu_wide_mimic dolphin_10m_pt_stages_arms4_smoke",
        DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        1800,
        "lever 1 (5 Sep 2026): the 10m post-trained recipe as four interleaved groups of 24, shm, 25 min",
    ),
)


def describe_plan(name: str, resources: Resources, seconds: float) -> str:
    """The line printed before a remote call (the numbers the user approves, rule 4)."""

    cost = format_usd(cost_estimate(resources, seconds))
    return (
        f"{name}: {resources.describe()}, planned {seconds / 60:.0f} min -> about {cost} (Modal prices of "
        f"{PRICES_DATE}, billed per second on max(request, usage)); environment {MODAL_ENVIRONMENT}, "
        f"app {APP_NAME}"
    )


def estimate_table() -> str:
    lines = [f"Planned Modal runs (ENV.md §5.3; prices of {PRICES_DATE}):"]
    for run in PLANNED_RUNS:
        lines.append(
            f"  {run.name:<26} {run.resources.describe():<18} {run.seconds / 60:>4.0f} min  "
            f"{format_usd(run.cost):>7}  {run.note}"
        )
    lines.append(
        "  image builds (first time, Modal's builder): CPU image about 5 min / about $0.01, Dolphin image "
        "about 5 min / about $0.02 (two AppImage downloads + libmelee), GPU Dolphin image about 5 min / "
        "about $0.01 (the cu128 torch wheel and its CUDA libraries, several GB); later runs reuse them"
    )
    return "\n".join(lines)


def _planned(name: str) -> PlannedRun | None:
    for run in PLANNED_RUNS:
        if run.name == name:
            return run
    return None


# ---------------------------------------------------------------------------
# the package mount
# ---------------------------------------------------------------------------


def mount_manifest(root: str | Path | None = None) -> list[tuple[str, int]]:
    """``(relative posix path, bytes)`` of every file the image mount carries, in walk order.

    The same rule Modal applies to ``add_local_dir``: ignored directories are pruned (never walked),
    the remaining files are filtered with :data:`IGNORE_PATTERNS`.
    """

    base = LOCAL_ROOT if root is None else Path(root)
    matcher = modal.FilePatternMatcher(*IGNORE_PATTERNS)
    files: list[tuple[str, int]] = []
    for dirpath, dirnames, filenames in os.walk(base, topdown=True):
        relative_dir = Path(dirpath).relative_to(base)
        dirnames[:] = sorted(name for name in dirnames if not matcher(relative_dir / name))
        for name in sorted(filenames):
            relative = relative_dir / name
            if matcher(relative):
                continue
            files.append((relative.as_posix(), (base / relative).stat().st_size))
    return files


def manifest_summary(files: Sequence[tuple[str, int]]) -> str:
    total = sum(size for _, size in files)
    return f"mount: {len(files)} files, {total / 1024:.0f} KiB, {LOCAL_ROOT} -> {REMOTE_ROOT}"


def manifest_text() -> str:
    files = mount_manifest()
    return "\n".join([manifest_summary(files), *(f"  {size:>8}  {name}" for name, size in files)])


# ---------------------------------------------------------------------------
# run files
# ---------------------------------------------------------------------------


def available_configs() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.toml"))


def resolve_config(config: str | Path) -> Path:
    """A path to a TOML run file, or a name (``smoke_3m`` / ``smoke_3m.toml``) under ``configs/rl``."""

    candidate = Path(config)
    if candidate.is_file():
        return candidate.resolve()
    for option in (CONFIG_DIR / candidate.name, CONFIG_DIR / f"{candidate.name}.toml"):
        if option.is_file():
            return option
    raise FileNotFoundError(
        f"no run file {str(config)!r}: give a path or one of {available_configs()} (in {CONFIG_DIR})"
    )


def config_env_type(path: str | Path) -> str:
    """The ``[env] type`` of a run file (``tomllib`` only -- no torch, usable at ``modal run`` time)."""

    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    env = raw.get("env", {})
    return str(env.get("type", "dummy")) if isinstance(env, Mapping) else "dummy"


def config_needs_mimic(path: str | Path) -> bool:
    """Whether a run file evaluates against ``mimic`` (``[eval] opponents``; ``tomllib`` only, usable at
    ``modal run`` time): such a run needs the MIMIC image (``dolphin_train_gpu_wide_mimic``)."""

    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    section = raw.get("eval", {})
    opponents = section.get("opponents", ()) if isinstance(section, Mapping) else ()
    return isinstance(opponents, list | tuple) and any(item == "mimic" for item in opponents)


KIND_RESOURCES: Final[Mapping[str, Resources]] = MappingProxyType(
    {
        "smoke_train": TRAIN_RESOURCES,
        "dolphin_train": DOLPHIN_TRAIN_RESOURCES,
        "dolphin_train_gpu": DOLPHIN_TRAIN_GPU_RESOURCES,
        "dolphin_train_gpu_wide": DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
        "dolphin_train_gpu_wide_mimic": DOLPHIN_TRAIN_GPU_WIDE_RESOURCES,
    }
)
"""What each training function's decorator requests (the ``kind`` prefix of a planned run's name)."""


def resource_overrides(plan: PlannedRun) -> dict[str, Any]:
    """The ``Function.with_options`` keywords that take a training function from its static request
    (:data:`KIND_RESOURCES`) to the planned row's resources; ``{}`` when they agree.  The 75m files ask
    for 96 GiB on the 64 GiB wide box (:data:`DOLPHIN_TRAIN_GPU_XWIDE_RESOURCES`)."""

    kind = plan.name.split(" ", 1)[0]
    static = KIND_RESOURCES.get(kind)
    if static is None:
        raise ValueError(f"{plan.name!r} is not a training kind ({', '.join(KIND_RESOURCES)})")
    wanted = plan.resources
    overrides: dict[str, Any] = {}
    if wanted.cpu != static.cpu:
        overrides["cpu"] = wanted.cpu
    if wanted.memory_mib != static.memory_mib:
        overrides["memory"] = wanted.memory_mib
    if wanted.gpu != static.gpu:
        overrides["gpu"] = wanted.gpu
    if wanted.timeout_s != static.timeout_s:
        overrides["timeout"] = wanted.timeout_s
    return overrides


def plan_for_config(
    path: str | Path, *, gpu: bool = False, wide: bool = False, mimic: bool = False
) -> PlannedRun:
    """The ENV.md §5.3 row of a run file (``smoke_train <stem>`` / ``dolphin_train <stem>`` /
    ``dolphin_train_gpu <stem>`` / ``dolphin_train_gpu_wide <stem>`` /
    ``dolphin_train_gpu_wide_mimic <stem>``), or a default with that kind's budget."""

    stem = Path(path).stem
    env_type = config_env_type(path)
    if wide and not gpu:
        raise ValueError("--wide is the 32-core / 64 GiB variant of the gpu function: pass --gpu as well")
    if mimic and not (env_type == "dolphin" and gpu and wide):
        raise ValueError(
            "a mimic evaluation runs on the wide GPU Dolphin box (dolphin_train_gpu_wide_mimic): "
            "pass --gpu --wide with a dolphin run file"
        )
    if env_type == "dolphin" and gpu and wide and mimic:
        kind, resources, seconds = "dolphin_train_gpu_wide_mimic", DOLPHIN_TRAIN_GPU_WIDE_RESOURCES, 900
    elif env_type == "dolphin" and gpu and wide:
        kind, resources, seconds = "dolphin_train_gpu_wide", DOLPHIN_TRAIN_GPU_WIDE_RESOURCES, 900
    elif env_type == "dolphin" and gpu:
        kind, resources, seconds = "dolphin_train_gpu", DOLPHIN_TRAIN_GPU_RESOURCES, 900
    elif env_type == "dolphin":
        kind, resources, seconds = "dolphin_train", DOLPHIN_TRAIN_RESOURCES, 900
    elif env_type == "dummy" and gpu and wide:
        raise ValueError(
            "--wide is for Dolphin run files: a dummy-env run takes the plain gpu box (dolphin_train_gpu)"
        )
    elif env_type == "dummy" and gpu:
        # 5 Sep 2026 (/rl-speedup): the dummy env on the L4 box benches the policy / learner paths alone.
        kind, resources, seconds = "dolphin_train_gpu", DOLPHIN_TRAIN_GPU_RESOURCES, 900
    elif gpu:
        raise ValueError(f"env.type {env_type!r} has no gpu training function (dummy | dolphin run files)")
    else:
        kind, resources, seconds = "smoke_train", TRAIN_RESOURCES, 600
    planned = _planned(f"{kind} {stem}")
    if planned is not None:
        return planned
    return PlannedRun(f"{kind} {stem}", resources, seconds, f"not in ENV.md §5.3; assuming the {kind} budget")


def volume_checkpoint_dir(run_name: str, checkpoint_subdir: str | None) -> Path:
    """Where a run's ``latest.pt`` goes on the Volume: ``/runs/<subdir or run name>``."""

    return Path(VOLUME_MOUNT) / (checkpoint_subdir or run_name)


def split_tags(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def split_overrides(value: Any) -> tuple[str, ...]:
    """``table.key=value`` overrides from the CLI: a comma-separated string, a sequence of them, or the
    mapping Modal's option parser makes of ``k=v,k=v`` (5 Sep 2026); the values keep their text
    (``parse_override`` types them)."""

    if value is None or value == "":
        return ()
    if isinstance(value, Mapping):
        return tuple(f"{key}={item}" for key, item in value.items())
    if isinstance(value, str):
        return _split_top_level(value)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise TypeError(f"overrides must be a string, a sequence or a mapping, got {type(value).__name__}")


def _split_top_level(text: str) -> tuple[str, ...]:
    """Split ``text`` on the commas outside ``[]`` / ``{}`` and quotes, so a TOML array or inline table value
    (``env.dolphin.characters=[18, 1]``, P23 8 Sep 2026) stays one override; blanks are dropped."""

    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for char in text:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return tuple(part.strip() for part in parts if part.strip())


# ---------------------------------------------------------------------------
# the work the remote functions do (plain functions, tested locally)
# ---------------------------------------------------------------------------


def pytest_command(extra_args: Sequence[str] = (), *, in_container: bool = False) -> list[str]:
    """``python -m pytest tests -q -p no:cacheprovider --deselect <id> ... <extra_args>``.

    ``in_container`` adds the :data:`GIT_DEPENDENT_TESTS` deselects (no git in the Modal container)."""

    node_ids = [*DESELECTED_TESTS, *(GIT_DEPENDENT_TESTS if in_container else ())]
    deselect = [arg for node_id in node_ids for arg in ("--deselect", node_id)]
    return [sys.executable, "-m", "pytest", *PYTEST_BASE_ARGS, *deselect, *extra_args]


def _stream(command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run ``command``, echo its output line by line (it lands in the Modal logs) and keep it."""

    started = time.perf_counter()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    returncode = process.wait()
    return {
        "command": list(command),
        "returncode": returncode,
        "output": "".join(lines),
        "duration_s": time.perf_counter() - started,
    }


_MYPY_ERROR: Final[re.Pattern[str]] = re.compile(r"^(?P<file>[^:\s]+):\d+(?::\d+)?: error:")


def mypy_error_counts(output: str) -> tuple[int, int]:
    """``(all errors, errors outside Ali's flat modules)`` in a mypy report (the 25 pre-existing
    ``model.py`` errors are not ours, ENV.md §3)."""

    total = ours = 0
    for line in output.splitlines():
        match = _MYPY_ERROR.match(line)
        if match is None:
            continue
        total += 1
        if match["file"] not in ALI_MODULES:
            ours += 1
    return total, ours


def _machine() -> dict[str, Any]:
    return {"python": platform.python_version(), "cpu_count": os.cpu_count(), "platform": platform.platform()}


def run_test_suite(
    pytest_args: Sequence[str] = (),
    *,
    lint: bool = False,
    cwd: str | Path | None = None,
    in_container: bool = False,
) -> dict[str, Any]:
    """The test suite (and, with ``lint``, ruff + mypy) as subprocesses; a JSON-friendly report.

    ``ok`` is pytest's success and, when linting, ruff's plus "no mypy errors outside Ali's
    modules".  Locally ``cwd`` defaults to the package root; in the container that *is*
    :data:`REMOTE_ROOT` and ``in_container`` also deselects :data:`GIT_DEPENDENT_TESTS`.
    """

    workdir = LOCAL_ROOT if cwd is None else Path(cwd)
    report: dict[str, Any] = {
        "pytest": _stream(pytest_command(pytest_args, in_container=in_container), cwd=workdir),
        "lint": None,
        "ok": False,
        "machine": {**_machine(), "cwd": str(workdir)},
    }
    ok = report["pytest"]["returncode"] == 0
    if lint:
        env = {**os.environ, "MYPY_CACHE_DIR": os.environ.get("MYPY_CACHE_DIR", REMOTE_MYPY_CACHE)}
        ruff = _stream([sys.executable, "-m", "ruff", "check", "."], cwd=workdir)
        mypy = _stream([sys.executable, "-m", "mypy"], cwd=workdir, env=env)
        total, ours = mypy_error_counts(mypy["output"])
        mypy.update({"errors": total, "errors_outside_ali_modules": ours})
        report["lint"] = {"ruff": ruff, "mypy": mypy}
        ok = ok and ruff["returncode"] == 0 and ours == 0
    report["ok"] = ok
    return report


class VolumeCommitHook:
    """``Trainer.on_save`` for the Modal functions: ``volume.commit()`` after every checkpoint write.

    Modal commits a Volume when the function returns; a container restart before that (the 23 Aug
    2026 self-play run had two) loses everything written since the previous commit -- that run's
    evaluation records among them.  A failed commit is printed and counted, never raised: the next
    save retries it, and a 23.5-hour run must not die on a transient Volume error.
    """

    def __init__(self, volume: Any, *, name: str = VOLUME_NAME) -> None:
        self.volume = volume
        self.name = name
        self.commits = 0
        self.failures = 0

    def __call__(self) -> None:
        try:
            self.volume.commit()
        except Exception as error:  # any Volume error is reported, not fatal
            self.failures += 1
            print(
                f"volume {self.name}: commit failed ({type(error).__name__}: {error}); the next save retries",
                flush=True,
            )
            return
        self.commits += 1

    def summary(self) -> dict[str, int]:
        return {"commits": self.commits, "failures": self.failures}


def volume_commit_hook(volume: Any, *, name: str = VOLUME_NAME) -> VolumeCommitHook:
    """A :class:`VolumeCommitHook` for ``volume`` (anything with a ``commit()`` method)."""

    return VolumeCommitHook(volume, name=name)


def run_smoke_train(
    config: str | Path,
    *,
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_dir: str | Path | None = None,
    tags: Sequence[str] = (),
    device: str | None = None,
    resume: bool = True,
    on_save: Callable[[], None] | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """``python -m melee_rl.train`` with these options; the ``TrainResult`` as a JSON-friendly dict.

    Exactly the launcher's semantics (``melee_rl.train.main``): the run name defaults to
    ``rl-<config>-<UTC>``, the config stem and ``tags`` become W&B tags, an existing ``latest.pt``
    in ``checkpoint_dir`` is resumed unless ``resume`` is false.  Works for every ``[env] type`` the
    launcher knows (the Dolphin run files need the Dolphin image and the ISO).  ``on_save`` is the
    trainer's per-checkpoint callback (the remote functions pass a :class:`VolumeCommitHook`).
    """

    started = time.perf_counter()
    path = resolve_config(config)
    import torch

    from melee_rl.logging import format_summary
    from melee_rl.train import default_run_name, main

    imported = time.perf_counter()
    name = run_name or default_run_name(path)
    argv = ["--config", str(path), "--wandb-name", name]
    if steps is not None:
        argv += ["--steps", str(steps)]
    if checkpoint_dir is not None:
        argv += ["--checkpoint-dir", str(checkpoint_dir)]
    if wandb_mode is not None:
        argv += ["--wandb-mode", wandb_mode]
    if device is not None:
        argv += ["--device", device]
    if not resume:
        argv.append("--no-resume")
    for tag in tags:
        argv += ["--wandb-tag", tag]
    for override in overrides:
        argv += ["--set", override]
    result = main(argv, on_save=on_save)
    finished = time.perf_counter()
    last = result.last_metrics
    return {
        "config": str(path),
        "config_name": path.stem,
        "env_type": result.config.env.type,
        "run_name": result.run_name,
        "tags": list(result.tags),
        "step": result.step,
        "steps_run": result.steps_run,
        "frames_seen": result.frames_seen,
        "history": result.history,
        "last_metrics": last,
        "summary": None if last is None else format_summary(last),
        "checkpoint": None if result.checkpoint is None else str(result.checkpoint),
        "wandb_run_id": result.wandb_run_id,
        "wandb_mode": result.config.logging.resolved_mode(wandb_mode),
        "argv": argv,
        "duration_s": {
            "imports": imported - started,
            "main": finished - imported,
            "total": finished - started,
        },
        "machine": {
            **_machine(),
            "torch": torch.__version__,
            "threads": result.config.runtime.threads,
            "device": device or result.config.runtime.device,
        },
    }


def run_dolphin_check(
    exe: str | Path,
    iso_path: str | Path,
    *,
    frames: int = 600,
    idle_seconds: float = 60.0,
    num_envs: int = 1,
    characters: tuple[int, int] = (1, 1),
    stage: int = 25,
    cpu_level: int = 9,
) -> dict[str, Any]:
    """The Dolphin boot smoke (P5-PLAN.md "runs (b)"); never raises -- errors land in the report.

    Records the raw ``--version`` output, the build libmelee detected, seconds to the first in-game
    frame, the fps of ``frames`` neutral-controller steps, the stages / characters / frame indices
    seen, item counts, then sleeps ``idle_seconds`` and steps once more (``idle_ok``: no relaunch
    was needed, i.e. an idle Dolphin keeps its connection), plus every per-environment counter.
    """

    started = time.perf_counter()
    report: dict[str, Any] = {
        "exe": str(exe),
        "iso_path": str(iso_path),
        "frames_requested": frames,
        "idle_seconds": idle_seconds,
        "num_envs": num_envs,
        "version_probe": version_probe(exe),
        "machine": _machine(),
        "error": None,
        "ok": False,
    }
    env: Any = None
    try:
        import torch

        from controller_codec import CustomV1Codec
        from melee_rl.env.dolphin import DolphinEnv, DolphinEnvConfig
        from melee_rl.frames import neutral_labels

        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        report["torch"] = torch.__version__
        config = DolphinEnvConfig(
            num_envs=num_envs,
            dolphin_path=str(exe),
            iso_path=str(iso_path),
            characters=characters,
            stage=stage,
            keepalive_seconds=0.0,
        )
        boot_started = time.perf_counter()
        env = DolphinEnv(config, cpu_level=cpu_level)
        report["boot_s"] = time.perf_counter() - boot_started
        report["build"] = asdict(env.build)
        report["launch_seconds"] = [row["launch_seconds"] for row in env.stats()]
        report["slippi_ports"] = [row["slippi_port"] for row in env.stats()]
        first = env.current()
        report["first_frame_index"] = first.frame_index.tolist()
        control = CustomV1Codec().decode(neutral_labels((num_envs,)))
        actions = {port: control for port in env.controlled_ports}
        stages: set[int] = set()
        characters_seen: set[tuple[int, int]] = set()
        resets = 0
        max_items = 0
        max_percent = {"p0": 0, "p1": 0}
        x_range = {"p0": [float("inf"), float("-inf")], "p1": [float("inf"), float("-inf")]}
        frame_min = frame_max = int(first.frame_index.min())
        step_started = time.perf_counter()
        for _ in range(frames):
            output = env.step(actions)
            tree = output.frames[0]
            stages.update(int(value) for value in tree["stage"].tolist())
            p0 = tree["p0"]["character"].tolist()
            p1 = tree["p1"]["character"].tolist()
            characters_seen.update(zip((int(a) for a in p0), (int(b) for b in p1), strict=True))
            resets += int(output.needs_reset.sum())
            max_items = max(max_items, int(tree["items"]["exists"].sum(dim=1).max()))
            for player in ("p0", "p1"):
                max_percent[player] = max(max_percent[player], int(tree[player]["percent"].max()))
                x_range[player][0] = min(x_range[player][0], float(tree[player]["x"].min()))
                x_range[player][1] = max(x_range[player][1], float(tree[player]["x"].max()))
            frame_min = min(frame_min, int(output.frame_index.min()))
            frame_max = max(frame_max, int(output.frame_index.max()))
        step_s = time.perf_counter() - step_started
        report.update(
            {
                "frames_stepped": frames,
                "step_s": step_s,
                "fps_total": frames * num_envs / step_s if step_s > 0 else None,
                "fps_per_env": frames / step_s if step_s > 0 else None,
                "stages_seen": sorted(stages),
                "characters_seen": sorted(characters_seen),
                "resets_seen": resets,
                "max_items": max_items,
                "max_percent": max_percent,
                "x_range": x_range,
                "cpu_levels": [row["cpu_levels"] for row in env.stats()],
                "frame_index_range": [frame_min, frame_max],
                "stats_after_steps": env.stats(),
            }
        )
        before = env.stats()
        last_frame = env.current().frame_index.tolist()
        time.sleep(idle_seconds)
        idle_started = time.perf_counter()
        output = env.step(actions)
        after = env.stats()
        report["idle_step_s"] = time.perf_counter() - idle_started
        report["idle_frame_delta"] = [
            int(b - a) for a, b in zip(last_frame, output.frame_index.tolist(), strict=True)
        ]
        report["idle_ok"] = all(
            a["relaunches"] == b["relaunches"] and b["faults"] == a["faults"]
            for a, b in zip(before, after, strict=True)
        )
        report["stats"] = after
        report["ok"] = True
    except Exception as error:  # the report is the deliverable; a crash must not lose the measurements
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:  # best effort on the way down
                report["close_error"] = repr(error)
    report["duration_s"] = time.perf_counter() - started
    return report


# ---------------------------------------------------------------------------
# P9: recording clips and rendering them
# ---------------------------------------------------------------------------


def find_playback_asset(releases: Sequence[Mapping[str, Any]]) -> dict[str, str] | None:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


def fetch_playback_asset(*, url: str = PLAYBACK_RELEASES_URL, timeout_s: float = 60.0) -> dict[str, Any]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


def install_dolphin(url: str, directory: str | Path, *, timeout_s: float = 900.0) -> dict[str, Any]:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


def _find_dolphin_exe(root: Path) -> Path | None:
    """The ``dolphin-emu`` binary of an unpacked build (AppImage layout or a plain directory)."""

    preferred = root / "squashfs-root" / "usr" / "bin" / "dolphin-emu"
    if preferred.is_file():
        return preferred
    for name in ("dolphin-emu", "dolphin-emu-nogui", "Slippi Dolphin"):
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    return None


def _sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_play(exe: str | Path, iso_path: str | Path, *, seconds: float, work_dir: Path) -> dict[str, Any]:
    """20 s of real play that produces the ``.slp`` the rest of the probe renders."""

    from controller_codec import CustomV1Codec
    from melee_rl.env.dolphin import DolphinEnv, DolphinEnvConfig
    from melee_rl.frames import neutral_labels
    from melee_rl.video import collect_replays

    replays = work_dir / "replays"
    report: dict[str, Any] = {"ok": False, "slp": None, "frames": 0, "error": None}
    env: Any = None
    frames = max(1, round(seconds * 60))
    try:
        config = DolphinEnvConfig(
            num_envs=1,
            dolphin_path=str(exe),
            iso_path=str(iso_path),
            save_replays=True,
            replay_dir=str(replays),
            stop_timeout_s=20.0,
            keepalive_seconds=0.0,
        )
        started = time.perf_counter()
        env = DolphinEnv(config, cpu_level=9)
        report["boot_s"] = time.perf_counter() - started
        control = CustomV1Codec().decode(neutral_labels((1,)))
        actions = {port: control for port in env.controlled_ports}
        stepped = time.perf_counter()
        for _ in range(frames):
            env.step(actions)
        report["frames"] = frames
        report["step_s"] = time.perf_counter() - stepped
        report["fps"] = frames / report["step_s"] if report["step_s"] > 0 else None
        report["stats"] = env.stats()
    except Exception as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            try:
                env.close()  # the graceful stop is what finishes the .slp
            except Exception as error:
                report["close_error"] = repr(error)
    found = collect_replays(replays, 1)[0]
    if found is None:
        report["slp_search"] = sorted(str(item) for item in work_dir.rglob("*.slp"))[:10]
        report["error"] = report["error"] or f"no .slp written under {replays}"
        return report
    report["slp"] = str(found)
    report["slp_bytes"] = found.stat().st_size
    report["ok"] = report["error"] is None
    return report


def run_video_check(
    exe: str | Path,
    iso_path: str | Path,
    *,
    seconds: float = 20.0,
    work_dir: str | Path = "/tmp/video-check",
    download_fallback: bool = False,
    render_timeout_s: float = 120.0,
    candidates: str | None = None,
    builds: str | None = None,
    skip_primary: bool = False,
) -> dict[str, Any]:
    """The P9 §6 probe: can anything in this container render a replay?  Never raises.

    Reports ``--help`` / ``ldd`` / the bundled ffmpeg libraries of the executable, plays ``seconds``
    of real Melee to produce a ``.slp``, then walks ``render.RENDER_CANDIDATES`` until one produces a
    video ffprobe accepts. The historical ``download_fallback`` option reaches a disabled acquisition
    helper in this release. Provide a compatible playback executable separately.
    """

    started = time.perf_counter()
    root = Path(work_dir)
    report: dict[str, Any] = {
        "exe": str(exe),
        "iso_path": str(iso_path),
        "machine": _machine(),
        "help": {},
        "ldd": {},
        "libs": [],
        "play": {},
        "attempts": [],
        "winner": None,
        "fallback": [],
        "error": None,
        "ok": False,
    }
    try:
        from melee_rl.render import RENDER_CANDIDATES, RenderConfig, run_command

        root.mkdir(parents=True, exist_ok=True)
        report["help"] = run_command([str(exe), "--help"], timeout_s=60.0).record(tail=4000)
        report["ldd"] = run_command(["ldd", str(exe)], timeout_s=60.0).record(tail=4000)
        lib_dir = Path(exe).resolve().parents[1] / "lib"
        if lib_dir.is_dir():
            report["libs"] = sorted(
                item.name
                for item in lib_dir.iterdir()
                if any(mark in item.name.lower() for mark in ("libav", "libsw", "libx264", "libvpx"))
            )
        report["play"] = _probe_play(exe, iso_path, seconds=seconds, work_dir=root)
        if not report["play"]["ok"]:
            report["error"] = report["play"]["error"]
            return report
        base = RenderConfig(
            dolphin_path=str(exe),
            iso_path=str(iso_path),
            timeout_s=render_timeout_s,
            caption=False,
            keep_work_dir=True,  # a timed-out attempt still shows whether frames were being dumped
        )
        slp = Path(report["play"]["slp"])
        frames = int(report["play"]["frames"])
        wanted = _selected(RENDER_CANDIDATES, candidates, key=lambda item: item.name)
        if not skip_primary:
            report["winner"] = _walk_candidates(base, wanted, slp, frames, root, report["attempts"])
        if report["winner"] is None and download_fallback:
            report["fallback"] = _walk_fallback_builds(
                report,
                slp,
                frames,
                root,
                iso_path,
                timeout_s=render_timeout_s,
                names=builds,
                candidates=wanted,
            )
        report["ok"] = report["winner"] is not None
        return report
    except Exception as error:  # the report is the deliverable
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        report["duration_s"] = time.perf_counter() - started


def _walk_fallback_builds(
    report: dict[str, Any],
    slp: Path,
    frames: int,
    root: Path,
    iso_path: str | Path,
    *,
    timeout_s: float = 120.0,
    names: str | None = None,
    candidates: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    """Historical fallback probe retained for compatibility.

    Acquisition helpers are disabled in this release, so this path raises before installing
    any emulator. Supply a compatible playback executable through the primary render path.
    """

    from melee_rl.render import RENDER_CANDIDATES, RenderConfig, run_command

    entries = _selected(
        [dict(candidate) for candidate in PLAYBACK_CANDIDATES], names, key=lambda item: item["name"]
    )
    if names is None:
        found = fetch_playback_asset()
        if "url" in found:
            entries.append(found)
        else:
            report["playback_asset_lookup"] = found
    ladder = RENDER_CANDIDATES if candidates is None else candidates
    records: list[dict[str, Any]] = []
    for entry in entries:
        directory = root / str(entry["name"])
        install = install_dolphin(str(entry["url"]), directory)
        record: dict[str, Any] = {"asset": entry, "install": install}
        records.append(record)
        exe = install.get("exe")
        if not exe:
            continue
        linked = run_command(["ldd", str(exe)], timeout_s=60.0)
        record["missing_libs"] = sorted(
            line.split("=>")[0].strip() for line in linked.stdout.splitlines() if "not found" in line
        )
        help_report = run_command([str(exe), "--help"], timeout_s=60.0).record(tail=4000)
        record["help"] = help_report
        text = str(help_report.get("stdout", "")) + str(help_report.get("stderr", ""))
        record["has_comm_file_option"] = "-i " in text or "--slippi" in text
        lib_dir = Path(exe).resolve().parents[1] / "lib"
        record["libs"] = (
            sorted(
                item.name
                for item in lib_dir.iterdir()
                if any(mark in item.name.lower() for mark in ("libav", "libsw", "libx264", "libvpx"))
            )
            if lib_dir.is_dir()
            else []
        )
        config = RenderConfig(
            dolphin_path=str(exe),
            iso_path=str(iso_path),
            timeout_s=timeout_s,
            caption=False,
            keep_work_dir=True,
        )
        attempts: list[Any] = []
        record["attempts"] = attempts
        record["winner"] = _walk_candidates(config, ladder, slp, frames, directory / "render", attempts)
        if record["winner"] is not None:
            report["winner"] = f"{entry['name']}:{record['winner']}"
            report["winner_exe"] = str(exe)
            break
    return records


T = TypeVar("T")


def _selected(items: Sequence[T], names: str | None, *, key: Callable[[T], str]) -> list[T]:
    """``items`` filtered by a comma-separated name list (``None`` keeps them all, in order)."""

    if names is None:
        return list(items)
    wanted = split_tags(names)
    return [item for name in wanted for item in items if key(item) == name]


def _walk_candidates(
    base: Any, candidates: Sequence[Any], slp: Path, frames: int, work_dir: Path, attempts: list[Any]
) -> str | None:
    """Try each rung of the ladder; the first that yields a verified video wins."""

    from melee_rl.render import (
        RenderJob,
        candidate_config,
        ffprobe_command,
        parse_ffprobe,
        render_replay,
        run_command,
    )

    for candidate in candidates:
        config = candidate_config(base, candidate)
        target = work_dir / f"probe-{candidate.name}.mp4"
        job = RenderJob(slp=slp, target=target, caption="", frames=frames)
        result = render_replay(config, job, work_dir=work_dir / candidate.name)
        dumps = work_dir / candidate.name / job.name / "user" / "Dump"
        # The render RATE, measured rather than inferred: how many seconds of footage the dump holds
        # against the seconds of wall clock it took.  Reading it off the file size assumes the encoder
        # hit its target bitrate, which a mostly static screen certainly does not.
        rate: dict[str, Any] = {}
        if result.dump:
            probe = run_command(ffprobe_command(config, result.dump), timeout_s=120.0)
            info = parse_ffprobe(probe.stdout)
            dumped = info["duration_s"]
            rate = {
                "dump_seconds": dumped,
                "dump_frames": info["frames"],
                "wall_seconds": round(result.seconds, 1),
                "slowdown_vs_realtime": (round(result.seconds / dumped, 2) if dumped else None),
            }
        attempts.append(
            {
                "name": candidate.name,
                "backend": candidate.backend,
                "platform": candidate.platform,
                "xvfb": candidate.xvfb,
                **result.record(),
                "dump_tree": sorted(
                    f"{item.relative_to(dumps)} ({item.stat().st_size} B)"
                    for item in dumps.rglob("*")
                    if item.is_file()
                )
                if dumps.is_dir()
                else [],
                "rate": rate,
                "commands": result.commands,
            }
        )
        if result.ok:
            return str(candidate.name)
    return None


def run_render_bench(
    exe: str | Path,
    iso_path: str | Path,
    *,
    render_exe: str | Path | None = None,
    seconds: float = 20.0,
    worker_counts: Sequence[int] = (1, 4),
    work_dir: str | Path = "/tmp/render-bench",
    backend: str = "OGL",
    xvfb: bool = True,
    platform: str | None = None,
    internal_resolution: int = 1,
    bitrate_kbps: int = 3000,
    render_seconds: float = 120.0,
) -> dict[str, Any]:
    """How fast does stage 2 render, and how does it scale with parallel workers?  Never raises.

    Plays ``seconds`` of Melee once to get a real ``.slp``, then renders that same replay ``n`` times
    concurrently for each ``n`` in ``worker_counts``.

    P9b: the render now ends when the replay does, because ``--cout`` makes Dolphin report every
    frame (``render.run_playback``), so each clip reports its own startup / steady-state split and
    ``render_seconds`` is only a safety net.  That makes the batch's wall clock the honest answer for
    that batch size, and lets the report project a real two-minute clip instead of bracketing it.
    ``nvidia-smi`` and ``glxinfo`` are recorded first, so "did the GPU engage" is answered rather than
    inferred -- under Xvfb, GLX normally lands on Mesa's llvmpipe regardless of what is attached.
    """

    from melee_rl.render import (
        RenderConfig,
        RenderJob,
        ffprobe_command,
        parse_ffprobe,
        render_many,
        run_command,
    )

    started = time.perf_counter()
    root = Path(work_dir)
    renderer = str(render_exe or RENDER_DOLPHIN_PATH)
    report: dict[str, Any] = {
        "exe": str(exe),
        "render_exe": renderer,
        "machine": _machine(),
        "backend": backend,
        "xvfb": xvfb,
        "seconds": seconds,
        "worker_counts": list(worker_counts),
        "batches": [],
        "error": None,
        "ok": False,
    }
    try:
        root.mkdir(parents=True, exist_ok=True)
        report["nvidia_smi"] = run_command(["nvidia-smi"], timeout_s=60.0).record(tail=1500)
        display = ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24"] if xvfb else []
        report["glxinfo"] = run_command([*display, "glxinfo", "-B"], timeout_s=120.0).record(tail=1500)
        report["play"] = _probe_play(exe, iso_path, seconds=seconds, work_dir=root)
        if not report["play"]["ok"]:
            report["error"] = report["play"]["error"]
            return report
        slp = Path(report["play"]["slp"])
        frames = int(report["play"]["frames"])
        config = RenderConfig(
            dolphin_path=renderer,
            iso_path=str(iso_path),
            backend=backend,
            platform=platform,
            xvfb=xvfb,
            internal_resolution=internal_resolution,
            bitrate_kbps=bitrate_kbps,
            caption=False,
            timeout_s=render_seconds,
            min_timeout_s=render_seconds,
            speed_factor=1.0,  # a flat safety cap: the frame markers are what end the render
            keep_work_dir=True,  # the raw dump is the measurement, not the trimmed mp4
        )
        for count in worker_counts:
            batch_dir = root / f"workers{count}"
            jobs = [
                RenderJob(slp=slp, target=batch_dir / f"clip{index}.mp4", frames=frames)
                for index in range(count)
            ]
            batch_started = time.perf_counter()
            results = render_many(config, jobs, work_dir=batch_dir / "work", workers=count)
            wall = time.perf_counter() - batch_started
            footage = []
            trees = []
            for result in results:
                work = batch_dir / "work" / Path(result.target).stem / "user" / "Dump"
                if work.is_dir():
                    trees += [
                        f"{item.relative_to(work)} ({item.stat().st_size} B)"
                        for item in sorted(work.rglob("*"))
                        if item.is_file()
                    ]
                if not result.dump:
                    continue
                probe = run_command(ffprobe_command(config, result.dump), timeout_s=120.0)
                duration = parse_ffprobe(probe.stdout)["duration_s"]
                if duration:
                    footage.append(float(duration))
            startups = [result.startup_s for result in results if result.startup_s is not None]
            rates = [result.render_fps for result in results if result.render_fps]
            slowdowns = [result.slowdown for result in results if result.slowdown is not None]
            row: dict[str, Any] = {
                "workers": count,
                "clips": len(jobs),
                "finished": sum(1 for result in results if result.finished),
                "dumped": len(footage),
                "wall_s": round(wall, 1),
                "stop_reasons": [result.stop_reason for result in results],
                "startup_s": [round(value, 1) for value in startups],
                "render_fps": [round(value, 2) for value in rates],
                "slowdown": [round(value, 2) for value in slowdowns],
                "footage_seconds": [round(value, 1) for value in footage],
                "errors": [result.error for result in results if not result.dump],
                "dump_tree": trees[:8],
            }
            if startups:
                row["median_startup_s"] = round(statistics.median(startups), 1)
            if rates:
                # Melee runs at 60 fps, so render_fps / 60 is the fraction of real time we manage.
                fps = statistics.median(rates)
                row["median_render_fps"] = round(fps, 2)
                row["render_slowdown"] = round(60.0 / fps, 2) if fps else None
                projected = row.get("median_startup_s", 0.0) + 7200.0 / fps
                row["minutes_per_2min_clip"] = round(projected / 60.0, 1)
            if slowdowns:
                row["median_slowdown"] = round(statistics.median(slowdowns), 2)
            report["batches"].append(row)
        report["ok"] = any(row.get("dumped") for row in report["batches"])
        return report
    except Exception as error:  # the report is the deliverable
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        return report
    finally:
        report["duration_s"] = time.perf_counter() - started


def format_render_bench(report: Mapping[str, Any]) -> str:
    """The bench as text, printed inside the container so a dead client loses nothing."""

    gl = str(report.get("glxinfo", {}).get("stdout", ""))
    renderer = next(
        (line.strip() for line in gl.splitlines() if "renderer string" in line.lower()), "unknown"
    )
    smi = report.get("nvidia_smi", {})
    lines = [
        f"render_exe {report['render_exe']} backend {report['backend']} xvfb {report['xvfb']}",
        f"GL {renderer}",
        f"nvidia-smi rc {smi.get('returncode')!r}: {str(smi.get('stdout', ''))[:300]}",
    ]
    play = report.get("play", {})
    lines.append(
        f"play: ok {play.get('ok')} {play.get('frames')} frames at {play.get('fps')} fps -> "
        f"{play.get('slp')}; error {play.get('error')}"
    )
    for row in report["batches"]:
        lines.append(
            f"  workers {row['workers']:<3} finished {row.get('finished')}/{row['clips']} "
            f"dumped {row['dumped']}/{row['clips']} in {row['wall_s']} s wall"
        )
        lines.append(
            f"    startup {row.get('median_startup_s')} s (all {row.get('startup_s')}), "
            f"render {row.get('median_render_fps')} fps = {row.get('render_slowdown')}x real time, "
            f"end to end {row.get('median_slowdown')}x -> {row.get('minutes_per_2min_clip')} min "
            f"per 2-min clip"
        )
        lines.append(f"    stop reasons {row.get('stop_reasons')}, footage {row['footage_seconds']} s")
        lines.append(f"    dump tree: {row.get('dump_tree')}")
        for error in row["errors"][:2]:
            lines.append(f"    error: {error}")
    lines.append(f"render_bench: ok {report['ok']}, error {report['error']}")
    return "\n".join(lines)


def format_video_check(report: Mapping[str, Any]) -> str:
    """The probe's findings as text -- printed inside the container too, so a dead client loses nothing."""

    lines = [
        f"--help rc {report['help'].get('returncode')!r}:\n{report['help'].get('stdout', '')}",
        f"ldd: {report['ldd'].get('stdout', '')[-1500:]}",
        f"bundled ffmpeg libraries: {report['libs']}",
    ]
    play = report["play"]
    lines.append(
        f"play: ok {play.get('ok')} {play.get('frames')} frames at {play.get('fps')} fps -> "
        f"{play.get('slp')} ({play.get('slp_bytes')} bytes); error {play.get('error')}"
    )
    if play.get("slp_search"):
        lines.append(f"  .slp found elsewhere: {play['slp_search']}")
    for attempt in report["attempts"]:
        lines.append(
            f"  {attempt.get('name'):<18} backend {attempt.get('backend'):<18} "
            f"xvfb {attempt.get('xvfb')!s:<5} ok {attempt.get('ok')!s:<5} rc {attempt.get('returncode')} "
            f"{attempt.get('seconds')} s {attempt.get('width')}x{attempt.get('height')} "
            f"{attempt.get('duration_s')} s: {attempt.get('error')}"
        )
    for record in report["fallback"] or ():
        install = record.get("install", {})
        asset = record.get("asset", {})
        lines.append(
            f"fallback {asset.get('name')}: sha256 {install.get('sha256')} ({install.get('bytes')} "
            f"bytes), exe {install.get('exe')}, error {install.get('error')}"
        )
        if install.get("tree"):
            lines.append(f"  tree: {install['tree']}")
        if "help" in record:
            lines.append(f"  missing shared libraries: {record.get('missing_libs')}")
            lines.append(f"  comm-file option present: {record.get('has_comm_file_option')}")
            lines.append(f"  bundled ffmpeg libraries: {record.get('libs')}")
            lines.append(f"  --help:\n{record['help'].get('stdout', '')}{record['help'].get('stderr', '')}")
        for attempt in record.get("attempts", []):
            lines.append(
                f"  {attempt.get('name'):<18} ok {attempt.get('ok')!s:<5} rc {attempt.get('returncode')} "
                f"{attempt.get('width')}x{attempt.get('height')}: {attempt.get('error')}"
            )
            lines.append(f"    dump: {attempt.get('dump_tree')} rate: {attempt.get('rate')}")
    lines.append(
        f"video_check: winner {report['winner']!r} ({report.get('winner_exe')}), error {report['error']}"
    )
    return "\n".join(lines)


def run_record_video(
    config: str | Path,
    *,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    render: bool = True,
    force_baseline: bool = False,
    wandb: bool = False,
    wandb_mode: str = "online",
    device: str | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Record every matchup of ``config``'s ``[video]`` table for ``checkpoint`` (both stages); ``overrides``
    are ``table.key=value`` lines applied to the run file first (5 Sep 2026: ``actor.decision_stride=k``)."""

    from dataclasses import replace as dataclass_replace

    from melee_rl.checkpoint import load_checkpoint
    from melee_rl.config import load_config
    from melee_rl.video import format_summary, record_matches, wandb_upload

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    video = run_config.video
    video_overrides: dict[str, Any] = {}
    if clips is not None:
        video_overrides["clips"] = clips
    if seconds is not None:
        video_overrides["seconds"] = seconds
    if matchups is not None:
        video_overrides["matchups"] = split_tags(matchups)
    if video_overrides:
        run_config = dataclass_replace(run_config, video=dataclass_replace(video, **video_overrides))
    source = Path(checkpoint)
    if step is None:
        try:
            step = int(load_checkpoint(source)["step"])
        except Exception:  # a BC export has no step; the caption then omits it
            step = None
    if label is None and source.stem in ("latest", "best"):
        label = source.parent.name
    run = record_matches(
        run_config,
        checkpoint=source,
        output_dir=output_dir,
        bc=bc,
        step=step,
        label=label,
        # The run file's [runtime] device, unless the caller overrides it: every P9/P11 recorder says
        # "cpu" (batch 2 is latency-bound), and P12 says "cuda" because MIMIC has no KV cache.
        device=device or run_config.runtime.device,
        render=render,
        force_baseline=force_baseline,
        log=print,
    )
    summary = format_summary(run)
    print(summary)
    upload = None
    if wandb or run_config.video.wandb:
        upload = wandb_upload(run_config, run, mode=wandb_mode)
        print(f"wandb upload: {upload}")
    return {
        **run.record(),
        "summary": summary,
        "wandb": upload,
        "duration_s": time.perf_counter() - started,
    }


def run_external_match(
    config: str | Path,
    *,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P25: real four-stock matches between two outside agents (``sides`` = ``"smashbot,cpu9"``).

    Neither side is one of our checkpoints, which the recorder's ``<policy|bc>:<opponent>`` grammar
    cannot express -- these are the calibration arms that place SmashBot on the same ladder as every
    model in ``RESULTS.md`` before any of our weights are pointed at it.
    """

    from melee_rl.config import load_config
    from melee_rl.external_match import record_external_matches

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    result = record_external_matches(
        run_config,
        sides=split_tags(sides),
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        log=print,
    )
    return {**result, "duration_s": time.perf_counter() - started}


def external_route(names: Sequence[str]) -> tuple[str, str]:
    """``(Modal function name, planned-run name)`` for an external match between ``names``.

    Which image a pair of sides needs is a fact about what each image carries, so it lives in one place:

    * a ``slippi_ai`` side -> the slippi image (it has MIMIC and SmashBot too, P27);
    * a ``phillip`` side -> a Phillip image: the GPU one carries MIMIC, SmashBot and the sidecar, the CPU one
      only Dolphin and the sidecar, so **any** other outside agent sends the pair to the GPU image (a
      ``phillip,smashbot`` match on the CPU image would die on the missing SmashBot checkout);
    * otherwise P25's pair: the GPU image when MIMIC plays, the CPU image for ``smashbot`` against a CPU.

    No image carries both the slippi-ai runtime and the Phillip sidecar.
    """

    wants_slippi = any(name == "slippi_ai" or name.startswith("slippi_ai:") for name in names)
    wants_phillip = any(name == "phillip" or name.startswith("phillip:") for name in names)
    if wants_phillip and wants_slippi:
        raise ValueError("no image carries both the slippi-ai runtime and the Phillip sidecar")
    if wants_phillip:
        needs_gpu = "mimic" in names or "smashbot" in names
        name = "external_match_phillip" if needs_gpu else "external_match_phillip_cpu"
        return name, f"{name} {'phillip_mimic' if needs_gpu else 'phillip_cpu9'}"
    if wants_slippi:
        return "external_match_slippi", "external_match_slippi slippi_ai_cpu9"
    needs_gpu = "mimic" in names
    name = "external_match" if needs_gpu else "external_match_cpu"
    return name, f"{name} {'smashbot_mimic' if needs_gpu else 'smashbot_cpu9'}"


def run_phillip_check(
    config: str | Path, *, agents: str = "", overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """P32: :func:`melee_rl.phillip.preflight` for every agent in ``agents`` (comma separated; empty = the
    seven delay-0 agents), on the run file's ``[phillip]`` table.  The report per agent is what the gate
    reads:
    a clean restore (no variable initialised or padded), zero exceptions, and the sidecar's cost per
    decision.  Nothing boots a Dolphin."""

    from dataclasses import replace as dataclass_replace

    from melee_rl.config import load_config
    from melee_rl.phillip import DELAY0_AGENTS, format_preflight, preflight

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    names = split_tags(agents) or DELAY0_AGENTS
    reports: dict[str, Any] = {}
    for name in names:
        settings = dataclass_replace(run_config.phillip, agent=name)
        try:
            report = preflight(settings, frames=180, log=print)
        except Exception as error:  # one broken agent must not hide the others' reports
            report = {"agent": name, "error": f"{type(error).__name__}: {error}"}
            print(f"phillip {name}: FAILED {report['error']}")
        else:
            print(format_preflight(report))
        reports[name] = report
    ok = all(
        "error" not in report
        and not report["restore"]["missing"]
        and not report["restore"]["padded"]
        and report["exceptions"] == 0
        for report in reports.values()
    )
    print(f"phillip_check: {'OK' if ok else 'FAILED'} for {len(reports)} agent(s)")
    return {"ok": ok, "agents": reports, "duration_s": time.perf_counter() - started}


def run_slippi_check(
    config: str | Path, *, models: str = "", overrides: Sequence[str] = ()
) -> dict[str, Any]:
    """22 Sep 2026: :func:`melee_rl.slippi_ai_agent.preflight` for every release in ``models`` (comma
    separated; empty = the run file's ``[slippi_ai] model``), on the run file's ``[slippi_ai]`` table.
    The report per release is what the gate reads: the sha256 verified, zero exceptions over the
    synthetic frames, a controller that moved.  Nothing boots a Dolphin; the delay-0 JAX file was the
    reason (its pip layer, its restore and its host copy all had to be proven before a campaign)."""

    from dataclasses import replace as dataclass_replace

    from melee_rl.config import load_config
    from melee_rl.slippi_ai_agent import format_preflight, preflight

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    names = split_tags(models) or (run_config.slippi_ai.model,)
    reports: dict[str, Any] = {}
    for name in names:
        settings = dataclass_replace(run_config.slippi_ai, model=name)
        try:
            report = preflight(settings, frames=180, num_envs=2, log=print)
        except Exception as error:  # one broken release must not hide the others' reports
            report = {"model": name, "error": f"{type(error).__name__}: {error}"}
            print(f"slippi-ai {name}: FAILED {report['error']}")
        else:
            print(format_preflight(report))
        reports[name] = report
    ok = all(
        "error" not in report
        and report["sha256_verified"]
        and report["exceptions"] == 0
        and report["rows_changed"] > 0
        for report in reports.values()
    )
    print(f"slippi_check: {'OK' if ok else 'FAILED'} for {len(reports)} release(s)")
    return {"ok": ok, "models": reports, "duration_s": time.perf_counter() - started}


def run_slippi_parity_job(
    config: str | Path,
    *,
    checkpoint: str,
    output: str,
    cases: str = "",
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """``melee_rl.slippi_ai_parity.run_slippi_parity`` on a run file (``slippi_parity``): the cases named in
    ``cases`` (comma separated release names; empty = all of ``DEFAULT_CASES``); returns the report's verdict
    per case."""

    from melee_rl.config import load_config
    from melee_rl.slippi_ai_parity import run_slippi_parity, select_cases

    started = time.perf_counter()
    run_config = load_config(resolve_config(config), overrides)
    report = run_slippi_parity(
        run_config,
        checkpoint,
        select_cases(cases),
        device=run_config.runtime.device,
        output=output,
        log=print,
    )
    verdicts = {
        case["release"]: {
            "pass": case["pass"],
            "frames": case["frames"],
            "worst_max_abs_logit": max(item["max_abs_logit"] for item in case["components"].values()),
            "worst_max_rel_logit": max(item["max_rel_logit"] for item in case["components"].values()),
            "worst_mean_kl": max(item["mean_kl"] for item in case["components"].values()),
        }
        for case in report["cases"]
    }
    return {
        "pass": report["pass"],
        "cases": verdicts,
        "output": output,
        "duration_s": time.perf_counter() - started,
    }


def run_render_slp(
    directory: str | Path,
    *,
    config: str | Path | None = None,
    force: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    """Stage 2 alone: render every ``.slp`` under ``directory`` that has no mp4 beside it yet."""

    from melee_rl.config import load_config
    from melee_rl.render import RenderConfig, RenderJob, render_many
    from melee_rl.slp import summarize_replay

    started = time.perf_counter()
    render_config = RenderConfig() if config is None else load_config(resolve_config(config)).video.render
    root = Path(directory)
    jobs = []
    for replay in sorted(root.rglob("*.slp")):
        target = replay.with_suffix(".mp4")
        if target.is_file() and not force:
            continue
        # The replay's own length is what bounds the render: with frames = 0 the bound falls back to
        # min_timeout_s (120 s), which silently truncates anything longer than that renders in
        # (measured 28 Aug 2026: four 2-3.5 minute matches all cut to 82.7 s).
        try:
            frames = summarize_replay(replay).frames
        except (OSError, ValueError, IndexError, struct.error) as error:
            print(f"could not read {replay.name} ({error}); rendering it unbounded")
            frames = 0
        jobs.append(RenderJob(slp=replay, target=target, caption=replay.stem, frames=frames))
    results = render_many(
        render_config, jobs, work_dir=Path(tempfile.gettempdir()) / "melee-rl-render", workers=workers
    )
    return {
        "directory": str(root),
        "jobs": len(jobs),
        "rendered": sum(1 for result in results if result.ok),
        "results": [result.record() for result in results],
        "duration_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# the Modal environment
# ---------------------------------------------------------------------------


def active_environment() -> str:
    """The Modal environment this process would use (``-e`` / ``MODAL_ENVIRONMENT`` / profile)."""

    return str(modal.config.config.get("environment") or "")


def pin_environment() -> str:
    """Select :data:`MODAL_ENVIRONMENT` unless ``MODAL_ENVIRONMENT`` was set explicitly; return the
    active one."""

    if not os.environ.get("MODAL_ENVIRONMENT"):
        modal.config.config.override_locally("environment", MODAL_ENVIRONMENT)
    return active_environment()


def check_environment() -> None:
    """Refuse to run outside :data:`MODAL_ENVIRONMENT` (the shared workspace's other environments
    belong to other people, ENV.md §5.0)."""

    active = active_environment()
    if active != MODAL_ENVIRONMENT:
        raise RuntimeError(
            f"melee_rl.modal_app runs only in the Modal environment {MODAL_ENVIRONMENT!r}; the active one is "
            f"{active or '<workspace default>'!r}: pass -e {MODAL_ENVIRONMENT} (ENV.md §5.0)"
        )


pin_environment()


# ---------------------------------------------------------------------------
# Modal objects: app, images, volumes, secret, functions
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME, include_source=False)


def _make_base_image(torch_index: str) -> modal.Image:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


def _dolphin_layers(image: modal.Image) -> modal.Image:
    raise RuntimeError("Provide the runtime and third-party assets separately; automatic acquisition is absent from this release.")


def _with_package(image: modal.Image) -> modal.Image:
    # The frozen text-file allowlist prevents local games, weights or upstream
    # checkouts added later from travelling with the application source.
    manifest = json.loads((LOCAL_ROOT / "SOURCE_MANIFEST.json").read_text())
    for entry in manifest["runtime_files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid source manifest path")
        local = LOCAL_ROOT / relative
        if (local.is_symlink() or not local.resolve().is_relative_to(LOCAL_ROOT.resolve())
                or hashlib.sha256(local.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError(f"source identity changed: {relative}")
        image = image.add_local_file(local, remote_path=f"{REMOTE_ROOT}/{relative.as_posix()}")
    image = image.add_local_file(LOCAL_ROOT / "SOURCE_MANIFEST.json",
                                 remote_path=f"{REMOTE_ROOT}/SOURCE_MANIFEST.json")
    runtime_environment = {"PYTHONPATH": REMOTE_ROOT, "PYTHONUNBUFFERED": "1"}
    roles = ("CPU", "GPU", "MIMIC", "SLIPPI", "PHILLIP_CPU", "PHILLIP_GPU",
             "SMASHBOT_CPU", "SMASHBOT_GPU", "PLAYBACK")
    setting_names = ["FAYNT_RUNTIME_IMAGE", "FAYNT_MODAL_ENVIRONMENT", "FAYNT_RUNS_VOLUME",
                     "FAYNT_GAME_VOLUME", "FAYNT_WANDB_SECRET"]
    setting_names.extend(f"FAYNT_{role}_IMAGE" for role in roles)
    runtime_environment.update({name: os.environ[name] for name in setting_names if name in os.environ})
    return image.env(runtime_environment).workdir(REMOTE_ROOT)


def _provided_image(role: str) -> modal.Image:
    image_ref = os.environ.get(f"FAYNT_{role}_IMAGE") or os.environ.get("FAYNT_RUNTIME_IMAGE")
    if not image_ref:
        raise RuntimeError("Set FAYNT_RUNTIME_IMAGE to your separately prepared private runtime image.")
    if "@sha256:" not in image_ref:
        raise ValueError("Use a digest-pinned runtime image, with @sha256: in its reference.")
    return _with_package(modal.Image.from_registry(image_ref))


cpu_image = _provided_image("CPU")
dolphin_image = _provided_image("CPU")
dolphin_gpu_image = _provided_image("GPU")
dolphin_mimic_image = _provided_image("MIMIC")
dolphin_smashbot_image = _provided_image("SMASHBOT_CPU")
dolphin_smashbot_gpu_image = _provided_image("SMASHBOT_GPU")
dolphin_slippi_gpu_image = _provided_image("SLIPPI")
dolphin_phillip_image = _provided_image("PHILLIP_CPU")
dolphin_phillip_gpu_image = _provided_image("PHILLIP_GPU")
dolphin_video_image = _provided_image("PLAYBACK")

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, environment_name=MODAL_ENVIRONMENT)
iso_volume = modal.Volume.from_name(
    ISO_VOLUME_NAME, create_if_missing=False, environment_name=MODAL_ENVIRONMENT
)
wandb_secret = modal.Secret.from_name(
    WANDB_SECRET_NAME, environment_name=MODAL_ENVIRONMENT, required_keys=[WANDB_SECRET_KEY]
)


@app.function(
    image=dolphin_image,
    cpu=TESTS_RESOURCES.cpu,
    memory=TESTS_RESOURCES.memory_mib,
    timeout=TESTS_RESOURCES.timeout_s,
)
def run_tests(pytest_args: Sequence[str] = (), lint: bool = False) -> dict[str, Any]:
    """``python -m pytest tests -q -p no:cacheprovider [pytest_args]`` in the container (libmelee present, no
    git: :data:`GIT_DEPENDENT_TESTS` deselected)."""

    return run_test_suite(pytest_args, lint=lint, cwd=REMOTE_ROOT, in_container=True)


@app.function(
    image=dolphin_gpu_image,
    gpu=TESTS_GPU_RESOURCES.gpu,
    cpu=TESTS_GPU_RESOURCES.cpu,
    memory=TESTS_GPU_RESOURCES.memory_mib,
    timeout=TESTS_GPU_RESOURCES.timeout_s,
)
def run_tests_gpu(pytest_args: Sequence[str] = (), lint: bool = False) -> dict[str, Any]:
    """``run_tests`` on the cu128 image with an L4: the CUDA-gated tests run instead of being skipped."""

    return run_test_suite(pytest_args, lint=lint, cwd=REMOTE_ROOT, in_container=True)


def _remote_train(
    config: str,
    *,
    steps: int | None,
    wandb_mode: str | None,
    run_name: str | None,
    checkpoint_subdir: str | None,
    tags: Sequence[str],
    device: str | None,
    resume: bool,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Shared body of ``smoke_train`` / ``dolphin_train``: the launcher + a checkpoint directory on the
    Volume."""

    from melee_rl.train import default_run_name

    path = resolve_config(config)
    name = run_name or default_run_name(path)
    checkpoint_dir = volume_checkpoint_dir(name, checkpoint_subdir)
    os.makedirs(REMOTE_WANDB_DIR, exist_ok=True)
    hook = volume_commit_hook(runs_volume)  # every checkpoint write becomes durable at once
    result = run_smoke_train(
        path,
        steps=steps,
        wandb_mode=wandb_mode,
        run_name=name,
        checkpoint_dir=checkpoint_dir,
        tags=tags,
        device=device,
        resume=resume,
        on_save=hook,
        overrides=overrides,
    )
    runs_volume.commit()
    result["volume"] = {
        "name": VOLUME_NAME,
        "mount": VOLUME_MOUNT,
        "checkpoint_dir": str(checkpoint_dir),
        "commits": hook.summary(),
    }
    return result


@app.function(
    image=cpu_image,
    cpu=TRAIN_RESOURCES.cpu,
    memory=TRAIN_RESOURCES.memory_mib,
    timeout=TRAIN_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume},
    secrets=[wandb_secret],
)
def smoke_train(
    config: str,
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_subdir: str | None = None,
    tags: Sequence[str] = (),
    device: str | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """A dummy-env training run of ``configs/rl/<config>`` with checkpoints on the Volume."""

    return _remote_train(
        config,
        steps=steps,
        wandb_mode=wandb_mode,
        run_name=run_name,
        checkpoint_subdir=checkpoint_subdir,
        tags=tags,
        device=device,
        resume=resume,
        overrides=overrides,
    )


@app.function(
    image=dolphin_image,
    cpu=DOLPHIN_TRAIN_RESOURCES.cpu,
    memory=DOLPHIN_TRAIN_RESOURCES.memory_mib,
    timeout=DOLPHIN_TRAIN_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
    secrets=[wandb_secret],
)
def dolphin_train(
    config: str,
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_subdir: str | None = None,
    tags: Sequence[str] = (),
    device: str | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """A Dolphin-env training run of ``configs/rl/<config>`` (ISO Volume at ``/iso``, checkpoints on the
    Volume)."""

    return _remote_train(
        config,
        steps=steps,
        wandb_mode=wandb_mode,
        run_name=run_name,
        checkpoint_subdir=checkpoint_subdir,
        tags=tags,
        device=device,
        resume=resume,
        overrides=overrides,
    )


@app.function(
    image=dolphin_gpu_image,
    gpu=DOLPHIN_TRAIN_GPU_RESOURCES.gpu,
    cpu=DOLPHIN_TRAIN_GPU_RESOURCES.cpu,
    memory=DOLPHIN_TRAIN_GPU_RESOURCES.memory_mib,
    timeout=DOLPHIN_TRAIN_GPU_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
    secrets=[wandb_secret],
)
def dolphin_train_gpu(
    config: str,
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_subdir: str | None = None,
    tags: Sequence[str] = (),
    device: str | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """``dolphin_train`` on the GPU Dolphin image with an L4: the learner runs on ``cuda`` (P6a)."""

    return _remote_train(
        config,
        steps=steps,
        wandb_mode=wandb_mode,
        run_name=run_name,
        checkpoint_subdir=checkpoint_subdir,
        tags=tags,
        device="cuda" if device is None else device,
        resume=resume,
        overrides=overrides,
    )


@app.function(
    image=dolphin_image,
    cpu=DOLPHIN_CHECK_RESOURCES.cpu,
    memory=DOLPHIN_CHECK_RESOURCES.memory_mib,
    timeout=DOLPHIN_CHECK_RESOURCES.timeout_s,
    volumes={ISO_MOUNT: iso_volume},
)
def dolphin_check(
    build: str = DEFAULT_DOLPHIN_BUILD,
    frames: int = 600,
    idle_seconds: float = 60.0,
    num_envs: int = 1,
) -> dict[str, Any]:
    """The Dolphin boot smoke for one build of :data:`DOLPHIN_BUILDS` (see :func:`run_dolphin_check`)."""

    return run_dolphin_check(
        dolphin_build(build).exe, ISO_PATH, frames=frames, idle_seconds=idle_seconds, num_envs=num_envs
    )


@app.function(
    image=dolphin_gpu_image,
    gpu=ENV_BENCH_RESOURCES.gpu,
    cpu=ENV_BENCH_RESOURCES.cpu,
    memory=ENV_BENCH_RESOURCES.memory_mib,
    timeout=ENV_BENCH_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def env_bench(specs: Sequence[Mapping[str, Any]], output: str | None = None) -> list[dict[str, Any]]:
    """The Dolphin-only frame-step bench (lever 0): one :class:`melee_rl.env_bench.EnvBenchSpec` per entry
    of ``specs`` (its fields as a mapping), measured in turn on this box; the reports are written to
    ``/runs/<output>`` on the Volume when ``output`` is given, and returned."""

    from melee_rl.env_bench import EnvBenchSpec, run_env_bench_sweep

    built = [
        EnvBenchSpec(**{**spec, "players": tuple(spec.get("players", ("policy", "policy")))})
        for spec in specs
    ]
    reports = run_env_bench_sweep(built, log=lambda message: print(message, flush=True))
    if output is not None:
        path = Path(VOLUME_MOUNT) / output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=1))
        runs_volume.commit()
        print(f"wrote {path}")
    return reports


def train_function_for(env_type: str, *, gpu: bool = False, wide: bool = False, mimic: bool = False) -> Any:
    """The remote training function for a run file's ``[env] type`` (``gpu``: the L4 Dolphin variant;
    ``wide``: its 32-core / 64 GiB sibling for 64 Dolphins; ``mimic``: the wide variant on the MIMIC
    image, for a run whose evaluation opponents include ``mimic``)."""

    if wide and not gpu:
        raise ValueError("--wide is the 32-core / 64 GiB variant of the gpu function: pass --gpu as well")
    if mimic:
        if env_type != "dolphin" or not (gpu and wide):
            raise ValueError(
                "a mimic evaluation runs on the wide GPU Dolphin box (dolphin_train_gpu_wide_mimic): "
                "pass --gpu --wide with a dolphin run file"
            )
        return dolphin_train_gpu_wide_mimic
    if env_type == "dummy":
        if wide:
            raise ValueError(
                "env.type 'dummy' runs on the plain gpu box (dolphin_train_gpu: no Dolphins to widen for); "
                "--wide is for Dolphin run files"
            )
        # 5 Sep 2026 (/rl-speedup): --gpu takes a dummy-env run file to the L4 box (no ISO is read).
        return dolphin_train_gpu if gpu else smoke_train
    if env_type == "dolphin":
        if gpu:
            return dolphin_train_gpu_wide if wide else dolphin_train_gpu
        return dolphin_train
    raise ValueError(f"env.type {env_type!r} has no Modal training function (dummy | dolphin)")


@app.function(
    image=dolphin_gpu_image,
    gpu=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.gpu,
    cpu=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.cpu,
    memory=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.memory_mib,
    timeout=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
    secrets=[wandb_secret],
)
def dolphin_train_gpu_wide(
    config: str,
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_subdir: str | None = None,
    tags: Sequence[str] = (),
    device: str | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """``dolphin_train_gpu`` with 32 cores / 64 GiB next to the L4 (P6b): 64 Dolphins + the eval ones."""

    return _remote_train(
        config,
        steps=steps,
        wandb_mode=wandb_mode,
        run_name=run_name,
        checkpoint_subdir=checkpoint_subdir,
        tags=tags,
        device="cuda" if device is None else device,
        resume=resume,
        overrides=overrides,
    )


@app.function(
    image=dolphin_mimic_image,
    gpu=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.gpu,
    cpu=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.cpu,
    memory=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.memory_mib,
    timeout=DOLPHIN_TRAIN_GPU_WIDE_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
    secrets=[wandb_secret],
)
def dolphin_train_gpu_wide_mimic(
    config: str,
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_subdir: str | None = None,
    tags: Sequence[str] = (),
    device: str | None = None,
    resume: bool = True,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """``dolphin_train_gpu_wide`` on the MIMIC image (2 Sep 2026): a run whose ``[eval] opponents`` name
    ``mimic`` needs the released checkout and ``fox-master`` bundle next to the learner."""

    return _remote_train(
        config,
        steps=steps,
        wandb_mode=wandb_mode,
        run_name=run_name,
        checkpoint_subdir=checkpoint_subdir,
        tags=tags,
        device="cuda" if device is None else device,
        resume=resume,
        overrides=overrides,
    )


@app.function(
    image=dolphin_video_image,
    cpu=VIDEO_RESOURCES.cpu,
    memory=VIDEO_RESOURCES.memory_mib,
    timeout=VIDEO_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
    secrets=[wandb_secret],
)
def record_video(
    config: str,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    render: bool = True,
    force_baseline: bool = False,
    wandb: bool = False,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P9: play every matchup of ``configs/rl/<config>.toml`` for ``checkpoint`` and render the clips.

    The Volume is reloaded first so a checkpoint a *live* training run has just written is visible, and
    committed afterwards so the clips survive the container."""

    runs_volume.reload()
    result = run_record_video(
        config,
        checkpoint=checkpoint,
        bc=bc,
        step=step,
        output_dir=output_dir,
        label=label,
        clips=clips,
        seconds=seconds,
        matchups=matchups,
        render=render,
        force_baseline=force_baseline,
        wandb=wandb,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_mimic_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def record_mimic(
    config: str,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    force_baseline: bool = False,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P12: the same recorder against the released MIMIC policy, on the GPU image that carries it.

    Recording only (``render = False``): the ``.slp`` files are the deliverable and rendering needs the
    playback build of the *video* image."""

    runs_volume.reload()
    result = run_record_video(
        config,
        checkpoint=checkpoint,
        bc=bc,
        step=step,
        output_dir=output_dir,
        label=label,
        clips=clips,
        seconds=seconds,
        matchups=matchups,
        render=False,
        force_baseline=force_baseline,
        wandb=False,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_smashbot_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def record_smashbot(
    config: str,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    force_baseline: bool = False,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P25: the recorder against altf4/SmashBot -- ``[video] matchups = ["policy:smashbot"]``.

    On the image that carries both outside agents, so a job may mix ``policy:smashbot`` with
    ``policy:mimic`` in one run file.  Recording only (``render = False``): the ``.slp`` files are the
    deliverable and rendering needs the playback build of the *video* image.
    """

    runs_volume.reload()
    result = run_record_video(
        config,
        checkpoint=checkpoint,
        bc=bc,
        step=step,
        output_dir=output_dir,
        label=label,
        clips=clips,
        seconds=seconds,
        matchups=matchups,
        render=False,
        force_baseline=force_baseline,
        wandb=False,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_smashbot_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def external_match(
    config: str,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P25 calibration: two outside agents against each other, on the GPU image (MIMIC needs it)."""

    runs_volume.reload()
    result = run_external_match(
        config,
        sides=sides,
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_slippi_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def record_slippi_ai(
    config: str,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    force_baseline: bool = False,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P27: the recorder against a released slippi-ai agent -- ``[video] matchups = ["policy:slippi_ai"]``.

    Which release plays comes from ``[slippi_ai] model`` (or ``--set slippi_ai.model=<name>``), so one
    run file covers every arm of the campaign.  Recording only (``render = False``): the ``.slp`` files
    are the deliverable and rendering needs the playback build of the *video* image.
    """

    runs_volume.reload()
    result = run_record_video(
        config,
        checkpoint=checkpoint,
        bc=bc,
        step=step,
        output_dir=output_dir,
        label=label,
        clips=clips,
        seconds=seconds,
        matchups=matchups,
        render=False,
        force_baseline=force_baseline,
        wandb=False,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_slippi_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def slippi_parity(
    config: str,
    checkpoint: str,
    output: str,
    cases: str = "",
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """The parity gate of the slippi-ai port (17 Sep 2026, LEAGUE-PLAN.md §3.4): our ``checkpoint`` against
    each case's TensorFlow agent, the PyTorch port shadowing it; the JSON report lands at ``output`` on the
    Volume."""

    runs_volume.reload()
    result = run_slippi_parity_job(
        config, checkpoint=checkpoint, output=output, cases=cases, overrides=overrides
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_slippi_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def external_match_slippi(
    config: str,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P27 calibration: a slippi-ai release against MIMIC, SmashBot or a Dolphin CPU.

    The same job as :func:`external_match` on the image that also carries the slippi-ai runtime.  These
    arms are what place the new rung on the existing ladder -- and what catch a broken integration: a
    release that cannot beat CPU 9 is not a measurement of anything.
    """

    runs_volume.reload()
    result = run_external_match(
        config,
        sides=sides,
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_smashbot_image,
    cpu=SMASHBOT_RESOURCES.cpu,
    memory=SMASHBOT_RESOURCES.memory_mib,
    timeout=SMASHBOT_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def external_match_cpu(
    config: str,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """The same job with no GPU and no MIMIC layer: the ``smashbot vs cpu<N>`` delay sweep.

    Three of the four calibration arms carry no neural network at all, so paying for an L4 on them
    would be about a third of the campaign's budget for nothing.
    """

    runs_volume.reload()
    result = run_external_match(
        config,
        sides=sides,
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_phillip_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def record_phillip(
    config: str,
    checkpoint: str,
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    force_baseline: bool = False,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P32: the recorder against a vladfi1/phillip agent -- ``[video] matchups = ["policy:phillip"]``.

    On the GPU image that carries MIMIC, SmashBot and the Phillip sidecar interpreter.  Recording only
    (``render = False``): the ``.slp`` files are the deliverable.
    """

    runs_volume.reload()
    result = run_record_video(
        config,
        checkpoint=checkpoint,
        bc=bc,
        step=step,
        output_dir=output_dir,
        label=label,
        clips=clips,
        seconds=seconds,
        matchups=matchups,
        render=False,
        force_baseline=force_baseline,
        wandb=False,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_phillip_gpu_image,
    gpu=MIMIC_RESOURCES.gpu,
    cpu=MIMIC_RESOURCES.cpu,
    memory=MIMIC_RESOURCES.memory_mib,
    timeout=MIMIC_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def external_match_phillip(
    config: str,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P32 calibration: a Phillip agent against MIMIC (or SmashBot), on the GPU image carrying all three."""

    runs_volume.reload()
    result = run_external_match(
        config,
        sides=sides,
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_phillip_image,
    cpu=PHILLIP_RESOURCES.cpu,
    memory=PHILLIP_RESOURCES.memory_mib,
    timeout=PHILLIP_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def external_match_phillip_cpu(
    config: str,
    sides: str,
    output_dir: str,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """P32 calibration: a Phillip agent against Dolphin's CPU, with no GPU and no MIMIC layer."""

    runs_volume.reload()
    result = run_external_match(
        config,
        sides=sides,
        output_dir=output_dir,
        label=label,
        matches=matches,
        seconds=seconds,
        overrides=overrides,
    )
    runs_volume.commit()
    return result


@app.function(
    image=dolphin_phillip_image,
    cpu=PHILLIP_CHECK_RESOURCES.cpu,
    memory=PHILLIP_CHECK_RESOURCES.memory_mib,
    timeout=PHILLIP_CHECK_RESOURCES.timeout_s,
)
def phillip_preflight(config: str, agents: str = "", overrides: Sequence[str] = ()) -> dict[str, Any]:
    """P32: the sidecar preflight of every named agent (restore report, decision timing); no Dolphin."""

    return run_phillip_check(config, agents=agents, overrides=overrides)


@app.function(
    image=dolphin_slippi_gpu_image,
    cpu=SLIPPI_CHECK_RESOURCES.cpu,
    memory=SLIPPI_CHECK_RESOURCES.memory_mib,
    timeout=SLIPPI_CHECK_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume},
)
def slippi_preflight(config: str, models: str = "", overrides: Sequence[str] = ()) -> dict[str, Any]:
    """22 Sep 2026: every named release built and stepped on synthetic frames; no Dolphin, no GPU."""

    runs_volume.reload()
    return run_slippi_check(config, models=models, overrides=overrides)


@app.function(
    image=dolphin_video_image,
    cpu=VIDEO_RESOURCES.cpu,
    memory=VIDEO_RESOURCES.memory_mib,
    timeout=VIDEO_RESOURCES.timeout_s,
    volumes={VOLUME_MOUNT: runs_volume, ISO_MOUNT: iso_volume},
)
def render_slp(
    directory: str, config: str | None = None, force: bool = False, workers: int | None = None
) -> dict[str, Any]:
    """P9 stage 2 alone: render every ``.slp`` under a Volume directory that has no mp4 yet."""

    runs_volume.reload()
    result = run_render_slp(directory, config=config, force=force, workers=workers)
    runs_volume.commit()
    return result


def _render_bench_body(seconds: float, worker_counts: str, render_seconds: float) -> dict[str, Any]:
    counts = [int(part) for part in split_tags(worker_counts)] or [1]
    report = run_render_bench(
        dolphin_build(DEFAULT_DOLPHIN_BUILD).exe,
        ISO_PATH,
        seconds=seconds,
        worker_counts=counts,
        render_seconds=render_seconds,
    )
    print(format_render_bench(report))
    return report


@app.function(
    image=dolphin_video_image,
    cpu=RENDER_BENCH_RESOURCES.cpu,
    memory=RENDER_BENCH_RESOURCES.memory_mib,
    timeout=RENDER_BENCH_RESOURCES.timeout_s,
    volumes={ISO_MOUNT: iso_volume},
)
def render_bench(
    seconds: float = 20.0, worker_counts: str = "1,4", render_seconds: float = 120.0
) -> dict[str, Any]:
    """The render rate and its parallel scaling on the CPU video box (see :func:`run_render_bench`)."""

    return _render_bench_body(seconds, worker_counts, render_seconds)


@app.function(
    image=dolphin_video_image,
    gpu=RENDER_BENCH_GPU_RESOURCES.gpu,
    cpu=RENDER_BENCH_GPU_RESOURCES.cpu,
    memory=RENDER_BENCH_GPU_RESOURCES.memory_mib,
    timeout=RENDER_BENCH_GPU_RESOURCES.timeout_s,
    volumes={ISO_MOUNT: iso_volume},
)
def render_bench_gpu(
    seconds: float = 20.0, worker_counts: str = "1,4", render_seconds: float = 120.0
) -> dict[str, Any]:
    """The same bench with an L4 attached: does hardware GL engage under Xvfb, and does it pay?"""

    return _render_bench_body(seconds, worker_counts, render_seconds)


@app.function(
    image=dolphin_video_image,
    cpu=VIDEO_CHECK_RESOURCES.cpu,
    memory=VIDEO_CHECK_RESOURCES.memory_mib,
    timeout=VIDEO_CHECK_RESOURCES.timeout_s,
    volumes={ISO_MOUNT: iso_volume},
)
def video_probe(
    build: str = DEFAULT_DOLPHIN_BUILD,
    seconds: float = 20.0,
    download_fallback: bool = False,
    render_timeout_s: float = 120.0,
    candidates: str | None = None,
    builds: str | None = None,
    skip_primary: bool = False,
) -> dict[str, Any]:
    """P9 §6: does any video backend render a replay in this container?  (see :func:`run_video_check`)"""

    report = run_video_check(
        dolphin_build(build).exe,
        ISO_PATH,
        seconds=seconds,
        download_fallback=download_fallback,
        render_timeout_s=render_timeout_s,
        candidates=candidates,
        builds=builds,
        skip_primary=skip_primary,
    )
    print(format_video_check(report))  # the findings land in the Modal logs, client or no client
    return report


# ---------------------------------------------------------------------------
# local entrypoints (modal run -e smash_melee_bot -m melee_rl.modal_app::<name>)
# ---------------------------------------------------------------------------


def _outcome(label: str, resources: Resources, remote_s: float, wall_s: float) -> str:
    return (
        f"{label}: {remote_s:.0f} s in the function, {wall_s:.0f} s wall -> measured cost about "
        f"{format_usd(cost_estimate(resources, wall_s))} ({resources.describe()}, {PRICES_DATE} prices)"
    )


@app.local_entrypoint()
def tests(k: str | None = None, lint: bool = False, gpu: bool = False) -> str:
    """Run the suite on Modal; ``--k EXPR`` narrows it, ``--lint`` adds ruff + mypy, ``--gpu`` takes the L4
    box (the CUDA-gated tests)."""

    check_environment()
    kind = "run_tests_gpu" if gpu else "run_tests"
    plan = _planned(kind)
    assert plan is not None
    print(describe_plan(plan.name, plan.resources, plan.seconds))
    print(manifest_summary(mount_manifest()))
    pytest_args: tuple[str, ...] = () if k is None else ("-k", k)
    started = time.perf_counter()
    function = run_tests_gpu if gpu else run_tests
    report = function.remote(pytest_args, lint)
    wall = time.perf_counter() - started
    remote_s = report["pytest"]["duration_s"]
    if report["lint"] is not None:
        remote_s += report["lint"]["ruff"]["duration_s"] + report["lint"]["mypy"]["duration_s"]
    print(_outcome(kind, plan.resources, remote_s, wall))
    status = "ok" if report["ok"] else "FAILED"
    last_line = report["pytest"]["output"].rstrip().splitlines()[-1:] or [""]
    print(f"{kind}: {status} (pytest exit {report['pytest']['returncode']}: {last_line[0]})")
    if not report["ok"]:
        raise SystemExit(f"{kind}: FAILED (see the output above)")
    return json.dumps(report, indent=1)


@app.local_entrypoint()
def train(
    config: str = "smoke_3m",
    steps: int | None = None,
    wandb_mode: str | None = None,
    run_name: str | None = None,
    checkpoint_subdir: str | None = None,
    tags: str | None = None,
    device: str | None = None,
    resume: bool = True,
    gpu: bool = False,
    wide: bool = False,
    spawn: bool = False,
    overrides: str = "",
) -> str:
    """Train ``configs/rl/<config>.toml`` on Modal (dummy or Dolphin env by ``[env] type``; ``--gpu`` = the L4
    Dolphin variant, ``--device cuda`` unless given; ``--gpu --wide`` = the same with 32 cores / 64 GiB for
    64 Dolphins); checkpoints on the Volume.

    ``--spawn`` starts the function and returns at once (no result JSON; the run is followed through
    ``modal app logs`` / W&B / the Volume) -- combine with ``modal run --detach`` so the app outlives this
    client (multi-day runs, P6b)."""

    check_environment()
    path = resolve_config(config)
    if path.parent != CONFIG_DIR:
        raise SystemExit(f"{path} is outside {CONFIG_DIR}: only configs/rl/*.toml travel with the image")
    env_type = config_env_type(path)
    mimic = config_needs_mimic(path)  # the run file decides: a mimic eval opponent needs the MIMIC image
    try:
        function = train_function_for(env_type, gpu=gpu, wide=wide, mimic=mimic)
        plan = plan_for_config(path, gpu=gpu, wide=wide, mimic=mimic)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    resource_changes = resource_overrides(plan)
    if resource_changes:  # the planned row asks for more than the function's decorator (the 75m's 96 GiB)
        function = function.with_options(**resource_changes)
        print(f"{plan.name}: Function.with_options({resource_changes}) -> {plan.resources.describe()}")
    print(describe_plan(plan.name, plan.resources, plan.seconds))
    where = volume_checkpoint_dir("<run name>", checkpoint_subdir)
    print(
        f"run file {path.name} ([env] type {env_type}), steps {'TOML' if steps is None else steps}, W&B mode "
        f"{wandb_mode or 'TOML/env'}, checkpoints on {VOLUME_NAME}:{where}, resume {resume}"
    )
    if env_type == "dolphin":
        flavour = "Dolphin image"
        if gpu:
            size = (
                f", {plan.resources.cpu:g} cores / {plan.resources.memory_mib / 1024:g} GiB)" if wide else ")"
            )
            flavour = "GPU Dolphin image (torch cu128, L4, --device cuda" + size
            if mimic:
                flavour += f" + the MIMIC checkout and bundle at {MIMIC_ROOT} (mimic eval opponent)"
        print(
            f"{flavour} ({DEFAULT_DOLPHIN_BUILD} default build); ISO Volume {ISO_VOLUME_NAME} at {ISO_MOUNT}"
        )
    elif gpu:
        print(
            "GPU Dolphin image (torch cu128, L4, --device cuda) for a dummy-env run: no Dolphins, the policy "
            "and learner paths alone (/rl-speedup bench)"
        )
    arguments: dict[str, Any] = {
        "steps": steps,
        "wandb_mode": wandb_mode,
        "run_name": run_name,
        "checkpoint_subdir": checkpoint_subdir,
        "tags": split_tags(tags),
        "device": device,
        "resume": resume,
        "overrides": split_overrides(overrides),  # comma-separated table.key=value pairs (5 Sep 2026)
    }
    if overrides:
        print(f"run-file overrides: {split_overrides(overrides)}")
    if spawn:
        call = function.spawn(path.name, **arguments)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned {plan.name}: function call {call.object_id} in app {app_id}; with `modal run --detach` "
            f"the run continues without this client. Follow it: modal app logs {app_id} "
            f"-e {MODAL_ENVIRONMENT}; stop it: modal app stop {app_id} -e {MODAL_ENVIRONMENT}; "
            f"checkpoints on {VOLUME_NAME}:{where}"
        )
        return json.dumps(
            {"spawned": True, "function_call_id": call.object_id, "app_id": app_id, "config": path.name},
            indent=1,
        )
    started = time.perf_counter()
    result = function.remote(path.name, **arguments)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, result["duration_s"]["total"], wall))
    print(
        f"run {result['run_name']} (W&B id {result['wandb_run_id']}, mode {result['wandb_mode']}): "
        f"{result['steps_run']} steps run, now at step {result['step']}, {result['frames_seen']} frames "
        f"seen; checkpoint {result['checkpoint']}"
    )
    if result["summary"] is not None:
        print(result["summary"])
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def bench_env(
    sizes: str = "96",
    workers: str = "16",
    chunks: str = "1",
    frames: int = 600,
    warmup: int = 60,
    players: str = "self",
    controllers: str = "random",
    hold: int = 1,
    seed: int = 0,
    stages: str = "25,24,18,26,8,6",
    output: str | None = None,
    shm: str = "0",
    spawn: bool = False,
) -> str:
    """The Dolphin-only frame-step bench on the wide L4 box (lever 0, 5 Sep 2026): ``--sizes 24,48,72,96``
    measures one pool per size (``--workers`` / ``--chunks`` one value or one per size; a chunk k > 1 is the
    chunked wire with the controller held k frames), ``--players self|cpu`` the arm layout; ``--output
    p20/env_bench/x.json`` keeps the reports on the Volume; ``--spawn`` returns at once (use ``--detach``)."""

    from melee_rl.env_bench import PLAYER_LAYOUTS, parse_specs

    check_environment()
    plan = _planned("env_bench sweep")
    assert plan is not None
    for layout in players.split(","):
        if layout not in PLAYER_LAYOUTS:
            raise SystemExit(f"--players entries must be one of {sorted(PLAYER_LAYOUTS)}, got {layout!r}")
    argv = [
        "--sizes",
        sizes,
        "--workers",
        workers,
        "--chunks",
        chunks,
        "--frames",
        str(frames),
        "--warmup",
        str(warmup),
        "--players",
        players,
        "--controllers",
        controllers,
        "--hold",
        str(hold),
        "--seed",
        str(seed),
        "--stages",
        stages,
    ]
    argv.extend(["--shm", shm])
    specs, _args = parse_specs(argv)
    payload = [asdict(spec) for spec in specs]
    print(describe_plan(plan.name, plan.resources, plan.seconds))
    print(f"{len(specs)} measurement(s): " + ", ".join(spec.name for spec in specs) + f"; output {output}")
    if spawn:
        call = env_bench.spawn(payload, output)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned env_bench: function call {call.object_id} in app {app_id}; follow it with "
            f"modal app logs {app_id} -e {MODAL_ENVIRONMENT}; the reports land on {VOLUME_NAME}:{output}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id}, indent=1)
    started = time.perf_counter()
    reports = env_bench.remote(payload, output)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, sum(report["duration_s"] for report in reports), wall))
    return json.dumps(reports, indent=1)


@app.local_entrypoint()
def dolphin(
    build: str = DEFAULT_DOLPHIN_BUILD,
    frames: int = 600,
    idle_seconds: float = 60.0,
    num_envs: int = 1,
) -> str:
    """The Dolphin boot smoke on Modal: ``--version``, boot seconds, fps, stages, idle survival."""

    check_environment()
    chosen = dolphin_build(build)
    plan = _planned("dolphin_check")
    assert plan is not None
    print(describe_plan(f"dolphin_check {chosen.name}", plan.resources, plan.seconds))
    print(f"build {chosen.name}: {chosen.exe} ({chosen.note}); ISO Volume {ISO_VOLUME_NAME} at {ISO_PATH}")
    print(f"{frames} neutral frames in {num_envs} env(s), then {idle_seconds:g} s idle and one more step")
    started = time.perf_counter()
    report = dolphin_check.remote(chosen.name, frames, idle_seconds, num_envs)
    wall = time.perf_counter() - started
    print(_outcome("dolphin_check", plan.resources, report["duration_s"], wall))
    probe = report["version_probe"]
    print(
        f"--version: rc {probe.get('returncode')!r} stdout {probe.get('stdout', '')!r} "
        f"stderr {probe.get('stderr', '')!r}"
    )
    if report["ok"]:
        print(
            f"build {report['build']}; boot {report['boot_s']:.1f} s (launches {report['launch_seconds']}); "
            f"{report['frames_stepped']} frames in {report['step_s']:.1f} s = "
            f"{report['fps_per_env']:.1f} fps/env "
            f"({report['fps_total']:.1f} total); stages {report['stages_seen']}, characters "
            f"{report['characters_seen']}, resets {report['resets_seen']}, "
            f"frames {report['frame_index_range']}, "
            f"max items {report['max_items']}, max percent {report['max_percent']}, "
            f"x ranges {report['x_range']}, cpu levels {report['cpu_levels']}; idle_ok {report['idle_ok']} "
            f"(frame delta {report['idle_frame_delta']}, step {report['idle_step_s']:.2f} s)"
        )
    else:
        print(f"dolphin_check FAILED: {report['error']}\n{report.get('traceback', '')}")
    for row in report.get("stats", report.get("stats_after_steps", [])):
        print(f"env {row['index']}: {row}")
    if not report["ok"]:
        raise SystemExit("dolphin_check: FAILED (see the output above)")
    return json.dumps(report, indent=1)


@app.local_entrypoint()
def video(
    config: str = "video_20m_bc",
    checkpoint: str = "/runs/selfplay/20m_bc/latest.pt",
    bc: str | None = None,
    step: int | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    clips: int | None = None,
    seconds: float | None = None,
    matchups: str | None = None,
    render: bool = True,
    force_baseline: bool = False,
    wandb: bool = False,
    spawn: bool = False,
    mimic: bool = False,
    smashbot: bool = False,
    slippi_ai: bool = False,
    phillip: bool = False,
    overrides: str = "",
) -> str:
    """P9: record clips of ``--checkpoint`` playing every matchup of ``configs/rl/<config>.toml``.

    ``--clips`` / ``--seconds`` / ``--matchups`` (comma separated) override the run file for a
    shake-out; ``--no-render`` stops after the ``.slp`` files; ``--force-baseline`` re-records the BC
    clips even when their sha256-keyed directory already has them.  ``--spawn`` starts the job and
    returns at once: a synchronous ``.remote()`` is **cancelled** when its client goes away even under
    ``modal run --detach`` (26 Aug 2026: an 11-minute bench was lost to exactly that), and a recording
    job outlives any attached client.  The summary is printed inside the container either way, so
    ``modal app logs`` has it."""

    check_environment()
    path = resolve_config(config)
    if path.parent != CONFIG_DIR:
        raise SystemExit(f"{path} is outside {CONFIG_DIR}: only configs/rl/*.toml travel with the image")
    if sum((mimic, smashbot, slippi_ai, phillip)) > 1:
        raise SystemExit("--mimic, --smashbot, --slippi-ai and --phillip are different images; pass one")
    # --mimic runs the GPU image that carries the released MIMIC policy; --smashbot the one that carries
    # both outside agents (P25).  Both record only: rendering needs the playback build of the video image.
    function: Any
    if phillip:
        # P32: the GPU image that also carries the Phillip checkout and its sidecar interpreter.
        function, function_name = record_phillip, "record_phillip"
        plan = _planned("record_phillip video_phillip")
    elif slippi_ai:
        # P27: the image that also carries the slippi-ai runtime and TensorFlow.
        function, function_name = record_slippi_ai, "record_slippi_ai"
        plan = _planned("record_slippi_ai video_slippi_ai")
    elif smashbot:
        function, function_name = record_smashbot, "record_smashbot"
        plan = _planned("record_smashbot video_smashbot")
    elif mimic:
        function, function_name = record_mimic, "record_mimic"
        plan = _planned("record_mimic video_mimic")
    else:
        function, function_name = record_video, "record_video"
        plan = _planned("record_video video_20m_bc")
    assert plan is not None
    print(describe_plan(f"{function_name} {path.stem}", plan.resources, plan.seconds))
    print(
        f"run file {path.name}, checkpoint {checkpoint}, bc {bc or 'from the run file'}, "
        f"render {render}, force_baseline {force_baseline}, wandb {wandb}; clips land on {VOLUME_NAME}"
    )
    arguments: tuple[Any, ...] = (
        path.name,
        checkpoint,
        bc,
        step,
        output_dir,
        label,
        clips,
        seconds,
        matchups,
        force_baseline,
    )
    if not (mimic or smashbot or slippi_ai or phillip):
        arguments = (*arguments[:9], render, force_baseline, wandb)
    # 5 Sep 2026: run-file overrides for the recorder (lever 8 check (ii): actor.decision_stride=k).
    arguments = (*arguments, split_overrides(overrides))
    if overrides:
        print(f"run-file overrides: {split_overrides(overrides)}")
    if spawn:
        call = function.spawn(*arguments)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned {function_name}: function call {call.object_id} in app {app_id}; "
            f"read it with modal app logs {app_id} -e {MODAL_ENVIRONMENT}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id})
    started = time.perf_counter()
    result = function.remote(*arguments)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, result["duration_s"], wall))
    print(result["summary"])
    print(f"fetch them: modal volume get {VOLUME_NAME} {result['directory']} -e {MODAL_ENVIRONMENT} .")
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def parity(
    config: str = "slippi_parity",
    checkpoint: str = "/runs/selfplay/10m_delay/d18_ema1/step_00000760.pt",
    output: str = "/runs/parity/league-port.json",
    cases: str = "",
    spawn: bool = False,
    overrides: str = "",
) -> str:
    """The parity gate of the slippi-ai port (17 Sep 2026, LEAGUE-PLAN.md §3.4): ``--checkpoint`` (our fork,
    at its own trained delay) against each case's TensorFlow agent with the PyTorch port shadowing it
    teacher-forced; ``--cases gm,fox_d18_ditto_v4`` narrows the default three; the report is JSON at
    ``--output`` on the Volume.  Pass: |Δ logit| <= 1e-3 * max(1, |logit|) and mean KL <= 1e-6 per component.
    ``--spawn`` as always (GOTCHAS)."""

    check_environment()
    path = resolve_config(config)
    if path.parent != CONFIG_DIR:
        raise SystemExit(f"{path} is outside {CONFIG_DIR}: only configs/rl/*.toml travel with the image")
    from melee_rl.slippi_ai_parity import select_cases

    selected = select_cases(cases)  # validates the names before anything is billed
    plan = _planned("slippi_parity slippi_parity")
    assert plan is not None
    print(describe_plan(f"slippi_parity {path.stem}", plan.resources, plan.seconds))
    print(
        f"run file {path.name}, checkpoint {checkpoint}, cases {[case.release for case in selected]}, "
        f"output {output}"
    )
    arguments: tuple[Any, ...] = (path.name, checkpoint, output, cases, split_overrides(overrides))
    if spawn:
        call = slippi_parity.spawn(*arguments)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned slippi_parity: function call {call.object_id} in app {app_id}; "
            f"read it with modal app logs {app_id} -e {MODAL_ENVIRONMENT}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id})
    started = time.perf_counter()
    result = slippi_parity.remote(*arguments)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, result["duration_s"], wall))
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def phillip_check(
    config: str = "phillip_cpu9", agents: str = "", spawn: bool = False, overrides: str = ""
) -> str:
    """P32: preflight the Phillip sidecar for ``--agents a,b,c`` (default: the seven delay-0 agents) on the
    CPU Phillip image -- a clean restore of every agent and its cost per decision, before any Dolphin is
    paid for.
    ``--spawn`` as always (GOTCHAS)."""

    check_environment()
    path = resolve_config(config)
    if path.parent != CONFIG_DIR:
        raise SystemExit(f"{path} is outside {CONFIG_DIR}: only configs/rl/*.toml travel with the image")
    plan = _planned("phillip_check phillip_cpu9")
    assert plan is not None
    print(describe_plan(f"phillip_check {path.stem}", plan.resources, plan.seconds))
    print(f"run file {path.name}, agents {split_tags(agents) or 'the seven delay-0 agents'}")
    arguments: tuple[Any, ...] = (path.name, agents, split_overrides(overrides))
    if spawn:
        call = phillip_preflight.spawn(*arguments)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned phillip_check: function call {call.object_id} in app {app_id}; "
            f"read it with modal app logs {app_id} -e {MODAL_ENVIRONMENT}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id})
    started = time.perf_counter()
    result = phillip_preflight.remote(*arguments)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, result["duration_s"], wall))
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def slippi_check(
    config: str = "slippi_ai_cpu9", models: str = "", spawn: bool = False, overrides: str = ""
) -> str:
    """22 Sep 2026: preflight ``--models a,b,c`` (default: the run file's ``[slippi_ai] model``) on the
    slippi-ai image -- the pip layer, the checkout, the sha256, the restore and 180 synthetic frames of
    the batched agent, before any Dolphin is paid for.  ``--spawn`` as always (GOTCHAS)."""

    check_environment()
    path = resolve_config(config)
    if path.parent != CONFIG_DIR:
        raise SystemExit(f"{path} is outside {CONFIG_DIR}: only configs/rl/*.toml travel with the image")
    plan = _planned("slippi_check slippi_ai_cpu9")
    assert plan is not None
    print(describe_plan(f"slippi_check {path.stem}", plan.resources, plan.seconds))
    chosen: Any = split_tags(models) or "the run file's [slippi_ai] model"
    print(f"run file {path.name}, models {chosen}")
    arguments: tuple[Any, ...] = (path.name, models, split_overrides(overrides))
    if spawn:
        call = slippi_preflight.spawn(*arguments)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned slippi_check: function call {call.object_id} in app {app_id}; "
            f"read it with modal app logs {app_id} -e {MODAL_ENVIRONMENT}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id})
    started = time.perf_counter()
    result = slippi_preflight.remote(*arguments)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, result["duration_s"], wall))
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def external(
    config: str = "smashbot_cpu9",
    sides: str = "smashbot,cpu9",
    output_dir: str = "/runs/matches/p25-calibration",
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    spawn: bool = False,
    overrides: str = "",
) -> str:
    """P25: four-stock matches between two agents that are neither of them ours.

    ``--sides`` is two of ``smashbot`` / ``mimic`` / ``slippi_ai[:<release>]`` / ``cpu<1-9>``, in port
    order: the first plays Dolphin port 1, the second port 2, and a ``cpu`` side has to be the second
    (Dolphin only takes a CPU level there).  ``slippi_ai:<release>`` names the release for that side,
    overriding ``[slippi_ai] model`` (P27).

    These are the calibration arms that place an outside agent on the same ladder as every model in
    ``RESULTS.md`` before any of our weights are pointed at it -- and the gate that has to pass
    before a "we beat X" number means anything.

    An arm with ``mimic`` in it needs the GPU image; the rest carry no network at all and run without
    one.  ``--spawn`` as always: a synchronous ``.remote()`` dies with its client (GOTCHAS).
    """

    check_environment()
    path = resolve_config(config)
    if path.parent != CONFIG_DIR:
        raise SystemExit(f"{path} is outside {CONFIG_DIR}: only configs/rl/*.toml travel with the image")
    names = split_tags(sides)
    from melee_rl.external_match import env_layout  # validates the pair before anything is billed

    env_layout(names)
    try:
        function_name, plan_name = external_route(names)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    function: Any = {
        "external_match": external_match,
        "external_match_cpu": external_match_cpu,
        "external_match_slippi": external_match_slippi,
        "external_match_phillip": external_match_phillip,
        "external_match_phillip_cpu": external_match_phillip_cpu,
    }[function_name]
    plan = _planned(plan_name)
    assert plan is not None
    print(describe_plan(f"{function_name} {path.stem}", plan.resources, plan.seconds))
    print(f"run file {path.name}, sides {names}, output {output_dir}, label {label or '-'.join(names)}")
    arguments: tuple[Any, ...] = (
        path.name,
        sides,
        output_dir,
        label,
        matches,
        seconds,
        split_overrides(overrides),
    )
    if overrides:
        print(f"run-file overrides: {split_overrides(overrides)}")
    if spawn:
        call = function.spawn(*arguments)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned {function_name}: function call {call.object_id} in app {app_id}; "
            f"read it with modal app logs {app_id} -e {MODAL_ENVIRONMENT}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id})
    started = time.perf_counter()
    result = function.remote(*arguments)
    wall = time.perf_counter() - started
    print(_outcome(plan.name, plan.resources, result["duration_s"], wall))
    print(
        f"fetch it: modal volume get {VOLUME_NAME} {output_dir.removeprefix(VOLUME_MOUNT).lstrip('/')} "
        f"-e {MODAL_ENVIRONMENT} ."
    )
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def render(directory: str, config: str | None = None, force: bool = False, workers: int | None = None) -> str:
    """P9 stage 2 over an existing Volume directory of ``.slp`` files."""

    check_environment()
    plan = _planned("record_video video_20m_bc (shake-out)")
    assert plan is not None
    print(describe_plan("render_slp", plan.resources, plan.seconds))
    name = None if config is None else resolve_config(config).name
    started = time.perf_counter()
    result = render_slp.remote(directory, name, force, workers)
    wall = time.perf_counter() - started
    print(_outcome("render_slp", plan.resources, result["duration_s"], wall))
    print(f"render_slp: {result['rendered']} of {result['jobs']} replays rendered in {directory}")
    for row in result["results"]:
        if not row["ok"]:
            print(f"  FAILED {row['slp']}: {row['error']}")
    return json.dumps(result, indent=1)


@app.local_entrypoint()
def bench_render(
    gpu: bool = False,
    seconds: float = 20.0,
    worker_counts: str = "1,4",
    render_seconds: float = 120.0,
    spawn: bool = False,
) -> str:
    """Measure stage 2: the render slowdown against real time and how it scales with workers.

    ``--gpu`` runs the identical bench with an L4 attached, which answers whether hardware GL engages
    at all under Xvfb (``glxinfo`` is in the report).  ``--spawn`` starts the function and returns at
    once -- a synchronous ``.remote()`` is **cancelled** when its client goes away even under
    ``modal run --detach``, and a bench outlives any attached client, so use it (26 Aug 2026: an
    11-minute run was lost to exactly this).  The report is printed inside the container either way."""

    check_environment()
    name = "render_bench_gpu" if gpu else "render_bench"
    plan = _planned(name)
    assert plan is not None
    print(describe_plan(plan.name, plan.resources, plan.seconds))
    print(
        f"{seconds:g} s of play -> one .slp, then batches of {worker_counts} concurrent renders; "
        f"each ends on its own frame markers, with {render_seconds:g} s as the safety cap"
    )
    function = render_bench_gpu if gpu else render_bench
    if spawn:
        call = function.spawn(seconds, worker_counts, render_seconds)
        app_id = getattr(app, "app_id", None)
        print(
            f"spawned {name}: function call {call.object_id} in app {app_id}; read it with "
            f"modal app logs {app_id} -e {MODAL_ENVIRONMENT}"
        )
        return json.dumps({"spawned": True, "function_call_id": call.object_id, "app_id": app_id})
    started = time.perf_counter()
    report = function.remote(seconds, worker_counts, render_seconds)
    wall = time.perf_counter() - started
    print(_outcome(name, plan.resources, report["duration_s"], wall))
    print(format_render_bench(report))
    if not report["ok"]:
        raise SystemExit(f"{name}: nothing rendered (see the report above)")
    return json.dumps(report, indent=1)


@app.local_entrypoint()
def video_check(
    build: str = DEFAULT_DOLPHIN_BUILD,
    seconds: float = 20.0,
    download_fallback: bool = False,
    render_timeout_s: float = 120.0,
    candidates: str | None = None,
    builds: str | None = None,
    skip_primary: bool = False,
) -> str:
    """P9 §6: the one probe that says whether a replay can be rendered in this container.

    ``--candidates`` / ``--builds`` narrow the ladder and the downloaded builds to a comma-separated
    subset, ``--skip-primary`` goes straight to them, ``--render-timeout-s`` bounds each attempt."""

    check_environment()
    plan = _planned("video_check")
    assert plan is not None
    print(describe_plan(plan.name, plan.resources, plan.seconds))
    print(f"build {build}: {dolphin_build(build).exe}; {seconds:g} s of play, then the backend ladder")
    started = time.perf_counter()
    report = video_probe.remote(
        build, seconds, download_fallback, render_timeout_s, candidates, builds, skip_primary
    )
    wall = time.perf_counter() - started
    print(_outcome("video_check", plan.resources, report["duration_s"], wall))
    print(format_video_check(report))
    if not report["ok"]:
        raise SystemExit("video_check: no backend rendered a video (see the report above)")
    return json.dumps(report, indent=1)


# ---------------------------------------------------------------------------
# local, token-free helpers: python -m melee_rl.modal_app estimate | manifest
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(
        prog="python -m melee_rl.modal_app",
        description="Local helpers of the Modal harness (no token, no network); runs go through modal run",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("estimate", help="the planned runs of ENV.md 5.3 with their cost")
    commands.add_parser("manifest", help="the files the image mount carries")
    args = parser.parse_args(argv)
    text = estimate_table() if args.command == "estimate" else manifest_text()
    print(text)
    return text


if __name__ == "__main__":
    main()


__all__ = [
    "ALI_MODULES",
    "APP_NAME",
    "CONFIG_DIR",
    "DEFAULT_DOLPHIN_BUILD",
    "DEFAULT_RENDER_BUILD",
    "DESELECTED_TESTS",
    "DOLPHIN_APT_PACKAGES",
    "DOLPHIN_BUILDS",
    "DOLPHIN_CHECK_RESOURCES",
    "DOLPHIN_ROOT",
    "DOLPHIN_TRAIN_GPU_RESOURCES",
    "DOLPHIN_TRAIN_RESOURCES",
    "DOLPHIN_VIDEO_APT_PACKAGES",
    "GIT_DEPENDENT_TESTS",
    "IGNORE_PATTERNS",
    "IMAGE_PACKAGES",
    "ISO_MOUNT",
    "ISO_PATH",
    "ISO_VOLUME_NAME",
    "LIBMELEE_PIN",
    "LOCAL_ROOT",
    "MODAL_ENVIRONMENT",
    "PLANNED_RUNS",
    "PLAYBACK_ASSET_MARKER",
    "PLAYBACK_CANDIDATES",
    "PLAYBACK_RELEASES_URL",
    "PRICES_DATE",
    "PRICES_PER_SECOND",
    "PYTEST_BASE_ARGS",
    "PYTHON_VERSION",
    "REMOTE_ROOT",
    "RENDER_BENCH_GPU_RESOURCES",
    "RENDER_BENCH_RESOURCES",
    "RENDER_DOLPHIN_PATH",
    "TESTS_RESOURCES",
    "TORCH_CPU_INDEX",
    "TORCH_GPU_INDEX",
    "TORCH_PIN",
    "TRAIN_RESOURCES",
    "VIDEO_CHECK_RESOURCES",
    "VIDEO_DOLPHIN_BUILDS",
    "VIDEO_RESOURCES",
    "VOLUME_MOUNT",
    "VOLUME_NAME",
    "WANDB_SECRET_KEY",
    "WANDB_SECRET_NAME",
    "DolphinBuild",
    "PlannedRun",
    "Resources",
    "VolumeCommitHook",
    "active_environment",
    "app",
    "available_configs",
    "bench_env",
    "bench_render",
    "check_environment",
    "config_env_type",
    "config_needs_mimic",
    "cost_estimate",
    "cpu_image",
    "describe_plan",
    "dolphin",
    "dolphin_build",
    "dolphin_check",
    "dolphin_gpu_image",
    "dolphin_image",
    "dolphin_image_commands",
    "dolphin_train",
    "dolphin_train_gpu",
    "dolphin_train_gpu_wide",
    "dolphin_train_gpu_wide_mimic",
    "dolphin_video_image",
    "env_bench",
    "estimate_table",
    "fetch_playback_asset",
    "find_playback_asset",
    "format_render_bench",
    "format_usd",
    "format_video_check",
    "install_dolphin",
    "iso_volume",
    "main",
    "manifest_summary",
    "manifest_text",
    "mount_manifest",
    "mypy_error_counts",
    "pin_environment",
    "plan_for_config",
    "pytest_command",
    "record_video",
    "render",
    "render_bench",
    "render_bench_gpu",
    "render_slp",
    "resolve_config",
    "run_dolphin_check",
    "run_record_video",
    "run_render_bench",
    "run_render_slp",
    "run_slippi_parity_job",
    "run_smoke_train",
    "run_test_suite",
    "run_tests",
    "run_tests_gpu",
    "run_video_check",
    "runs_volume",
    "slippi_parity",
    "smoke_train",
    "split_tags",
    "tests",
    "train",
    "train_function_for",
    "version_probe",
    "video",
    "video_check",
    "video_probe",
    "volume_checkpoint_dir",
    "volume_commit_hook",
    "wandb_secret",
]
