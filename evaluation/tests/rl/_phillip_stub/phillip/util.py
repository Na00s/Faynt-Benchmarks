"""``load_params`` as upstream defines it (``phillip/util.py:219-229``)."""

from __future__ import annotations

import json
from typing import Any


def load_params(path: str, key: str | None = None) -> dict[str, Any]:
    with open(path + "/params") as handle:
        params: dict[str, Any] = json.load(handle)
    if key and key in params:
        params.update(params[key])
    params.update(path=path)
    return params
