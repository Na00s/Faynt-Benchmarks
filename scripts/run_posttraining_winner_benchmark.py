#!/usr/bin/env python3
"""Run the frozen benchmark on the post-training 75M winner, then the 10M winner."""

from __future__ import annotations

import importlib
import os

os.environ["MELEE_FINAL_BENCHMARK_VARIANT"] = "posttraining"
main = importlib.import_module("run_final_winner_benchmark").main


if __name__ == "__main__":
    main()
