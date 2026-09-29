#!/usr/bin/env python3
"""Run the existing frozen suite with P21 RL weights and SLP-only output."""
import os

os.environ["MELEE_FINAL_BENCHMARK_VARIANT"] = "postrl"

from run_final_winner_benchmark import main

if __name__ == "__main__":
    main()
