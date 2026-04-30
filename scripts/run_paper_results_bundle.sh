#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-configs/launch/pooltest02_paper_results_bundle_7gpu_v1.yaml}"

cd "${REPO_ROOT}"
PYTHONPATH=src python3 -m uwnav_dynamics.cli.paper_results_bundle -c "${CONFIG_PATH}"
