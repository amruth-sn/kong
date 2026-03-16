#!/bin/bash
set -euo pipefail

# Fix imports first since other tools depend on proper import organization
ruff check --select I --fix
ruff format
ruff check
typos
pyright
