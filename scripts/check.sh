#!/usr/bin/env bash
# The repository checks: adapter lint + tests, ASCII-only markdown.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
"$root/scripts/adapter-test.sh"
cd "$root"
sw-markdown-checker -f "docs/*.md"
sw-markdown-checker -f README.md
for f in xetal/README.md experiments/README.md; do sw-markdown-checker -f "$f"; done
