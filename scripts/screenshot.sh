#!/usr/bin/env bash
# Retake the README's screenshot of the live demo's model map (images/demo.png):
# build the site, render map.html with headless Chrome, and record which map
# it shows (images/demo.png.source, checked by pages/tests).
#   scripts/screenshot.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
[ -x "$chrome" ] || { echo "screenshot: no Chrome at $chrome (set CHROME)" >&2; exit 1; }
site="$root/work/site"
"$root/scripts/pages.sh" "$site" >/dev/null
"$chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1000,1240 \
    --screenshot="$root/images/demo.png" "file://$site/map.html" 2>/dev/null
cd "$root"
printf '%s  %s\n' "$(shasum -a 256 images/gpt2-model-map.svg | cut -d' ' -f1)" images/gpt2-model-map.svg > images/demo.png.source
echo "$root/images/demo.png"
