#!/bin/zsh
set -eu
project_root="${0:A:h:h:h}"
cd "$project_root"
fixture="${1:-build/live-validation/five_channels.wav}"
if [[ ! -x build/release/dvbt_capture ]]; then
    print -u2 'Build first: cmake --build --preset release --target dvbt_capture -j 6'
    exit 1
fi
if [[ ! -f "$fixture" ]]; then
    print -u2 'Generate the simulation fixture as described in support/live/README.md.'
    exit 1
fi
if [[ -z "${VK_ICD_FILENAMES:-}" && -f build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json ]]; then
    export VK_ICD_FILENAMES="$project_root/build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
fi
exec build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
    --simulate "$fixture" --archive none \
    --output "captures/simulation-$(date +%Y%m%d-%H%M%S)-$$"
