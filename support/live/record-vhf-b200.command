#!/bin/zsh
# Five VHF channels on the local B200; all scientific settings stay explicit.
set -eu
project_root="${0:A:h:h:h}"
cd "$project_root"
if [[ ! -x build/release/dvbt_capture ]]; then
    print -u2 'Build first: cmake --build --preset release --target dvbt_capture -j 6'
    exit 1
fi
if [[ -z "${VK_ICD_FILENAMES:-}" && -f build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json ]]; then
    export VK_ICD_FILENAMES="$project_root/build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
fi
# Keep the Mac awake while receiving. Extra CLI arguments can set a duration,
# output directory or other explicit override; the capture parser validates them.
exec /usr/bin/caffeinate -i build/release/dvbt_capture \
    --metadata recordings/live_mt_dandenong.json \
    --output "captures/live-vhf-b200-$(date +%Y%m%d-%H%M%S)-$$" \
    --archive profiles --range-metres 100000 \
    --integration 3 --doppler 100 --update 0.25 --remove-dc true \
    --fec-workers 5 --frame-slots 9 \
    "$@"
