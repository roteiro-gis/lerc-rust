#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${LERC_REFERENCE_DOCKER_IMAGE:-lerc-rust-reference}"
cargo_term_color="${CARGO_TERM_COLOR:-always}"
rustflags="${RUSTFLAGS:--D warnings}"

docker build -f "${repo_root}/docker/reference.Dockerfile" -t "${image_name}" "${repo_root}"
docker run --rm \
  -e CARGO_TERM_COLOR="${cargo_term_color}" \
  -e RUSTFLAGS="${rustflags}" \
  -v "${repo_root}:/workspace" \
  -w /workspace \
  "${image_name}" bash -lc '
  helper="$(./scripts/build-reference-helper.sh)"
  LERC_READER_REFERENCE_HELPER="${helper}" cargo test -p lerc-reader --test reference_parity
  LERC_READER_REFERENCE_HELPER="${helper}" cargo test -p lerc-writer --test reference_parity
'
