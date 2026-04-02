#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${LERC_REFERENCE_BUILD_DIR:-${repo_root}/target/reference}"
binary_path="${out_dir}/lerc-reference"
lib_dir="${LERC_REFERENCE_LIB_DIR:-}"
lib_name="${LERC_REFERENCE_LIB_NAME:-Lerc}"
cxx="${CXX:-c++}"

mkdir -p "${out_dir}"

compile_args=(
  -std=c++17
  -O3
  -Wall
  -Wextra
  -pedantic
  "${repo_root}/tools/lerc_reference_helper.cpp"
  -o "${binary_path}"
)

if [[ -n "${lib_dir}" ]]; then
  compile_args+=("-L${lib_dir}" "-Wl,-rpath,${lib_dir}")
fi

compile_args+=("-l${lib_name}")

"${cxx}" "${compile_args[@]}"

if [[ "$(uname -s)" == "Darwin" && -n "${lib_dir}" && -f "${lib_dir}/lib${lib_name}.dylib" ]]; then
  install_name_tool \
    -change "/usr/local/lib/lib${lib_name}.dylib" "${lib_dir}/lib${lib_name}.dylib" "${binary_path}" \
    2>/dev/null || true
fi

echo "${binary_path}"
