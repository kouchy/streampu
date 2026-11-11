#!/bin/bash
# Format the C++ sources with the clang-format version pinned in `.clang-format-version` (same version locally
# and in the CI, see .github/workflows/ci.yml and .gitlab-ci.yml). Different clang-format releases format the same
# code differently, so the version must match exactly.
#
# The binary is taken, in this order, from: $CLANG_FORMAT, `clang-format` in the PATH (if its version matches),
# `uvx`/`pipx` (PyPI wheel `clang-format==<version>`, downloaded on the fly). Otherwise:
#   python3 -m pip install clang-format==<version>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="$(tr -d '[:space:]' < .clang-format-version)"
version_of() { "$@" --version 2>/dev/null | sed -n 's/.*clang-format version \([0-9][0-9.]*\).*/\1/p'; }

if [[ -n "${CLANG_FORMAT:-}" ]]; then
    CF=(${CLANG_FORMAT})
elif command -v clang-format > /dev/null && [[ "$(version_of clang-format)" == "${VERSION}" ]]; then
    CF=(clang-format)
elif command -v uvx > /dev/null; then
    CF=(uvx "clang-format==${VERSION}")
elif command -v pipx > /dev/null; then
    CF=(pipx run "clang-format==${VERSION}")
else
    echo "error: clang-format ${VERSION} not found (found: $(version_of clang-format || true))." >&2
    echo "       install it with: python3 -m pip install clang-format==${VERSION}" >&2
    exit 1
fi
FOUND="$(version_of "${CF[@]}")"
if [[ "${FOUND}" != "${VERSION}" ]]; then
    echo "error: ${CF[*]} is clang-format ${FOUND}, expected ${VERSION} (.clang-format-version)." >&2
    exit 1
fi
echo "Using clang-format ${FOUND} (${CF[*]})"

find include -iname '*.h' -o -iname '*.hpp' -o -iname '*.hxx' | xargs "${CF[@]}" -i
find src -iname '*.cpp' -o -iname '*.h' -o -iname '*.hpp' -o -iname '*.hxx' | xargs "${CF[@]}" -i
find tests -iname '*.cpp' -o -iname '*.h' -o -iname '*.hpp' -o -iname '*.hxx' | xargs "${CF[@]}" -i
find signal_tracer -iname '*.cpp' -o -iname '*.h' -o -iname '*.hpp' -o -iname '*.hxx' | xargs "${CF[@]}" -i
