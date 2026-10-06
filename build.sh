#!/usr/bin/env bash
set -Eeuo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/reg-elegantbook.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT
cd -- "$project_dir"
latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$build_dir" main.tex
cp -- "$build_dir/main.pdf" "$project_dir/回归分析前三章.pdf"
printf '已生成：%s\n' "$project_dir/回归分析前三章.pdf"
