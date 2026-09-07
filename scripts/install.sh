#!/bin/sh
set -eu

repository="https://github.com/tahaenesaslanturk/Adaptea"
site="https://adaptea.dev"

say() {
  printf '%s\n' "$1"
}

fail() {
  printf 'adaptea installer: %s\n' "$1" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is required"

# Check if desktop app download was requested via flag or environment variable
download_desktop=0
for arg in "$@"; do
  case "$arg" in
    --desktop|--app|--dmg|--gui)
      download_desktop=1
      ;;
  esac
done

if [ "${ADAPTEA_DESKTOP:-0}" = "1" ] || [ "$download_desktop" = "1" ]; then
  case "$(uname -s 2>/dev/null || true)" in
    Darwin)
      say "Downloading Adaptea for macOS (Apple silicon)..."
      target_file="Adaptea-macos-arm64.dmg"
      curl -fL -o "$target_file" "$site/api/download/macos"
      say "✓ Downloaded $target_file to $(pwd)/$target_file"
      say "Open $target_file to install Adaptea in /Applications"
      exit 0
      ;;
    *)
      say "Downloading Adaptea for Windows (x64)..."
      target_file="Adaptea-windows-x64-setup.exe"
      curl -fL -o "$target_file" "$site/api/download/windows"
      say "✓ Downloaded $target_file to $(pwd)/$target_file"
      say "Run $target_file to install Adaptea"
      exit 0
      ;;
  esac
fi

command -v git >/dev/null 2>&1 || fail "Git is required; install Git and run this command again"

case "$(uname -s 2>/dev/null || true)" in
  Darwin|Linux) ;;
  *) fail "this installer currently supports macOS and Linux" ;;
esac

temporary_directory=$(mktemp -d 2>/dev/null || mktemp -d -t adaptea)
trap 'rm -rf "$temporary_directory"' EXIT HUP INT TERM

uv_bin=$(command -v uv 2>/dev/null || true)
if [ -z "$uv_bin" ]; then
  say "Preparing Adaptea's isolated Python runtime..."
  UV_UNMANAGED_INSTALL="$temporary_directory/uv" \
    sh -c "$(curl -LsSf https://astral.sh/uv/install.sh)"
  uv_bin="$temporary_directory/uv/uv"
fi
[ -x "$uv_bin" ] || fail "could not prepare the isolated runtime"

latest_url=$(curl -LsSf -o /dev/null -w '%{url_effective}' "$repository/releases/latest" 2>/dev/null || true)
latest_tag="${latest_url##*/}"

case "$latest_tag" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *)
    # Fallback to remote git tags if no GitHub release object was published yet
    latest_tag=$(git ls-remote --tags --sort=v:refname "$repository.git" 2>/dev/null | grep -o 'refs/tags/v[0-9]*\.[0-9]*\.[0-9]*' | tail -n 1 | sed 's#refs/tags/##' || true)
    ;;
esac

case "$latest_tag" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *)
    latest_tag="main"
    ;;
esac

say "Installing Adaptea ${latest_tag}..."
"$uv_bin" tool install --force --python 3.12 "git+$repository.git@${latest_tag}"

tool_bin=$HOME/.local/bin
if command -v adaptea >/dev/null 2>&1; then
  say "Adaptea is ready. Run: adaptea"
elif [ -x "$tool_bin/adaptea" ]; then
  say "Adaptea is ready at $tool_bin/adaptea"
  say "Add $tool_bin to PATH, then run: adaptea"
else
  fail "installation finished but the adaptea command was not found"
fi
