# Sets up a shell for this repository.  The script is sourced, not run, as the activation
# and exports of a script that is run are lost when it exits:
#
#     source scripts/ehl_env.sh
#
# It changes to the repository root, activates the conda environment (ehl-fenics, or the
# name in EHL_CONDA_ENV), and exports PYTHONPATH (the repository root) and DIJITSO_CACHE_DIR
# ($HOME/.cache/dijitso, created if missing).

# Refuses to be run.
if ! (return 0 2>/dev/null); then
    echo "ehl_env.sh: source it, do not run it:  source ${0}" >&2
    exit 1
fi

_ehl_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_ehl_envname="${EHL_CONDA_ENV:-ehl-fenics}"

# Makes 'conda activate' available where the shell has not run 'conda init'.
if ! declare -F conda >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(command conda shell.bash hook)"
    else
        for _ehl_base in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
            if [ -f "$_ehl_base/etc/profile.d/conda.sh" ]; then
                . "$_ehl_base/etc/profile.d/conda.sh"
                break
            fi
        done
    fi
fi

if ! declare -F conda >/dev/null 2>&1; then
    echo "ehl_env.sh: conda not found (looked on PATH and in ~/miniforge3, ~/mambaforge, ~/miniconda3, ~/anaconda3)" >&2
    unset _ehl_root _ehl_envname _ehl_base
    return 1
fi

if [ "${CONDA_DEFAULT_ENV:-}" != "$_ehl_envname" ]; then
    if ! conda activate "$_ehl_envname"; then
        echo "ehl_env.sh: 'conda activate $_ehl_envname' failed" >&2
        unset _ehl_root _ehl_envname _ehl_base
        return 1
    fi
fi

cd "$_ehl_root" || { unset _ehl_root _ehl_envname _ehl_base; return 1; }
export PYTHONPATH="$_ehl_root"
export DIJITSO_CACHE_DIR="$HOME/.cache/dijitso"
mkdir -p "$DIJITSO_CACHE_DIR"

echo "ehl env ready: $(pwd)"
echo "  conda env  $CONDA_DEFAULT_ENV  ($(command -v python))"
echo "  PYTHONPATH $PYTHONPATH"
echo "  DIJITSO    $DIJITSO_CACHE_DIR"

unset _ehl_root _ehl_envname _ehl_base
