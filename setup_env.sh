#!/bin/bash
set -euo pipefail

echo "Setting up DyLLM environment for TPU (torch_xla)..."

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: activate your conda env first (e.g., conda activate dyllm_tpu)."
  exit 1
fi

TORCH_VERSION="2.9.0"
TORCH_XLA_VERSION="2.9.0"

echo "Upgrading pip tooling..."
python3 -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch ${TORCH_VERSION}..."
python3 -m pip install --upgrade "torch==${TORCH_VERSION}"

echo "Installing torch_xla ${TORCH_XLA_VERSION}..."
python3 -m pip install --upgrade "torch-xla==${TORCH_XLA_VERSION}"

echo "Installing JAX TPU runtime..."
python3 -m pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

TORCH_LIB_DIR="$(python3 - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"

TPU_LIB_PATH=""
for p in /lib/libtpu.so /usr/lib/libtpu.so "${CONDA_PREFIX}/lib/libtpu.so"; do
  if [[ -f "$p" ]]; then
    TPU_LIB_PATH="$p"
    break
  fi
done

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/dyllm_tpu_env.sh" <<EOF
#!/bin/bash
export _OLD_DYLLM_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:\${LD_LIBRARY_PATH:-}"
if [[ -f "${TPU_LIB_PATH}" ]]; then
  export TPU_LIBRARY_PATH="${TPU_LIB_PATH}"
  export PTXLA_TPU_LIBRARY_PATH="${TPU_LIB_PATH}"
fi
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/dyllm_tpu_env.sh" <<'EOF'
#!/bin/bash
if [[ -n "${_OLD_DYLLM_LD_LIBRARY_PATH+x}" ]]; then
  export LD_LIBRARY_PATH="${_OLD_DYLLM_LD_LIBRARY_PATH}"
  unset _OLD_DYLLM_LD_LIBRARY_PATH
fi
unset TPU_LIBRARY_PATH
unset PTXLA_TPU_LIBRARY_PATH
EOF
chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/dyllm_tpu_env.sh" "${CONDA_PREFIX}/etc/conda/deactivate.d/dyllm_tpu_env.sh"

echo "Installing project runtime dependencies..."
python3 -m pip install --upgrade \
  transformers==4.57.6 \
  accelerate>=1.12.0 \
  numpy>=2.1.2 \
  pandas>=2.3.3 \
  scipy>=1.15.3 \
  datasets>=3.6.0 \
  tqdm>=4.67.1 \
  einops>=0.8.1 \
  pillow>=11.3.0

echo "Installing DyLLM in editable mode (without CUDA extensions)..."
export DYLLM_BUILD_CUDA_EXT=0
python3 -m pip install -e . --no-build-isolation

echo "Verifying torch/torch_xla version match..."
python3 - <<'PY'
import torch
import torch_xla


def major_minor(v: str) -> str:
    return ".".join(v.split("+")[0].split(".")[:2])

print("torch:", torch.__version__)
print("torch_xla:", torch_xla.__version__)
if major_minor(torch.__version__) != major_minor(torch_xla.__version__):
    raise SystemExit("Version mismatch between torch and torch_xla")
print("Version check passed.")
PY

echo "TPU setup complete."
echo "Run: conda deactivate && conda activate $(basename "$CONDA_PREFIX")"
echo "Then: PJRT_DEVICE=TPU python3 run.py"
