# CUDA runtime libraries are pinned Python wheels from JAX/PyTorch in requirements.lock.
# A host NVIDIA driver and NVIDIA Container Toolkit are required at runtime.
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.11.13-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/project_minimal:/opt/project_minimal/openpi:/opt/project_minimal/openpi/src:/opt/project_minimal/openpi/packages/openpi-client/src
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates git build-essential ffmpeg libgl1 libglib2.0-0 \
    openssh-client rsync procps && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/requirements.lock
RUN uv venv /opt/venv --python /usr/local/bin/python \
    && uv pip install --python /opt/venv/bin/python --no-deps -r /opt/requirements.lock
COPY openpi /opt/project_minimal/openpi
COPY conversion /opt/project_minimal/conversion
COPY training /opt/project_minimal/training
COPY configs /opt/project_minimal/configs
COPY SOURCE_PROVENANCE.md /opt/project_minimal/SOURCE_PROVENANCE.md
# The upstream metadata lists simulation extras; source imports use PYTHONPATH.
# Do not install that metadata and inadvertently pull simulator dependencies back in.
RUN python -c "import jax, torch, yaml, mcap, mcap_ros1; from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; from openpi.training import config; print('Tuning imports OK')" \
    && python -c "import importlib.util; from pathlib import Path; from conversion import pipeline; from training import config; assert all(importlib.util.find_spec(p) is None for p in ('mujoco', 'robosuite', 'gym_aloha', 'dm_control', 'pymunk')); assert all(not Path('/opt/project_minimal', p).exists() for p in ('piper_pi0', 'openpi/examples', 'openpi/third_party')); print('Standalone conversion/training imports OK')"
WORKDIR /workspace/project_minimal
CMD ["sleep", "infinity"]
