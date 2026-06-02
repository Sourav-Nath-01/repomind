FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create conda environment for the agent
RUN conda create -n agent_env python=3.11 pip -y --override-channels -c conda-forge && \
    conda clean -afy

# Set up the non-root agent user
RUN groupmod -n agent ubuntu 2>/dev/null || groupadd -g 1000 agent 2>/dev/null || true && \
    usermod -l agent -d /home/agent ubuntu 2>/dev/null || \
    useradd -u 1000 -g 1000 -m -s /bin/bash agent 2>/dev/null || true

# Ensure permissions
RUN mkdir -p /home/agent && chown -R agent:agent /home/agent
RUN chown -R agent:agent /opt/conda

USER agent

# Add conda to agent's bashrc and set default env
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate agent_env" >> ~/.bashrc

# Ensure our run commands use the conda environment's python/pip by default
ENV PATH="/opt/conda/envs/agent_env/bin:${PATH}"

# Install common test dependencies
RUN pip install --no-cache-dir \
    pytest pytest-xdist pytest-timeout "setuptools==65.5.0" wheel hypothesis \
    pytz asgiref sqlparse requests six \
    attrs more-itertools packaging pyparsing \
    setuptools_scm cython numpy extension-helpers pytest-astropy

# Workspace setup
USER root
RUN mkdir -p /workspace && chown agent:agent /workspace
USER agent
WORKDIR /workspace

RUN git config --global user.email "agent@code-agent" && \
    git config --global user.name "Code Agent" && \
    git config --global safe.directory '/workspace'

CMD ["python", "-m", "pytest", "--help"]
