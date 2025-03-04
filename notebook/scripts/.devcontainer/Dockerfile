# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.234.0/containers/python-3/.devcontainer/base.Dockerfile

ARG VARIANT="3.10"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

ENV POETRY_REQUESTS_TIMEOUT=240

RUN apt-get update && \
    apt-get -y install --no-install-recommends default-jre && \
    pip install poetry