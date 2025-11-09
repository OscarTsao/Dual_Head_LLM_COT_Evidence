# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces
