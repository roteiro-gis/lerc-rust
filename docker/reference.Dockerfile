FROM rust:1.77-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG LERC_VERSION=v4.1.0

RUN mkdir -p /tmp/lerc-src \
    && curl -L --max-redirs 5 "https://api.github.com/repos/Esri/lerc/tarball/refs/tags/${LERC_VERSION}" \
      | tar -xz -C /tmp/lerc-src --strip-components=1 \
    && cmake -S /tmp/lerc-src -B /tmp/lerc-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/lerc \
    && cmake --build /tmp/lerc-build --target install --parallel \
    && rm -rf /tmp/lerc-src /tmp/lerc-build

ENV LERC_REFERENCE_LIB_DIR=/opt/lerc/lib

WORKDIR /workspace
