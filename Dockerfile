# Render worker image: the usd-synth package plus a pinned Blender LTS (GPL, shipped unmodified
# under its own license; this project's code stays MIT). CPU rendering works out of the box.
# It also carries the optional Strata backend (render.engine: strata, MIT): the release .so
# plus Mesa's EGL and llvmpipe, so one image serves either engine.
FROM python:3.12-slim-bookworm

ARG BLENDER_VERSION=5.2.2
ARG BLENDER_SHA256=84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168
ARG STRATA_VERSION=0.5.0
ARG STRATA_SHA256=44f8263426b546c04295b1033032fded97541060c8e804873894d43e80fcffcb

# Shared libraries the official Linux build links against (X11/GL stubs; no display is used).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils \
        libx11-6 libxext6 libxrender1 libxfixes3 libxi6 libxkbcommon0 libsm6 libice6 \
        libgl1 libegl1 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/blender.tar.xz \
        "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
    && echo "${BLENDER_SHA256}  /tmp/blender.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/blender \
    && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && /opt/blender/blender --version | head -1

RUN mkdir -p /opt/strata/lib \
    && curl -fsSL -o /opt/strata/lib/libstrata.so \
        "https://github.com/Preponderous-Software/strata/releases/download/v${STRATA_VERSION}/libstrata-${STRATA_VERSION}-linux-x64.so" \
    && echo "${STRATA_SHA256}  /opt/strata/lib/libstrata.so" | sha256sum -c -

ENV USD_SYNTH_BLENDER=/opt/blender/blender \
    USD_SYNTH_STRATA=/opt/strata/lib/libstrata.so \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

ENTRYPOINT ["usd-synth"]
CMD ["--help"]
