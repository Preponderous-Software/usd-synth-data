# MinIO for the kind integration test only (any S3-compatible store works for real runs).
# Upstream stopped publishing MinIO server images and binaries and archived the repository
# (October 2025), so the server is built from source at its final release tag, pinned by
# commit. MinIO is AGPL-3.0 and is used here unmodified as a separate test service.
FROM golang:1.24.8-bookworm AS build
ARG MINIO_TAG=RELEASE.2025-10-15T17-29-55Z
ARG MINIO_COMMIT=9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a
RUN git clone --depth 1 --branch "${MINIO_TAG}" https://github.com/minio/minio.git /src \
    && test "$(git -C /src rev-parse HEAD)" = "${MINIO_COMMIT}" \
    && cd /src && CGO_ENABLED=0 go build -trimpath -o /minio .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /minio /minio
ENTRYPOINT ["/minio"]
