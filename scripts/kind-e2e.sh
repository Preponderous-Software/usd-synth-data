#!/usr/bin/env bash
# M7 integration test: a 4-shard Indexed Job on a local kind cluster, writing to MinIO running
# inside that cluster; one render pod is deleted mid-run and must resume; the merged dataset is
# compared with a single-process local run of the same config and seed.
#
# Needs docker, kind and kubectl. Touches ONLY the kind context it creates ("kind-$CLUSTER");
# every kubectl call names that context explicitly. Usage: scripts/kind-e2e.sh [--keep]
set -euo pipefail

CLUSTER="${CLUSTER:-usd-synth-e2e}"
CTX="kind-${CLUSTER}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="tests/data/kind.yaml"
SEED=42
COUNT=12
SHARDS=4
BUCKET=datasets
PREFIX=kind-e2e
OUT="s3://${BUCKET}/${PREFIX}"
WORK="${WORK:-$(mktemp -d)}"
KEEP="${1:-}"

k() { kubectl --context "${CTX}" "$@"; }
log() { printf '\n=== %s\n' "$*"; }

case "${CTX}" in kind-*) ;; *) echo "refusing: context ${CTX} is not a kind context" >&2; exit 2;; esac

cleanup() {
  if [ "${KEEP}" != "--keep" ]; then kind delete cluster --name "${CLUSTER}" >/dev/null 2>&1 || true; fi
  [ -n "${PF_PID:-}" ] && kill "${PF_PID}" 2>/dev/null || true
}
trap cleanup EXIT

log "build images"
docker build -q -t usd-synth:kind "${ROOT}"
docker build -q -t usd-synth-minio:kind "${ROOT}/deploy/minio"

log "create kind cluster ${CLUSTER}"
kind create cluster --name "${CLUSTER}" --wait 120s
kind load docker-image --name "${CLUSTER}" usd-synth:kind usd-synth-minio:kind

log "deploy MinIO in the cluster"
k apply -f "${ROOT}/deploy/kind/minio.yaml"
k rollout status deployment/minio --timeout=180s
k create secret generic usd-synth-s3 \
  --from-literal=AWS_ENDPOINT_URL=http://minio:9000 \
  --from-literal=AWS_ACCESS_KEY_ID=usd-synth-test \
  --from-literal=AWS_SECRET_ACCESS_KEY=usd-synth-test-secret
k run mkbucket --image=usd-synth:kind --image-pull-policy=Never --restart=Never \
  --env=AWS_ENDPOINT_URL=http://minio:9000 --env=AWS_ACCESS_KEY_ID=usd-synth-test \
  --env=AWS_SECRET_ACCESS_KEY=usd-synth-test-secret --command -- \
  python -c "import boto3; boto3.client('s3', endpoint_url='http://minio:9000', region_name='us-east-1').create_bucket(Bucket='${BUCKET}')"
k wait --for=jsonpath='{.status.phase}'=Succeeded pod/mkbucket --timeout=120s

log "render ${COUNT} scenes as a ${SHARDS}-shard Indexed Job"
(cd "${ROOT}" && usd-synth k8s-manifest --config "${CONFIG}" --seed "${SEED}" --count "${COUNT}" \
  --shards "${SHARDS}" --out "${OUT}" --image usd-synth:kind --image-pull-policy Never \
  --cpu 1 --memory 1Gi --name e2e) > "${WORK}/manifests.yaml"
grep -q PersistentVolumeClaim "${WORK}/manifests.yaml" && { echo "manifest needs a PVC" >&2; exit 1; }
k apply -f "${WORK}/manifests.yaml"

log "delete the shard-1 pod after its first scene is DONE"
SEL="job-name=e2e-render,batch.kubernetes.io/job-completion-index=1"
timeout 300 bash -c "until kubectl --context ${CTX} get pod -l '${SEL}' -o name | grep -q pod/; do sleep 1; done"
k wait --for=condition=Ready pod -l "${SEL}" --timeout=300s
POD="$(k get pod -l "${SEL}" -o jsonpath='{.items[0].metadata.name}')"
timeout 600 bash -c "until kubectl --context ${CTX} logs ${POD} 2>/dev/null | grep -q 'shard 1: scene 3 done'; do sleep 1; done"
k delete pod "${POD}" --wait=false
echo "deleted ${POD}"

k wait --for=condition=Complete job/e2e-render --timeout=1200s
k wait --for=condition=Complete job/e2e-merge --timeout=600s
k get pods -l "job-name=e2e-render" -o wide
RETRY_LOG="$(k logs -l "${SEL}" --tail=-1)"
echo "${RETRY_LOG}"
echo "${RETRY_LOG}" | grep -Eq 'shard 1/4: scenes \[3, 6\): [0-9]+ rendered, [1-9][0-9]* already done' \
  || { echo "retried shard 1 did not resume from its DONE objects" >&2; exit 1; }
k logs job/e2e-merge | tail -3

log "single-process local reference run (same config and seed)"
mkdir -p "${WORK}/local"
docker run --rm -u "$(id -u):$(id -g)" -v "${ROOT}/tests/data:/cfg:ro" -v "${WORK}/local:/out" \
  usd-synth:kind generate --config "/cfg/$(basename "${CONFIG}")" --seed "${SEED}" \
  --count "${COUNT}" --out /out/ds --threads 2

log "compare the merged cluster dataset with the local run"
k port-forward svc/minio 19000:9000 >/dev/null 2>&1 &
PF_PID=$!
sleep 3
AWS_ENDPOINT_URL=http://127.0.0.1:19000 AWS_ACCESS_KEY_ID=usd-synth-test \
AWS_SECRET_ACCESS_KEY=usd-synth-test-secret \
  python "${ROOT}/scripts/kind_compare.py" "${OUT}" "${WORK}/local/ds" "${SHARDS}"
log "kind e2e: PASS"
