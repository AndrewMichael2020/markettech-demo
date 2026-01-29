#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-markettech}"
SERVICE="${SERVICE:-markettech-truth-engine}"
TAG="${TAG:-v1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"

echo "Destroying Terraform-managed resources..."
pushd terraform >/dev/null
terraform init -upgrade
terraform destroy -auto-approve   -var="project_id=${PROJECT_ID}"   -var="region=${REGION}"   -var="repo=${REPO}"   -var="service_name=${SERVICE}"   -var="image=${IMAGE}"
popd >/dev/null

echo "Optional hard stop: unlink billing (commented out by default)"
echo "  gcloud billing projects unlink ${PROJECT_ID}"
echo "Optional nuclear option: delete project (commented out by default)"
echo "  gcloud projects delete ${PROJECT_ID}"
