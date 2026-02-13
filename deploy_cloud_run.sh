#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-markettech}"
SERVICE="${SERVICE:-markettech-truth-engine}"
TAG="${TAG:-v1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${TAG}"

echo "Using image: ${IMAGE}"
gcloud config set project "${PROJECT_ID}"

echo "Building and pushing image with Cloud Build..."
gcloud builds submit --tag "${IMAGE}" .

echo "Applying Terraform..."
terraform init -upgrade
terraform apply -auto-approve   -var="project_id=${PROJECT_ID}"   -var="region=${REGION}"   -var="repo=${REPO}"   -var="service_name=${SERVICE}"   -var="image=${IMAGE}"
