#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-markettech-truth-engine}"
SECRET_NAME="${SECRET_NAME:-openai-api-key}"

gcloud config set project "${PROJECT_ID}"
gcloud run services update "${SERVICE}" \
	--region "${REGION}" \
	--update-secrets="OPENAI_API_KEY=${SECRET_NAME}:latest"

echo "Updated Cloud Run to read OPENAI_API_KEY from Secret Manager secret ${SECRET_NAME} for service ${SERVICE} in ${REGION}."
