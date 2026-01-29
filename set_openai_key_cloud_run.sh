#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-markettech-truth-engine}"
OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

gcloud config set project "${PROJECT_ID}"
gcloud run services update "${SERVICE}" --region "${REGION}" --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY}"

echo "Updated Cloud Run env var OPENAI_API_KEY for service ${SERVICE} in ${REGION}."
