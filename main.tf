provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "run" {
  project = var.project_id
  service = "run.googleapis.com"
}

resource "google_project_service" "artifactregistry" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"
}

resource "google_project_service" "cloudbuild" {
  project = var.project_id
  service = "cloudbuild.googleapis.com"
}

# Artifact Registry repo for container images
resource "google_artifact_registry_repository" "repo" {
  depends_on = [google_project_service.artifactregistry]
  location   = var.region
  repository_id = var.repo
  format     = "DOCKER"
  description = "MarketTech workshop images"
}

# Cloud Run service (2nd gen)
resource "google_cloud_run_v2_service" "service" {
  depends_on = [google_project_service.run]
  name       = var.service_name
  location   = var.region

  template {
    containers {
      image = var.image
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
  }

  ingress = "INGRESS_TRAFFIC_ALL"
}

# Allow unauthenticated access by granting run.invoker to allUsers
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  location = google_cloud_run_v2_service.service.location
  name     = google_cloud_run_v2_service.service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "service_url" {
  value = google_cloud_run_v2_service.service.uri
}
