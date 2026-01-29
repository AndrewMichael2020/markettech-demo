variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type        = string
  description = "GCP region (e.g., us-central1)"
  default     = "us-central1"
}

variable "repo" {
  type        = string
  description = "Artifact Registry repo name"
  default     = "markettech"
}

variable "service_name" {
  type        = string
  description = "Cloud Run service name"
  default     = "markettech-truth-engine"
}

variable "image" {
  type        = string
  description = "Full container image URI (Artifact Registry)"
}
