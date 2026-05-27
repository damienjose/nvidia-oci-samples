# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

variable "compartment_id" {
  description = "OCI compartment OCID"
  type        = string
}

variable "label_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = "aiq"
}

variable "vault_type" {
  description = "OCI Vault type: DEFAULT or VIRTUAL_PRIVATE"
  type        = string
  default     = "DEFAULT"
}

variable "key_shape_algorithm" {
  description = "Master encryption key algorithm"
  type        = string
  default     = "AES"
}

variable "key_shape_length" {
  description = "Master encryption key length in bytes (16, 24, or 32)"
  type        = number
  default     = 32
}

variable "secrets" {
  description = "Map of secret names to their base64-encoded content. Keys are resource identifiers (not secret), values are the secret payloads."
  type        = map(string)
  default     = {}
}

variable "freeform_tags" {
  description = "Freeform tags"
  type        = map(string)
  default     = {}
}
