# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

ESA_GATE_MIN = 0.80
ESA_GATE_MAX = 0.995
ESA_SCAN_EPS = 1e-6
SUPPORTED_BACKENDS = {"thunder"}
SUPPORTED_PRECISIONS = {"fp8", "fp16", "bf16", "fp32", "fp64"}
