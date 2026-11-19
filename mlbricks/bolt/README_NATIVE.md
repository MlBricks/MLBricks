# BOLT native execution

Modern BOLT attention uses PyTorch SDPA. Optional shared native operators remain for standalone AVQ and other components that depend on them; they do not implement modern BOLT attention. CUDA qualification of the merged library is required before release.
