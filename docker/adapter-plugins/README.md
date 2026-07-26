# Trusted Adapter plugin build inputs

The default image contains no external plugin. `scripts/build_adapter_job_image.py`
creates a temporary Docker context, verifies a signed/allowlisted bundle, and places
only that public bundle here before building an immutable GPU Job image.
