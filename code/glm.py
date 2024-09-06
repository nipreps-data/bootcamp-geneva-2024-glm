#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "nibabel",
#     "niworkflows",
#     "nilearn",
#     "pybids",
# ]
# ///
import bids
import numpy as np
import nibabel as nb
import nilearn as nl
import niworkflows as nw
import niworkflows.data

layout = bids.BIDSLayout("sourcedata/raw")
layout.add_derivatives(
    "sourcedata/derivatives/fmriprep-24.0.0",
    config=[nw.data.load("nipreps.json")],
    validate=False,
)

bold_files = layout.get(
    suffix="bold", task="mixed", desc="preproc", extension=".nii.gz"
)
events = layout.get_collections(level="run", task="mixed")
