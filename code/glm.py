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
import os
from pathlib import Path
import bids
import bids.modeling.transformations
import nibabel as nb
import niworkflows as nw
import niworkflows.data


def get_duration(img_path: os.PathLike | Path) -> float:
    img = nb.load(img_path)
    if img.ndim < 4:
        return 0.0
    return float(img.shape[3] * img.header.get_zooms()[3])


layout = bids.BIDSLayout("sourcedata/raw")
layout.add_derivatives(
    "sourcedata/derivatives/fmriprep-24.0.0",
    config=[nw.data.load("nipreps.json")],
    validate=False,
)

orig_bold = layout.get(suffix="bold", task="mixed", desc=None, extension=".nii.gz")

bold_files = layout.get(
    suffix="bold", task="mixed", space="MNI152NLin2009cAsym", extension=".nii.gz"
)

for bold_file in bold_files:
    # Extract entities that identify a given run
    bold_entities = bold_file.get_entities()
    bold_metadata = bold_file.get_metadata()

    selectors = {
        key: bold_entities[key]
        for key in ("subject", "session", "task", "run")
        if key in bold_entities
    }

    orig_bold = layout.get(suffix="bold", desc=None, extension=".nii.gz", **selectors)[0]

    # Collect the scan length from the original BOLD
    # This is just helpful to construct the design matrix without retrieving the resampled
    # BOLD series.
    scan_length = get_duration(orig_bold)

    variables = layout.get_collections(level="run", scan_length=scan_length, **selectors)[0]

    if "StartTime" in bold_metadata:  # SliceTiming correction shifted the effective time
        # There is currently no way in PyBIDS to update the target index with a non-zero
        # onset, so subtract the shift from the onset of the variables.
        # This must be done prior to convolution.
        variables["trial_type"].onset -= bold_metadata["StartTime"]

    # Transformations are in-place operations
    # Factor trial_type into dummy variables
    bids.modeling.transformations.Factor(variables, "trial_type")
    # We now have several columns, glob to convolve all of them
    bids.modeling.transformations.Convolve(variables, variables.match_variables("trial_type.*"))

    # variables.to_df converts various types of variables into a pandas DataFrame with a
    # common index. Set the sampling rate to TR to resample dynamically according to
    # the TR of the BOLD series.
    df = variables.to_df(sampling_rate="TR")

    # The design matrix is a DataFrame with the columns of interest, nuissance regressors,
    # and an intercept.
    design_matrix = df[variables.match_variables(["trial_type.*", "rot_?", "trans_?"])]
    design_matrix["intercept"] = 1

    dmat_fname = bids.layout.writing.build_path(
        {
            "suffix": "design",
            "extension": ".tsv",
            "datatype": "func",
            **selectors,
        },
        path_patterns=[
            "sub-{subject}/[ses-{session}/]{datatype}/sub-{subject}_[ses-{session}_]task-{task}_[run-{run}_]{suffix}{extension}",
        ],
    )
    Path(dmat_fname).parent.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(dmat_fname, sep="\t", index=False)
