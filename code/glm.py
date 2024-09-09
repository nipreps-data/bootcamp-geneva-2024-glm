#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "nibabel",
#     "niworkflows",
#     "nilearn",
#     "pybids",
#     "fitlins",
# ]
# ///
import os
from pathlib import Path
import bids
import bids.modeling.transformations
import nibabel as nb
from nilearn import glm
import niworkflows as nw
import niworkflows.data


def get_duration(img_path: os.PathLike | Path) -> float:
    img = nb.load(img_path)
    if img.ndim < 4:
        return 0.0
    return float(img.shape[3] * img.header.get_zooms()[3])


if os.getcwd().endswith('code'):
    os.chdir('..')

layout = bids.BIDSLayout("sourcedata/raw")
layout.add_derivatives(
    "sourcedata/derivatives/fmriprep-24.0.0",
    config=[nw.data.load("nipreps.json")],
    validate=False,
)

# Temporarily get raw files for testing
bold_files = layout.get(
    suffix="bold", task="mixed", space=None, extension=".nii.gz"
)
# bold_files = layout.get(
#     suffix="bold", task="mixed", space="MNI152NLin2009cAsym", extension=".nii.gz"
# )

# +
first_level_models = []

for bold_file in bold_files:
# bold_file = bold_files[0]
# if True:
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

    # Rename columns not to start with trial_type, for easier contrasts
    design_matrix.columns = [col[11:] if col.startswith("trial_type.") else col for col in design_matrix.columns]

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

    model = glm.first_level.FirstLevelModel(
        smoothing_fwhm=5.0,  # FWHM, this should be a parameter
    )
    first_level_models.append(model)

    img = nb.load(bold_file)
    data = img.get_fdata(dtype="float32")
    ## Run the GLM
    model.fit(img, design_matrices=design_matrix)

    path_patterns=[
        "sub-{subject}/[ses-{session}/]sub-{subject}_[ses-{session}_][task-{task}_][run-{run}_]contrast-{contrast}_stat-{stat}_{suffix}{extension}",
    ]
    
    for contrast in ("motor", "music", "visual", "motor - music", "motor - visual", "music - visual"):
        effect = model.compute_contrast(contrast, output_type="effect_size")
        variance = model.compute_contrast(contrast, output_type="effect_variance")
        contrastValue = ''.join(("Vs" if part == "-" else part.capitalize()) for part in contrast.split())
        effect_fname = bids.layout.writing.build_path(
            {
                "suffix": "statmap",
                "extension": ".nii.gz",
                "contrast": contrastValue,
                "stat": "effect",
                **selectors,
            },
            path_patterns=path_patterns,
        )
        var_fname = bids.layout.writing.build_path(
            {
                "suffix": "statmap",
                "extension": ".nii.gz",
                "contrast": contrastValue,
                "stat": "variance",
                **selectors,
            },
            path_patterns=path_patterns,
        )
        effect.to_filename(effect_fname)
        variance.to_filename(var_fname)
# -

import acres

first_level_layout = bids.BIDSLayout('.', config=['bids', acres.Loader('fitlins.data').cached('fitlins.json')], validate=False)

first_level_layout.get(contrast='Motor')

second_level = glm.second_level.SecondLevelModel()

second_level.fit(first_level_models[:1] * 4)

music = second_level.compute_contrast(first_level_contrast='music')

plotting.plot_stat_map(
    motor,
    colorbar=True,
    title="First-level contrast: Motor",
)

# %matplotlib inline

motor_vs_visual = model.compute_contrast("motor - visual")

plotting.plot_stat_map(
    music,
    colorbar=True,
    title="Second-level contrast: Music",
)


