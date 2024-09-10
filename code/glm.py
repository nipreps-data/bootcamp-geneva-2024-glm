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
import logging
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


# Ignore this. Just helps makes it so we can use this script as a Jupyter notebook,
# which always sets the CWD to the directory containing the notebook.
if os.getcwd().endswith("code"):
    os.chdir("..")

# Collect raw data and derivatives in a single layout
layout = bids.BIDSLayout("sourcedata/raw")
layout.add_derivatives(
    "sourcedata/fmriprep-patch",
    config=[nw.data.load("nipreps.json")],
    validate=False,
)
logging.debug("Loaded BIDS dataset. Found %d files.", len(layout.files))

bold_files = layout.get(
    suffix="bold", task="mixed", space="MNI152NLin2009cAsym", extension=".nii.gz"
)

contrasts = (
    "motor",
    "music",
    "visual",
    "motor - music",
    "motor - visual",
    "music - visual",
)
# PyBIDS doesn't know how to write design matrices or statistical maps.
stat_patterns = [
    "[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_]task-{task}_[run-{run}_]{suffix<design>}{extension}",
    "[sub-{subject}/][ses-{session}/][sub-{subject}_][ses-{session}_][task-{task}_][run-{run}_]contrast-{contrast}_stat-{stat}_{suffix<statmap>}{extension}",
]

first_level_models = []

for bold_file in bold_files:
    logging.debug("Processing %s...", bold_file.path)
    # Extract entities that identify a given run
    bold_entities = bold_file.get_entities()
    selectors = {
        key: bold_entities[key]
        for key in ("subject", "session", "task", "run")
        if key in bold_entities
    }

    # Collect the scan length from the BOLD file.
    # We will be collecting variables from events.tsv (associated with the raw BOLD)
    # and timeseries.tsv (associated with the resampled BOLD). By default, the scan
    # length is retrieved from the associated BOLD file. We will collect it now so that
    # it is not necessary to have the raw BOLD file downloaded to run this script.
    scan_length = get_duration(bold_file)

    # get_collections() is a general tool that will find *any* variable in BIDS that
    # describes a file or event. It can include sparse variables (onset, duration, *metadata)
    # or dense variables (n vars * t time points). Entities and other metadata fields are
    # also available, but not very useful at the run level, because they are constant within
    # runs. These become more useful at higher levels.
    variables = layout.get_collections(
        level="run", scan_length=scan_length, **selectors
    )[0]

    # There is currently no way in PyBIDS to update the target index with a non-zero
    # onset, so subtract the shift from the onset of the variables.
    # This must be done prior to convolution.
    if start_time := bold_file.get_metadata().get("StartTime"):
        variables["trial_type"].onset -= start_time

    # Transformations are in-place operations:
    #
    # Factor trial_type into dummy variables
    bids.modeling.transformations.Factor(variables, "trial_type")
    # We now have several columns, glob to convolve all of them
    bids.modeling.transformations.Convolve(
        variables, variables.match_variables("trial_type.*")
    )

    # variables.to_df aligns all variables to a common index,
    # converting sparse variables to dense, if needed.
    # The special value TR is equivalent to `1./bold_file.get_metadata()['RepetitionTime']`.
    df = variables.to_df(sampling_rate="TR")

    # The design matrix is a DataFrame with the columns of interest, nuissance regressors,
    # and an intercept.
    design_matrix = df[
        variables.match_variables(["trial_type.*", "rot_?", "trans_?"])
    ].assign(intercept=1)

    # Rename columns not to start with trial_type. This makes life easier when writing
    # contrasts as combinations of columns.
    design_matrix.columns = [
        col[11:] if col.startswith("trial_type.") else col
        for col in design_matrix.columns
    ]
    logging.debug("Design matrix constructed. Columns: %s", list(design_matrix.columns))

    # Save design matrix
    dmat_fname = bids.layout.writing.build_path(
        {
            "suffix": "design",
            "extension": ".tsv",
            "datatype": "func",
            **selectors,
        },
        path_patterns=stat_patterns,
    )
    Path(dmat_fname).parent.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(dmat_fname, sep="\t", index=False)

    # Use the FirstLevelModel object, which allows us to nest the results in
    # a SecondLevelModel object below. We don't pass a mask, so that activations
    # outside the brain mask will be obvious and could indicate a preprocessing
    # issue.
    model = glm.first_level.FirstLevelModel(
        smoothing_fwhm=5.0,  # FWHM, this should be a parameter
    )
    first_level_models.append(model)

    # model.fit takes a nibabel image or a path. PyBIDS returns PathLike objects
    # (https://docs.python.org/3/library/os.html#os.PathLike), so we do not
    # need to load the data ourselves.
    model.fit(bold_file, design_matrices=design_matrix)

    logging.debug("First level model fit.")

    # Save the effect size and variance of contrasts.
    # This would allow us to run a second-level model without rerunning, if found,
    # but this script currently just recomputes them.
    for contrast in contrasts:
        effect = model.compute_contrast(contrast, output_type="effect_size")
        variance = model.compute_contrast(contrast, output_type="effect_variance")
        contrastValue = "".join(
            ("Vs" if part == "-" else part.capitalize()) for part in contrast.split()
        )
        effect_fname = bids.layout.writing.build_path(
            {
                "suffix": "statmap",
                "extension": ".nii.gz",
                "contrast": contrastValue,
                "stat": "effect",
                **selectors,
            },
            path_patterns=stat_patterns,
        )
        var_fname = bids.layout.writing.build_path(
            {
                "suffix": "statmap",
                "extension": ".nii.gz",
                "contrast": contrastValue,
                "stat": "variance",
                **selectors,
            },
            path_patterns=stat_patterns,
        )
        effect.to_filename(effect_fname)
        variance.to_filename(var_fname)

## If you want to load files with contrast and stat entities, you need a config.
## We could use FitLins' (https://github.com/poldracklab/fitlins/):
# import acres
#
# first_level_layout = bids.BIDSLayout(
#     '.',
#     config=['bids', acres.Loader('fitlins.data').cached('fitlins.json')],
#     validate=False,
# )
# first_level_layout.get(contrast='Motor')  # For example
#
## Instead we'll just use our list of first-level models...

logging.debug("Running second level model.")

second_level = glm.second_level.SecondLevelModel()
second_level.fit(first_level_models)

# This should look familiar from above. Since it's the top-level, we don't need
# any selectors to distinguish one run/session/subject from another.
for contrast in contrasts:
    zscore = second_level.compute_contrast(
        first_level_contrast=contrast, output_type="z_score"
    )
    tdp = glm.cluster_level_inference(zscore, threshold=[1, 2, 3], alpha=0.05)
    contrastValue = "".join(
        ("Vs" if part == "-" else part.capitalize()) for part in contrast.split()
    )
    zscore_fname = bids.layout.writing.build_path(
        {
            "suffix": "statmap",
            "extension": ".nii.gz",
            "contrast": contrastValue,
            "stat": "zscore",
        },
        path_patterns=stat_patterns,
    )
    tdp_fname = bids.layout.writing.build_path(
        {
            "suffix": "statmap",
            "extension": ".nii.gz",
            "contrast": contrastValue,
            "stat": "tdp",
        },
        path_patterns=stat_patterns,
    )
    zscore.to_filename(zscore_fname)
    tdp.to_filename(tdp_fname)

logging.debug("Done!")
