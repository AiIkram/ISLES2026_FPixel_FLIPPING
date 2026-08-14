"""
ISLES'26 algorithm for Grand Challenge — nnU-Net (ResEncUNetL, 3d_fullres) backend,
2-channel input version (T1 + bilateral-asymmetry difference map).

DEBUG/TIMING BUILD:
  - N4 bias-field correction REMOVED (channel 0 = reorientation only). This was
    cut to reclaim time-budget margin after two "Time limit exceeded" failures
    that both died silently inside N4/registration before nnU-Net ever started.
    NOTE: if the model was trained on N4-corrected inputs, this is a train/infer
    distribution mismatch and may reduce accuracy. Re-add n4_bias_correction()
    (still defined below, just unused) if accuracy suffers and there's budget.
  - Registration (asymmetry-diff map) parameters tightened to cap worst-case
    runtime: fewer iterations, looser convergence, one fewer pyramid level,
    lower sampling percentage. This bounds the previously-unbounded cost that
    caused timeouts (registration is an iterative optimizer with no fixed
    runtime — it varies per case).
  - [TIMING] print statements added around every major step so future failures
    are diagnosable from the log instead of guessing.
  - Postprocessing (largest-connected-component etc., from
    nnUNetv2_find_best_configuration) applied after prediction, if
    postprocessing.pkl/plans.json are found under MODEL_DIR. This is optional —
    missing files just skip postprocessing rather than crashing.

Pipeline:
  1) reorientation to a canonical orientation (channel 0)
  2) bilateral flip-register-difference asymmetry map, computed on the
     reoriented image, saved as channel 1 (mirrors compute_asymmetry_diff.py)
  3) run the trained nnU-Net model (Dataset112, 3d_fullres, folds 0-4,
     nnUNetResEncUNetLPlans, 2 input channels)
  4) apply postprocessing (if available) determined via
     nnUNetv2_find_best_configuration on the 5-fold ensemble
  5) map the prediction back onto the original input image's grid/orientation

init_model() — called once at server startup. Loads the nnU-Net predictor.
run(model)   — called once per case. Reads /input, runs inference, writes /output.

Your algorithm's interface (interf0):
  Inputs:
    - /input/images/t1-brain-mri
    - /input/stroke-metadata.json
  Outputs:
    - /output/images/stroke-lesion-segmentation
    - /output/images/lesion-probability-map

To test locally: ./do_test_run.sh
To save for upload to GC: ./do_save.sh
"""

import glob
import json
import os
import tempfile
import time
from pathlib import Path

import numpy
import SimpleITK

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

# Where Grand Challenge (and do_test_run.sh, via the ./model bind mount) makes
# your uploaded model tarball available at runtime.
MODEL_DIR = Path("/opt/ml/model")

# -----------------------------------------------------------------------------
# nnU-Net identifiers — must match what you trained with.
# -----------------------------------------------------------------------------
NNUNET_CONFIGURATION = "3d_fullres"
NNUNET_TRAINER = "nnUNetTrainer"
NNUNET_PLANS = "nnUNetResEncUNetLPlans"
NNUNET_FOLDS = (0, 1, 2, 3, 4)  # confirmed to fit the time budget with N4 removed + trimmed registration
NNUNET_CHECKPOINT = "checkpoint_best.pth"

# -----------------------------------------------------------------------------
# Manual preprocessing to replicate before calling nnU-Net.
# nnU-Net's own predict pipeline (resampling to target spacing, intensity
# normalization, cropping) runs automatically inside the predictor — you do
# NOT need to reimplement that here.
#
# N4 bias field correction is currently DISABLED (see module docstring).
#
# IMPORTANT: set this to whatever orientation code you standardized to during
# training (common choices: "RAS", "LPS"). Double check this against your
# training preprocessing script.
TARGET_ORIENTATION = "RAS"

N4_SHRINK_FACTOR = 4  # unused while N4 is disabled; kept for easy re-enable


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def _find_trained_model_folder():
    """Locate .../nnUNet_results/<Dataset>/<Trainer>__<Plans>__<Config>/ under MODEL_DIR.

    Supports either:
      - MODEL_DIR/nnUNet_results/<Dataset>/<Trainer>__<Plans>__<Config>/
      - MODEL_DIR/<Dataset>/<Trainer>__<Plans>__<Config>/               (flatter tarball layout)
    """
    candidates = list(MODEL_DIR.glob(f"**/*__{NNUNET_PLANS}__{NNUNET_CONFIGURATION}"))
    candidates = [c for c in candidates if c.is_dir() and c.name.startswith(NNUNET_TRAINER)]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a trained model folder matching "
            f"'{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIGURATION}' under {MODEL_DIR}. "
            f"Make sure you copied your full nnUNet_results/<Dataset...>/ folder "
            f"(containing fold_0/, plans.json, dataset.json) into model/nnUNet_results/."
        )
    if len(candidates) > 1:
        print(f"WARNING: multiple candidate model folders found, using the first: {candidates}")
    return candidates[0]


def _find_postprocessing_files():
    """Locate postprocessing.pkl + plans.json produced by
    nnUNetv2_find_best_configuration, from the crossval_results folder.

    Expected layout under MODEL_DIR:
      .../nnUNet_results/Dataset112_miccai/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/
          crossval_results_folds_0_1_2_3_4/postprocessing.pkl
          crossval_results_folds_0_1_2_3_4/plans.json

    Returns (pkl_path, plans_path) or (None, None) if not found. Does not raise —
    postprocessing is treated as optional so a missing file doesn't crash inference,
    it just skips postprocessing (logged clearly either way).
    """
    pkl_candidates = list(MODEL_DIR.glob("**/crossval_results_folds_*/postprocessing.pkl"))
    if not pkl_candidates:
        return None, None
    if len(pkl_candidates) > 1:
        print(f"WARNING: multiple postprocessing.pkl found, using the first: {pkl_candidates}")
    pp_pkl = pkl_candidates[0]
    pp_plans = pp_pkl.parent / "plans.json"
    if not pp_plans.exists():
        print(f"WARNING: found {pp_pkl} but no matching plans.json alongside it")
        return pp_pkl, None
    return pp_pkl, pp_plans


def init_model():
    """Load and return the nnU-Net predictor.

    Called once by app.py during server startup (before /health returns 200).
    Reused across all /invoke calls.
    """
    _show_torch_cuda_info()

    # nnunetv2 reads these env vars at import time in some versions. We don't
    # need real raw/preprocessed data for inference, but the paths must exist.
    os.environ.setdefault("nnUNet_raw", "/tmp/nnUNet_raw")
    os.environ.setdefault("nnUNet_preprocessed", "/tmp/nnUNet_preprocessed")
    os.environ.setdefault("nnUNet_results", str(MODEL_DIR / "nnUNet_results"))
    Path(os.environ["nnUNet_raw"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["nnUNet_preprocessed"]).mkdir(parents=True, exist_ok=True)

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=torch.cuda.is_available(),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    model_training_output_dir = _find_trained_model_folder()
    print(f"Loading nnU-Net model from: {model_training_output_dir}")
    predictor.initialize_from_trained_model_folder(
        str(model_training_output_dir),
        use_folds=NNUNET_FOLDS,
        checkpoint_name=NNUNET_CHECKPOINT,
    )

    return predictor


def run(model):
    """Called each time the /invoke endpoint is called.

    Reads input from /input, runs inference, writes output to /output.
    """
    interface_key = get_interface_key()

    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]

    return handler(model)


def interf0_handler(predictor):
    handler_t0 = time.time()

    # Read the input, keeping the original SimpleITK image to preserve its
    # affine/header for the final output.
    t1_image_path, _ = load_image_file_as_array_and_image(
        location=INPUT_PATH / "images/t1-brain-mri",
    )
    original_t1_image = SimpleITK.ReadImage(t1_image_path)

    input_stroke_metadata = load_json_file(
        location=INPUT_PATH / "stroke-metadata.json",
    )
    # NOTE: the acquisition-site key is 'CENTER' here (was 'SITE' in the
    # training .csv files). DAYS_POST_STROKE / CHRONICITY may be null.
    print("=+=" * 10)
    print("Loaded Stroke Metadata:")
    print(json.dumps(input_stroke_metadata, indent=2))
    print("=+=" * 10)

    # -------------------------------------------------------------------
    # 1) Reorientation ONLY (N4 disabled — see module docstring).
    #    This becomes channel 0 ("_0000") of the model's input.
    # -------------------------------------------------------------------
    t0 = time.time()
    channel0_image = original_t1_image
    if TARGET_ORIENTATION is not None:
        channel0_image = SimpleITK.DICOMOrient(channel0_image, TARGET_ORIENTATION)
    print(f"[TIMING] Reorientation took {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------
    # 2) Bilateral-asymmetry difference map, computed on channel 0.
    #    This becomes channel 1 ("_0001").
    # -------------------------------------------------------------------
    t1 = time.time()
    channel1_image = compute_asymmetry_diff_map(channel0_image)
    print(f"[TIMING] Asymmetry map (registration) took {time.time() - t1:.1f}s")

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        nnunet_input_dir = tmp_dir / "input"
        nnunet_output_dir = tmp_dir / "output"
        nnunet_input_dir.mkdir()
        nnunet_output_dir.mkdir()

        case_id = "case"
        input_file_0000 = nnunet_input_dir / f"{case_id}_0000.nii.gz"
        input_file_0001 = nnunet_input_dir / f"{case_id}_0001.nii.gz"
        SimpleITK.WriteImage(channel0_image, str(input_file_0000), useCompression=True)
        SimpleITK.WriteImage(channel1_image, str(input_file_0001), useCompression=True)

        # -----------------------------------------------------------
        # 3) Run the actual trained nnU-Net model on both channels.
        #    Channel order here MUST match training: 0000 = T1, 0001 = asym-diff.
        # -----------------------------------------------------------
        t2 = time.time()
        predictor.predict_from_files(
            [[str(input_file_0000), str(input_file_0001)]],
            str(nnunet_output_dir),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        print(f"[TIMING] nnU-Net predict_from_files took {time.time() - t2:.1f}s")

        # -----------------------------------------------------------
        # 4) Apply postprocessing determined during training
        #    (nnUNetv2_find_best_configuration -> postprocessing.pkl),
        #    from the 5-fold crossval_results_folds_0_1_2_3_4/ folder.
        # -----------------------------------------------------------
        t_pp = time.time()
        pp_pkl, pp_plans_json = _find_postprocessing_files()
        if pp_pkl is not None and pp_plans_json is not None:
            from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
            from batchgenerators.utilities.file_and_folder_operations import load_pickle, load_json

            pp_fns_and_kwargs = load_pickle(str(pp_pkl))
            pp_plans = load_json(str(pp_plans_json))

            postprocessed_dir = tmp_dir / "postprocessed"
            postprocessed_dir.mkdir()

            # NOTE: apply_postprocessing_to_folder's exact kwarg names vary across
            # nnunetv2 versions (some want plans_manager + dataset_json_file_or_dict
            # instead of plans_file_or_dict). Run:
            #   import inspect; from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
            #   print(inspect.signature(apply_postprocessing_to_folder))
            # against your installed version and adjust the call below if it doesn't match.
            apply_postprocessing_to_folder(
                str(nnunet_output_dir),
                str(postprocessed_dir),
                pp_fns_and_kwargs[0],
                pp_fns_and_kwargs[1],
                plans_file_or_dict=pp_plans,
                num_processes=1,  # low on purpose — GC inference containers have limited
                                  # CPU; -np 8 from local CLI runs was for offline batch
                                  # processing, not appropriate for a single-case container.
            )
            print(f"[TIMING] Postprocessing took {time.time() - t_pp:.1f}s")
            # IMPORTANT: apply_postprocessing_to_folder only writes the postprocessed
            # SEGMENTATION (case.nii.gz) — it does not touch/copy the probability .npz.
            # So the segmentation comes from postprocessed_dir, but the probability
            # map still has to be read from the original (pre-postprocessing) output.
            predicted_seg_path = postprocessed_dir / f"{case_id}.nii.gz"
            predicted_npz_path = nnunet_output_dir / f"{case_id}.npz"
        else:
            print("[TIMING] No postprocessing.pkl/plans.json found under MODEL_DIR — skipping postprocessing")
            predicted_seg_path = nnunet_output_dir / f"{case_id}.nii.gz"
            predicted_npz_path = nnunet_output_dir / f"{case_id}.npz"

        predicted_seg_image = SimpleITK.ReadImage(str(predicted_seg_path))
        probability_map_data = load_lesion_probability_map(predicted_npz_path)

    probability_map_image = SimpleITK.GetImageFromArray(probability_map_data.astype(numpy.float32))
    probability_map_image.CopyInformation(predicted_seg_image)

    # -------------------------------------------------------------------
    # 5) Map prediction back onto the ORIGINAL input image's grid.
    #    Reorientation/resampling preserves physical space, so resampling
    #    with an identity transform against the original image as reference
    #    correctly undoes step (1)'s reorientation and any spacing changes
    #    nnU-Net applied internally.
    # -------------------------------------------------------------------
    t3 = time.time()
    segmentation_in_original_space = SimpleITK.Resample(
        predicted_seg_image,
        original_t1_image,
        SimpleITK.Transform(),
        SimpleITK.sitkNearestNeighbor,
        0,
        predicted_seg_image.GetPixelID(),
    )
    probability_in_original_space = SimpleITK.Resample(
        probability_map_image,
        original_t1_image,
        SimpleITK.Transform(),
        SimpleITK.sitkLinear,
        0.0,
        SimpleITK.sitkFloat32,
    )

    binary_segmentation_mask = SimpleITK.GetArrayFromImage(segmentation_in_original_space).astype(numpy.uint8)
    probability_map_data = SimpleITK.GetArrayFromImage(probability_in_original_space).astype(numpy.float32)

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=binary_segmentation_mask,
        reference_image=original_t1_image,
    )
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/lesion-probability-map",
        array=probability_map_data,
        reference_image=original_t1_image,
    )
    print(f"[TIMING] Resample-back + write outputs took {time.time() - t3:.1f}s")
    print(f"[TIMING] TOTAL interf0_handler took {time.time() - handler_t0:.1f}s")

    return 0


def n4_bias_correction(image):
    """Apply N4 bias field correction, matching the preprocessing used before training.

    NOT CURRENTLY CALLED — disabled to reclaim time-budget margin (see module
    docstring). Kept here so it's a one-line change to re-enable if accuracy
    without N4 turns out to be unacceptable and there's time budget for it
    (e.g. after a time-limit increase, or after further trimming registration).

    Uses an Otsu-thresholded brain/head mask (standard default when no explicit
    brain mask is available) and a shrink factor to keep runtime reasonable.
    """
    image_float = SimpleITK.Cast(image, SimpleITK.sitkFloat32)

    mask = SimpleITK.OtsuThreshold(image_float, 0, 1, 200)

    shrunk_image = SimpleITK.Shrink(image_float, [N4_SHRINK_FACTOR] * image_float.GetDimension())
    shrunk_mask = SimpleITK.Shrink(mask, [N4_SHRINK_FACTOR] * mask.GetDimension())

    corrector = SimpleITK.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(shrunk_image, shrunk_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image_float)
    corrected_image = image_float / SimpleITK.Exp(log_bias_field)
    corrected_image = SimpleITK.Cast(corrected_image, image.GetPixelID())
    corrected_image.CopyInformation(image)
    return corrected_image


# =============================================================================
# Bilateral-asymmetry difference map (channel 1), ported in-process from
# compute_asymmetry_diff.py. Operates on `channel0_image` (reoriented, N4 no
# longer applied) as the "original".
#
# Registration parameters below are TIGHTENED from the original script to cap
# worst-case runtime (this optimizer's cost is per-case variable — it doesn't
# have a fixed runtime, which is what caused the timeouts). If accuracy proves
# sensitive to these, loosen gradually rather than reverting to the original
# 300-iteration / 3-level / 0.3-sampling settings, which had no margin.
# =============================================================================

def _reorient_to_ras(image):
    """Reorient image to canonical RAS so axis 0 = Left-Right."""
    return SimpleITK.DICOMOrient(image, "RAS")


def _mirror_flip_lr(image_ras):
    """Mirror image content along the Left-Right axis (axis 0 in RAS).

    Only pixel content is flipped; origin/spacing/direction stay the same, so
    the result occupies the SAME physical space, mirrored — exactly what's
    needed before registering back to the original.
    """
    return SimpleITK.Flip(image_ras, [True, False, False])


def _rigid_register(fixed, moving):
    """Rigid (6 DOF) registration of `moving` (flipped image) onto `fixed`.

    Mattes Mutual Information is used because near a lesion, flipped vs.
    original intensities may not correspond in a simple linear way, which MI
    handles better than an intensity-difference-based metric.
    """
    reg = SimpleITK.ImageRegistrationMethod()

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)  # was 0.3 — fewer samples per iteration

    reg.SetInterpolator(SimpleITK.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-3,               # was 1e-4 — looser convergence, stops sooner
        numberOfIterations=150,     # was 300 — hard cap on worst-case cost
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])       # was [4, 2, 1] — dropped finest level
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1])   # was [2, 1, 0]
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial_transform = SimpleITK.CenteredTransformInitializer(
        fixed,
        moving,
        SimpleITK.Euler3DTransform(),
        SimpleITK.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(
        SimpleITK.Cast(fixed, SimpleITK.sitkFloat32),
        SimpleITK.Cast(moving, SimpleITK.sitkFloat32),
    )
    return final_transform


def compute_asymmetry_diff_map(preprocessed_image, mask_image=None):
    """Compute the absolute flip-register-difference asymmetry map.

    `preprocessed_image` is treated as the "original" (reoriented; N4 no
    longer applied at this point).

    Steps: reorient a working copy to RAS -> mirror-flip L-R -> rigidly
    register the flipped copy back onto `preprocessed_image` -> resample
    onto `preprocessed_image`'s grid -> absolute difference -> (optional)
    brain-mask -> force identical header/geometry to `preprocessed_image`.
    """
    original = SimpleITK.Cast(preprocessed_image, SimpleITK.sitkFloat32)

    original_ras = _reorient_to_ras(original)
    flipped_ras = _mirror_flip_lr(original_ras)

    transform = _rigid_register(fixed=original, moving=flipped_ras)

    registered_flip = SimpleITK.Resample(
        flipped_ras,
        original,
        transform,
        SimpleITK.sitkLinear,
        0.0,
        original.GetPixelID(),
    )
    registered_flip.CopyInformation(original)

    diff_abs = SimpleITK.Abs(SimpleITK.Subtract(original, registered_flip))

    if mask_image is not None:
        mask = SimpleITK.Resample(
            mask_image, original, SimpleITK.Transform(), SimpleITK.sitkNearestNeighbor, 0
        )
        diff_abs = SimpleITK.Mask(diff_abs, mask)

    diff_abs.CopyInformation(original)
    return diff_abs


def load_lesion_probability_map(npz_path):
    """Extract the foreground-lesion probability map from nnU-Net's saved .npz.

    nnU-Net saves per-class probabilities under the 'probabilities' key, shaped
    (num_classes, Z, Y, X). Assumes a binary task with class 1 = lesion.
    """
    with numpy.load(str(npz_path)) as data:
        if "probabilities" in data:
            probabilities = data["probabilities"]
        else:
            # Fallback for older/renamed keys.
            probabilities = data[list(data.keys())[0]]

    if probabilities.ndim == 4:
        return probabilities[1]  # class 1 = lesion, class 0 = background
    return probabilities


def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array_and_image(*, location):
    input_files = (
        glob.glob(str(location / "*.mha"))
        + glob.glob(str(location / "*.nii.gz"))
        + glob.glob(str(location / "*.nii"))
    )
    if not input_files:
        raise FileNotFoundError(f"No valid image file found in {location}")

    image_path = input_files[0]
    image = SimpleITK.ReadImage(image_path)
    return image_path, SimpleITK.GetArrayFromImage(image)


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"

    image = SimpleITK.GetImageFromArray(array)

    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())

    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
