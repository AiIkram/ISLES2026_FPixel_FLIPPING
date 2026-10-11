"""
compute_asymmetry_diff.py

Computes a bilateral-asymmetry ("flip - register - difference") map from
T1w brain MRI, intended as a second nnU-Net input channel alongside the
original image.

This version ALWAYS saves the ABSOLUTE difference map only, as
"<case_id>_0001.nii.gz" — the original image ("<case_id>_0000.nii.gz")
is never touched or overwritten by this script.

Pipeline per case:
    1. Load original T1w image.
    2. Reorient a working copy to canonical RAS orientation so "flip
       along the left-right axis" is well defined regardless of how the
       input NIfTI was originally stored.
    3. Mirror-flip that canonical image along the L-R axis.
    4. Rigidly register the flipped image back onto the ORIGINAL image
       (registration works in physical/world space, correcting for the
       reorientation and any head tilt/misalignment).
    5. Resample the registered flipped image onto the original image's
       grid (this is the only resampling step - required to bring the
       transform's result onto the original voxel grid before any
       subtraction can happen).
    6. Compute the ABSOLUTE difference between the original and the
       registered-flipped image (no further resampling needed).
    
    7. Save the abs-diff map as "<case_id>_0001.nii.gz", with the EXACT
       same header/affine as the original image, so it lines up
       voxel-for-voxel (required by nnU-Net).
"""

import argparse
import os
import glob
import SimpleITK as sitk
import numpy as np


def reorient_to_ras(image: sitk.Image) -> sitk.Image:
    """Reorient image to canonical RAS so axis 0 = Left-Right."""
    return sitk.DICOMOrient(image, "RAS")


# def mirror_flip_lr(image_ras: sitk.Image) -> sitk.Image:
#     """
#     Mirror the image content along the Left-Right axis (axis 0 in RAS).
#     Only pixel content is flipped; origin/spacing/direction stay the
#     same, so the result occupies the SAME physical space, mirrored -
#     exactly what we need before registering back to the original.
#     """
#     return sitk.Flip(image_ras, [True, False, False])
def mirror_flip_lr(image_ras: sitk.Image) -> sitk.Image:
    """
    Mirror the image content along the Left-Right axis (axis 0 in RAS/ axis 2 in numpy).
    By extracting to numpy and forcing the original geometry back onto it, 
    we physically swap the hemispheres in world space.
    """
    # SimpleITK arrays are in (Z, Y, X) order
    arr = sitk.GetArrayFromImage(image_ras)
    
    # Flip along the X axis (Left-Right)
    arr_flipped = np.flip(arr, axis=2)
    
    # Convert back to SimpleITK image
    flipped_image = sitk.GetImageFromArray(arr_flipped)
    
    # CRITICAL: Copy the original geometry so the flipped pixels occupy the original grid
    flipped_image.CopyInformation(image_ras)
    
    return flipped_image

def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """
    Rigid (6 DOF) registration of `moving` (flipped image) onto `fixed`
    (original image). Mattes Mutual Information is used because near a
    lesion, flipped vs. original intensities may not correspond in a
    simple linear way, which MI handles better than an intensity-
    difference-based metric.
    """
    reg = sitk.ImageRegistrationMethod()

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3)

    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution for robustness
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    return final_transform
def affine_register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """
    Affine (12 DOF) registration of `moving` (flipped image) onto `fixed`
    (original image). Mattes Mutual Information is used because near a
    lesion, flipped vs. original intensities may not correspond in a
    simple linear way. Affine adds scaling and shearing to better handle
    global mass effect while preserving local lesion differences.
    """
    reg = sitk.ImageRegistrationMethod()

    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.3)

    reg.SetInterpolator(sitk.sitkLinear)

    # Lowered learning rate to 1.0 for Affine optimizer stability
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=300,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution for robustness
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Changed to AffineTransform(3) and MOMENTS initialization
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(
        sitk.Cast(fixed, sitk.sitkFloat32),
        sitk.Cast(moving, sitk.sitkFloat32),
    )
    return final_transform

def compute_asymmetry_diff_abs(
    original_path: str,
    output_path: str,
    mask_path: str = None,
) -> None:
    """
    Full pipeline for one case. Saves the ABSOLUTE asymmetry-diff map to
    `output_path` (expected to be "<case_id>_0001.nii.gz"), in the same
    space/header as `original_path`. The original image itself is only
    ever read, never modified or re-saved.
    """
    original = sitk.ReadImage(original_path, sitk.sitkFloat32)

    # 1-3: reorient + flip
    original_ras = reorient_to_ras(original)
    flipped_ras = mirror_flip_lr(original_ras)

    # 4: register flipped -> original (native/original space & orientation)
    transform = rigid_register(fixed=original, moving=flipped_ras)

    # 5: resample flipped image onto the ORIGINAL grid (only resample call)
    registered_flip = sitk.Resample(
        flipped_ras,
        original,               # reference grid = original image
        transform,
        sitk.sitkLinear,
        0.0,
        original.GetPixelID(),
    )
    registered_flip.CopyInformation(original)

    # 6: absolute difference only (no resampling needed here)
    diff_abs = sitk.Abs(sitk.Subtract(original, registered_flip))

    # 7: optional brain-mask to suppress skull/background artifacts
    if mask_path is not None:
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        mask = sitk.Resample(
            mask, original, sitk.Transform(), sitk.sitkNearestNeighbor, 0
        )
        diff_abs = sitk.Mask(diff_abs, mask)

    # 8: force identical header/geometry to the original image, then save
    diff_abs.CopyInformation(original)
    sitk.WriteImage(diff_abs, output_path)


def batch_process(
    images_dir: str,
    masks_dir: str = None,
    original_suffix: str = "_0000.nii.gz",
    output_suffix: str = "_0001.nii.gz",
):
    """
    Scans `images_dir` for files ending in `original_suffix`
    (e.g. case_001_0000.nii.gz), and writes the matching absolute
    asymmetry-diff map as case_001_0001.nii.gz next to it.
    The original "_0000" files are never overwritten - only read.

    If `masks_dir` is given, expects a brain mask per case named
    "<case_id>_mask.nii.gz" (adjust the pattern to your naming as needed).
    """
    originals = sorted(glob.glob(os.path.join(images_dir, f"*{original_suffix}")))
    if not originals:
        print(f"No files matching *{original_suffix} found in {images_dir}")
        return

    for orig_path in originals:
        case_id = os.path.basename(orig_path)[: -len(original_suffix)]
        out_path = os.path.join(images_dir, f"{case_id}{output_suffix}")
        if os.path.exists(out_path):
            print(f"[{case_id}] already done, skipping ({out_path} exists)")
            continue
        mask_path = None
        if masks_dir is not None:
            candidate = os.path.join(masks_dir, f"{case_id}_mask.nii.gz")
            if os.path.exists(candidate):
                mask_path = candidate

        print(f"[{case_id}] registering + computing absolute asymmetry diff ...")
        try:
            compute_asymmetry_diff_abs(
                original_path=orig_path,
                output_path=out_path,
                mask_path=mask_path,
            )
            print(f"[{case_id}] saved -> {out_path}")
        except Exception as e:
            print(f"[{case_id}] FAILED: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute absolute flip-register-difference asymmetry "
        "maps for nnU-Net, saved as <case_id>_0001.nii.gz."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Folder containing case_XXX_0000.nii.gz original images "
        "(e.g. nnUNet imagesTr/imagesTs folder).",
    )
    parser.add_argument(
        "--masks_dir",
        default=None,
        help="Optional folder containing brain masks named "
        "<case_id>_mask.nii.gz, used to zero-out non-brain diff noise.",
    )
    args = parser.parse_args()

    batch_process(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
    )
    
# pip install SimpleITK --break-system-packages

# python compute_asymmetry_diff.py \
#     --images_dir /path/to/nnUNet_raw/Dataset001_StrokeLesion/imagesTr 

# Example background run (replace the path with your own dataset directory):
# nohup python preprocessing/compute_asymmetry_diff.py --images_dir /path/to/nnUNet_raw/DatasetXXX/imagesTr > asymmetry.log 2>&1 &
