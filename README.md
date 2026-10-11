# FPixel — ISLES’26 Stroke Lesion Segmentation

**From Baseline to Bilateral Asymmetry: A Systematic nnU-Net Comparison for Ischemic Stroke Lesion Segmentation in ISLES’26**  
**Challenge result:** Rank #17

[Paper](https://papers.miccai.org/miccai-2026-sat/ISLES_028.html) · [PDF](https://papers.miccai.org/miccai-2026-sat/paper/ISLES_028.pdf) · [OpenReview](https://openreview.net/forum?id=Hs8smEehRx) · [ISLES’26 challenge rules](https://isles-26.grand-challenge.org/challenge-rules/)

## About the project

This repository contains the inference code for our ISLES’26 submission, along with the custom nnU-Net trainer and loss code used by the project. The main idea explored in the paper is to give the segmentation model an additional view of the MRI: a **bilateral asymmetry map** made by mirroring the T1-weighted image, registering the mirrored image to the original, and taking the absolute intensity difference.

The model uses the original T1-weighted MRI and this difference map as two input channels. The repository also includes a script for generating the asymmetry map separately.

Trained model weights and medical-image data are not included in this release.

## How inference works

1. The input T1-weighted MRI is loaded and reoriented to RAS orientation.
2. A left–right mirrored version is registered to the original image.
3. The absolute difference between the registered image and the original is saved as the asymmetry channel.
4. The two channels are passed to the trained nnU-Net model.
5. The predicted lesion segmentation and lesion-probability map are written back on the input image grid.

## Repository contents

```text
.
├── app.py
├── inference.py
├── preprocessing/
│   ├── README.md
│   └── compute_asymmetry_diff.py
├── custom_trainers/
│   ├── nnUNetTrainerDiceFocal.py
│   ├── compound_losses.py
│   └── README.md
├── requirements.txt
├── Dockerfile
├── do_build.sh
├── do_test_run.sh
├── do_save.sh
├── CITATION.cff
└── LICENSE
```

- `inference.py` contains the preprocessing and prediction pipeline.
- `app.py` provides the HTTP interface used by the Grand Challenge container.
- `preprocessing/compute_asymmetry_diff.py` generates the asymmetry channel for a directory of images.
- `custom_trainers/` contains the custom trainer and loss components needed by the model configuration.

## Running the container

You will need Docker, a Linux environment targeting `linux/amd64`, the required model files, and a compatible input case. A supported NVIDIA GPU runtime is recommended for testing inference performance.

### 1. Add the model files

The trained checkpoints are not part of this repository. Provide the model export in the layout expected by `inference.py`, or configure the model archive through Grand Challenge. The code expects the trainer, plans, and configuration names to match those used for the submitted model.

Please do not add private patient scans, credentials, or model files that you are not permitted to redistribute.

### 2. Build the image

```bash
./do_build.sh
```

### 3. Test locally

Place a valid test case in the structure expected by `do_test_run.sh`, for example:

```text
test/input/interf0/
├── stroke-metadata.json
└── images/
    └── t1-brain-mri/
        └── <input MRI file>
```

Then run:

```bash
./do_test_run.sh
```

The script starts the container, checks the service, runs inference, and collects the outputs under `test/output/`. A successful run depends on having the required model files and a valid test case; these are not included here.

### 4. Export for Grand Challenge

After testing the container, export it with:

```bash
./do_save.sh
```

Then follow the [Grand Challenge algorithm documentation](https://grand-challenge.org/documentation/create-your-own-algorithm/) to upload the container and model archive.

## Generate the asymmetry channel separately

The standalone script uses SimpleITK and NumPy. Install them in your Python environment:

```bash
python -m pip install SimpleITK numpy
```

Run it on a directory containing nnU-Net images named like `<case_id>_0000.nii.gz`:

```bash
python preprocessing/compute_asymmetry_diff.py \
  --images_dir /path/to/nnUNet_raw/DatasetXXX/imagesTr
```

The script writes the corresponding `<case_id>_0001.nii.gz` files beside the original images. Existing outputs are skipped, and the original `_0000.nii.gz` files are not overwritten. You can optionally pass `--masks_dir` for a directory of masks named `<case_id>_mask.nii.gz`.

### A note on preprocessing settings

The standalone script and the challenge inference pipeline do not use identical registration settings. Inference uses reduced registration settings to keep processing within the challenge time limit, and N4 bias-field correction is currently disabled there. If you retrain the model or regenerate its input channels, make sure the preprocessing settings match those used for training and record any changes. Different settings can change the resulting asymmetry map.

## Outputs

The container produces:

- `images/stroke-lesion-segmentation` — the predicted lesion mask
- `images/lesion-probability-map` — the predicted foreground lesion probabilities

Both outputs are mapped to the original input image grid.

## Notes on reproduction

- **Weights and data:** neither trained checkpoints nor challenge data are distributed here. You need authorized access to the model and a valid ISLES’26 case to run inference.
- **Runtime:** registration and 3D segmentation can be computationally expensive. CPU-only runs may be slow and do not reflect expected GPU performance.
- **Model compatibility:** the checkpoint, trainer, plans, configuration, channel order, and preprocessing must agree.
- **Postprocessing:** the inference code skips postprocessing when the expected postprocessing files are not available.

This repository shares the implementation of our challenge submission. It should not be interpreted as evidence of independent clinical validation or as a medical device.

## Citation

If this code is useful for your research, please cite the paper:

```bibtex
@inproceedings{
  zayim2026from,
  title     = {From Baseline to Bilateral Asymmetry: A Systematic nnU-Net Comparison for Ischemic Stroke Lesion Segmentation in {ISLES}'26},
  author    = {Beyza Zayim and Ikram Aissiou and Stephan Collins and Alain Lalande and Fabrice Meriaudeau},
  booktitle = {Stroke and Neurovascular Diseases Workshop on Imaging and Treatment Challenges},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Hs8smEehRx}
}
```

GitHub also reads the `CITATION.cff` file and can use it to show a **Cite this repository** option.

## License

The code is released under the Apache License 2.0; see [LICENSE](LICENSE). Before publishing, make sure that every included file can be redistributed under this license and that any third-party code or assets retain their required notices. The model weights and dataset are not included. Please check the [official ISLES’26 rules](https://isles-26.grand-challenge.org/challenge-rules/) for the challenge’s repository and sharing requirements.

## Acknowledgements

Thanks to the ISLES’26 organizers, the nnU-Net authors, and the developers of the open-source medical-imaging libraries used in this project.
