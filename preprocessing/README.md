# Bilateral-asymmetry preprocessing

`compute_asymmetry_diff.py` creates an absolute-difference image from a left–right mirrored T1-weighted MRI registered to the original image. The output is intended as nnU-Net channel `0001`; the original image remains channel `0000` and is not overwritten.

See the repository root `README.md` for command-line usage and the note on the difference between standalone registration settings and the runtime-constrained challenge inference implementation.
