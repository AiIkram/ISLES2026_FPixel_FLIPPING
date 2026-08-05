# Custom nnU-Net trainer classes

nnU-Net's `nnUNetPredictor` needs to *import* the actual trainer class used
during training (it defines more than just the loss — network config, deep
supervision, etc.), even though the loss function itself plays no role at
inference time.

`nnUNetTrainerDiceFocal` is a custom trainer, so it isn't part of the
`nnunetv2` pip package installed in the container. Drop the exact `.py` file
you used for training here (plus any helper file it imports, e.g. a custom
loss class), keeping the class name matching the folder name in your
nnUNet_results checkpoint exactly:

    custom_trainers/
        nnUNetTrainerDiceFocal.py

`inference.py` sets `nnUNet_extTrainer=/opt/app/custom_trainers` so nnU-Net's
`recursive_find_trainer_class_by_name` searches this folder when it can't
find the trainer in its own package — this is a first-class nnU-Net
extension point (see `nnunetv2/paths.py` / `nnunetv2/utilities/find_objects.py`),
not a hack.

Multiple custom trainers can go here side by side.
