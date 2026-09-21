# Real-IAD single-view PatchCore protocol

## Scope

Real-IAD `transistor1` and `plastic_nut` are registered as new external categories:

- `realiad_transistor1` is a multi-pin surface-mount integrated circuit, not the three-lead MVTec transistor.
- `realiad_plastic_nut` is not the MVTec metal nut.

Their results must not be presented as same-product holdout validation of the existing MVTec models.

## Frozen data contract

- Source release: Real-IAD 256-pixel archives.
- Annotation variant: official `realiad_jsons_sv` single-view split.
- Camera view: C1 only.
- Training input: entries labelled `OK` by the official view-level annotation.
- Calibration: 20% of official normal training images, selected by the SHA-256 rank of `42:path`.
- Test: the untouched official single-view test entries.
- Pixel masks: used only for anomalous test entries with an official mask.
- Pixel mask binarization: values greater than or equal to 128 are anomalous. The 256-pixel release contains grayscale boundary values introduced by resizing.
- An official anomaly mask that becomes empty at the fixed threshold remains valid for image-level evaluation and is recorded as a pixel-label warning.
- Image and mask files are checksum-locked in the generated manifests.

An image may reside under an `NG` source directory while the single-view JSON labels it `OK`. This means the selected C1 view has no visible annotated anomaly; the JSON label, not the source directory name, is authoritative.

## Frozen model contract

- Model: PatchCore
- Backbone: pretrained `wide_resnet50_2`
- Layers: `layer2`, `layer3`
- Input: 256 x 256 RGB
- Coreset ratio: 10%
- Nearest neighbours: 9
- Seeds: 42, 43, 44
- Image threshold: higher 95th percentile of the normal calibration scores
- Pixel threshold: higher 99.5th percentile of pooled normal calibration-map pixels

All six fits and calibration thresholds must finish before the evaluation script is run.

## Windows order

1. Run `scripts\prepare_realiad.ps1` to verify archive hashes, extract only referenced C1 files, and generate manifests.
2. Run `scripts\run_realiad_patchcore_fits.ps1` to complete all six fits without loading test images.
3. Review the console output.
4. Run `scripts\run_realiad_patchcore_evaluations.ps1` once to evaluate the frozen test splits and build the handoff ZIP.
