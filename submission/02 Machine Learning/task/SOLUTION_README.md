# BeeX Task 2: Sonar Segmentation Training Notes

## Objective

I trained the provided semantic segmentation model to segment:

- `anode`
- `member`
- background for everything else

I kept the model family as `UnetPlusPlus('resnet34', classes=3)`. I changed the training process to fix data, loss, metric, and reproducibility issues rather than spending excessive time on model training.

## Dataset Insights

The dataset is COCO 1.1 style at `interview_dataset/annotations/instances_default.json`.

I observed the following from the provided archive:

- Images: `1111`
- Annotations: `1363`
- Categories: `anode` and `member`
- `anode` annotations: `112`
- `member` annotations: `1251`
- Background-only images: `115`
- Image-level class presence:
  - `884` images contain only `member`
  - `112` images contain both `anode` and `member`
  - `115` images contain no target annotation

The main dataset challenge I saw is imbalance: `anode` is much rarer and smaller than `member`. The images are also wide sonar frames, so center-cropping can discard useful context.

In my first CPU run, the model learned `member` but predicted zero anode pixels. I checked the split and confirmed that validation did contain anode images, so the failure came from pixel-level imbalance rather than a missing validation class. After resizing to `192 x 192`, validation had only `346` anode pixels across `222` images.

## Changes Made

- I rewrote COCO parsing around image IDs instead of assuming `image_id - 1`.
- I build semantic masks as class indices:
  - `0 = background`
  - `1 = anode`
  - `2 = member`
- I draw `member` before `anode` so rare anode pixels are preserved if polygons overlap.
- I use aspect-preserving letterbox resize instead of center crop.
- I resize masks with nearest-neighbor interpolation.
- I keep image interpolation separate from mask interpolation.
- I use ImageNet normalization for the `resnet34` encoder.
- I keep the required Unet++/ResNet34 model type, but use logits with no final activation.
- I use weighted cross entropy plus soft Dice loss.
- I skip absent foreground classes in Dice loss so batches without anode no longer reward suppressing the anode channel.
- I add a small anode-specific Dice term that only activates when a batch contains anode pixels.
- I use a weighted sampler to show anode-present images more often during training.
- I boost the anode cross-entropy class weight moderately, with CLI flags to tune the recall/false-positive trade-off.
- I add optional train-only sonar augmentation for hidden-test robustness.
- I keep augmentation conservative: horizontal flips, mild affine jitter, intensity/gamma jitter, light noise, and mild blur.
- I reject geometric augmentation if it erases tiny anode masks or drops too much foreground.
- I stratify the train/validation/test split by image class presence when possible.
- I report IoU and F1 separately for `anode` and `member`.
- I save target/predicted pixel counts so zero-anode predictions are easy to detect.
- I save reproducible artifacts: loss curve, metrics JSON/text, and best checkpoint.

## Setup

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-cpu.txt
```

To rerun training, place the provided `interview_dataset.zip` in this folder. The script will extract it to `interview_dataset/` if the extracted dataset is not already present.

I use `requirements-cpu.txt` for CPU-only PyTorch wheels during smoke tests and correctness checks. This avoids pulling the very large CUDA/NVIDIA wheel stack when disk space is limited.

If I want GPU-specific PyTorch wheels, I use the official PyTorch install command for the target CUDA version, then install the non-Torch dependencies:

```bash
pip install -r requirements.txt
```

If a previous broad PyTorch install ran out of disk space, I clear pip's cache and remove partial CUDA packages before retrying:

```bash
pip cache purge
pip freeze | grep -E '^(nvidia-|cuda-|torch|torchvision|triton)' | cut -d= -f1 | xargs -r pip uninstall -y
```

## Smoke Test

I use a tiny run first to prove parsing, training, validation, metrics, and artifact writing:

```bash
python task1_ml.py --smoke --epochs 1 --batch-size 2 --image-size 224 --no-pretrained --output-dir outputs_smoke
```

I expect these artifacts:

- `outputs_smoke/loss_curves.png`
- `outputs_smoke/metrics.json`
- `outputs_smoke/metrics.txt`
- `outputs_smoke/best_model.pt`

## Unit Tests

I added lightweight unit tests for the data pipeline and metrics without adding a new test framework:

```bash
python -B -m unittest test_task1_ml.py
```

These tests cover polygon mask priority, letterbox mask resizing, stratified splitting, Dice-loss edge cases, per-class metrics, and augmentation mask safety. They do not train a model.

## Modest Training Run

The assignment explicitly discourages excessive training. I use validation to select the checkpoint and report final metrics on the test split. A reasonable reproducible CPU run is:

```bash
python task1_ml.py --epochs 5 --batch-size 2 --image-size 224 --no-pretrained --num-workers 0 --output-dir outputs_cpu_anode
```

For a faster anode-focused check, I use a balanced subset:

```bash
python task1_ml.py --epochs 3 --batch-size 2 --image-size 192 --no-pretrained --max-samples 256 --num-workers 0 --output-dir outputs_anode_subset
```

If ImageNet weights are available locally or can be downloaded, I omit `--no-pretrained`; that should help the tiny `anode` class more than training a ResNet34 encoder from scratch.

## Hidden-Test Robustness

For underwater sonar, I do recommend data augmentation. The hidden dataset may differ in brightness, contrast, speckle/noise, slight vehicle pose, and left-right presentation. I only apply augmentation to the training split so validation and test metrics remain clean.

I use this command for a modest augmented experiment:

```bash
python task1_ml.py --epochs 5 --batch-size 2 --image-size 224 --no-pretrained --num-workers 0 --augment --output-dir outputs_cpu_anode_aug
```

The default augmentation is intentionally mild because anode masks are rare and tiny after resizing:

- horizontal flip probability: `0.5`
- affine probability: `0.35`
- max rotation: `3` degrees
- scale jitter: `0.04`
- translation jitter: `0.03`
- intensity jitter probability: `0.5`
- noise probability: `0.3`
- blur probability: `0.2`

I avoid vertical flips, large rotations, random crops, MixUp/CutMix, and aggressive blur because they can break sonar geometry or remove the anode target entirely.

My augmented CPU run completed with best validation checkpoint at epoch `5`:

- validation `anode IoU=0.3556`, `anode F1=0.5247`
- validation `member IoU=0.6140`, `member F1=0.7608`
- test `anode IoU=0.4775`, `anode F1=0.6464`
- test `member IoU=0.5811`, `member F1=0.7351`
- test predicted anode pixels: `500`, with `388` target anode pixels

Compared with my non-augmented run, augmentation improved held-out `member` IoU slightly but reduced held-out `anode` IoU on this visible split. I would keep the non-augmented checkpoint as my best visible-test result, and keep the augmented run as a robustness experiment for possible hidden-set intensity/pose shift.

## Metrics

I report assignment metrics for the two target classes only:

- `anode` IoU and F1
- `member` IoU and F1

The script writes metrics to `metrics.txt` and `metrics.json`. I use background during training and confusion-matrix computation, but I do not include it in the final reported assignment metrics. The JSON includes `validation_metrics`, `test_metrics`, `validation_pixel_counts`, and `test_pixel_counts`.

In my full CPU run at `224 x 224`, anode was no longer collapsed to zero. The best checkpoint was selected at validation epoch `4`, then evaluated on the held-out test split:

- validation `anode IoU=0.3849`, `anode F1=0.5558`
- validation `member IoU=0.6040`, `member F1=0.7531`
- test `anode IoU=0.5570`, `anode F1=0.7155`
- test `member IoU=0.5694`, `member F1=0.7256`
- test predicted anode pixels: `336`, with `388` target anode pixels

This run used `--no-pretrained`, so the result is still conservative. With cached ImageNet encoder weights, I would expect the tiny anode class to improve further.

For final visible-split reporting, I would use the non-augmented `outputs_cpu_anode` checkpoint because it gives the stronger anode result. For a hidden test set with sonar appearance shift, I would also evaluate `outputs_cpu_anode_aug` because it trades some visible anode precision/IoU for more train-time variation.

## Maintainability Notes

- `load_coco_records()` and `draw_polygon_mask()` own COCO parsing and mask rasterization.
- `letterbox_image_and_mask()` preserves wide sonar context while keeping class masks discrete.
- `split_records()` owns deterministic train/validation/test splitting.
- `soft_dice_loss()`, `binary_dice_loss()`, and class/sample weighting handle the rare anode class.
- `apply_train_augmentation()` is train-only and is disabled unless `--augment` is passed.
- `test_task1_ml.py` covers the pure data/metric behavior without running training.
- The submitted primary artifacts are in `outputs_cpu_anode`; I did not include `.venv`, the dataset archive, or exploratory output folders.

## LLM Use

I used LLM tools, including OpenAI Codex/ChatGPT-style assistance, under my direction while completing this task. I provided the ML goals, training constraints, debugging observations, experiment results, and review feedback. I reviewed plans and code changes frequently, gave corrections where needed, and approved the direction before moving on. I reviewed and verified the submitted code, metrics, and selected artifacts before packaging.

## Notes

I focused on correctness and reproducibility of the ML pipeline:

- correct train/validation pairing
- correct mask interpolation
- correct multi-class loss
- class imbalance handling
- separate target-class metrics
