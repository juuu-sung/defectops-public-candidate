"""Decode integrity checks for all files, quality descriptors for normal development data only."""

import hashlib
import importlib.metadata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from defectops.manifest import summary, verify_manifest


def read_image(path: Path, expected_formats: set[str]) -> Image.Image:
    with Image.open(path) as image:
        if image.format not in expected_formats:
            raise ValueError(f"Expected {'/'.join(sorted(expected_formats))} encoding")
        image.verify()
    with Image.open(path) as image:
        image.load()
        return image.copy()


def read_png(path: Path) -> Image.Image:
    """Backward-compatible strict PNG decoder used for masks and MVTec images."""
    return read_image(path, {"PNG"})


def normal_quality(image: Image.Image) -> dict:
    """Native-resolution luminance and 4-neighbor Laplacian, excluding the border."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    if min(gray.shape) < 3:
        raise ValueError("Image must be at least 3x3 for quality descriptors")
    laplacian = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
                 + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return {"brightness_mean": float(gray.mean()), "brightness_std": float(gray.std()),
            "laplacian_variance": float(laplacian.var())}


def distribution(values: list[float]) -> dict:
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    [round(float(v), 6) for v in np.quantile(values, [0, .05, .5, .95, 1])]))


def audit_dataset(root: Path, manifest: dict) -> dict:
    verify_manifest(root, manifest)
    root = root.resolve()
    errors = []
    warnings = []
    image_specs = Counter()
    mask_specs = Counter()
    types = Counter()
    decoded_hashes = {}
    qualities = defaultdict(list)
    images_decoded = masks_decoded = 0
    pixel_duplicates = []
    expected_paths = {r["image_path"] for r in manifest["records"]}
    expected_paths.update(r["mask_path"] for r in manifest["records"] if r["mask_path"])
    is_real_iad = manifest.get("dataset") in {"real_iad", "real_iad_followup"}
    if manifest.get("dataset") == "real_iad":
        inventory_roots = [root / "real_iad" / "images" / manifest["source_object"]]
        expected_image_formats = {"JPEG"}
    elif manifest.get("dataset") == "real_iad_followup":
        source_objects = {record.get("source_object") for record in manifest["records"]}
        if len(source_objects) != 1 or None in source_objects:
            raise ValueError("Real-IAD follow-up manifest must contain one source object")
        inventory_roots = [
            root / "real_iad_followup" / "images" / next(iter(source_objects))
        ]
        expected_image_formats = {"JPEG"}
    else:
        inventory_roots = [root / category for category in manifest["categories"]]
        expected_image_formats = {"PNG"}
    for inventory_root in inventory_roots:
        for path in inventory_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                if path.relative_to(root).as_posix() not in expected_paths:
                    errors.append({"path": path.relative_to(root).as_posix(), "error": "Image not in manifest"})
    for record in manifest["records"]:
        rel = record["image_path"]
        types[f"{record['category']}/{record['split']}/{record['defect_type']}"] += 1
        image = None
        try:
            image = read_image(root / rel, expected_image_formats)
            images_decoded += 1
            image_specs[f"{image.width}x{image.height}/{image.mode}"] += 1
            if image.mode not in {"RGB", "L"}:
                raise ValueError(f"Unsupported source image mode: {image.mode}")
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + b":" + rgb.tobytes()).hexdigest()
            if pixel_hash in decoded_hashes:
                pixel_duplicates.append([decoded_hashes[pixel_hash], rel])
                errors.append({"path": rel, "error": "Duplicate decoded image pixels"})
            else:
                decoded_hashes[pixel_hash] = rel
            if record["split"] in {"train", "calibration"}:
                qualities[f"{record['category']}/{record['split']}"].append(
                    {"image_path": rel, **normal_quality(image)})
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            errors.append({"path": rel, "error": str(exc)})
        if record["mask_path"]:
            mask_rel = record["mask_path"]
            try:
                mask = read_png(root / mask_rel)
                masks_decoded += 1
                mask_specs[f"{mask.width}x{mask.height}/{mask.mode}"] += 1
                if mask.mode not in {"1", "L"}:
                    raise ValueError(f"Expected single-channel mask, got {mask.mode}")
                pixels = np.asarray(mask.convert("L"))
                if is_real_iad:
                    binary_pixels = pixels >= 128
                    if not binary_pixels.any():
                        warnings.append({
                            "path": mask_rel,
                            "warning": (
                                "Official 256-pixel Real-IAD anomaly mask is empty after the "
                                "frozen >=128 threshold; image label is retained"
                            ),
                        })
                else:
                    if not set(np.unique(pixels).tolist()) <= {0, 255}:
                        raise ValueError("Mask values must be binary 0/255")
                    if not pixels.any():
                        raise ValueError("Anomalous image has an empty mask")
                if image is not None and mask.size != image.size:
                    raise ValueError(f"Mask/image size mismatch: {mask.size} vs {image.size}")
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                errors.append({"path": mask_rel, "error": str(exc)})
    quality_summary = {}
    for key, rows in sorted(qualities.items()):
        quality_summary[key] = {"count": len(rows), **{
            field: distribution([row[field] for row in rows])
            for field in ("brightness_mean", "brightness_std", "laplacian_variance")}}
    return {"schema_version": 1, "status": "passed" if not errors else "failed",
            "dataset": manifest.get("dataset"), "categories": manifest.get("categories"),
            **summary(manifest), "defect_counts": dict(sorted(types.items())),
            "images_decoded": images_decoded, "masks_decoded": masks_decoded,
            "image_specs": dict(image_specs), "mask_specs": dict(mask_specs),
            "decoded_pixel_duplicates": pixel_duplicates, "errors": errors,
            "warnings": warnings,
            "quality_summary": quality_summary, "normal_quality_rows": dict(qualities),
            "quality_method": "Pillow L, native resolution, float64 4-neighbor Laplacian variance, interior only",
            "limitations": ["Test data decoded for integrity only; no test quality statistics or previews.",
                            "No semantic inspection of defect labels or mask boundaries.",
                            "Quality descriptors are not rejection thresholds or model anomaly scores.",
                            "Checksums freeze local files; original download authenticity is not established."],
            "versions": {p: importlib.metadata.version(p) for p in ("Pillow", "numpy")}}


def markdown_report(report: dict) -> str:
    lines = ["# 데이터 준비 결과", "", f"점검 상태: **{report['status']}**", "",
             "실제 로컬 파일을 검사한 결과다. 모델 학습·성능 평가 결과는 아니다.", "",
             f"- 이미지 디코딩: {report['images_decoded']}장",
             f"- 결함 마스크 디코딩: {report['masks_decoded']}장",
             f"- 오류: {len(report['errors'])}건",
             f"- 경고: {len(report.get('warnings', []))}건",
             f"- 디코딩 픽셀 중복: {len(report['decoded_pixel_duplicates'])}쌍", "",
             "## 고정된 데이터 목록", "", "| 구분 | 장수 |", "|---|---:|"]
    for key, count in report["counts"].items():
        lines.append(f"| {key} | {count} |")
    lines += ["", "목록만 나눴으며 원본 파일을 이동·수정하지 않았다.", "",
              "## 이미지와 마스크 형식", ""]
    for title, field in (("이미지", "image_specs"), ("마스크", "mask_specs")):
        for spec, count in report[field].items():
            lines.append(f"- {title}: {spec}, {count}장")
    lines += ["", "## 유형별 수량", "", "| 목록/유형 | 장수 |", "|---|---:|"]
    for key, count in report["defect_counts"].items():
        lines.append(f"| {key} | {count} |")
    lines += ["", "## 정상 데이터 밝기·선명도 기초 점검", "",
              "테스트는 무결성만 확인했다. 아래 통계는 train/calibration만 사용한다.", "",
              "밝기는 0–255 회색조 평균, 선명도 지표는 원본 해상도의 Laplacian 분산이다.",
              "이는 기술 통계이며 불량 점수나 품질 거절 기준이 아니다.", "",
              "| 목록 | 지표 | 최소 | p05 | 중앙값 | p95 | 최대 |", "|---|---|---:|---:|---:|---:|---:|"]
    for split, stats in report["quality_summary"].items():
        for field in ("brightness_mean", "brightness_std", "laplacian_variance"):
            values = stats[field]
            lines.append(f"| {split} | {field} | " + " | ".join(f"{values[k]:.3f}" for k in
                         ("min", "p05", "median", "p95", "max")) + " |")
    lines += ["", "## 재현 정보", "", f"- Dataset: `{report.get('dataset', 'unknown')}`",
              f"- Manifest SHA-256: `{report['manifest_sha256']}`",
              f"- Pillow: {report['versions']['Pillow']}, NumPy: {report['versions']['numpy']}",
              "- 분할 seed 42, 정상 80/20, calibration 수 올림", "",
              "## 한계와 다음 단계", "",
              "파일 디코딩·마스크 이진값·크기·대응 관계를 점검했으나, 마스크가 실제 결함 경계를",
              "정확하게 표시하는지는 판단하지 않았다. 체크섬은 로컬 데이터 버전을 고정하며 공식 원본 인증은 아니다.",
              "정상 calibration 표본 수는 위 분할 표에 기록했다. 95% 분위수는 초기 경험적 운영점이며 오탐률 5%를 보장하지 않는다.",
              (
                  "다음은 고정된 Real-IAD loader·전처리 규칙으로 PatchCore fit/calibration을 실행하는 단계다."
                  if report.get("dataset") in {"real_iad", "real_iad_followup"}
                  else "다음은 데이터 loader·전처리 규칙·PaDiM 학습 연결과 장비별 ML 환경 고정이다."
              )]
    if report["errors"]:
        lines += ["", "## 발견한 오류", ""]
        lines += [f"- {e['path']}: {e['error']}" for e in report["errors"]]
    if report.get("warnings"):
        lines += ["", "## 데이터 경고", ""]
        lines += [f"- {e['path']}: {e['warning']}" for e in report["warnings"]]
    return "\n".join(lines) + "\n"
