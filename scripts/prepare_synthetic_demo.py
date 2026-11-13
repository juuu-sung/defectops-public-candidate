#!/usr/bin/env python3
"""Build a self-contained synthetic UI fixture; never run or load an ML model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from defectops.review_store import ReviewStore


def prepare(output: Path) -> Path:
    # A new folder is mandatory: never overwrite existing opinions or a real database.
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / 'synthetic-transistor-001'
    bundle.mkdir()
    image = Image.new('RGB', (640, 300), '#e9eef4')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    draw.rectangle((0, 0, 639, 36), fill='#172b43')
    draw.text((18, 8), 'SYNTHETIC INPUT - NO REAL PRODUCT', font=font, fill='white')
    for x in (210, 320, 430):
        draw.ellipse((x-15, 247, x+15, 277), fill='#8195a9')
    draw.rounded_rectangle((200, 64, 440, 174), radius=14, fill='#26384b')
    draw.text((255, 103), 'DEMO PART', font=font, fill='white')
    draw.line([(245, 174), (245, 202), (180, 229), (180, 264)], fill='#77899c', width=13)
    draw.line([(320, 174), (320, 262)], fill='#77899c', width=13)
    draw.line([(395, 174), (395, 202), (430, 230), (430, 262)], fill='#77899c', width=13)
    image.save(bundle/'input.png')
    yy, xx = np.mgrid[:300, :640]
    anomaly_map = np.exp(-((xx-184)**2/1100 + (yy-232)**2/900)).astype(np.float32)
    heat = np.zeros((300, 640, 3), dtype=np.uint8)
    heat[:, :, 0] = (255*anomaly_map).astype(np.uint8)
    heat[:, :, 2] = (90*(1-anomaly_map)).astype(np.uint8)
    Image.fromarray(heat).save(bundle/'heatmap.png')
    alpha = (anomaly_map*.60)[:, :, None]
    pixels = np.array(image).astype(float)*(1-alpha) + np.array([255, 85, 50])*alpha
    overlay = Image.fromarray(pixels.astype(np.uint8))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((147, 190, 253, 283), outline='#e75d3b', width=3)
    overlay_draw.rectangle((0, 0, 639, 36), fill='#172b43')
    overlay_draw.text((18, 8), 'SYNTHETIC MAP - NOT MODEL INFERENCE', font=font, fill='white')
    overlay.save(bundle/'overlay.png')
    np.savez_compressed(bundle/'anomaly-map.npz', anomaly_map=anomaly_map)
    artifacts = {}
    for key, name in [('input','input.png'), ('overlay','overlay.png'),
                      ('heatmap','heatmap.png'), ('anomaly_map','anomaly-map.npz')]:
        artifacts[key] = name
        artifacts[key+'_sha256'] = hashlib.sha256((bundle/name).read_bytes()).hexdigest()
    record = dict(schema_version=1, inspection_id=bundle.name, status='completed',
                  source_kind='SYNTHETIC_DEMO', model_inference_performed=False,
                  demo_notice='Generated shapes, map and scores; no real product or model inference.',
                  category='transistor', model='patchcore',
                  comparison_status='exploratory_after_padim_test_observed',
                  decision='REVIEW_REQUIRED', decision_rule='score >= threshold',
                  raw_score=0.8, image_threshold=0.5,
                  automatic_pass=False, automatic_pass_allowed=False,
                  input={'sha256': artifacts['input_sha256']}, artifacts=artifacts,
                  trace={'fit_run_id': 'synthetic-ui-fixture-no-model',
                         'checkpoint_sha256': hashlib.sha256(b'NO CHECKPOINT: synthetic fixture').hexdigest(),
                         'manifest_sha256': hashlib.sha256(b'NO DATASET: generated shapes').hexdigest()})
    (bundle/'inspection.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    db = output/'reviews.sqlite3'
    store = ReviewStore(db)
    store.initialize()
    store.register(bundle)
    return db


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/demo/synthetic-review-v1')
    args = parser.parse_args()
    db = prepare(args.output)
    print(f'Synthetic fixture ready; no inference performed. Demo DB: {db}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
