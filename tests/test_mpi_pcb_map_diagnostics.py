import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts import analyze_mpi_pcb_maps as diagnostic


def png(pixels, mode=None):
    stream = io.BytesIO()
    Image.fromarray(pixels, mode=mode).save(stream, format="PNG")
    return stream.getvalue()


def region_fixture(gt_hot=True):
    values = np.ones((256, 256), dtype=np.float32)
    mask = np.zeros(values.shape, dtype=np.bool_)
    mask[:2, :2] = True
    values[mask if gt_hot else ~mask] = 10
    score = diagnostic.candidate_image_score(0, values, diagnostic.CANDIDATE)
    return values, mask, score


class RegionStatisticsTests(unittest.TestCase):
    def test_hot_small_gt_has_high_rank_but_diluted_score_share(self):
        values, mask, score = region_fixture()
        result = diagnostic.region_statistics(values, mask, score)
        gt = result['gt']
        self.assertEqual(result['top1pct_pixels'], 656)
        self.assertEqual(gt['pixels'], 4)
        self.assertEqual(gt['within_region_rank_auc'], 1)
        self.assertEqual(gt['top1pct_overlap_pixels_min'], 4)
        self.assertEqual(gt['top1pct_overlap_pixels_max'], 4)
        self.assertAlmostEqual(gt['top1pct_score_share_min'], 40 / (40 + 652))
        self.assertEqual(gt['bbox_xyxy_exclusive'], [0, 0, 2, 2])
        self.assertTrue(gt['smaller_than_top1pct_pool'])

    def test_no_gt_response_is_separate_from_dilution(self):
        values, mask, score = region_fixture(False)
        result = diagnostic.region_statistics(values, mask, score)['gt']
        self.assertEqual(result['within_region_rank_auc'], 0)
        self.assertEqual(result['top1pct_overlap_pixels_max'], 0)
        self.assertEqual(result['top1pct_score_share_max'], 0)

    def test_cutoff_ties_report_bounds_not_arbitrary_pixel_selection(self):
        values = np.ones((256, 256), dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.bool_)
        mask.flat[:1000] = True
        result = diagnostic.region_statistics(values, mask, 1)['gt']
        self.assertEqual(result['within_region_rank_auc'], .5)
        self.assertEqual(result['top1pct_overlap_pixels_min'], 0)
        self.assertEqual(result['top1pct_overlap_pixels_max'], 656)
        self.assertEqual(result['top1pct_score_share_min'], 0)
        self.assertEqual(result['top1pct_score_share_max'], 1)

    def test_empty_full_unavailable_masks_and_zero_scores(self):
        values = np.zeros((256, 256), dtype=np.float32)
        for mask in (np.zeros_like(values, dtype=bool), np.ones_like(values, dtype=bool), None):
            with self.subTest(mask=None if mask is None else bool(mask.all())):
                result = diagnostic.region_statistics(values, mask, 0)
                if mask is None:
                    self.assertFalse(result['gt_available'])
                    self.assertIsNone(result['gt'])
                else:
                    self.assertIsNone(result['gt']['within_region_rank_auc'])
                    self.assertIsNone(result['gt']['top1pct_score_share_max'])
                json.dumps(result, allow_nan=False)

    def test_invalid_maps_masks_scores_fail(self):
        values, mask, score = region_fixture()
        for kind in ('nan', 'negative', 'shape', 'mask_shape', 'mask_dtype', 'score', 'infinite_score'):
            with self.subTest(kind=kind):
                a, m, s = values.copy(), mask.copy(), score
                if kind == 'nan': a[0, 0] = np.nan
                if kind == 'negative': a[0, 0] = -1
                if kind == 'shape': a = a[:128]
                if kind == 'mask_shape': m = m[:128]
                if kind == 'mask_dtype': m = m.astype(np.uint8)
                if kind == 'score': s += 0.1
                if kind == 'infinite_score': s = float('inf')
                with self.assertRaises(ValueError):
                    diagnostic.region_statistics(a, m, s)

    def test_missing_gt_excludes_entire_photo_from_location_statistics(self):
        values, mask, score = region_fixture()
        photo = {'score': score, 'threshold': 2., 'decision': 'BELOW_THRESHOLD_NOT_AUTO_PASS',
                 'regions': [{'score': score} for _ in range(12)]}
        result = diagnostic.describe_photo(photo, np.stack([values]*12), [mask]*11+[None], False)
        self.assertIsNone(result['highest_region_gt_any'])
        self.assertIsNone(result['best_gt_region_score'])
        self.assertTrue(all(r['gt'] is None for r in result['regions']))

    def test_normal_region_can_dominate_despite_gt_response(self):
        values, mask, score = region_fixture()
        maps = np.stack([values] * 12)
        maps[1] = 20
        masks = [np.zeros((256, 256), dtype=bool) for _ in range(12)]
        masks[0] = mask
        photo = {'score': 20., 'threshold': 30., 'decision': 'BELOW_THRESHOLD_NOT_AUTO_PASS',
                 'regions': [{'score': 20. if i == 1 else score} for i in range(12)]}
        result = diagnostic.describe_photo(photo, maps, masks, True)
        self.assertFalse(result['highest_region_gt_any'])
        self.assertEqual(result['highest_regions'], [diagnostic.REGIONS[1]])
        self.assertAlmostEqual(result['maximum_minus_best_gt_region_score'], 20-score)
        self.assertFalse(result['all_gt_regions_have_zero_top1pct_overlap'])


class EvidenceTests(unittest.TestCase):
    def test_map_hash_shape_order_pickle_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in ('valid', 'hash', 'shape', 'order', 'dtype', 'nan', 'negative', 'object', 'extras'):
                with self.subTest(kind=kind):
                    values = np.zeros((12, 256, 256), dtype=np.float32)
                    regions = np.array(diagnostic.REGIONS)
                    if kind == 'shape': values = values[:11]
                    if kind == 'order': regions = regions[::-1]
                    if kind == 'dtype': values = values.astype(np.float64)
                    if kind == 'nan': values[0, 0, 0] = np.nan
                    if kind == 'negative': values[0, 0, 0] = -1
                    if kind == 'object': regions = regions.astype(object)
                    path = root / (kind + '.npz')
                    np.savez_compressed(path, anomaly_maps=values, regions=regions,
                                        **({'extra': np.zeros(1)} if kind == 'extras' else {}))
                    photo = {'key': 'good/0000', 'map_file': path.name,
                             'map_sha256': diagnostic.sha256_file(path) if kind != 'hash' else 'wrong'}
                    if kind == 'valid':
                        self.assertEqual(diagnostic.load_maps(root, photo).shape, (12, 256, 256))
                    else:
                        with self.assertRaises(ValueError): diagnostic.load_maps(root, photo)
            child = root / 'child'
            child.mkdir()
            with self.assertRaises(ValueError):
                diagnostic.load_maps(child, {'map_file': '../valid.npz'})

    def test_source_image_mask_crc_and_pixels_are_rechecked(self):
        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0, 0] = 1  # Native 0/1, not >=128.
        raw, raw_mask = png(pixels), png(mask)
        row = {'member': 'image.png', 'member_sha256': diagnostic.digest(raw),
               'decoded_rgb_sha256': diagnostic.digest(pixels.tobytes()),
               'mask_member': 'mask.png', 'mask_sha256': diagnostic.digest(raw_mask),
               'mask': diagnostic.mask_details(raw_mask)}
        source = mock.Mock()
        source.read.side_effect = lambda name: raw if name == 'image.png' else raw_mask
        images, masks = diagnostic.read_photo(source, {'regions': [row]}, b'')
        self.assertEqual(masks[0].sum(), 1)
        self.assertEqual(images[0].size, (256, 256))
        for field in ('member_sha256', 'decoded_rgb_sha256', 'mask_sha256'):
            with self.subTest(field=field):
                bad = {**row, field: 'wrong'}
                with self.assertRaises(ValueError): diagnostic.read_photo(source, {'regions': [bad]}, b'')

    def test_only_known_broken_mask_can_be_unavailable(self):
        pixels = np.zeros((256, 256, 3), dtype=np.uint8)
        raw, broken = png(pixels), b'broken mask'
        row = {'member': 'image.png', 'member_sha256': diagnostic.digest(raw),
               'decoded_rgb_sha256': diagnostic.digest(pixels.tobytes()),
               'mask_member': diagnostic.MASK_MEMBER, 'mask_sha256': diagnostic.digest(broken)}
        source = mock.Mock()
        source.read.side_effect = lambda name: raw if name == 'image.png' else broken
        with mock.patch.object(diagnostic, 'MASK_SHA256', diagnostic.digest(broken)):
            _, masks = diagnostic.read_photo(source, {'regions': [row]}, b'')
        self.assertIsNone(masks[0])
        row['mask_member'] = 'unexpected-broken-mask.png'
        with self.assertRaises((ValueError, OSError)):
            diagnostic.read_photo(source, {'regions': [row]}, b'')

    def test_wrong_handoff_fails_before_reading_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'wrong.zip'
            path.write_bytes(b'not a zip')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                diagnostic.load_evidence(Path(temp), path)


class WorkflowTests(unittest.TestCase):
    def test_groups(self):
        for label in (0, 1):
            groups = []
            for first, second in ((False, False), (False, True), (True, False), (True, True)):
                photos = {key: {'label': label, 'decision': 'REVIEW_REQUIRED' if flag else 'BELOW_THRESHOLD_NOT_AUTO_PASS'}
                          for key, flag in zip(diagnostic.MODEL_KEYS, (first, second))}
                groups.append(diagnostic.comparison_group(photos))
            self.assertEqual(groups, ['both_missed','visa_only_caught','realiad_only_caught','both_caught'] if label
                             else ['both_below_threshold','visa_only_false_alarm','realiad_only_false_alarm','both_false_alarm'])

    def test_cli_argument_mapping(self):
        with mock.patch.object(diagnostic, 'run', return_value={'status': 'synthetic'}) as run:
            self.assertEqual(diagnostic.main(['--project', 'C:/project', '--archive', 'C:/data.zip']), 0)
            run.assert_called_once_with(Path('C:/project'), Path('C:/data.zip'))
        with mock.patch.object(diagnostic, 'run', side_effect=ValueError('invalid')):
            self.assertEqual(diagnostic.main(['--archive', 'C:/data.zip']), 1)

    def test_publish_manifest_and_never_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage, output, handoff = root/'stage', root/'out', root/'result.zip'
            stage.mkdir()
            (stage/'analysis.json').write_text('{}')
            (stage/'cards').mkdir()
            (stage/'cards/card.jpg').write_bytes(b'synthetic card')
            diagnostic.publish(stage, output, handoff)
            with zipfile.ZipFile(handoff) as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read('handoff-manifest.json'))
                for item in manifest['members']:
                    self.assertEqual(diagnostic.digest(archive.read(item['name'])), item['sha256'])
            before = handoff.read_bytes()
            with self.assertRaises(FileExistsError): diagnostic.publish(stage, output, handoff)
            self.assertEqual(handoff.read_bytes(), before)

    def test_existing_output_fails_before_any_input_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'artifacts/analyses'/diagnostic.NAME).mkdir(parents=True)
            with mock.patch.object(diagnostic, 'load_evidence') as load:
                with self.assertRaises(FileExistsError): diagnostic.run(root, root/'missing.zip')
                load.assert_not_called()

    def test_full_cohort_packaging_without_gpu_or_network(self):
        """Synthetic I/O boundary smoke; real scalar math, summaries, publish and JSON."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evaluation = root/'artifacts/evaluations'/diagnostic.EVALUATION_NAME
            (evaluation/'recovery').mkdir(parents=True)
            pixels = np.zeros((256, 256, 3), dtype=np.uint8)
            raw = png(pixels)
            (evaluation/'recovery/grid1-3-defect-0054.png').write_bytes(raw)
            source = root/'source.zip'
            with zipfile.ZipFile(source, 'w') as writer: writer.writestr('synthetic', 'unchanged')
            reports = {'evaluation.json': {'archive_sha256': diagnostic.sha256_file(source)},
                       'recovery.json': {'recovered_png_sha256': diagnostic.digest(raw)},
                       'cohort-amendment.json': {'photo_keys': [], 'localization_ineligible_photos': ['defect/0006']},
                       'test-audit.json': {'photos': []}}
            for key in diagnostic.MODEL_KEYS: reports[key+'-scores.json'] = {'photos': []}
            for label, count in ((0, 54), (1, 55)):
                for i in range(count):
                    key = f"{'defect' if label else 'good'}/{i:04d}"
                    reports['cohort-amendment.json']['photo_keys'].append(key)
                    reports['test-audit.json']['photos'].append({'key':key, 'label':label})
                    for model in diagnostic.MODEL_KEYS:
                        flagged = label == 0 and (i < 2 if model == diagnostic.MODEL_KEYS[0] else 2 <= i < 6)
                        score = 10. if flagged else 1.
                        reports[model+'-scores.json']['photos'].append({'key':key,'label':label,'score':score,
                            'threshold':5.,'decision':'REVIEW_REQUIRED' if flagged else 'BELOW_THRESHOLD_NOT_AUTO_PASS',
                            'regions':[{'score':score} for _ in range(12)]})
            masks = [np.zeros((256,256), dtype=bool) for _ in range(12)]
            masks[0][0,0] = True
            before = source.read_bytes()
            with mock.patch.object(diagnostic, 'load_evidence', return_value=reports), \
                 mock.patch.object(diagnostic, 'REPAIRED_RGB_SHA256', diagnostic.digest(pixels.tobytes())), \
                 mock.patch.object(diagnostic, 'read_photo', return_value=([Image.fromarray(pixels)]*12,masks)), \
                 mock.patch.object(diagnostic, 'load_maps', side_effect=lambda _,p:np.full((12,256,256),p['score'],dtype=np.float32)), \
                 mock.patch.object(diagnostic, 'render_page', return_value=Image.new('RGB',(20,20))), \
                 mock.patch.object(diagnostic, 'git_state', return_value={'commit':'synthetic'}):
                result = diagnostic.run(root, source)
            self.assertEqual(result['map_files_verified'],218)
            self.assertEqual(result['review_card_photos'],61)
            self.assertEqual(result['summary']['models']['realiad_pcb']['all_defects']['gt_photo_denominator'],54)
            self.assertEqual(source.read_bytes(),before)
            with zipfile.ZipFile(result['handoff']) as archive:
                analysis=json.loads(archive.read('analysis.json'))
                self.assertFalse(analysis['inference_performed'])
                self.assertFalse(analysis['resolution_causality_identified'])
                self.assertFalse(any(n.endswith(('.npz','.pt')) for n in archive.namelist()))
                self.assertEqual(len([n for n in archive.namelist() if n.endswith('.jpg')]),122)


if __name__ == '__main__':
    unittest.main()
