"""Map-only precision controls; never change the backbone or live map module."""
from contextlib import contextmanager
import copy
import hashlib

import numpy as np


ATOL = 1e-9
RTOL = 1e-10


@contextmanager
def map_tf32(torch, enabled):
    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


def distances_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4096,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid stable-map distances')
    return values


def kernel_array(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    if (kernel.shape != (33, 33) or not np.isfinite(kernel).all() or np.any(kernel < 0)
            or abs(kernel.sum()-1) > 1e-6):
        raise ValueError('Expected normalized positive 33x33 installed Gaussian kernel')
    return kernel


def numpy_reference(distances, kernel):
    """Independent float64 FFT cross-correlation, nearest4x, reflect16.

    Uses the EXACT installed kernel, no regeneration or renormalization.
    Reversal implements conv2d cross-correlation even for asymmetric test kernels.
    """
    values = distances_array(distances).reshape(64, 64)
    kernel = kernel_array(kernel)
    expanded = np.repeat(np.repeat(values, 4, axis=0), 4, axis=1)
    padded = np.pad(expanded, 16, mode='reflect')
    spectrum = np.fft.rfft2(padded, s=(320, 320))
    spectrum *= np.fft.rfft2(kernel[::-1, ::-1], s=(320, 320))
    result = np.fft.irfft2(spectrum, s=(320, 320))[32:288, 32:288].copy()
    # FFT roundoff can create tiny negative values around exact zeros.
    if result.min() < -ATOL:
        raise ValueError('Negative independent reference beyond roundoff')
    return np.maximum(result, 0)


def compare(actual, expected, enforce=False):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != (256, 256) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid stable-map comparison')
    diff = np.abs(a-b)
    outside = int(np.sum(diff > ATOL+RTOL*np.abs(b)))
    if enforce and (outside or np.any(a < 0)):
        pos = np.unravel_index(int(diff.argmax()), diff.shape)
        raise ValueError('CPU float64 installed map differs from independent reference: '
                         f'max={diff.max():.12g}, outside={outside}, yx={tuple(map(int, pos))}, '
                         f'actual={a[pos]:.12g}, reference={b[pos]:.12g}, minimum={a.min():.12g}')
    top = lambda x: float(np.partition(x.ravel(), -656)[-656:].mean())
    return {'maximum_absolute_difference': float(diff.max()), 'outside_float64_tolerance': outside,
            'top656_a': top(a), 'top656_b': top(b), 'top656_delta': top(a)-top(b)}


class StableMaps:
    def __init__(self, runtime):
        self.runtime = runtime
        self.torch = runtime.torch
        live = runtime.model.model.anomaly_map_generator
        blur = live.blur
        if (blur.border_type != 'reflect' or blur.padding != 'same' or blur.channels != 1
                or list(blur.padding_shape) != [16, 16, 16, 16]):
            raise ValueError('Installed map geometry differs')
        self.kernel = kernel_array(blur.kernel.detach().cpu().numpy().reshape(33, 33)).copy()
        self.live = live
        self.module = copy.deepcopy(live).to(device='cpu', dtype=self.torch.float64).eval()
        self.initial_tf32 = self.torch.backends.cudnn.allow_tf32
        self.meta = {'dtype': 'float64', 'device': 'cpu', 'interpolation': 'nearest64to256',
                     'border': 'reflect16', 'kernel_sha256': hashlib.sha256(self.kernel.tobytes()).hexdigest(),
                     'kernel_sum': float(self.kernel.sum()), 'atol': ATOL, 'rtol': RTOL,
                     'original_cudnn_allow_tf32': self.initial_tf32}

    def check_unchanged(self):
        if (self.torch.backends.cudnn.allow_tf32 != self.initial_tf32
                or not np.array_equal(self.kernel, self.live.blur.kernel.detach().cpu().numpy().reshape(33, 33))):
            raise ValueError('Live map kernel or TF32 setting mutated')

    def stable(self, distances):
        values = distances_array(distances)
        with self.torch.inference_mode():
            tensor = self.torch.as_tensor(values.copy().reshape(1, 1, 64, 64), dtype=self.torch.float64, device='cpu')
            result = self.module(tensor, (256, 256)).detach().numpy().reshape(256, 256).copy()
        compare(result, numpy_reference(values, self.kernel), enforce=True)
        self.check_unchanged()
        return result

    def gate(self, native, double, saved_map):
        """Only map calls happen under the temporary GPU precision setting."""
        arrays, report = {}, {}
        for name, values in [('native', native), ('double', double)]:
            for enabled in (True, False):
                with map_tf32(self.torch, enabled):
                    result = self.runtime.maps(values)
                arrays[f'{name}-gpu-tf32-{str(enabled).lower()}.npy'] = result
            stable = self.stable(values)
            arrays[f'{name}-cpu-float64.npy'] = stable
            report[name] = {
                'gpu_tf32_on_vs_cpu64': compare(arrays[f'{name}-gpu-tf32-true.npy'], stable),
                'gpu_tf32_off_vs_cpu64': compare(arrays[f'{name}-gpu-tf32-false.npy'], stable),
                'cpu64_vs_reference': compare(stable, numpy_reference(values, self.kernel), enforce=True),
            }
        replay = arrays['native-gpu-tf32-true.npy']
        delta = float(np.max(np.abs(replay.astype(np.float64)-saved_map)))
        if not np.isfinite(delta) or delta > 1e-4:
            raise ValueError(f'Official stored-distance map replay differs: max={delta:.12g}, limit=0.0001')
        report['official_replay_maximum_difference'] = delta
        report['cross_precision_diagnostic_only'] = compare(saved_map, arrays['double-cpu-float64.npy'])
        self.check_unchanged()
        return report, arrays
