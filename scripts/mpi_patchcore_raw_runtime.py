"""Native-feature global greedy; CPU corpus, bounded synchronous CUDA transfers."""
from __future__ import annotations

import gc
import json
import time
import numpy as np

from scripts import mpi_patchcore_raw_bank as b
from scripts.mpi_patchcore_balanced_runtime import Runtime as Base
p, fit = b.p, b.fit


def load_training(paths, check, rows=12288, dim=768):
    """Preserve photo/region/patch order without a second full corpus allocation."""
    features = np.empty((len(paths)*rows, dim), dtype=np.float32)
    for i, path in enumerate(paths):
        check()
        block = p.array(path, (rows, dim))
        features[i*rows:(i+1)*rows] = block
        del block
    check()
    return features


def update_streamed(torch, features, minimum, center_index, chunk, check=lambda:None):
    """Same L2 recurrence as resident selection, but the CPU matrix is never uploaded whole."""
    if chunk < 1:
        raise ValueError('Positive chunk required')
    center = torch.from_numpy(features[center_index]).to('cuda')
    for a,z in p.probe.slices(len(features), chunk):
        check()
        block = torch.from_numpy(features[a:z]).to('cuda')
        distances = torch.linalg.norm(block-center, ord=2, dim=1, keepdim=True)
        torch.minimum(minimum[a:z], distances, out=minimum[a:z])
        del block, distances
    index = torch.max(minimum.squeeze(1), dim=0).indices
    minimum[index,0] = 0.
    return int(index.item())


def select_raw(torch, root, signature, features_factory, n, maximum, start, check,
                  interval=1000, chunk=65536, preview=lambda x:None):
    """Global native-feature recurrence; CPU corpus and bounded GPU transfers."""
    root.mkdir(exist_ok=True)
    if not 0 <= start < n or not 0 < maximum <= n:
        raise ValueError('Invalid selection size/start')
    steps = p.checkpoint_steps(root, signature, n, maximum, interval)
    if steps and steps[0][1]['pending_center'] != start:
        raise ValueError('Selection initialization differs')
    if steps and steps[-1][1]['step'] == maximum:
        return steps[-1][1]['selected'], [x[2] for x in steps]
    features = minimum = None
    try:
        check()
        cpu_features = features_factory()
        if cpu_features.shape[0] != n or cpu_features.dtype != np.float32 or not np.isfinite(cpu_features).all():
            raise ValueError('Invalid region features')
        features = cpu_features
        del cpu_features
        if steps:
            name, progress, receipt = steps[-1]
            minimum = torch.from_numpy(np.array(p.array(root/name/'minimum.npy',(n,1),finite=False))).to('cuda')
            selected, center, previous = list(progress['selected']), progress['pending_center'], receipt['receipt_sha256']
        else:
            minimum = torch.full((n,1), float('inf'), dtype=torch.float32, device='cuda')
            selected, center, previous = [], start, None
        seen = set(selected)

        def save():
            nonlocal previous
            name = f'step-{len(selected):06d}'
            cpu = minimum.cpu().numpy().copy()
            progress = {'step':len(selected), 'selected':list(selected), 'pending_center':center, 'previous_receipt':previous}
            p.validate_selection(progress,cpu,n,maximum)
            with fit.phase_stage(root,name) as stage:
                np.save(stage/'minimum.npy',cpu,allow_pickle=False)
                fit.write_json(stage/'progress.json',progress)
                receipt = fit.seal_phase(stage,name,signature,{},['minimum.npy','progress.json'])
            if not np.array_equal(p.array(root/name/'minimum.npy',(n,1),finite=False),cpu):
                raise ValueError('Selection checkpoint reload differs')
            previous = receipt['receipt_sha256']

        if not steps: save()
        began = time.perf_counter()
        measured = []
        with torch.no_grad():
            while len(selected) < maximum:
                check()
                before = time.perf_counter()
                center = update_streamed(torch,features,minimum,center,chunk,check)
                measured.append(time.perf_counter()-before)
                if center in seen: raise ValueError('Repeated greedy center; no replacement')
                selected.append(center); seen.add(center)
                if len(measured) == 25:
                    preview({'iterations':25, 'mean_seconds':float(np.mean(measured)),
                             'remaining_linear_estimate_seconds':float(np.mean(measured)*(maximum-len(selected))),
                             'scope':'actual full-scan first25 of attempt; estimate only, excludes calibration/evaluation',
                             'selected_at_measurement':len(selected)})
                if len(selected)%25 == 0:
                    print(f'raw768 global representatives {len(selected)}/{maximum}; attempt {time.perf_counter()-began:.1f}s',flush=True)
                if len(selected)%interval == 0 or len(selected)==maximum:
                    save()
                    print(f'{root.name}: representatives {len(selected)}/{maximum}; attempt {time.perf_counter()-began:.1f}s',flush=True)
    finally:
        del features, minimum
        gc.collect(); torch.cuda.empty_cache()
    steps = p.checkpoint_steps(root,signature,n,maximum,interval)
    return selected, [x[2] for x in steps]


class Runtime(Base):
    def parity(self, check):
        """Installed Anomalib distance/update path; no projection in either comparator."""
        from anomalib.models.components.sampling.k_center_greedy import KCenterGreedy
        torch = self.torch
        rows = []
        self.release()
        with torch.no_grad():
            for seed in (42,43,44):
                values = np.random.default_rng(seed).standard_normal((97,768)).astype(np.float32)
                full = torch.from_numpy(values).to('cuda')
                official = KCenterGreedy(full, sampling_ratio=17/97)
                official.features = full
                official.reset_distances()
                center, ids, states = 3, [], []
                for _ in range(17):
                    check()
                    official.update_distances(center)
                    center = int(torch.as_tensor(official.get_new_idx()).item())
                    official.min_distances[center,0] = 0.
                    ids.append(center)
                    states.append(official.min_distances.clone())
                for chunk in (7,31,65536):
                    minimum = torch.full((97,1),float('inf'),dtype=torch.float32,device='cuda')
                    center, actual, error = 3, [], 0.
                    for state in states:
                        center = update_streamed(torch,values,minimum,center,chunk,check)
                        actual.append(center)
                        error = max(error,float((minimum-state).abs().max().item()))
                    if actual != ids or not np.isfinite(error) or error > 1e-5:
                        raise ValueError('Raw768 streamed recurrence differs from installed resident reference')
                    rows.append({'seed':seed,'chunk':chunk,'indices':actual,'max_abs_difference':error})
                    del minimum
                del full, official, states
        self.release()
        return {'passed':True,'cases':rows,'scope':'same native synthetic inputs, installed distance recurrence, not projected-selection equivalence or model accuracy'}

    def raw_select(self, root, signature, paths, start, check):
        self.release()
        if len(paths) != 256:
            raise ValueError('Expected all256 frozen training feature files')
        def preview(value):
            # Once saved, keep the first timing observation on resume.
            if not (root/'timing-preview').exists():
                p.publish(root,'timing-preview',signature,value)
            else:
                p.value(root,'timing-preview',signature)
            print(json.dumps(value,indent=2),flush=True)
        ids, receipts = select_raw(self.torch,root,signature,lambda:load_training(paths,check),
            3145728,78643,start,check,preview=preview)
        if len(ids)!=78643 or len(set(ids))!=78643:
            raise ValueError('Raw bank count/uniqueness changed')
        b.distribution(ids)
        if (root/'timing-preview').exists():
            p.value(root,'timing-preview',signature)
        return ids, receipts
