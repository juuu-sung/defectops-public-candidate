"""Training-cache-only selection. No query images, maps, labels or scores accepted."""
from __future__ import annotations

import gc
import json
import time

import numpy as np

from scripts import mpi_patchcore_balanced_bank as b
from scripts.mpi_patchcore_scaling_runtime import Runtime as Base
from scripts.mpi_patchcore_scaling_probe_runtime import update_chunked

p, fit = b.p, b.fit


def projected_region(paths, region, photos=256, regions=12, patches=1024, dim=369, chunk=65536):
    ids = b.global_indices(region, photos, regions, patches)
    n = photos*regions*patches
    intervals = p.probe.slices(n, chunk)
    if len(paths) != len(intervals): raise ValueError('Missing projected chunks')
    values = np.empty((len(ids), dim), dtype=np.float32)
    filled = np.zeros(len(ids), dtype=bool)
    for path,(a,z) in zip(paths, intervals, strict=True):
        block = p.array(path, (z-a,dim))
        lo,hi = np.searchsorted(ids, [a,z])
        values[lo:hi] = block[ids[lo:hi]-a]
        filled[lo:hi] = True
        del block
    if not filled.all(): raise ValueError('Region candidate coverage incomplete')
    return values


def select_region(torch, root, signature, features_factory, n, maximum, start, check,
                  interval=1000, chunk=65536):
    """Same greedy recurrence/tie handling as parent; checkpoint restart tested on CPU."""
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
        features = torch.from_numpy(cpu_features).to('cuda')
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
        with torch.no_grad():
            while len(selected) < maximum:
                check()
                center = update_chunked(torch,features,minimum,center,chunk,check)
                if center in seen: raise ValueError('Repeated greedy center; no replacement')
                selected.append(center); seen.add(center)
                if len(selected)%interval == 0 or len(selected)==maximum:
                    save()
                    print(f'{root.name}: representatives {len(selected)}/{maximum}; attempt {time.perf_counter()-began:.1f}s',flush=True)
    finally:
        del features, minimum
        gc.collect(); torch.cuda.empty_cache()
    steps = p.checkpoint_steps(root,signature,n,maximum,interval)
    return selected, [x[2] for x in steps]


class Runtime(Base):
    def balanced_select(self, root, signature, paths, start, check):
        self.release()
        root.mkdir(exist_ok=True)
        selected, receipts = [], {}
        for region,quota in enumerate(b.POLICY['quotas']):
            ids, saved = select_region(self.torch,root/fit.REGIONS[region],signature,
                lambda r=region:projected_region(paths,r),256*1024,quota,start%(256*1024),check)
            selected.extend(b.global_indices(region)[ids].tolist())
            receipts[fit.REGIONS[region]] = saved
        if b.distribution(selected)['regions'] != b.POLICY['quotas'] or len(selected)!=78643:
            raise ValueError('Balanced bank quota/count mismatch')
        return selected, receipts

    def verify_bank(self, checkpoint, parent_checkpoint, raw_paths, selected):
        torch = self.torch
        candidate = torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
        parent = torch.load(parent_checkpoint,map_location='cpu',weights_only=True)['state_dict']
        if set(candidate)!=set(parent): raise ValueError('Model tensor keys changed')
        bank_key = 'model.memory_bank'
        if bank_key not in candidate: raise ValueError('Bank tensor missing')
        unchanged = [k for k in candidate if k!=bank_key]
        if any(not torch.equal(candidate[k],parent[k]) for k in unchanged):
            raise ValueError('Non-bank model tensors changed')
        ids = np.asarray(selected)
        bank = candidate[bank_key].numpy()
        if bank.shape!=(78643,768) or bank.dtype!=np.float32: raise ValueError('Bank shape/dtype changed')
        for i,path in enumerate(raw_paths):
            targets = np.flatnonzero(ids//12288==i)
            raw = p.array(path,(12288,768))
            if not np.array_equal(bank[targets],raw[ids[targets]%12288]):
                raise ValueError('Bank does not match selected training rows')
        return {'passed':True,'non_bank_tensors_unchanged':len(unchanged),
                'selected_raw_rows_exact':78643,'feature_dimension':768}
