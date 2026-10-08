"""Single candidate: global greedy in native 768D, no projection or region quotas."""
from __future__ import annotations

import hashlib

from scripts import mpi_patchcore_balanced_bank as prior

p, fit, development, evaluation = prior.p, prior.fit, prior.development, prior.evaluation
NAME = 'mpi-patchcore-raw-bank-v1'
POLICY = {**prior.POLICY,
    'selection': 'global k-center greedy in original cached float32 768D features; no projection, quotas or prefilter',
    'quotas': None, 'projected_dimension': None, 'selection_dimension': 768,
    'initial_center': 'exact parent global start_index; no modulo, new seed or best-start search',
    'projection_caveat': 'Parent projection verified for provenance only; never used for candidate distances',
    'selection_inputs': 'only frozen 256 training feature caches and parent global start_index',
    'selection_storage': '9 GiB CPU array; synchronous 65536-row GPU transfers, full candidate scan per representative',
    'timing_preview_selections': 25,
}
distribution = prior.distribution
validate_raw = prior.validate_raw
contract_replay = prior.contract_replay
stability = prior.stability


def budget():
    n, d, k = POLICY['candidates'], POLICY['selection_dimension'], POLICY['representatives']
    raw = n*d*4
    return {'raw_cpu_bytes': raw, 'gpu_feature_chunk_bytes': POLICY['chunk_rows']*d*4,
            'gpu_minimum_bytes': n*4, 'selection_host_to_device_bytes': raw*k,
            'selection_host_to_device_TiB': raw*k/1024**4,
            'distance_elements': n*d*k,
            'checkpoint_minimum_bytes': (2+k//1000)*n*4,
            'scope': 'Array and transfer arithmetic, NOT measured process peaks or runtime; no time cap'}


def prepare(project, handoffs=None):
    source, identity = prior.prepare(project, handoffs)
    paths = [project/'scripts/run_mpi_patchcore_raw_bank.ps1',
             project/'docs/mpi-patchcore-raw-bank-v1.md']
    identity = {**identity, 'name': NAME, 'policy': POLICY, 'budget': budget(),
                'sources_lf_sha256': {**identity['sources_lf_sha256'],
                    **{f.relative_to(project).as_posix(): hashlib.sha256(
                        f.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for f in paths}}}
    return source, identity


def compare(source, candidate, method):
    result = prior.compare(source, candidate, method)
    result['comparison'] = 'Global projected369D greedy versus global native768D greedy; same256 development/169 calibration'
    return result
