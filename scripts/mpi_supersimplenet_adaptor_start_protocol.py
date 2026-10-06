"""Matched 500-update arms: only the adaptor's starting state differs."""
from copy import deepcopy
import hashlib
import json
import zipfile

from scripts import mpi_supersimplenet_parity_protocol as parity
from scripts import mpi_supersimplenet_varying_protocol as v
from scripts.mpi_supersimplenet_matched_eval_runtime import validate_public_differences

fit = parity.fit
NAME = "mpi-supersimplenet-adaptor-start-v1"
AUDIT_NAME = "mpi-supersimplenet-update-audit-v1"
AUDIT_SHA = "e2300535f78dc88c892162e2c06c5e829d1187a4485c36823c3f268cdbf5a179"
ARMS = ("trained_adaptor", "initial_adaptor")
POLICY = deepcopy(v.POLICY)
POLICY.update({"arms":list(ARMS), "initialization":"shared formal step0 head and backbone; adaptor from final or step0",
    "loss":"Anomalib 2.2.0 SSNLoss unchanged; official public training forward",
    "selection":"first eight formal-fit training rows, one normal photo; same as implementation audit",
    "fixed_control":None, "maximum_attempted_updates":1300, "primary_comparison_step":500,
    "automatic_model_selection":False, "later_phases_automatically_started":False,
    "interpretation":"matched synthetic learnability; held-out seeds on the same training photo, not unseen photos or defects"})
POLICY["resource_plan"]["comparison_seconds"] = None
FILES, EVIDENCE_FILES = v.FILES, v.EVIDENCE_FILES


def validate_initialization(record,identity,arm):
    if arm not in ARMS or record['arm'] != arm or not record['optimizer_empty']:
        raise ValueError('Invalid arm or nonempty initial optimizer')
    expected = dict(identity['checkpoints']['0']['fingerprints'])
    if arm == 'trained_adaptor':
        expected['adaptor'] = identity['checkpoints']['28800']['fingerprints']['adaptor']
    if record['fingerprints'] != expected:
        raise ValueError('Only the registered adaptor starting state may differ')


def validate_progress(progress,initialization):
    step=progress['step']
    if step not in POLICY['snapshots'] or len(progress['losses']) != step:
        raise ValueError('Invalid checkpoint step/history')
    if progress['fingerprints']['backbone'] != initialization['fingerprints']['backbone']:
        raise ValueError('Frozen backbone changed')
    if step == 0 and progress['fingerprints'] != initialization['fingerprints']:
        raise ValueError('Step zero differs from registered initialization')
    for seed in POLICY['evaluation_seeds']:
        validate_public_differences(progress['evaluations'][str(seed)]['checks']['public_batch_comparison'])


def validate_matched(a,b):
    if a['step'] != b['step']:
        raise ValueError('Different update counts cannot form the primary comparison')
    for seed in POLICY['evaluation_seeds']:
        if a['evaluations'][str(seed)]['target_sha256'] != b['evaluations'][str(seed)]['target_sha256']:
            raise ValueError('Evaluation synthetic targets differ between arms')
    for x,y in zip(a['losses'],b['losses'],strict=True):
        if any(x[k] != y[k] for k in ('step','seed','target_sha256','positive_samples','positive_pixels')):
            raise ValueError('Training synthetic schedule/targets differ between arms')


def read_checkpoints(root,signature,initialization):
    progress=v.read_checkpoints(root,signature)
    for row in progress:validate_progress(row,initialization)
    return progress


def prepare(project,handoffs,fit_root):
    path=handoffs/(AUDIT_NAME+'.zip')
    if fit.sha256_file(path) != AUDIT_SHA:
        raise ValueError('Reviewed update audit ZIP missing or changed; do not rerun it')
    with zipfile.ZipFile(path) as z:
        report=json.loads(z.read('report.json'));prior=json.loads(z.read('plan.json'))
    if (not report['pairs_all_jobs_matched']['repeat'] or not report['pairs_all_jobs_matched']['same_loss']
            or report['completed_jobs'] != 2):
        raise ValueError('Implementation audit prerequisites did not pass')
    identity=parity.prepare(project,handoffs,fit_root)
    for key in ('fit_signature','checkpoints','contract_sha256','rows','archive_sha256'):
        if prior[key] != identity[key]:raise ValueError(f'Prior audit provenance differs: {key}')
    if (identity['checkpoints']['0']['fingerprints']['backbone'] != identity['checkpoints']['28800']['fingerprints']['backbone']
            or identity['checkpoints']['0']['fingerprints']['adaptor'] == identity['checkpoints']['28800']['fingerprints']['adaptor']):
        raise ValueError('Need same frozen backbone and different registered adaptor states')
    if len(identity['rows']) != 8 or len({r['photo_id'] for r in identity['rows']}) != 1:
        raise ValueError('Expected the same eight tiles of one normal photo')
    identity.pop('jobs')
    identity.update(name=NAME,policy=POLICY,update_audit_sha256=AUDIT_SHA)
    for name in ('scripts/run_mpi_supersimplenet_adaptor_start.ps1','docs/mpi-supersimplenet-adaptor-start-v1.md'):
        identity['sources_lf_sha256'][name]=hashlib.sha256((project/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return identity
