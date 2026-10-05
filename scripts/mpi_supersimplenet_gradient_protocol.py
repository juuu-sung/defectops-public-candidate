"""Read-only loss-gradient attribution, not a new training candidate."""
from copy import deepcopy
import hashlib
import io
import json
import zipfile

import numpy as np

from scripts import mpi_supersimplenet_noise_protocol as old

fit = old.fit
NAME = "mpi-supersimplenet-gradient-audit-v1"
NOISE_SHA = "f058d2c8c067f4d0f9441a016d4bf578c68388dfe5a49f4dd0b22ca8c2c943dd"
STATES = ["original", "varying015", "varying060"]
STRENGTHS = ["std015", "std060"]
SEEDS = [900001, 900002, 900003, 900004]
TERMS = ["map_l1_normal", "map_l1_anomaly", "map_focal_normal", "map_focal_anomaly",
         "classification_focal_normal", "classification_focal_anomaly", "official_total"]
GROUPS = ["adaptor", "segmentation", "classification_conv", "classification_fc"]
POLICY = {"states": STATES, "strengths": STRENGTHS, "seeds": SEEDS, "terms": TERMS,
    "input_tiles": 8, "effective_batch": 16, "optimizer_updates": 0, "reset_parameters": False,
    "mode": "official training forward; frozen backbone; restore buffers, modes and RNG",
    "resource_plan": {"comparison_seconds": 2700, "rss_GiB": 20, "selection_and_inference_reserved_GiB": 6},
    "maximum_attempted_jobs": 30, "free_disk_GiB": 2,
    "gradient_sum_relative_tolerance": .001, "gradient_sum_absolute_tolerance": 1e-5,
    "reload_tolerance": 1e-4, "test_images_decoded": 0, "test_masks_decoded": 0,
    "threshold_changed": False, "service_model_changed": False, "automatic_pass_allowed": False,
    "interpretation": "previously observed synthetic seeds on one training photo; gradients are not defect accuracy or AdamW update directions"}


def jobs():
    return [{"name": f"{state}-{strength}-s{seed}", "state": state, "strength": strength, "seed": seed}
            for state in STATES for strength in STRENGTHS for seed in SEEDS]


def prepare(project, fit_root, fixed_root, varying_root, noise_root):
    handoffs = project/"artifacts/handoff"
    records, noise_identity, _ = old.prepare(project, handoffs, fit_root, fixed_root, varying_root)
    path = handoffs/(old.NAME+".zip")
    if fit.sha256_file(path) != NOISE_SHA:
        raise ValueError("Reviewed noise-strength handoff required")
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        if (len(z.namelist()) != len(manifest)+1 or set(z.namelist()) != set(manifest)|{"handoff-manifest.json"}
                or any(hashlib.sha256(z.read(n)).hexdigest() != h for n,h in manifest.items())):
            raise ValueError("Noise handoff inventory/checksum differs")
        plan = json.loads(z.read("plan.json"))
        if any(plan.get(k) != v for k,v in noise_identity.items()):
            raise ValueError("Noise experiment provenance differs")
        signature = fit.digest(plan)
        if old.v.old.e.value(noise_root, "identity", signature) != plan:
            raise ValueError("Noise local identity differs")
        receipt = fit.read_phase(noise_root, "std060-step-0500", signature, old.FILES)
        if receipt != json.loads(z.read("std060-step-0500/receipt.json")):
            raise ValueError("Strong-noise model differs from reviewed checkpoint")
        references, fingerprints = {}, {}
        for state,phase in (("original","std060-step-0000"), ("varying015","baseline"), ("varying060","std060-step-0500")):
            progress = json.loads(z.read(phase+"/progress.json"))
            fingerprints[state] = progress["fingerprints"]
            with np.load(io.BytesIO(z.read(phase+"/arrays.npz")), allow_pickle=False) as f:
                references[state] = {k:f[k] for k in f.files}
    identity = deepcopy(noise_identity)
    identity.update(name=NAME, policy=POLICY, noise_zip_sha256=NOISE_SHA, noise_signature=signature,
                    strong_checkpoint_sha256=receipt["files"]["model.pt"], state_fingerprints=fingerprints)
    for name in ("scripts/mpi_supersimplenet_gradient_protocol.py", "scripts/mpi_supersimplenet_gradient_runtime.py",
                 "scripts/run_mpi_supersimplenet_gradients.py", "scripts/run_mpi_supersimplenet_gradients.ps1",
                 "docs/mpi-supersimplenet-gradient-audit-v1.md"):
        identity["sources_lf_sha256"][name] = hashlib.sha256((project/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    return identity, references


def gram_matrix(vectors):
    """CPU chunks bound temporary memory; preserve float32 gradients and reduce in float64."""
    if len(vectors) != len(TERMS) or len({v.size for v in vectors}) != 1:
        raise ValueError("Gradient inventory differs")
    result = np.zeros((len(TERMS),len(TERMS)), np.float64)
    for start in range(0, vectors[0].size, 65536):
        block = np.stack([v[start:start+65536] for v in vectors]).astype(np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite gradient")
        result += block @ block.T
    return result


def summarize_gram(gram):
    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (7,7) or not np.isfinite(gram).all() or not np.allclose(gram,gram.T,atol=1e-10):
        raise ValueError("Invalid gradient Gram matrix")
    if np.linalg.eigvalsh(gram).min() < -1e-8 * max(1., float(np.trace(gram))):
        raise ValueError("Gradient Gram matrix is not positive semidefinite")
    normal = np.array([1,0,1,0,1,0,0.],float)
    anomaly = np.array([0,1,0,1,0,1,0.],float)
    def norm(c): return float(np.sqrt(max(0., c @ gram @ c)))
    n,a,total = norm(normal),norm(anomaly),norm(normal+anomaly)
    cosine = float(np.clip((normal @ gram @ anomaly)/(n*a),-1,1)) if n and a else None
    return {"term_gradient_l2": {k:float(np.sqrt(max(0.,gram[i,i]))) for i,k in enumerate(TERMS)},
        "normal_gradient_l2":n, "anomaly_gradient_l2":a, "combined_gradient_l2":total,
        "normal_anomaly_cosine":cosine, "retained_norm_fraction":total/(n+a) if n+a else None,
        "note":"Negative cosine suggests local opposition, not proof of training failure. Zero-gradient cosine is undefined."}


def read_job(root, job, signature):
    fit.read_phase(root,job["name"],signature,["result.json","arrays.npz"])
    result = json.loads((root/job["name"]/"result.json").read_bytes())
    if result["job"] != job or result["optimizer_updates"] != 0 or not result["state_unchanged"]:
        raise ValueError("Job identity/state differs")
    if result["state_before_sha256"] != result["state_after_sha256"]:
        raise ValueError("Model state changed")
    for group in GROUPS:
        if summarize_gram(result["groups"][group]["gram"]) != result["groups"][group]["summary"]:
            raise ValueError("Gradient summary cannot be reproduced")
    return result
