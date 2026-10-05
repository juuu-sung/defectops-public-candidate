"""Four matched GPU comparisons. Disposable updates are restored, never saved."""
import gc
import random
from unittest.mock import patch

import numpy as np

from scripts import mpi_supersimplenet_parity_protocol as p
from scripts.mpi_supersimplenet_evaluation_runtime import Runtime as BaseRuntime, environment
from scripts.mpi_supersimplenet_parity_reference import build_reference, reference_loss


class Runtime(BaseRuntime):
    def compare_job(self, images, job, checkpoint, signature, expected):
        torch, model = self.torch, self.model
        from anomalib.models.image.supersimplenet.loss import SSNLoss
        from torch.nn import functional as f
        if job not in p.jobs() or len(images) != 8:
            raise ValueError("Unregistered parity job")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["purpose"] != p.e.p.NAME or payload["identity"] != signature
                or payload["step"] != job["step"]):
            raise ValueError("Checkpoint identity differs")
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False).eval()
        before = self.fingerprints()
        if before != expected:
            raise ValueError("Checkpoint tensor fingerprints differ")
        initial = {k: v.clone() for k, v in payload["model"].items() if k.startswith(("adaptor.", "segdec."))}
        reference = None
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        try:
            random.seed(job["seed"]); np.random.seed(job["seed"])
            torch.manual_seed(job["seed"]); torch.cuda.manual_seed_all(job["seed"])
            x = self.tensor(images)
            manual = torch.from_numpy(np.stack([np.asarray(im, np.float32)/255 for im in images])).permute(0,3,1,2).to(x.device)
            manual = (manual - x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            preprocessing = p.compare(x.cpu().numpy(), manual.cpu().numpy(), "outputs")
            del manual
            with torch.no_grad():
                raw = list(model.feature_extractor.feature_extractor(x).values())
                size = tuple(2*d for d in raw[0].shape[-2:])
                def expanded(corners):
                    return f.avg_pool2d(torch.cat([f.interpolate(v, size=size, mode="bilinear",
                        align_corners=corners) for v in raw], dim=1), 3, stride=1, padding=1)
                features = expanded(False)
                official_features = model.feature_extractor(x)
                feature_check = p.compare(features.cpu().numpy(), official_features.cpu().numpy(), "outputs")
                author_features = expanded(True)
                # Deliberate upstream difference; not included in the pass/fail gate.
                interpolation = p.compare(official_features.cpu().numpy(), author_features.cpu().numpy(), "outputs")
                del raw, official_features, author_features
                zero = torch.zeros_like(features)
                noise, masks, labels = model.anomaly_generator(zero, x.new_zeros((8,1,*size)), x.new_zeros(8))
                del zero
            if (features.shape != (8,1536,64,64) or noise.shape != (16,1536,64,64)
                    or not torch.isfinite(noise).all() or not ((masks == 0)|(masks == 1)).all()
                    or not torch.equal(labels.bool(), masks.flatten(1).any(1))
                    or not labels.any() or labels.all()):
                raise ValueError("Invalid/missing-class fixed synthetic draw; no reroll")
            noise_hash = p.hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            reference = build_reference()
            reference.load_state_dict(initial, strict=True)
            reference = reference.to("cuda")
            model.adaptor.requires_grad_(True); model.segdec.requires_grad_(True)
            loss_fn = SSNLoss()

            def optimizer_for(module):
                optimizer = torch.optim.AdamW([
                    {"params": module.adaptor.parameters(), "lr": 1e-4, "weight_decay": .01},
                    {"params": module.segdec.parameters(), "lr": 2e-4, "weight_decay": 1e-5}])
                optimizer.load_state_dict(payload["optimizer"])
                # AdamW moments must be private, not aliased across the two paths.
                for state in optimizer.state.values():
                    for key, value in list(state.items()):
                        if torch.is_tensor(value): state[key] = value.clone()
                return optimizer

            arrays = {"masks": masks[:,0].cpu().numpy().astype(bool), "labels": labels.cpu().numpy().astype(bool)}
            official_loss_parts = lambda m,s: (loss_fn(m,s,masks,labels), {
                "map_focal": loss_fn.focal_loss(m,masks), "map_truncated_l1": loss_fn.trunc_l1_loss(m,masks),
                "classification_focal": loss_fn.focal_loss(s,labels)})
            calls = []

            def fixed_generator(adapted, ignored_mask, ignored_label):
                if adapted.shape != features.shape: raise ValueError("Generator boundary shape changed")
                calls.append(True)
                return torch.cat((adapted,adapted)) + noise, masks, labels

            def one(module, name, forward, loss):
                module.train()
                if hasattr(module, "feature_extractor"): module.feature_extractor.eval()
                optimizer = optimizer_for(module)
                parameters = {k:v for k,v in module.named_parameters() if v.requires_grad}
                if set(parameters) != {k for k in initial if k in dict(reference.named_parameters())}:
                    raise ValueError("Trainable parameter inventory differs")
                optimizer.zero_grad(set_to_none=True)
                maps, scores = forward()
                total, parts = loss(maps, scores)
                if not torch.isfinite(total): raise ValueError("Nonfinite loss")
                arrays[name+"_maps"] = maps[:,0].detach().cpu().numpy().copy()
                arrays[name+"_scores"] = scores.detach().cpu().numpy().copy()
                values = {k:float(v.detach()) for k,v in parts.items()}
                values["total"] = float(total.detach())
                total.backward()
                if any(v.grad is None or not torch.isfinite(v.grad).all() for v in parameters.values()):
                    raise ValueError("Missing/nonfinite trainable gradient")
                gradients = {k:v.grad.detach().cpu().numpy().copy() for k,v in parameters.items()}
                optimizer.step()
                updates = {k:(v.detach().cpu()-initial[k]).numpy().copy() for k,v in parameters.items()}
                buffers = {k:v.detach().cpu().numpy().copy() for k,v in module.named_buffers() if k.startswith("segdec.")}
                optimizer.zero_grad(set_to_none=True)
                del optimizer, total, parts, maps, scores
                return {"loss":values, "gradients":gradients, "updates":updates, "buffers":buffers}

            with patch.object(model.anomaly_generator, "forward", side_effect=fixed_generator):
                def official_forward():
                    maps, scores, got_masks, got_labels = model(x, masks=x.new_zeros((8,256,256)), labels=x.new_zeros(8))
                    if not torch.equal(got_masks,masks) or not torch.equal(got_labels,labels):
                        raise ValueError("Official targets changed")
                    return maps,scores
                official = one(model, "official", official_forward, official_loss_parts)
            if len(calls) != 1 or any(v.grad is not None for v in model.feature_extractor.parameters()):
                raise ValueError("Official boundary/backbone freeze changed")
            independent = one(reference, "reference", lambda: reference(features,noise),
                              lambda m,s: reference_loss(m,s,masks,labels))
            checks = {"preprocessing":preprocessing, "features":feature_check}
            checks.update({k:p.compare(arrays["official_"+k],arrays["reference_"+k],"outputs") for k in ("maps","scores")})
            checks["loss"] = {k:p.compare(official["loss"][k],independent["loss"][k],"loss") for k in official["loss"]}
            for kind in ("gradients","updates","buffers"):
                if set(official[kind]) != set(independent[kind]): raise ValueError("Comparison inventory differs")
                checks[kind] = {k:p.compare(official[kind][k],independent[kind][k],kind) for k in official[kind]}
            gates = [checks[k] for k in ("preprocessing","features","maps","scores")]
            gates += [v for k in ("loss","gradients","updates","buffers") for v in checks[k].values()]
            # Reduce by sample: no batch-sized float64 temporary on the 8 GiB GPU.
            noise_squares = sum(float(sample.double().square().sum()) for sample in noise)
            result = {"job":job, "passed":all(v["passed"] for v in gates), "checks":checks,
                "loss_values":{"official":official["loss"],"reference":independent["loss"]},
                "known_interpolation_difference_not_a_gate":interpolation,
                "noise_sha256":noise_hash, "noise_rms":(noise_squares/noise.numel())**.5,
                "synthetic_pixel_fraction":float(masks.mean()), "synthetic_samples":int(labels.sum()),
                "optimizer_initial_step":job["step"], "disposable_optimizer_updates":2,
                "original_checkpoints_updated":False, "independent_backbone_equivalence_proven":False}
            del official, independent, features, noise, masks, labels, x
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(payload["model"], strict=True)
            model.requires_grad_(False).eval()
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
            if reference is not None: del reference
            gc.collect(); torch.cuda.empty_cache()
        if self.fingerprints() != before: raise ValueError("Audit failed to restore model state")
        result["model_state_restored"] = True
        result["initial_fingerprints"] = before
        self.synchronize()
        return result, arrays
