"""Four disposable step-zero paths, shared data/noise and private optimizer state."""
import gc
import random
from unittest.mock import patch

import numpy as np

from scripts import mpi_supersimplenet_update_audit_protocol as p
from scripts.mpi_supersimplenet_evaluation_runtime import Runtime as BaseRuntime, environment
from scripts.mpi_supersimplenet_parity_reference import build_reference, reference_loss


class Runtime(BaseRuntime):
    def compare_job(self, images, job, checkpoint, signature, expected):
        torch, model = self.torch, self.model
        from anomalib.models.image.supersimplenet.loss import SSNLoss
        if job not in p.jobs() or len(images) != 8:
            raise ValueError("Unregistered update audit")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["purpose"] != p.parity.e.p.NAME or payload["identity"] != signature
                or payload["step"] != 0 or payload["optimizer"]["state"]):
            raise ValueError("Need preserved step-zero weights and empty optimizer moments")
        model.load_state_dict(payload["model"], strict=True)
        model.requires_grad_(False).eval()
        if self.fingerprints() != expected:
            raise ValueError("Step-zero fingerprint differs")
        initial = {k: v.clone() for k,v in payload["model"].items() if k.startswith(("adaptor.","segdec."))}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        reference = None
        try:
            random.seed(job["seed"]); np.random.seed(job["seed"])
            torch.manual_seed(job["seed"]); torch.cuda.manual_seed_all(job["seed"])
            x = self.tensor(images)
            with torch.no_grad():
                features = model.feature_extractor(x)
                zero = torch.zeros_like(features)
                noise, masks, labels = model.anomaly_generator(zero, x.new_zeros((8,1,64,64)), x.new_zeros(8))
                del zero
            if (features.shape != (8,1536,64,64) or noise.shape != (16,1536,64,64)
                    or not torch.isfinite(noise).all() or not ((masks == 0)|(masks == 1)).all()
                    or not torch.equal(labels.bool(), masks.flatten(1).any(1))
                    or not labels.any() or labels.all()):
                raise ValueError("Invalid fixed synthetic draw; no reroll")
            noise_hash = p.hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            reference = build_reference().to("cuda")
            arrays = {"masks": masks[:,0].cpu().numpy().astype(bool), "labels": labels.cpu().numpy().astype(bool)}
            loss_fn = SSNLoss()
            records = {}

            def logits_loss(maps, scores):
                return loss_fn(maps,scores,masks,labels), {
                    "map_focal": loss_fn.focal_loss(maps,masks),
                    "map_truncated_l1": loss_fn.trunc_l1_loss(maps,masks),
                    "classification_focal": loss_fn.focal_loss(scores,labels)}

            def save_output(arm, suffix, maps, scores):
                arrays[arm+"_"+suffix+"_maps"] = maps[:,0].detach().cpu().numpy().copy()
                arrays[arm+"_"+suffix+"_scores"] = scores.detach().cpu().numpy().copy()

            def snapshot_outputs(module, arm, stage, mode):
                # Train-mode diagnostics must not contaminate the next input's BN buffers.
                saved = {k:v.clone() for k,v in module.segdec.named_buffers()}
                module.train(mode == "train")
                if hasattr(module, "feature_extractor"): module.feature_extractor.eval()
                with torch.no_grad():
                    adapted = module.adaptor(features)
                    for kind in ("clean", "synthetic"):
                        for k,v in module.segdec.named_buffers(): v.copy_(saved[k])
                        inputs = torch.cat((adapted, adapted))
                        if kind == "synthetic": inputs.add_(noise)
                        maps,scores = module.segdec(inputs)
                        save_output(arm, f"{stage}_{mode}_{kind}", maps,scores)
                        del inputs,maps,scores
                    for k,v in module.segdec.named_buffers(): v.copy_(saved[k])

            for arm in p.ARMS:
                official = arm.startswith("official")
                module = model if official else reference
                module.zero_grad(set_to_none=True)
                module.load_state_dict(payload["model"] if official else initial, strict=True)
                module.requires_grad_(False).eval()
                module.adaptor.requires_grad_(True); module.segdec.requires_grad_(True)
                snapshot_outputs(module, arm, "pre", "eval")
                module.train()
                if official: module.feature_extractor.eval()
                optimizer = torch.optim.AdamW([
                    {"params": module.adaptor.parameters(), "lr": 1e-4, "weight_decay": .01},
                    {"params": module.segdec.parameters(), "lr": 2e-4, "weight_decay": 1e-5}])
                optimizer.load_state_dict(payload["optimizer"])
                if optimizer.state:
                    raise ValueError("Step-zero optimizer was not reset")
                parameters = {k:v for k,v in module.named_parameters() if v.requires_grad}
                if list(parameters) != list(dict(reference.named_parameters())):
                    raise ValueError("Parameter order differs between implementations")
                optimizer.zero_grad(set_to_none=True)
                calls = []
                def fixed_generator(adapted, ignored_masks, ignored_labels):
                    if adapted.shape != features.shape: raise ValueError("Generator boundary differs")
                    calls.append(True)
                    return torch.cat((adapted,adapted))+noise, masks, labels
                if official:
                    with patch.object(model.anomaly_generator,"forward",side_effect=fixed_generator):
                        maps,scores,got_masks,got_labels = model(x,masks=x.new_zeros((8,256,256)),labels=x.new_zeros(8))
                    if len(calls) != 1 or not torch.equal(got_masks,masks) or not torch.equal(got_labels,labels):
                        raise ValueError("Official target injection differs")
                else:
                    maps,scores = reference(features,noise)
                total,parts = (reference_loss(maps,scores,masks,labels) if arm == "reference_author"
                               else logits_loss(maps,scores))
                if not torch.isfinite(total): raise ValueError("Nonfinite loss")
                save_output(arm,"pre_train_synthetic",maps,scores)
                values = {k:float(v.detach()) for k,v in parts.items()}; values["total"] = float(total.detach())
                total.backward()
                if any(v.grad is None or not torch.isfinite(v.grad).all() for v in parameters.values()):
                    raise ValueError("Missing/nonfinite gradients")
                if official and any(v.grad is not None for v in model.feature_extractor.parameters()):
                    raise ValueError("Frozen backbone received gradient")
                gradients = {k:v.grad.detach().cpu().numpy().copy() for k,v in parameters.items()}
                optimizer.step()
                updates = {k:(v.detach().cpu()-initial[k]).numpy().copy() for k,v in parameters.items()}
                buffers = {k:v.detach().cpu().numpy().copy() for k,v in module.named_buffers() if k.startswith("segdec.")}
                records[arm] = {"loss":values,"gradients":gradients,"updates":updates,"buffers":buffers}
                optimizer.zero_grad(set_to_none=True)
                del total,parts,maps,scores,optimizer
                snapshot_outputs(module,arm,"post","eval")
                snapshot_outputs(module,arm,"post","train")
                module.zero_grad(set_to_none=True)
                self.synchronize()
                print(f"  {arm}: one update and matched post-update outputs recorded",flush=True)

            checks,details,passed = {},{},{}
            output_checks = p.output_checks(arrays)
            for pair,(a,b) in p.PAIRS.items():
                checks[pair] = {kind:{k:p.parity.compare(records[a][kind][k],records[b][kind][k],kind)
                                     for k in records[a][kind]} for kind in ("loss","gradients","updates","buffers")}
                details[pair] = {k:p.selected_differences(records[a]["gradients"][k],records[b]["gradients"][k],
                                                        records[a]["updates"][k],records[b]["updates"][k])
                                 for k in records[a]["gradients"]
                                 if not checks[pair]["gradients"][k]["passed"] or not checks[pair]["updates"][k]["passed"]}
                gates = list(output_checks[pair].values())+[v for c in checks[pair].values() for v in c.values()]
                passed[pair] = all(g["passed"] for g in gates)
            effects = {arm:{f"{kind}_{out}":p.parity.compare(arrays[f"{arm}_post_eval_{kind}_{out}"],
                        arrays[f"{arm}_pre_eval_{kind}_{out}"],"outputs")
                        for kind in ("clean","synthetic") for out in ("maps","scores")} for arm in p.ARMS}
            result = {"job":job,"pair_passed":passed,"output_checks":output_checks,"parameter_checks":checks,
                      "largest_difference_values":details,"within_arm_eval_change_not_a_gate":effects,
                      "loss_values":{arm:records[arm]["loss"] for arm in p.ARMS},
                      "noise_sha256":noise_hash,"initial_fingerprints":expected,
                      "disposable_optimizer_updates":4,"original_checkpoints_updated":False,
                      "synthetic_pixel_fraction":float(masks.mean()),"synthetic_samples":int(labels.sum())}
            del records,features,noise,masks,labels,x
        finally:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(payload["model"],strict=True)
            model.requires_grad_(False).eval()
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
            if reference is not None: del reference
            gc.collect(); torch.cuda.empty_cache()
        if self.fingerprints() != expected:
            raise ValueError("Original state restoration failed")
        result["model_state_restored"] = True
        p.verify_arrays(result,arrays)
        self.synchronize()
        return result,arrays
