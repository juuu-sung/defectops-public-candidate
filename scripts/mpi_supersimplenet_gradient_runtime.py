"""Calculate, but never apply, separate normal/anomaly loss gradients."""
import hashlib
import random

import numpy as np

from scripts import mpi_supersimplenet_gradient_protocol as p
from scripts.mpi_supersimplenet_noise_runtime import Runtime as NoiseRuntime, noise_level, environment
from scripts.mpi_supersimplenet_runtime import state_digest
from scripts.mpi_supersimplenet_synthetic_runtime import preserved_buffers_and_modes
from scripts.run_mpi_supersimplenet_varying import compare_arrays


class Runtime(NoiseRuntime):
    def step(self, *args, **kwargs):
        raise ValueError("Gradient audit is read-only; optimizer updates forbidden")

    def step_at_strength(self, *args, **kwargs):
        raise ValueError("Gradient audit is read-only; optimizer updates forbidden")

    def probe(self, images, job, expected):
        if job not in p.jobs() or len(images) != 8:
            raise ValueError("Unregistered diagnostic job")
        torch, model = self.torch, self.model
        from torchvision.ops import sigmoid_focal_loss
        self.optimizer.zero_grad(set_to_none=True)
        model.eval()
        before = state_digest(model)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        grouped = {"adaptor":list(model.adaptor.parameters()), "segmentation":list(model.segdec.seg_head.parameters()),
            "classification_conv":list(model.segdec.cls_conv.parameters()), "classification_fc":list(model.segdec.cls_fc.parameters())}
        params = [t for group in grouped.values() for t in group]
        if any(not t.requires_grad for t in params) or any(t.requires_grad for t in model.feature_extractor.parameters()):
            raise ValueError("Trainable/frozen parameter contract differs")
        try:
            seed = job["seed"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with preserved_buffers_and_modes(model), noise_level(model.anomaly_generator,p.old.STRENGTHS[job["strength"]]):
                model.train(); model.feature_extractor.eval()
                maps,scores,masks,labels = model(self.tensor(images),
                    masks=torch.zeros((8,256,256),device="cuda"),labels=torch.zeros(8,device="cuda"))
                arrays = {"masks":masks[:,0].detach().cpu().numpy().astype(bool),
                    "labels":labels.detach().cpu().numpy().astype(bool),
                    "train_mixed_maps":maps[:,0].detach().cpu().numpy().copy(),
                    "train_mixed_scores":scores.detach().cpu().numpy().copy()}
                replay = compare_arrays(arrays,{k:expected[k] for k in arrays})
                target = hashlib.sha256(arrays["masks"].tobytes()+arrays["labels"].tobytes()).hexdigest()
                mi,ci = masks.bool(),labels.bool()
                if not mi.any() or mi.all() or not ci.any() or ci.all():
                    raise ValueError("Missing class; do not reroll synthetic seeds")
                mf = sigmoid_focal_loss(maps,masks,alpha=-1,gamma=4.,reduction="none")
                cf = sigmoid_focal_loss(scores,labels,alpha=-1,gamma=4.,reduction="none")
                # L1 uses separate class means; focal retains the official global denominator.
                terms = [torch.clamp(maps[~mi]+.5,min=0).mean(),torch.clamp(-maps[mi]+.5,min=0).mean(),
                         mf[~mi].sum()/mf.numel(),mf[mi].sum()/mf.numel(),
                         cf[~ci].sum()/cf.numel(),cf[ci].sum()/cf.numel()]
                terms.append(self.loss(maps,scores,masks,labels))
                values = {name:float(term.detach()) for name,term in zip(p.TERMS,terms,strict=True)}
                loss_difference = abs(sum(values[k] for k in p.TERMS[:-1])-values["official_total"])
                if loss_difference > 1e-5:
                    raise ValueError("Loss decomposition differs from official total")
                vectors = {name:[] for name in grouped}
                missing = {name:[] for name in grouped}
                for i,term in enumerate(terms):
                    grads = torch.autograd.grad(term,params,retain_graph=i<len(terms)-1,allow_unused=True)
                    offset = 0
                    for group,tensors in grouped.items():
                        selected = grads[offset:offset+len(tensors)]
                        missing[group].append(sum(g is None for g in selected))
                        parts = [np.zeros(t.numel(),np.float32) if g is None else
                                 g.detach().cpu().numpy().reshape(-1).copy() for g,t in zip(selected,tensors,strict=True)]
                        vectors[group].append(np.concatenate(parts))
                        offset += len(tensors)
                    del grads, selected, parts
                groups = {}
                for name,vec in vectors.items():
                    error_sq, total_sq, max_error = 0.,0.,0.
                    for start in range(0,vec[0].size,65536):
                        block = np.stack([x[start:start+65536] for x in vec]).astype(np.float64)
                        error = block[:-1].sum(axis=0)-block[-1]
                        error_sq += float(error@error); total_sq += float(block[-1]@block[-1])
                        max_error = max(max_error,float(np.abs(error).max()))
                    rel_error = (error_sq/total_sq)**.5 if total_sq else None
                    if max_error > p.POLICY["gradient_sum_absolute_tolerance"] and (rel_error is None or rel_error > p.POLICY["gradient_sum_relative_tolerance"]):
                        raise ValueError(f"Component gradients do not reconstruct total in {name}")
                    gram = p.gram_matrix(vec)
                    groups[name] = {"gram":gram.tolist(),"summary":p.summarize_gram(gram),
                        "parameter_values":vec[0].size,"unused_parameter_counts_by_term":missing[name],
                        "gradient_sum_max_abs_difference":max_error,"gradient_sum_relative_l2_difference":rel_error}
                self.synchronize()
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        after = state_digest(model)
        if before != after or model.training or any(t.grad is not None for t in model.parameters()):
            raise ValueError("Read-only gradient audit changed model or left gradients")
        result = {"job":job,"optimizer_updates":0,"loss_terms":values,"loss_sum_max_abs_difference":loss_difference,
            "groups":groups,"target_sha256":target,"positive_pixels":int(mi.sum()),"positive_samples":int(ci.sum()),
            "previous_train_forward_max_abs_difference":replay,"state_before_sha256":before,
            "state_after_sha256":after,"state_unchanged":True,"no_parameter_gradients_left":True}
        return result,arrays
