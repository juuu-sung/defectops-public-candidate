"""Bounded full-adaptor/head learning on a fixed, reproducible synthetic task."""
import hashlib
import random

import numpy as np

from scripts import mpi_supersimplenet_learnability_protocol as p
from scripts.mpi_supersimplenet_synthetic_runtime import Runtime as FrozenRuntime, environment


class Runtime(FrozenRuntime):
    def __init__(self, checkpoint, signature):
        super().__init__(checkpoint, signature)
        from anomalib.models.image.supersimplenet.loss import SSNLoss
        self.loss = SSNLoss()
        self.model.adaptor.requires_grad_(True)
        self.model.segdec.requires_grad_(True)
        self.optimizer = self.new_optimizer()

    def new_optimizer(self):
        return self.torch.optim.AdamW([
            {"params": self.model.adaptor.parameters(), "lr": p.POLICY["adaptor_lr"],
             "weight_decay": p.POLICY["adaptor_weight_decay"]},
            {"params": self.model.segdec.parameters(), "lr": p.POLICY["segdec_lr"],
             "weight_decay": p.POLICY["segdec_weight_decay"]}])

    def reset_task_rng(self):
        seed = p.POLICY["seed"]
        random.seed(seed); np.random.seed(seed)
        self.torch.manual_seed(seed); self.torch.cuda.manual_seed_all(seed)

    def components(self, maps, scores, masks, labels):
        with self.torch.no_grad():
            return {"map_focal": float(self.loss.focal_loss(maps, masks)),
                    "map_truncated_l1": float(self.loss.trunc_l1_loss(maps, masks)),
                    "classification_focal": float(self.loss.focal_loss(scores, labels))}

    def step(self, images, index, target):
        torch, model = self.torch, self.model
        self.reset_task_rng()
        model.train(); model.feature_extractor.eval()
        self.optimizer.zero_grad(set_to_none=True)
        maps, scores, masks, labels = model(self.tensor(images),
            masks=torch.zeros((8,256,256), device="cuda"), labels=torch.zeros(8, device="cuda"))
        if (maps.shape != (16,1,64,64) or masks.shape != maps.shape or scores.shape != (16,)
                or labels.shape != scores.shape or not ((masks == 0)|(masks == 1)).all()
                or not torch.equal(labels.bool(), masks.flatten(1).any(1))
                or not all(torch.isfinite(t).all() for t in (maps,scores,masks,labels))):
            raise ValueError("Official fixed-task output contract differs")
        target_now = hashlib.sha256(masks[:,0].bool().cpu().numpy().tobytes()+labels.bool().cpu().numpy().tobytes()).hexdigest()
        if target_now != target:
            raise ValueError("Synthetic targets changed; no reroll")
        components = self.components(maps,scores,masks,labels)
        loss = self.loss(maps,scores,masks,labels)
        if loss.ndim or not torch.isfinite(loss): raise ValueError("Nonfinite scalar loss")
        loss.backward()
        gradients = {}
        for name,module in (("adaptor",model.adaptor),("segmentation",model.segdec.seg_head),
                            ("classification_conv",model.segdec.cls_conv),("classification_fc",model.segdec.cls_fc)):
            params = list(module.parameters())
            if any(t.grad is None or not torch.isfinite(t.grad).all() for t in params):
                raise ValueError(f"Missing or nonfinite {name} gradients")
            # Zero at convergence is evidence, not permission to extend training.
            gradients[name] = sum(float(t.grad.detach().double().square().sum()) for t in params)**.5
        if any(t.grad is not None for t in model.feature_extractor.parameters()):
            raise ValueError("Frozen backbone received gradients")
        self.optimizer.step()
        if any(not torch.isfinite(t).all() for group in self.optimizer.param_groups for t in group["params"]):
            raise ValueError("Nonfinite updated parameters")
        self.synchronize()
        return {"step":index+1,"loss":float(loss.detach()),"components":components,"gradient_l2":gradients}

    @staticmethod
    def tensor_stats(tensor):
        # Per-sample reductions avoid a second batch-sized double allocation.
        count, total, squares = tensor.numel(), 0., 0.
        for sample in tensor:
            v = sample.detach().double()
            total += float(v.sum()); squares += float(v.square().sum())
        mean = total/count
        return {"shape":list(tensor.shape),"mean":mean,"rms":(squares/count)**.5,
                "std":max(0.,squares/count-mean*mean)**.5}

    def snapshot(self, images, measure=False):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        flags = [(t,t.requires_grad) for t in self.model.parameters()]
        self.model.requires_grad_(False)
        stats, handles = {}, []
        if measure:
            def adaptor_hook(module,args,out):
                if "features_before_adaptor" not in stats:
                    stats["features_before_adaptor"] = self.tensor_stats(args[0])
                    stats["features_after_adaptor"] = self.tensor_stats(out)
            def noise_hook(module,args,out):
                clean, mixed, masks = args[0], out[0], out[1]
                noise_sq, feature_sq, count, outside = 0., 0., 0, 0.
                # The official generator concatenates two copies, not interleaved copies.
                for i in range(16):
                    c = clean[i%8].detach()
                    d = mixed[i].detach()-c
                    inside = masks[i,0].bool()
                    if inside.any():
                        noise_sq += float(d[:,inside].double().square().sum())
                        feature_sq += float(c[:,inside].double().square().sum())
                        count += int(inside.sum())*c.shape[0]
                    if (~inside).any(): outside = max(outside,float(d[:,~inside].abs().max()))
                if not count or noise_sq <= 0 or outside != 0:
                    raise ValueError("No valid fixed synthetic perturbation")
                stats["realized_noise"] = {"masked_feature_values":count,
                    "masked_rms":(noise_sq/count)**.5,"matched_clean_rms":(feature_sq/count)**.5,
                    "noise_to_matched_feature_rms":(noise_sq/feature_sq)**.5 if feature_sq else None,
                    "outside_mask_max_abs":outside,
                    "interpretation":"small relative RMS alone does not prove unlearnability"}
            handles = [self.model.adaptor.register_forward_hook(adaptor_hook),
                       self.model.anomaly_generator.register_forward_hook(noise_hook)]
        try:
            arrays, checks = super().trial(images,p.POLICY["seed"])
        finally:
            for handle in handles: handle.remove()
            for tensor, flag in flags: tensor.requires_grad_(flag)
        return arrays, checks, stats

    def save(self, path, signature, step):
        self.optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        self.torch.save({"purpose":p.NAME,"identity":signature,"step":step,
            "model":self.model.state_dict(),"optimizer":self.optimizer.state_dict()},path)

    def load(self, path, signature, step):
        payload = self.torch.load(path,map_location="cpu",weights_only=True)
        if payload["purpose"] != p.NAME or payload["identity"] != signature or payload["step"] != step:
            raise ValueError("Limited experiment checkpoint identity differs")
        del self.optimizer
        self.torch.cuda.empty_cache()
        self.model.load_state_dict(payload["model"],strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(payload["optimizer"])
        params = [t for g in self.optimizer.param_groups for t in g["params"]]
        if step and (len(self.optimizer.state) != len(params) or
                     any(int(s["step"].item()) != step for s in self.optimizer.state.values())):
            raise ValueError("Optimizer step restoration differs")
        self.model.eval()
        # No stochastic continuation is needed: every forward resets all RNGs to 42.

    def release(self):
        if hasattr(self,"optimizer"): del self.optimizer
        super().release()
