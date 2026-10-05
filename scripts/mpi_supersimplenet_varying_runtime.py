"""One changed condition: deterministic new synthetic draw per optimizer update."""
import hashlib
import random

import numpy as np

from scripts import mpi_supersimplenet_varying_protocol as p
from scripts.mpi_supersimplenet_learnability_runtime import Runtime as FixedRuntime, environment
from scripts.mpi_supersimplenet_synthetic_runtime import Runtime as FrozenRuntime


class Runtime(FixedRuntime):
    # Optimizer construction/components are inherited unchanged from the fixed-task run.
    def step(self, images, index):
        if getattr(self,"control_only",False):
            raise ValueError("Fixed control is read-only; optimizer updates forbidden")
        torch, model = self.torch, self.model
        seed = p.training_seed(index)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model.train(); model.feature_extractor.eval()
        self.optimizer.zero_grad(set_to_none=True)
        maps,scores,masks,labels = model(self.tensor(images),
            masks=torch.zeros((8,256,256),device="cuda"),labels=torch.zeros(8,device="cuda"))
        if (maps.shape != (16,1,64,64) or masks.shape != maps.shape or scores.shape != (16,)
                or labels.shape != scores.shape or not ((masks == 0)|(masks == 1)).all()
                or not torch.equal(labels.bool(),masks.flatten(1).any(1))
                or not all(torch.isfinite(t).all() for t in (maps,scores,masks,labels))):
            raise ValueError("Official synthetic output contract differs")
        if not labels.bool().any() or labels.bool().all():
            raise ValueError(f"Seed {seed} has a missing class; preserve, do not reroll")
        target = hashlib.sha256(masks[:,0].bool().cpu().numpy().tobytes()+labels.bool().cpu().numpy().tobytes()).hexdigest()
        components = self.components(maps,scores,masks,labels)
        loss = self.loss(maps,scores,masks,labels)
        if loss.ndim or not torch.isfinite(loss): raise ValueError("Nonfinite scalar loss")
        loss.backward()
        gradients = {}
        for name,module in (("adaptor",model.adaptor),("segmentation",model.segdec.seg_head),
                            ("classification_conv",model.segdec.cls_conv),("classification_fc",model.segdec.cls_fc)):
            params = list(module.parameters())
            if any(t.grad is None or not torch.isfinite(t.grad).all() for t in params):
                raise ValueError(f"Missing/nonfinite {name} gradients")
            gradients[name] = sum(float(t.grad.detach().double().square().sum()) for t in params)**.5
        if any(t.grad is not None for t in model.feature_extractor.parameters()):
            raise ValueError("Frozen backbone received gradients")
        self.optimizer.step()
        if any(not torch.isfinite(t).all() for g in self.optimizer.param_groups for t in g["params"]):
            raise ValueError("Nonfinite updated parameters")
        self.synchronize()
        return {"step":index+1,"seed":seed,"target_sha256":target,
                "positive_samples":int(labels.sum()),"positive_pixels":int(masks.sum()),
                "loss":float(loss.detach()),"components":components,"gradient_l2":gradients}

    def evaluate_seed(self, images, seed):
        if seed not in p.POLICY["evaluation_seeds"]: raise ValueError("Unregistered evaluation seed")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        flags = [(t,t.requires_grad) for t in self.model.parameters()]
        self.model.requires_grad_(False)
        try:
            return FrozenRuntime.trial(self,images,seed)
        finally:
            for tensor,flag in flags: tensor.requires_grad_(flag)

    def save(self, path, signature, step):
        self.optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        self.torch.save({"purpose":p.NAME,"identity":signature,"step":step,
            "model":self.model.state_dict(),"optimizer":self.optimizer.state_dict()},path)

    def load(self, path, signature, step):
        payload = self.torch.load(path,map_location="cpu",weights_only=True)
        if payload["purpose"] != p.NAME or payload["identity"] != signature or payload["step"] != step:
            raise ValueError("Varying-task checkpoint identity differs")
        del self.optimizer
        self.torch.cuda.empty_cache()
        self.model.load_state_dict(payload["model"],strict=True)
        self.optimizer = self.new_optimizer()
        self.optimizer.load_state_dict(payload["optimizer"])
        params = [t for g in self.optimizer.param_groups for t in g["params"]]
        if step and (len(self.optimizer.state) != len(params) or
                     any(int(s["step"].item()) != step for s in self.optimizer.state.values())):
            raise ValueError("Optimizer restoration differs")
        self.model.eval()
        # The next update always resets all RNGs from its locked update index.

    def load_fixed_control(self, path, signature):
        FixedRuntime.load(self,path,signature,500)
        self.control_only = True
