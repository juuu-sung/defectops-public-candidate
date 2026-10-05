"""Exercise official frozen model with synthetic features; no optimizer/backward."""
from contextlib import contextmanager
import random

import numpy as np

from scripts import mpi_supersimplenet_synthetic_protocol as q
from scripts.mpi_supersimplenet_evaluation_runtime import Runtime as BaseRuntime, environment
from scripts.mpi_supersimplenet_runtime import state_digest


@contextmanager
def preserved_buffers_and_modes(model):
    modes = [(m, m.training) for m in model.modules()]
    buffers = {name: value.detach().clone() for name,value in model.named_buffers()}
    try:
        yield buffers
    finally:
        for name,value in model.named_buffers():
            value.copy_(buffers[name])
        for module, mode in modes:
            module.training = mode


class Runtime(BaseRuntime):
    def trial(self, images, seed):
        torch, model = self.torch, self.model
        if len(images) != 8 or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Eight normals and frozen parameters required")
        before = state_digest(model)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        captured, handles = {}, []
        try:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.no_grad(), preserved_buffers_and_modes(model):
                x = self.tensor(images)
                manual = (torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32)/255 for im in images]))
                          .permute(0,3,1,2).to(x.device))
                manual = (manual-torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
                preprocessing_difference = float((manual-x).abs().max())
                if preprocessing_difference > 1e-6:
                    raise ValueError("ImageNet preprocessing is not applied exactly once")
                del manual
                # Hooks capture precisely the features used by the official training forward.
                handles.append(model.anomaly_generator.register_forward_pre_hook(lambda module,args: captured.update(clean=args[0])))
                handles.append(model.anomaly_generator.register_forward_hook(lambda module,args,out: captured.update(mixed=out[0])))
                model.train(); model.feature_extractor.eval()
                with preserved_buffers_and_modes(model) as buffers:
                    train_map, train_score, masks, labels = model(x, masks=torch.zeros((8,256,256),device=x.device), labels=torch.zeros(8,device=x.device))
                    changed = [n for n,b in model.named_buffers() if not torch.equal(b,buffers[n])]
                    if any(not n.startswith("segdec.") for n in changed):
                        raise ValueError("Unexpected non-head buffer mutation")
                for handle in handles: handle.remove()
                handles.clear()
                model.eval()
                mixed, clean = captured["mixed"], captured["clean"]
                if (mixed.shape != (16,1536,64,64) or clean.shape != (8,1536,64,64)
                        or masks.shape != (16,1,64,64) or labels.shape != (16,)
                        or not ((masks == 0)|(masks == 1)).all()
                        or not torch.equal(labels.bool(), masks.flatten(1).any(1))):
                    raise ValueError("Official synthetic feature or target shape differs")
                if not torch.isfinite(mixed).all():
                    raise ValueError("Nonfinite mixed features")
                duplicated = torch.cat((clean,clean))
                # Chunk by sample to avoid materializing a second large delta tensor.
                outside_max, inside_max = 0., 0.
                for i in range(16):
                    delta = (mixed[i]-duplicated[i]).abs()
                    inside = masks[i,0].bool()
                    if (~inside).any(): outside_max = max(outside_max,float(delta[:,~inside].max()))
                    if inside.any(): inside_max = max(inside_max,float(delta[:,inside].max()))
                if outside_max != 0 or (labels.any() and inside_max == 0):
                    raise ValueError("Synthetic noise not confined to its recorded mask")
                del delta
                eval_map, eval_score = model.segdec(mixed)
                clean_map, clean_score = model.segdec(duplicated)
                unchanged = ~labels.bool()
                normal_score_delta = float((eval_score[unchanged]-clean_score[unchanged]).abs().max()) if unchanged.any() else None
                normal_map_delta = float((eval_map[unchanged]-clean_map[unchanged]).abs().max()) if unchanged.any() else None
                if unchanged.any() and max(normal_score_delta,normal_map_delta) > 1e-4:
                    raise ValueError("Unperturbed eval features changed prediction")
                # Native full public eval vs manually separated path, same batch and transforms.
                public = model(x)
                expanded = model.anomaly_map_generator(clean_map[:8], final_size=(256,256))
                full_map_difference = float((expanded-public.anomaly_map).abs().max())
                full_score_difference = float((clean_score[:8]-public.pred_score.reshape(-1)).abs().max())
                if max(full_map_difference,full_score_difference) > 1e-4:
                    raise ValueError("Separated eval path differs from official public forward")
                arrays = {"masks":masks[:,0].cpu().numpy().astype(bool), "labels":labels.cpu().numpy().astype(bool)}
                losses = {}
                from anomalib.models.image.supersimplenet.loss import SSNLoss
                loss = SSNLoss()
                for name, maps, scores in (("train_mixed",train_map,train_score),("eval_mixed",eval_map,eval_score),("eval_clean",clean_map,clean_score)):
                    arrays[name+"_maps"] = maps[:,0].cpu().numpy().copy()
                    arrays[name+"_scores"] = scores.reshape(16).cpu().numpy().copy()
                    if name != "eval_clean":
                        losses[name] = {"map_focal":float(loss.focal_loss(maps,masks)),
                            "map_truncated_l1":float(loss.trunc_l1_loss(maps,masks)),
                            "classification_focal":float(loss.focal_loss(scores,labels)),
                            "total":float(loss(maps,scores,masks,labels))}
                checks = {"preprocessing_max_abs_difference":preprocessing_difference,
                    "public_eval_map_max_abs_difference":full_map_difference,"public_eval_score_max_abs_difference":full_score_difference,
                    "temporary_changed_buffers":changed, "noise_outside_mask_max":outside_max,"noise_inside_mask_max":inside_max,
                    "unperturbed_eval_score_max_abs_difference":normal_score_delta,
                    "unperturbed_eval_map_max_abs_difference":normal_map_delta,
                    "no_gradients":not any(p.grad is not None for p in model.parameters()),
                    "loss_components":losses, "same_features_both_modes":True}
        finally:
            for handle in handles: handle.remove()
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        after = state_digest(model)
        if before != after or model.training or not checks["no_gradients"]:
            raise ValueError("Model parameters/buffers/mode were not preserved")
        checks.update(state_before_sha256=before,state_after_sha256=after,state_unchanged=True)
        return arrays, checks
