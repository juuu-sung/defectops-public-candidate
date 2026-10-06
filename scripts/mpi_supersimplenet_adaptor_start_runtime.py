"""Same preserved step-zero head in both arms; fresh matched AdamW states."""
from scripts import mpi_supersimplenet_adaptor_start_protocol as p
from scripts.mpi_supersimplenet_varying_runtime import Runtime as VaryingRuntime, environment
from scripts.mpi_supersimplenet_matched_eval_runtime import Runtime as MatchedRuntime


class Runtime(VaryingRuntime):
    def initialize(self,step_zero,identity,arm):
        if hasattr(self,'arm') or arm not in p.ARMS:
            raise ValueError('Unregistered or repeated initialization')
        if self.fingerprints() != identity['checkpoints']['28800']['fingerprints']:
            raise ValueError('Original final model differs before initialization')
        payload=self.torch.load(step_zero,map_location='cpu',weights_only=True)
        if (payload['purpose'] != p.parity.e.p.NAME or payload['identity'] != identity['fit_signature']
                or payload['step'] != 0 or payload['optimizer']['state']):
            raise ValueError('Need preserved formal step-zero checkpoint')
        final_adaptor={k:v.detach().cpu().clone() for k,v in self.model.adaptor.state_dict().items()}
        self.model.load_state_dict(payload['model'],strict=True)
        if self.fingerprints() != identity['checkpoints']['0']['fingerprints']:
            raise ValueError('Step-zero tensor fingerprints differ')
        if arm=='trained_adaptor':self.model.adaptor.load_state_dict(final_adaptor,strict=True)
        self.model.requires_grad_(False)
        self.model.adaptor.requires_grad_(True);self.model.segdec.requires_grad_(True)
        del self.optimizer
        self.optimizer=self.new_optimizer()
        self.model.eval();self.arm=arm
        record={'arm':arm,'fingerprints':self.fingerprints(),'optimizer_empty':not bool(self.optimizer.state),
                'head_source':'preserved formal step-000000, including BatchNorm buffers'}
        p.validate_initialization(record,identity,arm)
        return record

    def evaluate_seed(self,images,seed):
        if seed not in p.POLICY['evaluation_seeds']:raise ValueError('Unregistered evaluation seed')
        self.optimizer.zero_grad(set_to_none=True);self.model.eval()
        flags=[(t,t.requires_grad) for t in self.model.parameters()]
        self.model.requires_grad_(False)
        try:return MatchedRuntime.trial(self,images,seed)
        finally:
            for t,flag in flags:t.requires_grad_(flag)

    def feature_statistics(self,images):
        self.model.eval()
        with self.torch.no_grad():
            features=self.model.feature_extractor(self.tensor(images))
            adapted=self.model.adaptor(features)
            return {'before_adaptor':self.tensor_stats(features),'after_adaptor':self.tensor_stats(adapted)}

    def step(self,images,index):
        if not hasattr(self,'arm'):raise ValueError('Initialize arm before training')
        return super().step(images,index)

    def save(self,path,signature,step):
        self.optimizer.zero_grad(set_to_none=True);self.synchronize()
        self.torch.save({'purpose':p.NAME,'identity':signature,'arm':self.arm,'step':step,
                         'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict()},path)

    def load(self,path,signature,step):
        payload=self.torch.load(path,map_location='cpu',weights_only=True)
        if (payload['purpose'] != p.NAME or payload['identity'] != signature
                or payload['arm'] != self.arm or payload['step'] != step):
            raise ValueError('Cross-arm or unregistered checkpoint')
        del self.optimizer;self.torch.cuda.empty_cache()
        self.model.load_state_dict(payload['model'],strict=True)
        self.optimizer=self.new_optimizer();self.optimizer.load_state_dict(payload['optimizer'])
        params=[t for group in self.optimizer.param_groups for t in group['params']]
        if (step == 0 and self.optimizer.state) or (step and (len(self.optimizer.state) != len(params)
                or any(int(s['step'].item()) != step for s in self.optimizer.state.values()))):
            raise ValueError('Optimizer restoration differs')
        self.model.eval()
