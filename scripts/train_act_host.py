"""Standalone ACT training for validated Dobot HDF5; no robot connections."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Windows(Dataset):
    def __init__(self, directory, ids, stats, chunk, validation=False):
        self.directory, self.ids, self.stats, self.chunk = directory, ids, stats, chunk
        self.windows = []
        if validation:
            for i in ids:
                with h5py.File(directory / f'episode_{i}.hdf5', 'r') as f:
                    n = len(f['action'])
                self.windows.extend((i, j) for j in sorted(set([0, n//4, n//2, 3*n//4, n-1])))

    def __len__(self):
        return len(self.windows) if self.windows else len(self.ids)

    def __getitem__(self, index):
        from dobot_control.act_dataset import load_act_window
        if self.windows:
            i, start = self.windows[index]
        else:
            i = self.ids[index]
            with h5py.File(self.directory / f'episode_{i}.hdf5', 'r') as f:
                start = np.random.randint(len(f['action']))
        image, qpos, action, mask = load_act_window(self.directory / f'episode_{i}.hdf5', start, self.chunk)
        return (torch.from_numpy(image.transpose(0,3,1,2).copy()).float()/255,
                torch.from_numpy((qpos-self.stats['qpos_mean'])/self.stats['qpos_std']),
                torch.from_numpy((action-self.stats['action_mean'])/self.stats['action_std']),
                torch.from_numpy(mask))


def make_policy(config, run):
    from policy import ACTPolicy
    saved = sys.argv
    # Upstream parser requires these CLI flags before applying args_override.
    sys.argv = ['act', '--ckpt_dir', str(run), '--policy_class', 'ACT', '--task_name', 'dobot',
                '--seed', str(config['seed']), '--num_epochs', '1']
    try:
        return ACTPolicy(config)
    finally:
        sys.argv = saved


def gpu_batch(batch):
    return [x.cuda(non_blocking=True) for x in batch]


def evaluate(policy, loader):
    policy.eval()
    totals, count = {}, 0
    # Fixed latent sampling makes validation comparable without consuming training RNG.
    with torch.random.fork_rng(devices=[0]), torch.inference_mode():
        torch.manual_seed(12345)
        for batch in loader:
            image, qpos, action, mask = gpu_batch(batch)
            loss = policy(qpos, image, action, mask)
            for k,v in loss.items(): totals[k] = totals.get(k,0.) + v.item()*len(image)
            count += len(image)
    policy.train()
    return {k:v/count for k,v in totals.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validate-every',type=int,default=250)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    args.run.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.dataset/'training_support'))
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.benchmark=False
    split=json.loads((args.dataset/'splits.json').read_text())
    stats={k:np.asarray(v,dtype=np.float32) for k,v in json.loads((args.dataset/'train_normalization_stats.json').read_text()).items()}
    train_ids,val_ids=split['train'],split['validation']
    if args.smoke:
        train_ids=list(dict.fromkeys(train_ids[i] for i in np.linspace(0, len(train_ids)-1, min(4, len(train_ids))).astype(int)))
        val_ids=list(dict.fromkeys([val_ids[0],val_ids[-1]]))
    config=dict(lr=1e-5,lr_backbone=1e-5,weight_decay=1e-4,backbone='resnet18',
                camera_names=['top','left'],num_queries=30,chunk_size=30,hidden_dim=512,
                dim_feedforward=3200,enc_layers=4,dec_layers=7,nheads=8,kl_weight=10,seed=0)
    settings=dict(vars(args), policy=config, train_ids=train_ids, validation_ids=val_ids,
                  precision='float32',image_size=[480,640],validation_positions='0,25%,50%,75%,last',
                  normalization=f'{len(split["train"])} training episodes only; ImageNet inside ACTPolicy once',
                  image_preprocess=json.loads((args.dataset/'image_preprocess.json').read_text()) if (args.dataset/'image_preprocess.json').exists() else {},
                  upstream_commit='742c753c0d4a5d87076c8f69e5628c79a8cc5488')
    (args.run/'config.json').write_text(json.dumps(settings,indent=2,default=str))
    (args.run/'splits.json').write_text(json.dumps(split,indent=2))
    (args.run/'normalization.json').write_text((args.dataset/'train_normalization_stats.json').read_text())
    # Sampling with replacement keeps batch size stable for small smoke subsets.
    data=Windows(args.dataset,train_ids,stats,30)
    sampler=torch.utils.data.RandomSampler(data,replacement=True,num_samples=args.steps*args.batch_size)
    loader=DataLoader(data,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,
                      pin_memory=True,persistent_workers=args.workers>0)
    val=DataLoader(Windows(args.dataset,val_ids,stats,30,True),batch_size=args.batch_size,
                   num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    policy=make_policy(config,args.run)
    optimizer=policy.configure_optimizers()
    start_step=0; best=float('inf')
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert ck['config']==config and ck['split']==split
        policy.load_state_dict(ck['policy']); optimizer.load_state_dict(ck['optimizer'])
        start_step=ck['step']; best=ck['best']
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
    def save(name,step):
        payload=dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
                     config=config,split=split,stats=stats,python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        temp=args.run/(name+'.tmp'); torch.save(payload,temp); temp.replace(args.run/name)
    policy.train(); torch.cuda.reset_peak_memory_stats()
    durations=[]; iterator=iter(loader); wall=time.perf_counter()
    with (args.run/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(start_step+1,args.steps+1):
            begin=time.perf_counter()
            image,qpos,action,mask=gpu_batch(next(iterator))
            optimizer.zero_grad(set_to_none=True)
            losses=policy(qpos,image,action,mask)
            assert torch.isfinite(losses['loss']).item(), 'nonfinite loss'
            losses['loss'].backward()
            grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            duration=time.perf_counter()-begin
            if step>start_step+3: durations.append(duration)
            row=dict(step=step,seconds=duration,grad_norm=float(grad),**{k:float(v) for k,v in losses.items()})
            if step%args.validate_every==0 or step==args.steps:
                row['validation']=evaluate(policy,val)
                assert all(np.isfinite(v) for v in row['validation'].values())
                if row['validation']['loss']<best:
                    best=row['validation']['loss']; save('best.pt',step)
            if step%args.save_every==0: save(f'step_{step}.pt',step)
            log.write(json.dumps(row)+'\n')
            if args.smoke or step%25==0: print(json.dumps(row),flush=True)
        save('last.pt',args.steps)
    peak=torch.cuda.max_memory_allocated()/1024**3
    # Check saved policy can be reconstructed and produces the same prior prediction.
    policy.eval()
    with torch.inference_mode(): reference=policy(qpos[:1],image[:1]).cpu()
    restored=make_policy(config,args.run)
    checkpoint=torch.load(args.run/'last.pt',map_location='cpu',weights_only=False)
    restored.load_state_dict(checkpoint['policy']); restored.eval()
    with torch.inference_mode(): prediction=restored(qpos[:1],image[:1]).cpu()
    torch.testing.assert_close(reference,prediction,rtol=0,atol=0)
    result=dict(passed=True,steps=args.steps-start_step,batch_size=args.batch_size,
                mean_step_seconds=float(np.mean(durations)),peak_allocated_gib=peak,
                elapsed_seconds=time.perf_counter()-wall,checkpoint_reload='exact prediction match',
                inference_shape=list(prediction.shape),best_validation_loss=best,
                train_episodes=len(train_ids),validation_episodes=len(val_ids))
    (args.run/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
