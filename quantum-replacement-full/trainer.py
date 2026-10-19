"""Resumable training; atomic checkpoints at optimizer-step boundaries."""
import bootstrap
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from config import write_json
from data import VSRDataset, audit_splits
from replacement_model import build, save_model, load_model
from settings import run_dir
from replacement_metrics import evaluate
from feature_cache import make_cache, scores_from_feature
from resource_monitor import Monitor, synchronize
from utils import seed_everything, parameter_counts, environment, sha256
from settings import source_hashes
from compare_results import compare_rows


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');write_json(tmp,value);os.replace(tmp,path)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda'] is not None: torch.cuda.set_rng_state_all(s['cuda'])

def checkpoint(m,path,c,state=None):
    path=Path(path);tmp=path.with_name(path.name+'.pending');old=path.with_name(path.name+'.previous')
    if tmp.exists(): shutil.rmtree(tmp)
    save_model(m,tmp,c)
    if state is not None:
        torch.save(state,tmp/'training_state.pt')
        write_json(tmp/'training_state_hash.json',{'sha256':sha256(tmp/'training_state.pt')})
    if old.exists(): shutil.rmtree(old)
    if path.exists(): path.rename(old)
    tmp.rename(path)
    if old.exists(): shutil.rmtree(old)

def recover(path):
    path=Path(path);old=path.with_name(path.name+'.previous')
    if not path.exists() and old.exists(): old.rename(path)

def dataset_signature(dataset):
    # Encoded image bytes are hashed once per unique image per invocation.
    return {'manifest':dataset.manifest_hash,'ids':[r['id'] for r in dataset.rows],
            'images':{n:sha256(dataset.root/'images'/n) for n in sorted({r['image'] for r in dataset.rows})}}


def train(c,resume=False):
    out=run_dir(c);out.mkdir(parents=True,exist_ok=True)
    trainset=VSRDataset(c['data']['root'],c['data']['track'],'train',c['data']['train_limit'])
    dev=VSRDataset(c['data']['root'],c['data']['track'],'dev',c['data']['eval_limit'])
    signature={'config':c,'environment':environment(),'source_hashes':source_hashes(),
               'train':dataset_signature(trainset),'dev':dataset_signature(dev)}
    registration=out/'registry.json'
    if registration.exists():
        saved=json.loads(registration.read_text())
        if saved['signature']!=signature: raise ValueError('Run configuration, code, environment or inputs changed. Choose output.run_name for a new run.')
        if (out/'completion.json').exists():
            print('Completed compatible run reused: '+str(out));return
        if not resume: raise ValueError('Interrupted run exists. Use --resume or a new output.run_name.')
    else:
        if resume: raise ValueError('No registered run to resume')
        audit=audit_splits(c['data']['root'],c['data']['track'],('train','dev'))
        if audit['overlaps']['train__dev']['image_caption_pairs']: raise ValueError('Train/dev duplicate image-caption pairs')
        atomic_json(registration,{'signature':signature,'audit':audit})
    # Advisory OS lock automatically releases on exit, including crashes.
    import fcntl
    lock=(out/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session=time.time_ns()
    try:
        with Monitor(c['monitor']['interval_seconds']) as monitor:
            _train(c,out,trainset,dev,resume)
    finally:
        if 'monitor' in locals() and hasattr(monitor,'result'):
            atomic_json(out/('resources_%d.json'%session),monitor.result)
        lock.close()


def _train(c,out,trainset,dev,resume):
    e,t=c['experiment'],c['training'];seed_everything(e['seed'])
    recover(out/'last');recover(out/'best')
    m=load_model(out/'last',c['model']['device']) if resume and (out/'last').exists() else build(c)
    if c['model']['gradient_checkpointing']:
        from replacement_model import base_of
        b=base_of(m)
        if b.settings.mode not in ('zero_shot','full_frozen','frozen_head'):
            b.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            b.settings.gradient_checkpointing=True
    write_json(out/'replacement_architecture.json',m.replacement_metadata)
    first=trainset[0];m.context(first['pixels'],first['caption'])
    params=[p for p in m.parameters() if p.requires_grad]
    write_json(out/'parameters.json',{'counts':parameter_counts(m),'trainable_names':[n for n,p in m.named_parameters() if p.requires_grad]})
    if not params:
        checkpoint(m,out/'best',c)
        result=evaluate(m,dev,draws=c['evaluation']['bootstrap_draws'],seed=e['seed'])
        write_json(out/'best_development.json',result)
        del m;gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        m=load_model(out/'best',c['model']['device'])
        reloaded=evaluate(m,dev,draws=c['evaluation']['bootstrap_draws'],seed=e['seed'])
        comparison=compare_rows(result['examples'],reloaded['examples'])
        write_json(out/'reload_comparison.json',comparison);write_json(out/'clean_results.json',reloaded)
        if not comparison['passed']: raise RuntimeError('Control reload mismatch')
        atomic_json(out/'completion.json',{'control':True,'optimizer_steps':0,'reload_verified':True,'metrics':reloaded['metrics']});return
    cache_start=time.monotonic()
    train_cache=None
    dev_cache=None
    cache_seconds=time.monotonic()-cache_start
    defaults={'projector':1e-5,'full_unfrozen':1e-5,'quantum_projector':1e-5,'quantum_full':1e-5}
    lr=t['lr'] if t['lr'] is not None else defaults.get(e['mode'],1e-4)
    if lr<=0: raise ValueError('Training LR must be positive')
    write_json(out/'optimizer.json',{'optimizer':'AdamW','resolved_learning_rate':lr,'weight_decay':t['weight_decay'],
               'effective_batch_size':t['accumulation'],'microbatch_size':1,'resolved_model':vars(m.settings)})
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=t['weight_decay'])
    total=math.ceil(len(trainset)/t['accumulation'])*t['epochs'];warmup=max(1,int(total*t['warmup_fraction']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min((s+1)/warmup,max(0.,(total-s)/max(1,total-warmup))))
    generator=torch.Generator().manual_seed(e['seed'])
    progress={'epoch':0,'cursor':0,'order':None,'epoch_loss_sum':0.,'epoch_correct':0,
              'updates':0,'best':float('inf'),'stale':0,'history':[],
              'training_seconds':0.,'validation_seconds':0.,'cache_seconds':cache_seconds}
    if resume and (out/'last/training_state.pt').exists():
        statepath=out/'last/training_state.pt'
        if sha256(statepath)!=json.loads((out/'last/training_state_hash.json').read_text())['sha256']: raise ValueError('Resume state hash mismatch')
        # Only load locally generated, trusted training state (contains Python/NumPy RNG).
        state=torch.load(statepath,map_location='cpu',weights_only=False)
        progress=state['progress'];optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['generator']);restore_rng(state['rng'])
        progress['cache_seconds']+=cache_seconds
    def save_last():
        checkpoint(m,out/'last',c,{'progress':progress,'optimizer':optimizer.state_dict(),
                   'scheduler':scheduler.state_dict(),'generator':generator.get_state(),'rng':rng_state()})
    while progress['epoch']<t['epochs'] and progress['stale']<t['patience']:
        if progress['order'] is None: progress['order']=torch.randperm(len(trainset),generator=generator).tolist()
        order=progress['order'];m.train()
        for start in range(progress['cursor'],len(order),t['accumulation']):
            synchronize(m.device_);tick=time.monotonic();group=order[start:start+t['accumulation']]
            optimizer.zero_grad(set_to_none=True)
            for i in group:
                if train_cache is None:
                    sample=trainset[i];ctx=m.context(sample['pixels'],sample['caption'])
                    scores=m(sample['pixels'].unsqueeze(0).to(m.device_),ctx);label=sample['label']
                else:
                    entry=train_cache[i];label=entry['row']['label']
                    scores=scores_from_feature(m,entry['feature'].to(m.device_),entry['label_ids'])
                loss=torch.nn.functional.cross_entropy(scores,torch.tensor([label],device=m.device_))
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                (loss/len(group)).backward();progress['epoch_loss_sum']+=loss.item()
                progress['epoch_correct']+=int(scores.argmax(-1).item()==label)
            norm=torch.nn.utils.clip_grad_norm_(params,t['clip_norm'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();progress['updates']+=1;progress['cursor']=start+len(group)
            synchronize(m.device_);progress['training_seconds']+=time.monotonic()-tick
            if progress['updates']%t['log_every_updates']==0:
                print('Epoch %d/%d, update %d, examples %d/%d, gradient norm %.5g'%(progress['epoch']+1,t['epochs'],progress['updates'],progress['cursor'],len(order),norm.item()),flush=True)
            if progress['updates']%t['save_every_updates']==0: save_last()
        result=evaluate(m,dev,dev_cache)
        progress['validation_seconds']+=result['seconds']
        row={'epoch':progress['epoch']+1,'train_loss':progress['epoch_loss_sum']/len(order),
             'train_accuracy_online':progress['epoch_correct']/len(order),'dev':result['metrics'],
             'updates':progress['updates'],'training_seconds_cumulative':progress['training_seconds'],
             'validation_seconds_cumulative':progress['validation_seconds'],'lr':scheduler.get_last_lr()[0]}
        progress['history'].append(row);print(row,flush=True)
        if result['metrics']['loss']<progress['best']-t['min_delta']:
            progress['best']=result['metrics']['loss'];progress['stale']=0
            checkpoint(m,out/'best',c);atomic_json(out/'best_development.json',result)
        else: progress['stale']+=1
        progress.update(epoch=progress['epoch']+1,cursor=0,order=None,epoch_loss_sum=0.,epoch_correct=0)
        atomic_json(out/'history.json',progress['history']);save_last()
    # Independent reconstruction verifies adapter, head, LoRA and full weights.
    device=m.device_;del params,optimizer,scheduler,m;gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    m=load_model(out/'best',str(device))
    reloaded=evaluate(m,dev,draws=c['evaluation']['bootstrap_draws'],seed=e['seed'])
    reference=json.loads((out/'best_development.json').read_text())
    comparison=compare_rows(reference['examples'],reloaded['examples'])
    atomic_json(out/'reload_comparison.json',comparison);atomic_json(out/'clean_results.json',reloaded)
    if not comparison['passed']: raise RuntimeError('Reload/cache parity failed; inspect reload_comparison.json')
    atomic_json(out/'completion.json',{'optimizer_steps':progress['updates'],'epochs_completed':progress['epoch'],
        'best_dev_loss':progress['best'],'training_seconds':progress['training_seconds'],
        'validation_seconds':progress['validation_seconds'],'cache_prepare_seconds':progress['cache_seconds'],
        'cached_training':train_cache is not None,'reload_verified':True,'metrics':reloaded['metrics'],
        'training_examples_per_second':len(trainset)*progress['epoch']/max(progress['training_seconds'],1e-9)})
