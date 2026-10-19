"""Exact frozen final-feature reuse, with per-example input hashes and parity checks."""
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from experiment_config import digest
from experiment_model import base_of
from quantum_shared import QuantumVLM
from utils import input_fingerprint, environment, source_hashes, sha256
from config import write_json


def eligible(m):
    b=base_of(m)
    return not any(p.requires_grad for p in b.backbone.parameters()) and (
        (isinstance(m,QuantumVLM) and m.placement=='q4') or
        (not isinstance(m,QuantumVLM) and b.head is not None))

def scores_from_feature(m,h,ids):
    b=base_of(m)
    if isinstance(m,QuantumVLM): h=m.adapter(h)
    if b.head is not None: return b.head(h.float())
    return F.linear(h,b.backbone.lm_head.weight[list(ids)]).float()

def extract(m,image,ctx):
    b=base_of(m)
    if isinstance(m,QuantumVLM): m.bypass=True
    try:
        with torch.no_grad(): return b._hidden(image,ctx)[:,-1].detach().clone()
    finally:
        if isinstance(m,QuantumVLM): m.bypass=False

def make_cache(m,dataset,c):
    from prefix_cache import eligible as prefix_eligible, make_cache as make_prefix_cache
    if prefix_eligible(m): return make_prefix_cache(m,dataset,c)
    if not eligible(m): return None
    cfg=c['cache'];start=time.monotonic();m.eval();b=base_of(m)
    key={'model':vars(b.settings),'environment':environment(),'manifest':dataset.manifest_hash,
         'ids':[r['id'] for r in dataset.rows],'sources':source_hashes()}
    # Learned readout/quantum weights are downstream and intentionally excluded.
    folder=Path(cfg['directory'])/digest(key);folder.mkdir(parents=True,exist_ok=True)
    result=[];checks=[]
    params=[p for p in m.parameters() if p.requires_grad]
    for i in range(len(dataset)):
        row=dataset[i];ctx=m.context(row['pixels'],row['caption'])
        ids=tuple(a[0] for a in ctx.label_ids)
        if any(len(a)!=1 for a in ctx.label_ids): raise ValueError('Cache requires single-token answers')
        fingerprint=input_fingerprint(row['pixels'],ctx.prompt)
        filename=folder/(digest({'id':row['id'],**fingerprint})+'.pt')
        image=row['pixels'].unsqueeze(0).to(m.device_)
        checksum=filename.with_suffix('.sha256')
        if filename.exists() and checksum.exists() and checksum.read_text()==sha256(filename):
            h=torch.load(filename,map_location=m.device_,weights_only=True)
        else:
            h=extract(m,image,ctx);tmp=filename.with_suffix('.tmp')
            torch.save(h.cpu(),tmp);tmp.replace(filename);checksum.write_text(sha256(filename))
        if i<cfg['verify_examples']:
            original=m(image,ctx);cached=scores_from_feature(m,h,ids)
            torch.testing.assert_close(original,cached,rtol=cfg['rtol'],atol=cfg['atol'])
            if params:
                label=torch.tensor([row['label']],device=m.device_)
                g1=torch.autograd.grad(F.cross_entropy(original,label),params,allow_unused=True)
                g2=torch.autograd.grad(F.cross_entropy(cached,label),params,allow_unused=True)
                for a,z in zip(g1,g2):
                    if a is None or z is None:
                        if a is not z: raise ValueError('Cache gradient connectivity mismatch')
                    else: torch.testing.assert_close(a,z,rtol=cfg['rtol'],atol=cfg['atol'])
            checks.append({'id':row['id'],'scores_and_trainable_gradients_passed':True})
        metadata={k:v for k,v in row.items() if k!='pixels'};metadata.update(fingerprint)
        result.append({'feature':h.cpu(),'label_ids':ids,'row':metadata})
        if (i+1)%100==0: print('Cached/verified features %d/%d'%(i+1,len(dataset)),flush=True)
    write_json(folder/'verification.json',{'key':key,'checks':checks,'seconds':time.monotonic()-start})
    return result


def scores_from_entry(m,entry):
    if "prefix" in entry:
        from prefix_cache import score
        return score(m,entry)
    return scores_from_feature(m,entry["feature"].to(m.device_),entry["label_ids"])
