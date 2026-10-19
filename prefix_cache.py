"""Disk-backed frozen prefixes for quantum_only q1/q2/q3; no attack caching."""
import contextlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from experiment_config import digest
from utils import environment, source_hashes, input_fingerprint, sha256


class Captured(Exception):
    pass


def eligible(m):
    from quantum_shared import QuantumVLM
    return (isinstance(m, QuantumVLM) and m.placement in ('q1','q2','q3')
            and m.base.settings.mode == 'full_frozen'
            and not any(p.requires_grad for p in m.base.backbone.parameters()))


def capture(m, image, ctx):
    """Stop at the boundary before the existing quantum hook executes."""
    core=m.base.backbone.model
    target=core.multi_modal_projector if m.placement in ('q1','q2') else core.language_model.layers[m.block_index]
    holder={}
    def pre(module,args):
        holder['hidden']=args[0].detach().cpu().clone();raise Captured()
    def post(module,args,value):
        holder['hidden']=(value[0] if isinstance(value,tuple) else value).detach().cpu().clone();raise Captured()
    hook=(target.register_forward_pre_hook(pre,prepend=True) if m.placement=='q1'
          else target.register_forward_hook(post,prepend=True))
    try:
        with torch.no_grad(): m(image,ctx)
    except Captured:
        pass
    finally:
        hook.remove()
    if 'hidden' not in holder: raise RuntimeError('Prefix boundary was not reached')
    return holder['hidden']


@contextlib.contextmanager
def skip_prefix(decoder,index):
    # Only used with this process's model, outside concurrent model execution.
    saved=[]
    try:
        for layer in decoder.layers[:index+1]:
            saved.append((layer,layer.__dict__.get('forward')))
            layer.forward=lambda hidden_states,*args,**kwargs: hidden_states
        yield
    finally:
        for layer,original in saved:
            if original is None: del layer.forward
            else: layer.forward=original


def score(m,entry):
    core=m.base.backbone.model;decoder=core.language_model
    hidden=entry['prefix'].to(m.device_)
    ids=entry['input_ids'].to(m.device_);mask=entry['attention_mask'].to(m.device_)
    m.position=ids.shape[1]-1;m.hook_calls=0
    if m.placement=='q3':
        # The cached tensor contains the entire sequence after the frozen block.
        # Rebuild masks/rotary positions in the native decoder, skipping only
        # frozen prefix blocks. The existing quantum hook runs at its usual site.
        with skip_prefix(decoder,m.block_index):
            result=decoder(inputs_embeds=hidden,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
    else:
        if m.placement=='q1': projected=core.multi_modal_projector(hidden)
        else: projected=m._adapt(hidden)
        packed,_=core.pack_image_features((projected,),entry['image_sizes'].to(m.device_),
            image_newline=core.image_newline,vision_aspect_ratio=core.config.vision_aspect_ratio)
        features=torch.cat(packed,dim=0)
        embeds=core.get_input_embeddings()(ids)
        features=features.to(device=embeds.device,dtype=embeds.dtype)
        image_mask,_=core.get_placeholder_mask(ids,inputs_embeds=embeds,image_features=features)
        embeds=embeds.masked_scatter(image_mask,features)
        result=decoder(inputs_embeds=embeds,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state
    if m.hook_calls!=1: raise RuntimeError('Cached scoring must apply exactly one quantum adapter')
    return F.linear(result[:,-1],m.base.backbone.lm_head.weight[list(entry['label_ids'])]).float()


class DiskPrefixes:
    """Keep only paths/metadata in RAM; load one prefix at a time."""
    def __init__(self,entries): self.entries=entries
    def __len__(self): return len(self.entries)
    def __getitem__(self,i):
        meta=self.entries[i]
        return dict(meta,prefix=torch.load(meta['path'],map_location='cpu',weights_only=True))


def make_cache(m,dataset,c):
    m.eval();start=time.monotonic();cfg=c['cache']
    if m.base.backbone.config.model_type!='llava_onevision': raise ValueError('Prefix cache requires LLaVA-OneVision')
    key={'schema':1,'model':vars(m.base.settings),'placement':m.placement,'block_index':m.block_index,
         'manifest':dataset.manifest_hash,'environment':environment(),'sources':source_hashes()}
    folder=Path(cfg['directory'])/'prefixes'/digest(key);folder.mkdir(parents=True,exist_ok=True)
    entries=[];checks=[];bytes_total=0
    for i in range(len(dataset)):
        row=dataset[i];ctx=m.context(row['pixels'],row['caption'])
        fingerprint=input_fingerprint(row['pixels'],ctx.prompt)
        path=folder/(digest({'id':row['id'],**fingerprint})+'.pt');checksum=path.with_suffix('.sha256')
        image=row['pixels'].unsqueeze(0).to(m.device_)
        if not (path.exists() and checksum.exists() and checksum.read_text()==sha256(path)):
            h=capture(m,image,ctx);tmp=path.with_suffix('.pending')
            torch.save(h,tmp);tmp.replace(path);checksum.write_text(sha256(path));del h
        entry={'path':str(path),'input_ids':ctx.input_ids.cpu(),'attention_mask':ctx.attention_mask.cpu(),
               'image_sizes':torch.tensor([[ctx.height,ctx.width]],dtype=torch.long),
               'label_ids':tuple(x[0] for x in ctx.label_ids),
               'row':dict({k:v for k,v in row.items() if k!='pixels'},**fingerprint)}
        if i<cfg['verify_examples']:
            params=[p for p in m.parameters() if p.requires_grad]
            label=torch.tensor([row['label']],device=m.device_)
            original=m(image,ctx);loss=F.cross_entropy(original,label)
            g1=torch.autograd.grad(loss,params)
            cached=score(m,dict(entry,prefix=torch.load(path,map_location='cpu',weights_only=True)))
            loss2=F.cross_entropy(cached,label);g2=torch.autograd.grad(loss2,params)
            torch.testing.assert_close(original,cached,rtol=cfg['rtol'],atol=cfg['atol'])
            torch.testing.assert_close(loss,loss2,rtol=cfg['rtol'],atol=cfg['atol'])
            for a,b in zip(g1,g2): torch.testing.assert_close(a,b,rtol=cfg['rtol'],atol=cfg['atol'])
            checks.append({'id':row['id'],'scores_loss_gradients_passed':True})
            del original,cached,loss,loss2,g1,g2
        bytes_total+=path.stat().st_size;entries.append(entry)
        if (i+1)%100==0: print('Frozen prefixes cached/verified %d/%d (%.2f GiB)'%(i+1,len(dataset),bytes_total/2**30),flush=True)
    (folder/'verification.json').write_text(json.dumps({'key':key,'checks':checks,'bytes':bytes_total,'seconds':time.monotonic()-start},indent=2))
    return DiskPrefixes(entries)
