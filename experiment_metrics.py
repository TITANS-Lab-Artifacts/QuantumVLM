"""Clean metrics shared by cached and full forwards."""
import time
import torch
from config import write_json
from utils import accuracy_summary, cluster_accuracy_interval, input_fingerprint
from feature_cache import scores_from_entry
from resource_monitor import synchronize


def metrics(rows):
    m=accuracy_summary(rows);cm=[[0,0],[0,0]]
    for r in rows: cm[r['label']][r['prediction']]+=1
    m.update(confusion_matrix_true_rows_pred_columns=cm,
             recall_false=cm[0][0]/sum(cm[0]) if sum(cm[0]) else None,
             recall_true=cm[1][1]/sum(cm[1]) if sum(cm[1]) else None,
             predicted_true_rate=sum(r['prediction']==1 for r in rows)/len(rows))
    return m

def evaluate(m,dataset,cache=None,draws=0,seed=0):
    m.eval();rows=[];loss=0.;latencies=[];start=time.monotonic()
    with torch.no_grad():
        for i in range(len(dataset)):
            synchronize(m.device_);t=time.monotonic()
            if cache is None:
                sample=dataset[i];ctx=m.context(sample['pixels'],sample['caption'])
                scores=m(sample['pixels'].unsqueeze(0).to(m.device_),ctx)
                row={k:v for k,v in sample.items() if k!='pixels'}
                row.update(input_fingerprint(sample['pixels'],ctx.prompt))
            else:
                entry=cache[i];row=dict(entry['row'])
                scores=scores_from_entry(m,entry)
            loss+=torch.nn.functional.cross_entropy(scores,torch.tensor([row['label']],device=m.device_)).item()
            row.update(prediction=scores.argmax(-1).item(),scores=scores[0].float().cpu().tolist());rows.append(row)
            synchronize(m.device_);latencies.append(time.monotonic()-t)
            if (i+1)%100==0: print('Evaluated %d/%d'%(i+1,len(dataset)),flush=True)
    elapsed=time.monotonic()-start;summary=metrics(rows);summary['loss']=loss/len(rows)
    if draws: summary['accuracy_ci95_image_clusters']=cluster_accuracy_interval(rows,seed=seed,draws=draws)
    times=torch.tensor(latencies)
    return {'metrics':summary,'examples':rows,'seconds':elapsed,'examples_per_second':len(rows)/elapsed,
            'latency_seconds':{'median':times.median().item(),'p95':times.quantile(.95).item(),
                              'scope':('disk prefix read + downstream scoring' if hasattr(cache,'entries') else 'cached adapter only') if cache else 'end-to-end including input preparation',
                              'includes_first_example':True}}
