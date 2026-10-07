"""Qualified raw-observation clone-centroid drift proxy, not identifiable stochastic SINDy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .distill import fit_field,errors,PolynomialField,SymbolicSDE
from .teacher import TeacherSDE,simulate_sde
from .run_distillation import export_equations


def centroid_pairs(x: np.ndarray,obs: pd.DataFrame) -> dict[str,tuple[np.ndarray,np.ndarray,np.ndarray]]:
    """Create earlier/later clone means (n,d); these are population pseudo-transitions."""
    rows=[]
    for (partition,clone),group in obs.groupby(['partition','clone_idx']):
        times=sorted(group['Time point'].unique())
        for t0,t1 in zip(times[:-1],times[1:]):
            if t1-t0!=2:continue
            a=group.index[group['Time point']==t0].to_numpy();b=group.index[group['Time point']==t1].to_numpy()
            rows.append((partition,clone,t0,x[a].mean(0),x[b].mean(0)))
    result={}
    for partition in ('train','validation','test'):
        selected=[r for r in rows if r[0]==partition]
        if not selected:raise ValueError(f'No clone-centroid pairs for {partition}')
        result[partition]=(np.stack([r[3] for r in selected]),np.stack([r[4] for r in selected]),
                           np.array([[r[1],r[2]] for r in selected]))
    return result


def main() -> None:
    """Fit/validate on clone-disjoint proxy targets and compare frozen models on test pairs."""
    root=Path('outputs/direct_baseline');root.mkdir(parents=True,exist_ok=True)
    x=np.load('outputs/distillation/observed_states.npy')
    obs=pd.read_csv('outputs/distillation/state_metadata.csv')
    pairs=centroid_pairs(x,obs)
    train,next_train,_=pairs['train'];validation,next_validation,_=pairs['validation']
    models=[];rows=[]
    config=json.loads(Path('configs/experiment.json').read_text())
    for degree in (1,2):
        for threshold in (.05,.1):
            print('proxy fit',degree,threshold,flush=True)
            model=fit_field(train,(next_train-train)/2,degree,threshold,1000.,unbias=False)
            prediction=validation+2*model.predict(validation)
            metrics=errors(next_validation-validation,prediction-validation)
            rows.append({'degree':degree,'threshold':threshold,'alpha':1000.,'library_size':model.metadata['library_size'],
                         'active_terms':model.metadata['active_terms'],'validation_displacement_nrmse':metrics['nrmse']})
            models.append(model)
    selected=int(np.argmin([r['validation_displacement_nrmse'] for r in rows]));model=models[selected]
    model.save(root/'model.npz');export_equations(model,root)
    pd.DataFrame(rows).to_csv(root/'sweep.csv',index=False)
    test,later,ids=pairs['test']
    proxy_prediction=test+2*model.predict(test)
    comparison=[{'source':'Raw clone-centroid polynomial proxy',**errors(later-test,proxy_prediction-test)}]
    teacher=TeacherSDE(Path('outputs/baseline/official/teacher.ckpt'),device='cuda:0')
    methods=[('Neural SDE teacher',teacher)]
    if Path('outputs/sindy/diffusion/model.npz').exists():
        methods.append(('Symbolic SDE',SymbolicSDE(PolynomialField.load(Path('outputs/sindy/drift/model.npz')),
                      PolynomialField.load(Path('outputs/sindy/diffusion/model.npz')),device='cuda:0')))
    for name,sde in methods:
        predictions=np.zeros_like(test)
        for t in (2.,4.):
            ix=np.flatnonzero(ids[:,1]==t)
            if not len(ix):continue
            initial=np.repeat(test[ix],16,axis=0)
            simulation=simulate_sde(sde,initial,np.array([t,t+2]),seed=305,dt=.05)
            predictions[ix]=simulation[-1].reshape(len(ix),16,50).mean(1)
        comparison.append({'source':name,**errors(later-test,predictions-test)})
    pd.DataFrame(comparison).to_csv(root/'test_displacement_comparison.csv',index=False)
    result={'selected':rows[selected],'pair_counts':{k:len(v[0]) for k,v in pairs.items()},
            'direct_stochastic_diffusion_baseline':'Not identifiable from clone-centroid pseudo-transitions; not fabricated.',
            'assumptions':['Different cells share a clone barcode, not a continuous individual trajectory.',
               'Centroid change reflects sampling, branching and growth as well as dynamics.',
               'Raw proxy predicts a finite two-day displacement via Euler; SDE predictions use integrated sample means.',
               'Teacher fit exposure and upstream PCA preclude fully independent biological inference.',
               'This is a qualified drift proxy and not a fair stochastic-SINDy benchmark or evidence of superiority.'],
            'test_comparison':comparison}
    (root/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
