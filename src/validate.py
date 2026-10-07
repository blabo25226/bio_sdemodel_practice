"""Compare observed snapshots, teacher and symbolic stochastic distributions."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from .teacher import TeacherSDE,simulate_sde
from .distill import PolynomialField,SymbolicSDE,errors


def sliced_wasserstein(a: np.ndarray,b: np.ndarray,scale: np.ndarray,
                       projections: np.ndarray) -> float:
    """Mean 1-D Wasserstein distance after training scale normalization (n,d)/(m,d)."""
    aa=(a/scale)@projections;bb=(b/scale)@projections
    return float(np.mean([wasserstein_distance(aa[:,i],bb[:,i]) for i in range(aa.shape[1])]))


def main() -> None:
    """Frozen-model test evaluation across seeds; retain simulation failures as results."""
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--state-dir',type=Path,default=Path('outputs/distillation'))
    parser.add_argument('--sindy-dir',type=Path,default=Path('outputs/sindy'))
    parser.add_argument('--checkpoint',type=Path,default=Path('outputs/baseline/official/teacher.ckpt'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/validation'))
    parser.add_argument('--figure-prefix',default='')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    for target in ('drift','diffusion'):
        result=json.loads((args.sindy_dir/target/'results.json').read_text())
        if not result['function_success']:raise RuntimeError(f'{target} failed the validation gate')
    drift=PolynomialField.load(args.sindy_dir/'drift/model.npz')
    diffusion=PolynomialField.load(args.sindy_dir/'diffusion/model.npz')
    teacher=TeacherSDE(args.checkpoint,device='cuda:0')
    symbolic=SymbolicSDE(drift,diffusion,device='cuda:0')
    obs=pd.read_csv(args.state_dir/'state_metadata.csv')
    x=np.load(args.state_dir/'observed_states.npy')
    initial_ix=np.load(args.state_dir/'simulated_test.npz')['initial_indices']
    initial=np.repeat(x[initial_ix],4,axis=0)
    times=np.array([2.,4.,6.])
    rng=np.random.default_rng(42)
    projections=rng.normal(size=(len(drift.mean),64));projections/=np.linalg.norm(projections,axis=0)
    classifier=joblib.load(args.state_dir/'fate_classifier.joblib')
    from sklearn.metrics import accuracy_score
    test=obs.partition.eq('test').to_numpy()
    classifier_accuracy=float(accuracy_score(obs.loc[test,'Cell type annotation'],classifier.predict(x[test])))
    trajectories={};stability=[]
    for name,sde in [('Neural SDE teacher',teacher),('Symbolic SDE',symbolic)]:
        runs=[]
        for seed in (202,203,204):
            print('simulate',name,seed,flush=True)
            try:
                trajectory=simulate_sde(sde,initial,times,seed,dt=.05)
                finite=bool(np.isfinite(trajectory).all())
                maximum=float(np.max(np.linalg.norm(trajectory,axis=-1)))
                stability.append({'source':name,'seed':seed,'finite':finite,'max_state_norm':maximum,
                                  'outside_training_domain_fraction':float(np.mean(np.linalg.norm(trajectory,axis=-1)>drift.metadata['domain_radius']))})
                if finite:runs.append(trajectory)
            except Exception as exc:
                stability.append({'source':name,'seed':seed,'finite':False,'error':repr(exc)})
        if runs:trajectories[name]=np.concatenate(runs,axis=1)
    pd.DataFrame(stability).to_csv(root/'stability.csv',index=False)
    if len(trajectories)!=2:
        (root/'results.json').write_text(json.dumps({'simulation_success':False,'stability':stability},indent=2))
        return
    distribution=[];fates=[];moments=[]
    for i,t in enumerate(times):
        observed=x[test & obs['Time point'].eq(t).to_numpy()]
        teacher_state=trajectories['Neural SDE teacher'][i]
        for name in ('Observed data','Neural SDE teacher','Symbolic SDE'):
            states=observed if name=='Observed data' else trajectories[name][i]
            distribution.append({'time':t,'source':name,'n_cells_or_simulations':len(states),
                 'sliced_wasserstein_to_observed':sliced_wasserstein(states,observed,drift.scale,projections),
                 'sliced_wasserstein_to_teacher':sliced_wasserstein(states,teacher_state,drift.scale,projections)})
            labels=obs.loc[test & obs['Time point'].eq(t).to_numpy(),'Cell type annotation'] if name=='Observed data' else classifier.predict(states)
            counts=pd.Series(labels).value_counts(normalize=True)
            for fate in ('Monocyte','Neutrophil','Undifferentiated'):
                fates.append({'time':t,'source':name,'fate':fate,'fraction':float(counts.get(fate,0))})
            moments.append({'time':t,'source':name,'mean_norm':float(np.linalg.norm(states.mean(0))),
                            'covariance_trace':float(np.trace(np.cov(states,rowvar=False))),
                            'mean_error_to_observed_training_scaled':float(np.linalg.norm((states.mean(0)-observed.mean(0))/drift.scale)),
                            'covariance_error_to_observed_training_scaled':float(np.linalg.norm(np.cov(states/drift.scale,rowvar=False)-np.cov(observed/drift.scale,rowvar=False))),
                            'covariance_error_to_teacher_training_scaled':float(np.linalg.norm(np.cov(states/drift.scale,rowvar=False)-np.cov(teacher_state/drift.scale,rowvar=False)))})
    for name,rows in [('distribution_distances',distribution),('fate_composition',fates),('moments',moments)]:
        pd.DataFrame(rows).to_csv(root/f'{name}.csv',index=False)
    # Same initial distribution and explicit Brownian seeds for both models; coupling is not identifiability.
    dt_results={}
    for name,sde in [('Neural SDE teacher',teacher),('Symbolic SDE',symbolic)]:
        coarse=simulate_sde(sde,initial[:32],times,202,dt=.05)
        fine=simulate_sde(sde,initial[:32],times,202,dt=.025)
        dt_results[name]=sliced_wasserstein(coarse[-1],fine[-1],drift.scale,projections)
    target=np.load(args.state_dir/'observed_targets.npz')
    field_rows=[]
    for partition in ('train','validation','test'):
        ix=obs.partition.eq(partition).to_numpy()
        field_rows.append({'partition':partition,'target':'drift',**errors(target['drift'][ix],drift.predict(x[ix]))})
        predicted_g=diffusion.predict(x[ix]).reshape(ix.sum(),len(drift.mean),-1)
        teacher_g=target['diffusion'][ix]
        field_rows.append({'partition':partition,'target':'G',**errors(teacher_g.reshape(ix.sum(),-1),predicted_g.reshape(ix.sum(),-1))})
        teacher_d=teacher_g@teacher_g.transpose(0,2,1)
        predicted_d=predicted_g@predicted_g.transpose(0,2,1)
        field_rows.append({'partition':partition,'target':'D=GG^T',**errors(teacher_d.reshape(ix.sum(),-1),predicted_d.reshape(ix.sum(),-1))})
    pd.DataFrame(field_rows).to_csv(root/'observed_function_fidelity.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True)
    for ax,name in zip(axes,('Observed data','Neural SDE teacher','Symbolic SDE')):
        states=x[test & obs['Time point'].eq(6).to_numpy()] if name=='Observed data' else trajectories[name][-1]
        ax.scatter(states[:,0],states[:,1],s=4,alpha=.5);ax.set(title=name,xlabel='PC1',ylabel='PC2')
    fig.tight_layout();fig.savefig(f'outputs/figures/{args.figure_prefix}endpoint_comparison.png',dpi=160);plt.close(fig)
    fate_table=pd.DataFrame(fates)
    fig,ax=plt.subplots(figsize=(8,4))
    endpoint=fate_table[fate_table.time==6].pivot(index='source',columns='fate',values='fraction')
    endpoint.loc[['Observed data','Neural SDE teacher','Symbolic SDE']].plot.bar(stacked=True,ax=ax)
    ax.set(ylabel='Fraction',title='Day 6 composition (training-only kNN for simulations)')
    ax.tick_params(axis='x',rotation=10);fig.tight_layout();fig.savefig(f'outputs/figures/{args.figure_prefix}fate_comparison.png',dpi=160);plt.close(fig)
    tv=float(.5*np.abs(endpoint.loc['Neural SDE teacher']-endpoint.loc['Symbolic SDE']).sum())
    result={'simulation_success':all(r['finite'] for r in stability),
            'seeds':[202,203,204],'initial_observed_cells':len(initial_ix),'replicates_per_initial':4,
            'dt':.05,'dt_halving_endpoint_distance':dt_results,
            'teacher_symbolic_day6_fate_total_variation':tv,
            'fate_classifier_test_accuracy':classifier_accuracy,
            'fate_preservation_demo':tv<=.15,
            'interpretation':'Teacher approximation in sampled domain; thresholds frozen before test. No gene regulation or intrinsic-noise identification.',
            'caveats':['Teacher was exposed to most distillation-test cells; upstream PCA was not fitted on independent training subset.',
                       'Only 32 initial cells and three Brownian seeds; this is a small proof-of-concept.',
                       'Simulated fate labels are descriptive kNN assignments, not observed fates.',
                       'Polynomial extrapolation outside training-defined domain may fail; no clipping was applied.',
                       'Drift fitted independently need not retain the conservative potential-gradient structure of the teacher.']}
    (root/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
