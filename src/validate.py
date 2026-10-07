"""Distribution validation retaining numerical failures and all simulated mass."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from .teacher import TeacherSDE, simulate_sde
from .distill import PolynomialField, SymbolicSDE, errors


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, scale: np.ndarray,
                       projections: np.ndarray) -> float:
    """Mean projected Wasserstein after training scaling; inputs (n,d)/(m,d)."""
    aa=(a/scale)@projections
    bb=(b/scale)@projections
    return float(np.mean([wasserstein_distance(aa[:,i],bb[:,i]) for i in range(aa.shape[1])]))


def failure_masks(trajectory: np.ndarray, radius: float) -> tuple[np.ndarray,np.ndarray]:
    """For (time,n,d), return cumulative failure mask (time,n) and float64 state norms."""
    norms=np.linalg.norm(trajectory.astype(np.float64),axis=-1)
    bad=(~np.isfinite(trajectory).all(axis=-1)) | (norms>10*radius)
    return np.maximum.accumulate(bad,axis=0),norms


def main() -> None:
    """Evaluate frozen models on test; failed paths remain explicit, never silently discarded."""
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
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=True)
    for target in ('drift','diffusion'):
        if not json.loads((args.sindy_dir/target/'results.json').read_text())['function_success']:
            raise RuntimeError(f'{target} failed its validation gate')
    drift=PolynomialField.load(args.sindy_dir/'drift/model.npz')
    diffusion=PolynomialField.load(args.sindy_dir/'diffusion/model.npz')
    teacher=TeacherSDE(args.checkpoint,device='cuda:0')
    symbolic=SymbolicSDE(drift,diffusion,device='cuda:0')
    obs=pd.read_csv(args.state_dir/'state_metadata.csv')
    x=np.load(args.state_dir/'observed_states.npy')
    initial_ix=np.load(args.state_dir/'simulated_test.npz')['initial_indices']
    initial=np.repeat(x[initial_ix],4,axis=0)
    times=np.array([2.,4.,6.])
    radius=drift.metadata['domain_radius']
    rng=np.random.default_rng(42)
    projections=rng.normal(size=(len(drift.mean),64))
    projections/=np.linalg.norm(projections,axis=0)
    classifier=joblib.load(args.state_dir/'fate_classifier.joblib')
    from sklearn.metrics import accuracy_score
    test=obs.partition.eq('test').to_numpy()
    classifier_accuracy=float(accuracy_score(obs.loc[test,'Cell type annotation'],classifier.predict(x[test])))
    # Function/covariance fidelity is still evaluated when simulation later fails.
    target=np.load(args.state_dir/'observed_targets.npz')
    field_rows=[]
    for partition in ('train','validation','test'):
        ix=obs.partition.eq(partition).to_numpy()
        predicted_g=diffusion.predict(x[ix]).reshape(ix.sum(),len(drift.mean),-1)
        teacher_g=target['diffusion'][ix]
        teacher_d=teacher_g@teacher_g.transpose(0,2,1)
        predicted_d=predicted_g@predicted_g.transpose(0,2,1)
        for name,truth,predicted in [('drift',target['drift'][ix],drift.predict(x[ix])),
                                     ('G',teacher_g,predicted_g),('D=GG^T',teacher_d,predicted_d)]:
            field_rows.append({'partition':partition,'target':name,
                               **errors(truth.reshape(ix.sum(),-1),predicted.reshape(ix.sum(),-1))})
    pd.DataFrame(field_rows).to_csv(root/'observed_function_fidelity.csv',index=False)
    trajectories={}
    failures={}
    stability=[]
    for name,sde in [('Neural SDE teacher',teacher),('Symbolic SDE',symbolic)]:
        runs=[]
        masks=[]
        for seed in (202,203,204):
            print('simulate',name,seed,flush=True)
            exception=None
            try:
                trajectory=simulate_sde(sde,initial,times,seed,dt=.05)
            except Exception as exc:
                exception=repr(exc)
                trajectory=np.full((len(times),len(initial),len(drift.mean)),np.nan)
                trajectory[0]=initial
            bad,norms=failure_masks(trajectory,radius)
            finite_norms=norms[np.isfinite(norms)]
            stability.append({'source':name,'seed':seed,'finite':bool(np.isfinite(trajectory).all()),
                'max_finite_state_norm':float(finite_norms.max()) if len(finite_norms) else None,
                'nonfinite_path_fraction':float(np.mean(~np.isfinite(trajectory).all(axis=(0,2)))),
                'evaluated_time_explosion_or_nonfinite_path_fraction':float(bad[-1].mean()),
                'explosion_threshold':10*radius,
                'evaluated_time_outside_training_domain_fraction':float(np.mean((norms>radius)|~np.isfinite(norms))),
                'exception':exception})
            runs.append(trajectory)
            masks.append(bad)
        trajectories[name]=np.concatenate(runs,axis=1)
        failures[name]=np.concatenate(masks,axis=1)
    pd.DataFrame(stability).to_csv(root/'stability.csv',index=False)
    np.savez_compressed(root/'trajectories.npz',times=times,teacher=trajectories['Neural SDE teacher'],
                        symbolic=trajectories['Symbolic SDE'],teacher_failed=failures['Neural SDE teacher'],
                        symbolic_failed=failures['Symbolic SDE'])
    distribution=[]
    fates=[]
    moments=[]
    names=('Observed data','Neural SDE teacher','Symbolic SDE')
    for i,t in enumerate(times):
        observed=x[test & obs['Time point'].eq(t).to_numpy()]
        teacher_states=trajectories['Neural SDE teacher'][i]
        for name in names:
            states=observed if name=='Observed data' else trajectories[name][i]
            bad=np.zeros(len(states),dtype=bool) if name=='Observed data' else failures[name][i]
            finite=np.isfinite(states).all(axis=1)
            all_finite=bool(finite.all())
            row={'time':t,'source':name,'n_cells_or_simulations':len(states),
                 'numerically_failed_fraction':float(bad.mean()),
                 'full_distribution_available':all_finite,
                 'sliced_wasserstein_to_observed':None,
                 'sliced_wasserstein_to_teacher':None}
            if all_finite:
                row['sliced_wasserstein_to_observed']=sliced_wasserstein(states,observed,drift.scale,projections)
                row['sliced_wasserstein_to_teacher']=sliced_wasserstein(states,teacher_states,drift.scale,projections)
            distribution.append(row)
            # Failure mass is kept in the denominator; no survivor renormalization.
            if name=='Observed data':
                labels=obs.loc[test & obs['Time point'].eq(t).to_numpy(),'Cell type annotation'].to_numpy()
            else:
                labels=np.full(len(states),'Numerical failure',dtype=object)
                valid=finite & ~bad
                if valid.any():labels[valid]=classifier.predict(states[valid])
            counts=pd.Series(labels).value_counts(normalize=True)
            for fate in ('Monocyte','Neutrophil','Undifferentiated','Numerical failure'):
                fates.append({'time':t,'source':name,'fate':fate,'fraction':float(counts.get(fate,0))})
            moment={'time':t,'source':name,'full_distribution_available':all_finite,
                    'mean_norm':None,'covariance_trace':None,'mean_error_to_observed_training_scaled':None,
                    'covariance_error_to_observed_training_scaled':None,'covariance_error_to_teacher_training_scaled':None}
            if all_finite:
                states=states.astype(np.float64)
                moment.update(mean_norm=float(np.linalg.norm(states.mean(0))),
                    covariance_trace=float(np.trace(np.cov(states,rowvar=False))),
                    mean_error_to_observed_training_scaled=float(np.linalg.norm((states.mean(0)-observed.mean(0))/drift.scale)),
                    covariance_error_to_observed_training_scaled=float(np.linalg.norm(np.cov(states/drift.scale,rowvar=False)-np.cov(observed/drift.scale,rowvar=False))),
                    covariance_error_to_teacher_training_scaled=float(np.linalg.norm(np.cov(states/drift.scale,rowvar=False)-np.cov(teacher_states/drift.scale,rowvar=False))))
            moments.append(moment)
    for name,rows in [('distribution_distances',distribution),('fate_composition',fates),('moments',moments)]:
        pd.DataFrame(rows).to_csv(root/f'{name}.csv',index=False)
    dt_results={}
    for name,sde in [('Neural SDE teacher',teacher),('Symbolic SDE',symbolic)]:
        coarse=simulate_sde(sde,initial[:32],times,202,dt=.05)
        fine=simulate_sde(sde,initial[:32],times,202,dt=.025)
        if np.isfinite(coarse).all() and np.isfinite(fine).all():
            dt_results[name]={'endpoint_sliced_wasserstein':sliced_wasserstein(coarse[-1],fine[-1],drift.scale,projections),
                              'coarse_and_fine_finite':True}
        else:
            dt_results[name]={'endpoint_sliced_wasserstein':None,'coarse_and_fine_finite':False}
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharex=True,sharey=True)
    for ax,name in zip(axes,names):
        states=x[test & obs['Time point'].eq(6).to_numpy()] if name=='Observed data' else trajectories[name][-1]
        bad=np.zeros(len(states),dtype=bool) if name=='Observed data' else failures[name][-1]
        valid=np.isfinite(states).all(axis=1) & ~bad
        ax.scatter(states[valid,0],states[valid,1],s=4,alpha=.5)
        title=name if not bad.any() else f'{name}: valid survivors only\n{valid.sum()}/{len(states)} (biased diagnostic)'
        ax.set(title=title,xlabel='PC1',ylabel='PC2')
    fig.tight_layout()
    fig.savefig(f'outputs/figures/{args.figure_prefix}endpoint_comparison.png',dpi=160)
    plt.close(fig)
    endpoint=pd.DataFrame(fates).query('time==6').pivot(index='source',columns='fate',values='fraction')
    fig,ax=plt.subplots(figsize=(9,4))
    endpoint.loc[list(names),['Monocyte','Neutrophil','Undifferentiated','Numerical failure']].plot.bar(stacked=True,ax=ax)
    ax.set(ylabel='Fraction of all cells/simulations',title='Day 6: numerical failure mass is retained')
    ax.tick_params(axis='x',rotation=10)
    fig.tight_layout()
    fig.savefig(f'outputs/figures/{args.figure_prefix}fate_comparison.png',dpi=160)
    plt.close(fig)
    simulation_success=not any(m.any() for m in failures.values())
    tv=float(.5*np.abs(endpoint.loc['Neural SDE teacher']-endpoint.loc['Symbolic SDE']).sum())
    result={'simulation_success':simulation_success,'seeds':[202,203,204],
        'initial_observed_cells':len(initial_ix),'replicates_per_initial':4,'dt':.05,
        'dt_halving_endpoint_distance':dt_results,
        'teacher_symbolic_day6_total_variation_with_failure_mass':tv,
        'full_fate_distribution_preservation_evaluable':simulation_success,
        'fate_classifier_test_accuracy':classifier_accuracy,
        'fate_preservation_demo':simulation_success and tv<=.15,
        'symbolic_day6_numerical_failure_fraction':float(failures['Symbolic SDE'][-1].mean()),
        'interpretation':'Negative simulation results are retained. No model tuning was performed using these test results.',
        'caveats':['Teacher exposure and upstream PCA prevent independent biological confirmation.',
            'Only 32 initial cells and three Brownian seeds; simulated data are not new observations.',
            'Missing full distribution distances/moments mean numerical failure, not zero error.',
            'Survivor scatter is a biased diagnostic; failed paths retain probability mass in composition tables.',
            'Numerical failure is not biological cell death or a fate class.',
            'No state clipping; polynomial PSD does not guarantee nonexplosive dynamics.',
            'Independent drift fits need not retain the teacher conservative potential structure.',
            'dt-halving is distribution-level sensitivity; common entropy alone is not a strong pathwise convergence proof.']}
    (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
