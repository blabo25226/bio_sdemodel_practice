"""Reproducible drift-first hyperparameter sweep and group bootstrap."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .distill import PolynomialField,errors,fit_field


def query_partition(partition: str, target: str, stride: int, state_root: Path = Path('outputs/distillation')) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return states (n,d), teacher targets (n,k), dependence groups (n,)."""
    root=state_root
    obs=pd.read_csv(root/'state_metadata.csv')
    mask=obs.partition.eq(partition).to_numpy()
    x=np.load(root/'observed_states.npy')[mask]
    y=np.load(root/'observed_targets.npz')[target][mask].reshape(mask.sum(),-1)
    groups=obs.loc[mask,'dependence_group' if 'dependence_group' in obs else 'clone_idx'].astype(str).to_numpy()
    simulated=np.load(root/f'simulated_{partition}.npz')
    sx=simulated['states'].reshape(-1,x.shape[1])[::stride]
    sy=simulated[target][::stride].reshape(len(sx),-1)
    return np.vstack([x,sx]),np.vstack([y,sy]),np.concatenate([groups,simulated['groups'][::stride]])


def export_equations(field: PolynomialField, root: Path) -> None:
    """Write complete sparse coefficients, standardized-coordinate equations and transform."""
    names=[]
    for powers in field.powers:
        factors=[f'z{i+1}'+(f'^{p}' if p>1 else '') for i,p in enumerate(powers) if p]
        names.append('*'.join(factors) or '1')
    rows=[];equations=['# Polynomial field equations\n','Output units: '+field.metadata.get('output_units','original PC coordinate / day (finite-interval proxy if applicable)'),
        'Only powers and their monomial interactions are used. Coordinate definitions:\n']
    for i,(mean,scale) in enumerate(zip(field.mean,field.scale)):
        equations.append(f'z{i+1} = (PC{i+1} - ({mean:.9g})) / ({scale:.9g})')
    for k,coefficients in enumerate(field.coefficients):
        terms=[]
        for j in np.flatnonzero(coefficients):
            rows.append({'output':k+1,'feature':names[j],'coefficient':coefficients[j]})
            terms.append(f'({coefficients[j]:.9g})*{names[j]}')
        equations.append(f'\nfield_{k+1} = '+(' + '.join(terms) or '0'))
    pd.DataFrame(rows).to_csv(root/'coefficients.csv',index=False)
    (root/'equations.md').write_text('\n'.join(equations))


def run_target(target: str, config: dict) -> dict:
    """Select on validation only, bootstrap training groups, then evaluate fixed model on test."""
    root=Path(config.get('sindy_root','outputs/sindy'))/target;root.mkdir(parents=True,exist_ok=True)
    stride=config['simulation_query_stride']
    x,y,groups=query_partition('train',target,stride,Path(config.get('state_root','outputs/distillation')))
    vx,vy,_=query_partition('validation',target,stride,Path(config.get('state_root','outputs/distillation')))
    domain_radius=float(np.quantile(np.linalg.norm(x,axis=1),config['domain_training_norm_quantile']))
    train_domain=np.linalg.norm(x,axis=1)<=domain_radius
    validation_domain=np.linalg.norm(vx,axis=1)<=domain_radius
    all_train_x,all_train_y=x,y
    x,y,groups=x[train_domain],y[train_domain],groups[train_domain]
    if validation_domain.mean()<config['minimum_validation_domain_coverage']:
        raise ValueError('Training-defined domain covers too little validation data')
    rows=[];models=[]
    degrees=list(config['degrees'])
    for degree in degrees:
        for alpha in config['alphas']:
            for threshold in config['thresholds']:
                print('fit',target,degree,alpha,threshold,flush=True)
                model=fit_field(x,y,degree,threshold,alpha,unbias=config['unbias'])
                train=errors(y,model.predict(x));validation=errors(vy[validation_domain],model.predict(vx[validation_domain]))
                row={k:model.metadata[k] for k in ('degree','alpha','threshold','library_size','active_terms')}
                row.update(train_mse=train['mse'],validation_mse=validation['mse'],
                           train_nrmse=train['nrmse'],validation_nrmse=validation['nrmse'])
                rows.append(row);models.append(model)
                pd.DataFrame(rows).to_csv(root/'sweep.csv',index=False)
                print(row,flush=True)
        if degree==2 and config.get('degree_3_if_needed',False):
            if min(r['validation_nrmse'] for r in rows)>config['drift_validation_nrmse_max']:
                print('Degree2 insufficient on validation; degree3 justified',flush=True)
                degrees.append(3)
    best_error=min(r['validation_nrmse'] for r in rows)
    acceptable=[i for i,r in enumerate(rows) if r['validation_nrmse']<=best_error*(1+config['selection_relative_error_tolerance'])]
    passing=[i for i in acceptable if rows[i]['validation_nrmse']<=config['drift_validation_nrmse_max']]
    if passing:acceptable=passing
    selected=min(acceptable,key=lambda i:rows[i]['active_terms'])
    model=models[selected]
    model.metadata['target']=target
    model.metadata['output_units']='PC coordinate / day' if target=='drift' else 'PC coordinate / sqrt(day)'
    model.save(root/'model.npz');export_equations(model,root)
    support=model.coefficients!=0
    frequencies=np.zeros_like(model.coefficients)
    jaccards=[];rng=np.random.default_rng(config['seed'])
    group_ids=np.unique(groups)
    for b in range(config['bootstrap_replicates']):
        sampled=rng.choice(group_ids,len(group_ids),replace=True)
        ix=np.concatenate([np.flatnonzero(groups==group) for group in sampled])
        print('bootstrap',target,b,flush=True)
        boot=fit_field(x[ix],y[ix],model.metadata['degree'],model.metadata['threshold'],model.metadata['alpha'],reference=model,unbias=config['unbias'])
        # Fix initial training transform and threshold units across resampled groups.
        frequencies+=boot.coefficients!=0
        b_support=boot.coefficients!=0
        jaccards.append(float(np.count_nonzero(support&b_support)/max(np.count_nonzero(support|b_support),1)))
    frequencies/=config['bootstrap_replicates']
    np.save(root/'bootstrap_frequency.npy',frequencies)
    pd.DataFrame([{'output':int(k+1),'feature_index':int(j),'frequency':float(frequencies[k,j])}
                  for k,j in zip(*np.nonzero(support))]).to_csv(root/'bootstrap_selected_terms.csv',index=False)
    jitter=[]
    for factor in (.9,1.1):
        neighbor=fit_field(x,y,model.metadata['degree'],model.metadata['threshold']*factor,
                           model.metadata['alpha'],reference=model,unbias=config['unbias'])
        neighbor_support=neighbor.coefficients!=0
        jitter.append({'threshold':model.metadata['threshold']*factor,
                       'active_terms':int(neighbor_support.sum()),
                       'support_jaccard':float(np.count_nonzero(support&neighbor_support)/max(np.count_nonzero(support|neighbor_support),1)),
                       'validation_nrmse':errors(vy[validation_domain],neighbor.predict(vx[validation_domain]))['nrmse']})
    tx,ty,_=query_partition('test',target,stride,Path(config.get('state_root','outputs/distillation')))
    result={'selected':rows[selected],'train':errors(y,model.predict(x)),
            'validation':errors(vy[validation_domain],model.predict(vx[validation_domain])),
            'validation_all_domain':errors(vy,model.predict(vx)),
            'test':errors(ty,model.predict(tx)),
            'test_in_domain':errors(ty[np.linalg.norm(tx,axis=1)<=domain_radius],model.predict(tx[np.linalg.norm(tx,axis=1)<=domain_radius])),
            'domain_radius':domain_radius,'domain_definition':'Euclidean PC state norm <= training-only quantile; no clipping in model',
            'validation_domain_coverage':float(validation_domain.mean()),
            'test_domain_coverage':float(np.mean(np.linalg.norm(tx,axis=1)<=domain_radius)),
            'bootstrap_replicates':config['bootstrap_replicates'],
            'bootstrap_jaccards':jaccards,'threshold_jitter':jitter,
            'mean_selected_frequency':float(frequencies[support].mean()) if support.any() else 0.,
            'sources':['observed states','teacher-simulated states'],
            'sample_counts':{'train':len(x),'validation':len(vx),'test':len(tx)},
            'normalization':'RMSE / RMS teacher target',
            'test_policy':'evaluated only after frozen validation selection; not used for tuning'}
    model.metadata['domain_radius']=domain_radius
    model.save(root/'model.npz')
    result['function_success']=(result['validation']['nrmse']<=config['drift_validation_nrmse_max'] and
        result['mean_selected_frequency']>=config['bootstrap_mean_selected_frequency_min'])
    (root/'results.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.DataFrame(rows);fig,ax=plt.subplots()
    for (degree,alpha),subset in table.groupby(['degree','alpha']):
        ax.scatter(subset.active_terms,subset.validation_nrmse,label=f'degree {degree}, alpha={alpha:g}')
    ax.set(xlabel='Active coefficients',ylabel='In-domain validation NRMSE',title=target+' fidelity / sparsity')
    ax.legend();fig.savefig(f"outputs/figures/{config.get('figure_prefix','')}{target}_sparsity.png",dpi=160,bbox_inches='tight');plt.close(fig)
    return result


def main() -> None:
    """Drift gate is applied before diffusion; preserve any negative result."""
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/experiment.json'))
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    drift=run_target('drift',config)
    print('DRIFT',json.dumps(drift),flush=True)
    if not drift['function_success']:
        raise SystemExit('Drift insufficient under predeclared validation criteria; diffusion deferred')
    diffusion=run_target('diffusion',config)
    print('DIFFUSION',json.dumps(diffusion),flush=True)

if __name__=='__main__':main()
