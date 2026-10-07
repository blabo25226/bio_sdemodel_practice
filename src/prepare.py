"""Freeze clone-disjoint distillation partitions and observed/simulated teacher queries."""
from __future__ import annotations
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scdiffeq as sdq
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from .data import quickstart_subset, sha256_file
from .teacher import TeacherSDE, simulate_sde


def clone_split(obs: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Return (n,) train/validation/test labels, keeping every clone in a single partition."""
    groups = obs.clone_idx.astype(str).to_numpy()
    missing = obs.clone_idx.isna().to_numpy()
    groups[missing] = [f'missing_cell_{i}' for i in np.flatnonzero(missing)]
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    train_val, test = next(splitter.split(obs, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed+1).split(
        train_val, groups=groups[train_val]))
    labels = np.full(len(obs), 'test', dtype=object)
    labels[train_val[tr]] = 'train'
    labels[train_val[va]] = 'validation'
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(groups[labels==a]) & set(groups[labels==b])
    return labels


def main() -> None:
    """Generate query cache and baseline plots from the restored teacher, never tune on test."""
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--state-dir",type=Path,default=Path("outputs/distillation"))
    parser.add_argument("--components",type=int,default=50)
    parser.add_argument("--checkpoint",type=Path,default=Path("outputs/baseline/official/teacher.ckpt"))
    parser.add_argument("--figure-prefix",default="")
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = args.state_dir; out.mkdir(parents=True,exist_ok=True)
    figures = Path('outputs/figures'); figures.mkdir(parents=True,exist_ok=True)
    sdq_data = sdq.datasets.larry(data_dir='data', variant=None)
    a = quickstart_subset(sdq_data)
    del sdq_data
    x = np.asarray(a.obsm['X_pca'][:,:args.components], dtype=np.float32)
    obs = a.obs.copy()
    obs['partition'] = clone_split(obs)
    exposure = pd.read_csv(args.checkpoint.parent/'split_membership.csv',index_col=0)
    exposure['source_cell_id'] = exposure.source_cell_id.astype(str)
    exposure = exposure.set_index('source_cell_id')
    for name in ('fit_train','fit_val','test'):
        obs[f'teacher_{name}'] = obs.source_cell_id.astype(str).map(exposure[name])
    obs.to_csv(out/'state_metadata.csv',index=False)
    np.save(out/'observed_states.npy',x)
    teacher = TeacherSDE(args.checkpoint,device='cuda:0')
    f,g = teacher.evaluate(x)
    np.savez_compressed(out/'observed_targets.npz',drift=f,diffusion=g)
    rng = np.random.default_rng(0)
    sim = {}
    times = np.linspace(2,6,41)
    for partition in ('train','validation','test'):
        initial_idx = np.flatnonzero((obs.partition==partition)&(obs['Time point']==2))
        chosen = rng.choice(initial_idx,size=min(32,len(initial_idx)),replace=False)
        initial = np.repeat(x[chosen],4,axis=0)
        print('simulate',partition,len(initial),flush=True)
        trajectories = simulate_sde(teacher,initial,times,seed=100+len(sim),dt=.05)
        assert np.isfinite(trajectories).all()
        sf,sg = teacher.evaluate(trajectories.reshape(-1,args.components))
        groups = np.tile(np.repeat(obs.iloc[chosen].clone_idx.astype(str).to_numpy(),4),len(times))
        np.savez_compressed(out/f'simulated_{partition}.npz',states=trajectories,
                            drift=sf,diffusion=sg,times=times,groups=groups.astype(str),initial_indices=chosen)
        sim[partition] = trajectories
    # Training-only PCA-space endpoint classifier: descriptive fate readout, not causal labels.
    mask = obs.partition.eq('train').to_numpy()
    classifier = KNeighborsClassifier(n_neighbors=15).fit(x[mask],obs.loc[mask,'Cell type annotation'])
    import joblib
    joblib.dump(classifier,out/'fate_classifier.joblib')
    rows=[]
    for t in (2.,4.,6.):
        y=sim['test'][int((t-2)*10)]
        for source,labels in [('Observed data',obs.loc[(obs.partition=='test')&(obs['Time point']==t),'Cell type annotation']),
                              ('Neural SDE teacher',classifier.predict(y))]:
            counts=pd.Series(labels).value_counts(normalize=True)
            for fate in ('Monocyte','Neutrophil','Undifferentiated'):
                rows.append({'time':t,'source':source,'fate':fate,'fraction':float(counts.get(fate,0))})
    pd.DataFrame(rows).to_csv(f'outputs/tables/{args.figure_prefix}teacher_observed_fates.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    colors={'Monocyte':'tab:orange','Neutrophil':'tab:blue','Undifferentiated':'gray'}
    for name,color in colors.items():
        ix=obs['Cell type annotation'].eq(name).to_numpy()
        axes[0].scatter(x[ix,0],x[ix,1],s=2,c=color,label=name,alpha=.4)
    axes[0].legend(fontsize=8); axes[0].set_title('Observed LARRY (PCs)')
    choose=rng.choice(len(x),200,replace=False)
    axes[1].quiver(x[choose,0],x[choose,1],f[choose,0],f[choose,1]);axes[1].set_title('Teacher drift, PC1/PC2 components')
    axes[2].scatter(x[:,0],x[:,1],s=1,c='gray',alpha=.1)
    for path in sim['test'][:,:32,:].transpose(1,0,2):
        axes[2].plot(path[:,0],path[:,1],alpha=.3)
    axes[2].set_title('Teacher trajectories from test-origin states')
    for ax in axes:ax.set(xlabel='PC1',ylabel='PC2')
    fig.tight_layout();fig.savefig(figures/f'{args.figure_prefix}teacher_baseline.png',dpi=160);plt.close(fig)
    audit={'checkpoint_sha256':sha256_file(args.checkpoint),
           'split_seed':0,'partition_counts':obs.partition.value_counts().to_dict(),
           'group_split':'clone-disjoint; missing IDs unique per cell',
           'teacher_exposure_by_partition':obs.groupby('partition')[['teacher_fit_train','teacher_fit_val']].sum().to_dict(),
           'state_dimension':args.components,'simulated_source':'teacher simulation, not additional observed data',
           'qualitative_assessment':'PC-space field/trajectories and class composition available for visual inspection; official UMAP layout not identical.',
           'independence_caveat':'Distillation test is held out from symbolic fit, not from original teacher/upstream PCA.',
           'selection_policy':'Use validation for symbolic hyperparameters per explicit instruction Step4/skill; never test.'}
    Path(f'outputs/logs/{args.figure_prefix}distillation_split_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
