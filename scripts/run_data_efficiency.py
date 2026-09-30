"""Group-only development learning curve (no final-test access)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.loading import load_uji_trajectories
from src.experiments.features import extract_features_for_windows
from src.experiments.baselines import build_baseline
from src.experiments.metrics import classification_metrics
R=Path(__file__).resolve().parents[1]; D=R/'data/external/ujiindoorloc_mag'
def main():
 w=pd.read_csv(D/'processed/window_manifest_2s_stride_1s.csv'); scope=json.loads((D/'manifests/label_scopes.json').read_text());w=w[w.label.isin(scope['cv_eligible'])].reset_index(drop=True); f=extract_features_for_windows(load_uji_trajectories(D),w,'body_xyz_magnitude').set_index('window_id'); names=[x for x in f if x not in {'dataset_id','trajectory_id','group_id','label','source_split'}]; folds=pd.read_csv(D/'manifests/development_group_folds.csv'); rows=[]
 for frac in [.1,.2,.4,.6,.8,1.]:
  for seed in range(5):
   for fold in sorted(folds.fold.unique()):
    tg=folds[(folds.fold==fold)&(folds.partition=='train')].group_id.astype(str).unique();vg=folds[(folds.fold==fold)&(folds.partition=='validation')].group_id.astype(str).unique(); rng=np.random.default_rng(seed); selected=rng.choice(tg,size=max(1,round(len(tg)*frac)),replace=False);tr=w[w.group_id.astype(str).isin(selected)];va=w[w.group_id.astype(str).isin(vg)];m=build_baseline('knn',{'n_neighbors':1,'metric':'manhattan','weights':'distance'},seed).fit(f.loc[tr.window_id,names],tr.label);met=classification_metrics(va.label,m.predict(f.loc[va.window_id,names]));rows.append({'requested_fraction':frac,'actual_trajectory_count':len(selected),'actual_window_count':len(tr),'class_coverage':tr.label.nunique(),'seed':seed,'fold':fold,'model':'knn_body_xyz','macro_f1':met['macro_f1'],'balanced_accuracy':met['balanced_accuracy']})
 out=R/'results/tables';out.mkdir(parents=True,exist_ok=True);x=pd.DataFrame(rows);x.to_csv(out/'data_efficiency.csv',index=False)
 g=x.groupby('actual_trajectory_count')[['macro_f1','balanced_accuracy']].mean();fig,ax=plt.subplots();ax.plot(g.index,g.macro_f1,marker='o');ax.set(xlabel='Training trajectories',ylabel='Macro-F1');fig.savefig(R/'results/figures/macro_f1_vs_training_trajectories.png',dpi=200);plt.close(fig);fig,ax=plt.subplots();ax.plot(g.index,g.balanced_accuracy,marker='o');ax.set(xlabel='Training trajectories',ylabel='Balanced accuracy');fig.savefig(R/'results/figures/accuracy_vs_training_trajectories.png',dpi=200)
if __name__=='__main__':main()
