"""Development-only feature/sensor ablation and window-length study."""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from src.data.loading import load_uji_trajectories
from src.data.splits import assert_no_group_overlap
from src.data.windows import WindowConfig, generate_windows
from src.experiments.baselines import build_baseline
from src.experiments.features import extract_features_for_windows
from src.experiments.metrics import classification_metrics, trajectory_balanced_metrics

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/external/ujiindoorloc_mag'

def prepare_features(samples, windows, feature_name):
    t0=time.perf_counter(); features=extract_features_for_windows(samples,windows,feature_name).set_index('window_id'); feature_ms=(time.perf_counter()-t0)*1000/len(windows)
    return features, feature_ms

def evaluate(windows, feature_name, model_key, spec, folds, seed, features, feature_ms):
    names=[c for c in features.columns if c not in {'dataset_id','trajectory_id','group_id','label','source_split'}]
    metrics=[]; predictions=[]
    for fold in sorted(folds.fold.unique()):
        tg=sorted(folds[(folds.fold==fold)&(folds.partition=='train')].group_id.astype(str).unique()); vg=sorted(folds[(folds.fold==fold)&(folds.partition=='validation')].group_id.astype(str).unique()); assert_no_group_overlap({'train':tg,'validation':vg})
        tr=windows[windows.group_id.astype(str).isin(tg)].reset_index(drop=True); va=windows[windows.group_id.astype(str).isin(vg)].reset_index(drop=True)
        model=build_baseline(spec['name'],spec['parameters'],seed+int(fold)); model.fit(features.loc[tr.window_id,names],tr.label)
        t0=time.perf_counter(); pred=model.predict(features.loc[va.window_id,names]); infer_ms=(time.perf_counter()-t0)*1000/len(va)
        part=pd.DataFrame({'model':model_key,'feature_set':feature_name,'fold':fold,'window_id':va.window_id,'trajectory_id':va.trajectory_id,'true_label':va.label,'predicted_label':pred})
        m=classification_metrics(va.label,pred);m.update(trajectory_balanced_metrics(part));m['inference_ms_per_window']=infer_ms;metrics.append(m);predictions.append(part)
    row={'model':model_key,'feature_set':feature_name,'n_windows':len(windows),'n_features':len(names),'feature_extraction_ms_per_window':feature_ms}
    for key in ['macro_f1','balanced_accuracy','accuracy','trajectory_accuracy_mean','trajectory_macro_f1_mean','inference_ms_per_window']: row[key+'_mean']=float(np.mean([m[key] for m in metrics]))
    return row,pd.concat(predictions,ignore_index=True)

def main():
    cfg=yaml.safe_load((ROOT/'configs/suites/feature_window_ablation.yaml').read_text()); samples=load_uji_trajectories(DATA); folds=pd.read_csv(ROOT/cfg['fold_manifest']); scopes=json.loads((ROOT/cfg['label_scopes']).read_text()); out=ROOT/'results/tables'; figs=ROOT/'results/figures'; out.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    default=pd.read_csv(DATA/'processed/window_manifest_2s_stride_1s.csv');default=default[default.label.isin(scopes['cv_eligible'])].reset_index(drop=True)
    rows=[];preds=[]
    for ablation,feature in cfg['ablations'].items():
        features,feature_ms=prepare_features(samples,default,feature)
        for model,spec in cfg['models'].items():
            row,pred=evaluate(default,feature,model,spec,folds,cfg['seed'],features,feature_ms);row['ablation']=ablation;rows.append(row);preds.append(pred.assign(study='feature_sensor',ablation=ablation))
    ab=pd.DataFrame(rows);ab.to_csv(out/'feature_sensor_ablation.csv',index=False)
    pivot=ab.pivot(index='ablation',columns='model',values='macro_f1_mean');pivot.plot.bar(figsize=(11,6));plt.ylabel('Grouped-CV macro-F1');plt.tight_layout();plt.savefig(figs/'feature_sensor_ablation.png',dpi=200);plt.close()
    winrows=[]
    for length in cfg['window_lengths_s']:
        stride=length*cfg['stride_fraction']; windows=generate_windows(samples,WindowConfig(length,stride));windows=windows[windows.label.isin(scopes['cv_eligible'])].reset_index(drop=True);windows.to_csv(DATA/'processed'/f"window_manifest_{length:g}s_stride_{stride:g}s.csv",index=False)
        features,feature_ms=prepare_features(samples,windows,cfg['window_study_feature'])
        for model,spec in cfg['models'].items():
            row,pred=evaluate(windows,cfg['window_study_feature'],model,spec,folds,cfg['seed'],features,feature_ms);row.update({'window_length_s':length,'stride_s':stride});winrows.append(row);preds.append(pred.assign(study='window_length',window_length_s=length))
    wr=pd.DataFrame(winrows);wr.to_csv(out/'window_length_results.csv',index=False);pd.concat(preds,ignore_index=True).to_csv(out/'ablation_fold_predictions.csv',index=False)
    for metric,name,ylabel in [('macro_f1_mean','window_length_vs_macro_f1.png','Grouped-CV macro-F1'),('inference_ms_per_window_mean','window_length_vs_latency.png','Inference ms/window')]:
        fig,ax=plt.subplots(figsize=(8,5));
        for model,g in wr.groupby('model'):ax.plot(g.window_length_s,g[metric],marker='o',label=model)
        ax.set(xlabel='Window length (s)',ylabel=ylabel);ax.legend();fig.tight_layout();fig.savefig(figs/name,dpi=200);plt.close(fig)
    print(json.dumps({'ablation_rows':len(ab),'window_rows':len(wr),'predictions':sum(map(len,preds))},indent=2))
if __name__=='__main__':main()
