"""Verify completed construction-sensitivity evidence without retraining models."""
import argparse,csv,json
from pathlib import Path
from itertools import combinations
import numpy as np
import run_sampling_check as sampling
from verify_submission import relocated
from tools.data_quality import check_dataset

c=sampling.c
HERE=Path(__file__).resolve().parent

def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if a==b:return True
    try:return abs(float(a)-float(b))<1e-10
    except (ValueError,TypeError):return False

def inspect_overlap(paths):
    datasets={}
    for seed,path in paths.items():
        with np.load(path,allow_pickle=False) as d:
            gids=d['global_node_id'];u=d['source'];v=d['target'];mask=u<v
            datasets[seed]={'ids':gids.copy(),'raw_features':d['raw_features'].copy(),
             'roots':gids[d['labels']>=0],'positive':gids[d['labels']==1],
             'test_positive':gids[(d['labels']==1)&d['test_mask']],
             'test_ids':gids[d['test_mask']],
             'pairs':np.sort(gids[u[mask]].astype(np.int64)*2973489+gids[v[mask]])}
    rows=[]
    for a,b in combinations(sorted(datasets),2):
        x,y=datasets[a],datasets[b]
        shared,ix,iy=np.intersect1d(x['ids'],y['ids'],return_indices=True)
        c.require(np.allclose(x['raw_features'][ix],y['raw_features'][iy],rtol=1e-12,atol=1e-12),'Raw features changed for shared global nodes.')
        shared_edges=np.intersect1d(x['pairs'],y['pairs']).size
        rows.append({'construction_a':a,'construction_b':b,'shared_nodes':len(shared),
            'node_jaccard':len(shared)/(len(x['ids'])+len(y['ids'])-len(shared)),
            'shared_positive_roots':np.intersect1d(x['positive'],y['positive']).size,
            'shared_positive_test_roots':np.intersect1d(x['test_positive'],y['test_positive']).size,
            'shared_supervised_roots':np.intersect1d(x['roots'],y['roots']).size,
            'shared_test_roots':np.intersect1d(x['test_ids'],y['test_ids']).size,
            'shared_undirected_pairs':shared_edges,
            'undirected_edge_jaccard':shared_edges/(len(x['pairs'])+len(y['pairs'])-shared_edges),
            'shared_node_raw_features_agree':True})
    return rows

def verify(root,core,output,baseline_dataset=None):
    root=Path(root).resolve();core=Path(core).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    state=c.read_json(root/'batch_state.json');p=c.read_json(root/'protocol.json');spec=c.read_json(HERE/'audited_inputs.json')
    c.require(state['status']=='complete' and state['verified_new_fits']==18 and state['self_check_passed'],'Incomplete sampling batch.')
    c.require(p['construction_seeds']==[1,2] and p['model_seeds']==[0,1,2] and p['models']==sampling.MODELS,'Construction/model design differs.')
    c.require(p['epochs']==100 and p['hidden']==64 and p['order']==2 and p['lr']==.01,'Training configuration differs.')
    c.require(p['audited_specification']==spec,'Audited source/input specification differs.')
    c.require(c.sha(core/'protocol.json')==p['core_protocol_sha256'],'Core protocol fingerprint differs.')
    for name,wanted in spec['sources_sha256_normalized_lf'].items():
        c.require(c.sha(root/'source_snapshot'/name,normalize=True)==wanted,'Snapshot source differs: '+name)
    for name,field in [('run_sampling_check.py','driver_sha256'),('run_assessment3_core.py','utility_sha256'),
       ('verify_submission.py','verifier_sha256'),('run_sensitivity.py','sensitivity_verifier_sha256'),('prepare_ethereum.py','preprocessor_sha256')]:
        c.require(c.sha(root/'source_snapshot'/name)==p[field],'Snapshot driver differs: '+name)
        c.require(c.sha(HERE/name)==p[field],'Trusted local verifier/trainer version differs: '+name)
    c.require(c.sha(root/'source_snapshot/data_quality.py')==p['quality_checker_sha256']==c.sha(HERE/'tools/data_quality.py'),'Dataset checker differs.')
    c.require(c.sha(root/'source_graph_audit.json')==p['audit_sha256'],'Source graph audit differs.')
    rows,core_protocol=sampling.reference_rows(core);c.require(len(rows)==9,'Expected nine reused references.')
    quality=[];paths={0:None};baseline=core_protocol['audited_specification']['datasets']['ethereum']
    # The sampling archive intentionally holds the new samples only. Locate the baseline from an optional archived input.
    baseline_candidates=[core/'input_dataset.npz',HERE/baseline['relative_path']]
    if baseline_dataset is not None:
        baseline_dataset=Path(baseline_dataset)
        c.require(c.sha(baseline_dataset)==baseline['sha256'],'Baseline dataset fingerprint differs.')
        baseline_candidates.insert(0,baseline_dataset)
    paths.pop(0)
    for path in baseline_candidates:
        if path.is_file() and c.sha(path)==baseline['sha256']:paths[0]=path;break
    for seed in [1,2]:
        entry=state['datasets']['sample_'+str(seed)];dataset=relocated(root,entry['dataset'],4)
        q=check_dataset(dataset);c.require(equivalent(q,entry['quality']),'Recorded sample quality differs.')
        c.require(q['construction_seed']==seed,'Construction seed differs.');quality.append(q);paths[seed]=dataset
        for model_seed in [0,1,2]:
            job=state['jobs']['sample_{}_model_seed{}'.format(seed,model_seed)];c.require(job['status']=='verified','Incomplete training job.')
            run=relocated(root,job['run_directory'],3)
            manifest=c.read_json(run/'training_manifest.json')
            c.require(manifest['dataset_sha256']==q['dataset_sha256'],'Wrong training dataset.')
            c.require(manifest['preprocessing_manifest_sha256']==c.sha(dataset.parent/'pipeline_manifest.json'),'Preprocessing manifest differs.')
            c.require(manifest['environment']['python']=='3.8.20' and manifest['environment']['device']=='cpu','Execution environment differs.')
            batch=sampling.training_result(run,model_seed);sampling.check_predictions(run,dataset,model_seed)
            for r in batch:r['construction_seed']=seed
            rows.extend(batch)
    c.require(len(list(root.rglob('result.json')))==18 and len(rows)==27,'Fit inventory differs.')
    sampling.summarize(output,rows)
    for name in ['sampling_results.csv','sampling_summary.csv']:
        with (root/name).open(newline='') as f:given=list(csv.DictReader(f))
        with (output/name).open(newline='') as f:actual=list(csv.DictReader(f))
        c.require(equivalent(given,actual),'Uploaded aggregate differs: '+name)
    overlap=inspect_overlap(paths);c.write_csv(output/'construction_overlap.csv',overlap)
    gaps=[]
    for sample in [0,1,2]:
        for comparator in ['gcn','mlp']:
            lookup={(r['model_seed'],r['model']):r for r in rows if r['construction_seed']==sample}
            changes=np.asarray([100*(lookup[s,'bwgnn']['macro_f1']-lookup[s,comparator]['macro_f1']) for s in [0,1,2]])
            gaps.append({'construction_seed':sample,'comparator':comparator,'n':3,'macro_f1_gap_mean_pp':float(changes.mean()),
               'macro_f1_gap_sample_sd_pp':float(changes.std(ddof=1)),'positive_pairs':int(np.sum(changes>0))})
    c.write_csv(output/'paired_model_gaps.csv',gaps)
    report={'status':'passed','new_fits':18,'new_epochs':1800,'recomputed_validation_test_primary_metric_values':144,
      'reused_reference_fits':9,'verified_validation_thresholds':18,'total_distinct_extension_fits':128,'total_extension_epochs':12800,
      'dataset_quality':quality,'construction_overlap':overlap,'paired_gaps':gaps,'baseline_in_overlap':0 in paths,
      'new_training_during_audit':0,
      'limits':['Same released snapshot; all positive identities shared, not independent datasets.',
       'Matching, context selection and possibly split membership vary jointly.',
       'Three model seeds per construction here; primary seed0-construction core summary still uses five.',
       'Default-width models differ in parameter count; capacity matching was checked separately on construction seed0.',
       'Full-graph raw-feature reconstruction is not repeated from the 13.55M transactions in this score audit.']}
    c.write_json(output/'independent_sampling_audit.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sampling',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-dataset',type=Path)
    a=p.parse_args();r=verify(a.sampling,a.core,a.output,a.baseline_dataset)
    print(json.dumps({'status':r['status'],'verified_new_fits':18,'new_epochs':1800,'combined_extension_fits':128,'paired_gaps':r['paired_gaps']},indent=2))
if __name__=='__main__':main()
