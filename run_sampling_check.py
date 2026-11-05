"""Two additional construction seeds, three model seeds, three models: 18 fits."""
import argparse,importlib.util,json,os,shutil,sys,time,traceback
from pathlib import Path
import run_assessment3_core as c
from verify_submission import relocated
from tools.data_quality import check_dataset

HERE=Path(__file__).resolve().parent
CONSTRUCTION_SEEDS=[1,2];MODEL_SEEDS=[0,1,2];MODELS=['bwgnn','gcn','mlp']
PREP_SHA='caf79dfc2c6d29013900dcf23decba1d5ca9995c43631fba5c00c79f659bde48'
AUDIT_SHA='664b5380c6fcb1e913190dfb7238efa87c8974355482f1f5a0d981d56be70d07'

def training_result(root,seed):
    return c.verify_job(root,{'id':'sampling_model_seed'+str(seed),'kind':'ethereum','model_seed':seed,'graph_seed':None,'fits':3})

def reference_rows(core):
    state=c.read_json(core/'batch_state.json');p=c.read_json(core/'protocol.json')
    c.require(state['status']=='complete' and state['verified_model_fits']==75,'Core batch incomplete.')
    result=[]
    for seed in MODEL_SEEDS:
        entry=state['jobs']['ethereum_seed'+str(seed)]
        rows=training_result(relocated(core,entry['run_directory'],2),seed)
        for row in rows:row['construction_seed']=0
        result.extend(rows)
    return result,p

def summarize(root,rows):
    import numpy as np
    c.write_csv(root/'sampling_results.csv',rows);summary=[]
    for sample in [0,1,2]:
        for model in MODELS:
            group=[r for r in rows if r['construction_seed']==sample and r['model']==model]
            if not group:continue
            item={'construction_seed':sample,'model':model,'n_model_seeds':len(group)}
            for metric in c.METRICS:
                values=np.asarray([100*r[metric] for r in group]);item[metric+'_mean_percent']=float(values.mean())
                item[metric+'_sample_sd_percent']=float(values.std(ddof=1)) if len(group)>1 else None
            summary.append(item)
    c.write_csv(root/'sampling_summary.csv',summary)

def execute(args):
    project=Path(args.project).resolve();core=Path(args.core).resolve();audit=Path(args.audit).resolve()
    specification=c.read_json(HERE/'audited_inputs.json');c.validate_inputs(project,specification)
    c.require(c.sha(project/'prepare_ethereum.py')==PREP_SHA,'Preprocessor differs.')
    c.require(c.sha(audit)==AUDIT_SHA,'Source graph audit differs.')
    audit_record=c.read_json(audit);compact=Path(audit_record['compact_directory'])
    c.require(compact.is_dir(),'Original compact arrays not found: '+str(compact))
    baseline,core_protocol=reference_rows(core)
    c.require(core_protocol['audited_specification']==specification,'Core input/source specifications differ.')
    protocol={'version':1,'construction_seeds':CONSTRUCTION_SEEDS,'model_seeds':MODEL_SEEDS,'models':MODELS,
      'hidden':64,'order':2,'epochs':100,'lr':.01,'threads':args.threads,'seed0_reference_model_seeds':MODEL_SEEDS,
      'audit_sha256':AUDIT_SHA,'preprocessor_sha256':PREP_SHA,'driver_sha256':c.sha(Path(__file__)),
      'utility_sha256':c.sha(HERE/'run_assessment3_core.py'),'quality_checker_sha256':c.sha(HERE/'tools/data_quality.py'),
      'verifier_sha256':c.sha(HERE/'verify_submission.py'),'sensitivity_verifier_sha256':c.sha(HERE/'run_sensitivity.py'),
      'core_protocol_sha256':c.sha(core/'protocol.json'),'audited_specification':specification,
      'scope':'Joint negative matching, root assignment and sampled-context sensitivity on the same released static snapshot; not pure topology or independent datasets. No configuration chosen using test scores.'}
    if args.preflight_only:
        print(json.dumps({'status':'preflight_passed','new_fits':18,'reference_fits_reused':9,'compact_exists':True},indent=2));return
    root=Path(args.resume).resolve() if args.resume else project/'assessment3_runs'/('sampling_'+c.stamp())
    root.mkdir(parents=True,exist_ok=bool(args.resume))
    if args.resume:
        c.require(c.read_json(root/'protocol.json')==protocol,'Resume configuration differs.');state=c.read_json(root/'batch_state.json')
    else:
        c.write_json(root/'protocol.json',protocol);state={'status':'created','datasets':{},'jobs':{},'verified_new_fits':0,'planned_new_fits':18}
        snap=root/'source_snapshot';snap.mkdir()
        for name in list(specification['sources_sha256_normalized_lf'])+['prepare_ethereum.py']:shutil.copy2(project/name,snap/name)
        for name in ['run_sampling_check.py','run_assessment3_core.py','verify_submission.py','run_sensitivity.py','audited_inputs.json']:shutil.copy2(HERE/name,snap/name)
        shutil.copy2(HERE/'tools/data_quality.py',snap/'data_quality.py')
        shutil.copy2(audit,root/'source_graph_audit.json');c.write_json(root/'batch_state.json',state)
    lock=root/'running.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    rows=list(baseline)
    print('BATCH DIRECTORY: '+str(root),flush=True)
    print('18 NEW FITS ONLY. Resume with -Resume "{}"'.format(root),flush=True)
    try:
        if not state.get('self_check_passed'):
            c.run_worker(['prepare_ethereum.py','--self-check'],project,root/'preprocessing_self_check.log')
            c.run_worker(['train_ethereum.py','--self-check'],project,root/'model_self_check.log')
            state['self_check_passed']=True;c.write_json(root/'batch_state.json',state)
        for sample in CONSTRUCTION_SEEDS:
            sample_id='sample_'+str(sample);folder=root/sample_id;folder.mkdir(exist_ok=True)
            if sample_id in state['datasets']:
                dataset=Path(state['datasets'][sample_id]['dataset']);quality=check_dataset(dataset)
                c.require(quality['dataset_sha256']==state['datasets'][sample_id]['quality']['dataset_sha256'],'Prepared sample changed.')
            else:
                prep=folder/('preparation_'+c.stamp())
                c.run_worker(['prepare_ethereum.py','--audit-report',str(audit),'--seed',str(sample),'--negative-ratio','4',
                    '--fanout','8','--hops','2','--max-nodes','50000','--max-edges','2000000','--output',str(prep)],project,folder/'preparation.log')
                candidates=list(prep.glob('ethereum_pilot_*/dataset.npz'));c.require(len(candidates)==1,'Expected one prepared sample.')
                dataset=candidates[0];quality=check_dataset(dataset);c.require(quality['construction_seed']==sample,'Sample seed differs.')
                state['datasets'][sample_id]={'dataset':str(dataset),'quality':quality};c.write_json(root/'batch_state.json',state)
            for seed in MODEL_SEEDS:
                job_id=sample_id+'_model_seed'+str(seed);entry=state['jobs'].get(job_id,{})
                if entry.get('status')=='verified':
                    run=Path(entry['run_directory']);verified=training_result(run,seed)
                else:
                    attempt=folder/('model_seed{}_attempt_{}'.format(seed,c.stamp()))
                    state['jobs'][job_id]={'status':'running'};state['status']='running';c.write_json(root/'batch_state.json',state)
                    print('\n=== {}: model seeds 0/1/2; construction seed {} ==='.format(job_id,sample),flush=True)
                    c.run_worker(['train_ethereum.py','--dataset',str(dataset),'--models']+MODELS+['--seeds',str(seed),
                        '--hidden','64','--order','2','--epochs','100','--lr','0.01','--threads',str(args.threads),'--output',str(attempt)],project,folder/(job_id+'.log'))
                    candidates=list(attempt.glob('ethereum_training_*'));c.require(len(candidates)==1,'Expected one training output.')
                    run=candidates[0];verified=training_result(run,seed)
                    state['jobs'][job_id]={'status':'verified','run_directory':str(run)}
                manifest=c.read_json(run/'training_manifest.json')
                c.require(manifest['dataset_sha256']==quality['dataset_sha256'],'Training used a different sampled dataset.')
                check_predictions(run,dataset,seed)
                for row in verified:row['construction_seed']=sample
                rows.extend(verified);state['verified_new_fits']=len(rows)-len(baseline)
                summarize(root,rows);c.write_json(root/'batch_state.json',state)
                print('VERIFIED NEW FITS: {}/18'.format(state['verified_new_fits']),flush=True)
        c.require(len(rows)==27,'Expected 18 new and nine reference fits.')
        state['status']='complete';c.write_json(root/'batch_state.json',state);summarize(root,rows)
        c.bundle(root,'COMP8240_A3_Sampling_Evidence.zip');print('SAMPLING CHECK COMPLETE: 18 new fits verified; nine core fits reused.',flush=True)
    except BaseException as error:
        state['status']='failed';state['error']=str(error);state['traceback']=traceback.format_exc()
        c.write_json(root/'batch_state.json',state);summarize(root,rows);c.bundle(root,'COMP8240_A3_Sampling_Evidence.zip');raise
    finally:lock.unlink()

def check_predictions(run,dataset,seed):
    import numpy as np
    import run_sensitivity as s
    counts={'bwgnn':17858,'gcn':1346,'mlp':5506}
    with np.load(dataset,allow_pickle=False) as d:
        for model in MODELS:
            tag='bwgnn_seed{}_C2'.format(seed) if model=='bwgnn' else '{}_seed{}'.format(model,seed)
            config={'id':'sampling_reference','model':model,'hidden':64,'order':2,'loops':'base','parameters':counts[model]}
            s.verify_result(run/tag/'result.json',config,seed,reference=True)
            with np.load(run/tag/'selected_scores.npz',allow_pickle=False) as predictions:
                for split in ['val','test']:
                    mask=d[split+'_mask']
                    c.require(np.array_equal(predictions[split+'_global_id'],d['global_node_id'][mask]),'Score IDs differ from constructed sample.')
                    c.require(np.array_equal(predictions[split+'_labels'],d['labels'][mask]),'Score labels differ from constructed sample.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',required=True);p.add_argument('--core',required=True)
    p.add_argument('--audit',required=True);p.add_argument('--resume');p.add_argument('--threads',type=int,default=16);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();c.require(args.threads>0,'Threads must be positive.');execute(args)

if __name__=='__main__':main()
