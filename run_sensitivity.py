"""Thirty-five prespecified Ethereum sensitivity fits; original trainer unchanged."""
import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
import run_assessment3_core as common

HERE=Path(__file__).resolve().parent
CONFIGS=[
    {'id':'matched_bwgnn','model':'bwgnn','hidden':16,'order':2,'loops':'one_each','parameters':1394},
    {'id':'matched_gcn','model':'gcn','hidden':66,'order':2,'loops':'one_each','parameters':1388},
    {'id':'matched_mlp','model':'mlp','hidden':28,'order':2,'loops':'not_applicable','parameters':1402},
    {'id':'bwgnn_one_loop','model':'bwgnn','hidden':64,'order':2,'loops':'one_each','parameters':17858},
    {'id':'order_C1','model':'bwgnn','hidden':64,'order':1,'loops':'base','parameters':13762},
    {'id':'order_C3','model':'bwgnn','hidden':64,'order':3,'loops':'base','parameters':21954},
    {'id':'order_C5','model':'bwgnn','hidden':64,'order':5,'loops':'base','parameters':30146},
]
BY_ID={config['id']:config for config in CONFIGS}
REFERENCE={'id':'core_C2','model':'bwgnn','hidden':64,'order':2,'loops':'base','parameters':17858}
SEEDS=[0,1,2,3,4]

def verify_result(path,config,seed,reference=False):
    import numpy as np
    from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,precision_recall_curve,auc
    path=Path(path);r=common.read_json(path)
    common.require(r['model']==config['model'] and r['seed']==seed,'Model/seed mismatch.')
    common.require(r['hidden']==config['hidden'] and r['order']==(config['order'] if config['model']=='bwgnn' else None),'Architecture configuration mismatch.')
    common.require(r['epochs']==100 and r['learning_rate']==.01,'Training configuration mismatch.')
    common.require(r['trainable_parameter_count']==config['parameters'],'Parameter count mismatch.')
    if not reference:
        common.require(r.get('configuration')==config,'Sensitivity configuration missing or changed.')
        common.require(r['observed_self_loops']==(None if config['model']=='mlp' else (49224 if config['loops']=='one_each' else 0)),'Actual graph loop count mismatch.')
    folder=path.parent;scores=folder/'selected_scores.npz'
    common.require(common.sha(scores)==r['scores_sha256'],'Score fingerprint differs.')
    common.require(common.sha(folder/'selected_parameters.pt')==r['checkpoint_sha256'],'Checkpoint fingerprint differs.')
    with np.load(str(scores),allow_pickle=False) as d:
        threshold=float(d['threshold']);common.require(abs(threshold-r['selected_threshold'])<1e-14,'Threshold mismatch.')
        common.require(int(d['selected_epoch'])==r['selected_epoch'],'Selected epoch mismatch.')
        actuals={}
        for split in ['val','test']:
            y,p,m=d[split+'_labels'],d[split+'_probability32'],d[split+'_margin64']
            common.require(set(np.unique(y))=={0,1} and np.isfinite(p).all() and np.isfinite(m).all(),'Invalid scored labels or non-finite scores.')
            precision,recall,_=precision_recall_curve(y,m)
            actual={'macro_f1':float(f1_score(y,p>threshold,average='macro',labels=[0,1],zero_division=0)),
                    'roc_auc_margin64':float(roc_auc_score(y,m)),
                    'average_precision_margin64':float(average_precision_score(y,m)),
                    'pr_auc_trapezoid_margin64':float(auc(recall,precision))}
            for metric,value in actual.items():common.require(abs(value-r['validation' if split=='val' else 'test'][metric])<1e-10,'Metric recomputation differs: '+metric)
            actuals[split]=actual
        grid=np.linspace(.05,.95,19)
        f1s=[f1_score(d['val_labels'],d['val_probability32']>t,average='macro',labels=[0,1],zero_division=0) for t in grid]
        common.require(abs(threshold-grid[int(np.argmax(f1s))])<1e-14,'Threshold was not selected on validation Macro-F1.')
        common.require(not np.intersect1d(d['val_global_id'],d['test_global_id']).size,'Validation/test overlap.')
        population_hash=__import__('hashlib').sha256(d['val_global_id'].tobytes()+d['val_labels'].tobytes()+d['test_global_id'].tobytes()+d['test_labels'].tobytes()).hexdigest()
    with (folder/'epochs.csv').open(encoding='utf-8-sig',newline='') as stream:epochs=list(csv.DictReader(stream))
    common.require(len(epochs)==100,'Incomplete epoch history.')
    common.require([int(e['epoch']) for e in epochs]==list(range(100)),'Epoch sequence differs.')
    best=max(range(100),key=lambda i:float(epochs[i]['val_macro_f1']))
    common.require(int(epochs[best]['epoch'])==r['selected_epoch'],'Not first-best validation epoch.')
    common.require(abs(float(epochs[best]['val_macro_f1'])-actuals['val']['macro_f1'])<1e-10,'Selected score and epoch history disagree.')
    return {'configuration':config['id'],'model':config['model'],'seed':seed,'hidden':config['hidden'],
            'order':config['order'] if config['model']=='bwgnn' else None,'loop_policy':config['loops'],
            'parameters':config['parameters'],'selected_epoch':r['selected_epoch'],'threshold':threshold,
            'validation_macro_f1':actuals['val']['macro_f1'],'scored_population_hash':population_hash,**actuals['test']}

def trainer_module(project):
    sys.path.insert(0,str(project))
    trainer=importlib.import_module('train_ethereum')
    trainer.verify_sources()
    return trainer

def configured_builder(original,config,info):
    def build(name,graph,in_features,hidden,order):
        import dgl
        common.require(name==config['model'] and hidden==config['hidden'] and order==config['order'],'Worker model configuration differs.')
        if config['loops']=='one_each' and name in ['bwgnn','gcn']:
            graph=dgl.remove_self_loop(graph)
            if name=='bwgnn':graph=dgl.add_self_loop(graph)
        model=original(name,graph,in_features,hidden,order)
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        common.require(count==config['parameters'],'Constructed model parameter count differs.')
        if name=='mlp':info['observed_self_loops']=None
        else:
            import torch
            used=model.g if name=='bwgnn' else model.graph
            u,v=used.edges();ids=u[u==v]
            wanted=used.num_nodes() if config['loops']=='one_each' else 0
            common.require(len(ids)==wanted,'Incorrect actual self-loop count.')
            if wanted:common.require(torch.equal(torch.sort(ids).values,torch.arange(used.num_nodes(),dtype=ids.dtype)),'Self-loops are duplicated or missing.')
            info['observed_self_loops']=len(ids)
        return model
    return build

def worker(args):
    import torch,dgl,numpy as np
    project=Path(args.project).resolve();config=BY_ID[args.worker_config]
    trainer=trainer_module(project)
    specification=common.read_json(HERE/'audited_inputs.json')
    common.validate_inputs(project,specification)
    torch.set_num_threads(args.threads)
    data,pipeline=trainer.load_dataset(project/specification['datasets']['ethereum']['relative_path'])
    graph=dgl.graph((torch.from_numpy(data['source'].astype(np.int64)),torch.from_numpy(data['target'].astype(np.int64))),num_nodes=len(data['labels']))
    output=Path(args.worker_output).resolve();output.mkdir(parents=True,exist_ok=False)
    info={};original=trainer.build_model
    trainer.build_model=configured_builder(original,config,info)
    try:
        result=trainer.train_one(config['model'],args.worker_seed,graph,data,
             argparse.Namespace(hidden=config['hidden'],order=config['order'],epochs=100,lr=.01),output)
    finally:trainer.build_model=original
    result['configuration']=config;result.update(info)
    result['architecture']='Released architecture with hidden width {} and graph policy {}; model={}, filter order={}.'.format(config['hidden'],config['loops'],config['model'],result['order'])
    run=output/('{}_seed{}_C{}'.format(config['model'],args.worker_seed,config['order']) if config['model']=='bwgnn' else '{}_seed{}'.format(config['model'],args.worker_seed))
    common.write_json(run/'result.json',result)
    common.write_json(output/'worker_manifest.json',{'status':'complete','configuration':config,'seed':args.worker_seed,'result_directory':str(run),
        'dataset_sha256':specification['datasets']['ethereum']['sha256'],'environment':{'python':sys.version,'torch':torch.__version__,'dgl':dgl.__version__},**info})

def self_check(project):
    import numpy as np,torch,dgl
    trainer=trainer_module(project);torch.set_num_threads(2)
    features=np.random.RandomState(123).normal(size=(8,18)).astype(np.float32)
    labels=np.asarray([0,1,0,1,0,1,-1,-1],dtype=np.int64)
    data={'features':features,'labels':labels,'global_node_id':np.arange(8,dtype=np.int32),
          'train_mask':np.isin(np.arange(8),[0,1]),'val_mask':np.isin(np.arange(8),[2,3]),'test_mask':np.isin(np.arange(8),[4,5])}
    graph=dgl.graph(([0,1,2,3,4,5,6,7],[1,0,3,2,5,4,7,6]),num_nodes=8)
    with tempfile.TemporaryDirectory() as td:
        for config in CONFIGS:
            output=Path(td)/config['id'];output.mkdir();info={};original=trainer.build_model
            trainer.build_model=configured_builder(original,config,info)
            try:
                r=trainer.train_one(config['model'],0,graph,data,argparse.Namespace(hidden=config['hidden'],order=config['order'],epochs=2,lr=.01),output)
                common.require(r['trainable_parameter_count']==config['parameters'],'Toy parameter count differs.')
            finally:trainer.build_model=original
    print('SENSITIVITY SELF-CHECK PASS: seven actual model configurations, parameter counts, loop policies, optimizer and checkpoint restoration.',flush=True)

def references(core_root):
    state=common.read_json(core_root/'batch_state.json');protocol=common.read_json(core_root/'protocol.json')
    common.require(state['status']=='complete' and state['verified_model_fits']==75,'The 75-fit core batch is not complete.')
    common.require(protocol['seeds']==SEEDS and protocol['order']==2,'Core reference configuration differs.')
    result=[]
    for seed in SEEDS:
        entry=state['jobs']['ethereum_seed'+str(seed)]
        common.require(entry['status']=='verified','Core reference job incomplete.')
        run=Path(entry['run_directory'])
        if not run.is_dir():
            # Allows a deliberately relocated complete batch, without selecting by score.
            run=core_root/('ethereum_seed'+str(seed))/Path(str(entry['run_directory']).replace('\\','/')).name
        result.append(verify_result(run/('bwgnn_seed{}_C2'.format(seed))/'result.json',REFERENCE,seed,reference=True))
    common.require(len({r['scored_population_hash'] for r in result})==1,'Core reference populations differ.')
    return result,protocol

def aggregate(root,rows,baseline):
    import numpy as np
    common.write_csv(root/'sensitivity_results.csv',rows)
    summary=[]
    for config in CONFIGS:
        group=[r for r in rows if r['configuration']==config['id']]
        if not group:continue
        item={'configuration':config['id'],'n':len(group),'parameters':config['parameters'],'loop_policy':config['loops']}
        for key in ['validation_macro_f1']+common.METRICS:
            values=np.asarray([r[key]*100 for r in group])
            item[key+'_mean_percent']=float(values.mean());item[key+'_sd_percent']=float(values.std(ddof=1)) if len(values)>1 else None
        summary.append(item)
    common.write_csv(root/'sensitivity_summary.csv',summary)
    candidates=[]
    for order in [1,2,3,5]:
        group=baseline if order==2 else [r for r in rows if r['configuration']=='order_C'+str(order)]
        if len(group)==5:candidates.append({'order':order,'mean_validation_macro_f1':float(np.mean([r['validation_macro_f1'] for r in group]))})
    if len(candidates)==4:
        selected=max(candidates,key=lambda r:r['mean_validation_macro_f1'])
        common.write_json(root/'order_selection_by_validation.json',{'criterion':'Mean validation Macro-F1 across five seeds; first/smallest order wins exact ties. No test metric used.','candidates':candidates,'selected_order':selected['order'],'scope':'Secondary sensitivity; primary core results remain C=2.'})

def execute(args):
    project=Path(args.project).resolve();specification=common.read_json(HERE/'audited_inputs.json')
    inputs=common.validate_inputs(project,specification)
    core_root=Path(args.core).resolve();baseline,core_protocol=references(core_root)
    common.require(core_protocol['audited_specification']==specification,'Core and sensitivity input/source specifications differ.')
    protocol={'version':1,'configurations':CONFIGS,'seeds':SEEDS,'threads':args.threads,'epochs':100,'learning_rate':.01,
        'driver_sha256':common.sha(HERE/'run_sensitivity.py'),'utility_sha256':common.sha(HERE/'run_assessment3_core.py'),
        'core_directory':str(core_root),'core_protocol_sha256':common.sha(core_root/'protocol.json'),'audited_specification':specification,
        'scope':'35 fixed-sample Ethereum sensitivity fits; near-equal low-budget parameters with one self-loop for both graph models; separate default-width loop sensitivity; fixed-width C=1/3/5 versus prior C=2, with parameter differences reported. No new Amazon training or graph/split resampling.'}
    if args.preflight_only:
        print(json.dumps({'status':'preflight_passed','planned_fits':35,'configuration_count':len(CONFIGS),'inputs':inputs},indent=2));return
    root=Path(args.resume).resolve() if args.resume else project/'assessment3_runs'/('sensitivity_'+common.stamp())
    root.mkdir(parents=True,exist_ok=bool(args.resume))
    if args.resume:
        common.require(common.read_json(root/'protocol.json')==protocol,'Resume configuration differs.')
        state=common.read_json(root/'batch_state.json')
    else:
        common.write_json(root/'protocol.json',protocol);common.write_json(root/'core_reference_results.json',baseline)
        state={'status':'created','jobs':{},'verified_fits':0,'planned_fits':35,'created_utc':common.stamp()}
        snapshot=root/'source_snapshot';snapshot.mkdir()
        for name in specification['sources_sha256_normalized_lf']:shutil.copy2(str(project/name),str(snapshot/name))
        for name in ['run_sensitivity.py','run_assessment3_core.py','audited_inputs.json']:shutil.copy2(str(HERE/name),str(snapshot/name))
        common.write_json(root/'batch_state.json',state)
    lock=root/'running.lock'
    try:descriptor=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Batch is already locked. Confirm the old process has stopped before removing a stale running.lock.')
    os.write(descriptor,str(os.getpid()).encode('ascii'));os.close(descriptor)
    print('BATCH DIRECTORY: '+str(root),flush=True)
    print('RESUME: use Run-Assessment3-Sensitivity.ps1 -Resume "{}"'.format(root),flush=True)
    rows=[]
    try:
        if not state.get('self_check_passed'):
            common.run_worker([str(Path(__file__).resolve()),'--project',str(project),'--self-check'],project,root/'self_check.log')
            state['self_check_passed']=True;common.write_json(root/'batch_state.json',state)
        for config in CONFIGS:
            for seed in SEEDS:
                job_id=config['id']+'_seed'+str(seed);previous=state['jobs'].get(job_id,{})
                if previous.get('status')=='verified':
                    row=verify_result(Path(previous['result_file']),config,seed)
                    common.require(row['scored_population_hash']==baseline[0]['scored_population_hash'],'Resumed/reference populations differ.')
                    rows.append(row);continue
                job=root/job_id;job.mkdir(exist_ok=True);attempt=job/('attempt_'+common.stamp())
                state['jobs'][job_id]={'status':'running'};state['status']='running';common.write_json(root/'batch_state.json',state)
                print('\n=== FIT {}/35: {} ==='.format(len(rows)+1,job_id),flush=True)
                started=time.monotonic()
                common.run_worker([str(Path(__file__).resolve()),'--project',str(project),'--worker-config',config['id'],
                    '--worker-seed',str(seed),'--worker-output',str(attempt),'--threads',str(args.threads)],project,job/'console.log')
                manifest=common.read_json(attempt/'worker_manifest.json')
                common.require(manifest['status']=='complete' and manifest['configuration']==config and manifest['seed']==seed,'Worker completion record differs.')
                result_file=Path(manifest['result_directory'])/'result.json'
                row=verify_result(result_file,config,seed)
                common.require(row['scored_population_hash']==baseline[0]['scored_population_hash'],'Sensitivity/reference populations differ.')
                rows.append(row);state['jobs'][job_id]={'status':'verified','result_file':str(result_file),'elapsed_seconds':time.monotonic()-started}
                state['verified_fits']=len(rows);aggregate(root,rows,baseline);common.write_json(root/'batch_state.json',state)
                print('VERIFIED FITS: {}/35'.format(len(rows)),flush=True)
        common.require(len(rows)==35,'Incomplete sensitivity batch.')
        state['status']='complete';state['verified_fits']=35;state['completed_utc']=common.stamp()
        common.write_json(root/'batch_state.json',state);aggregate(root,rows,baseline)
        destination=common.bundle(root,'COMP8240_A3_Sensitivity_Evidence.zip')
        print('SENSITIVITY EVIDENCE ZIP: '+str(destination),flush=True)
        print('SENSITIVITY BATCH COMPLETE: 35 verified fits.',flush=True)
    except BaseException as error:
        state['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
        state['error']={'message':str(error),'traceback':traceback.format_exc()}
        for entry in state['jobs'].values():
            if entry.get('status')=='running':entry['status']=state['status']
        common.write_json(root/'batch_state.json',state);aggregate(root,rows,baseline)
        destination=common.bundle(root,'COMP8240_A3_Sensitivity_Evidence.zip')
        print('SENSITIVITY EVIDENCE ZIP: '+str(destination),flush=True);raise
    finally:
        if lock.exists():lock.unlink()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    parser.add_argument('--core')
    parser.add_argument('--resume')
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--worker-config',choices=sorted(BY_ID))
    parser.add_argument('--worker-seed',type=int,choices=SEEDS)
    parser.add_argument('--worker-output')
    args=parser.parse_args();common.require(args.threads>0,'Thread count must be positive.')
    if args.self_check:self_check(Path(args.project).resolve())
    elif args.worker_config:
        common.require(args.worker_seed is not None and args.worker_output,'Worker arguments missing.');worker(args)
    else:
        common.require(args.core,'--core is required.');execute(args)

if __name__=='__main__':main()
