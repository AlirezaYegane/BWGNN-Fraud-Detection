"""Portable saved-score verification. This does not claim to retrain checkpoints."""
import argparse,ast,importlib.metadata,json,platform,subprocess,sys
from pathlib import Path,PureWindowsPath
import run_assessment3_core as c
import run_sensitivity as sensitivity

ROOT=Path(__file__).resolve().parent
def relocated(root,value,last_parts):
    original=Path(value)
    if original.is_file() or original.is_dir():return original
    return root.joinpath(*PureWindowsPath(value).parts[-last_parts:])

def verify_core(root):
    root=Path(root);state=c.read_json(root/'batch_state.json');protocol=c.read_json(root/'protocol.json')
    c.require(state['status']=='complete' and state['verified_model_fits']==75,'Incomplete core batch.')
    rows=[]
    for job in c.jobs(ROOT,protocol['audited_specification'],protocol['threads']):
        entry=state['jobs'][job['id']];c.require(entry['status']=='verified','Unverified core job.')
        rows.extend(c.verify_job(relocated(root,entry['run_directory'],2),job))
    c.require(len(rows)==75,'Expected 75 core fits.');return rows

def verify_sensitivity(root):
    root=Path(root);state=c.read_json(root/'batch_state.json');rows=[]
    c.require(state['status']=='complete' and state['verified_fits']==35,'Incomplete sensitivity batch.')
    for config in sensitivity.CONFIGS:
        for seed in sensitivity.SEEDS:
            entry=state['jobs'][config['id']+'_seed'+str(seed)]
            c.require(entry['status']=='verified','Unverified sensitivity job.')
            rows.append(sensitivity.verify_result(relocated(root,entry['result_file'],4),config,seed))
    c.require(len({r['scored_population_hash'] for r in rows})==1,'Scored populations differ.')
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path);parser.add_argument('--sensitivity',type=Path)
    parser.add_argument('--runtime',action='store_true',help='Run real CPU model and preprocessing self-checks in the pinned environment.')
    parser.add_argument('--output',type=Path,default=ROOT/'validation_receipt_local.json')
    args=parser.parse_args();receipt={'status':'passed','python':sys.version,'checks':[],'scientific_retraining':False}
    for path in sorted(ROOT.rglob('*.py')):
        if '__pycache__' not in path.parts:ast.parse(path.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    receipt['checks'].append('Python 3.8 syntax')
    from bootstrap_upstream import EXPECTED,fingerprint
    for name,wanted in EXPECTED.items():c.require(fingerprint((ROOT/name).read_bytes())==wanted,'Original source differs: '+name)
    receipt['checks'].append('Audited original source fingerprints')
    if args.core:receipt['core_fits']=len(verify_core(args.core))
    if args.sensitivity:receipt['sensitivity_fits']=len(verify_sensitivity(args.sensitivity))
    if args.runtime:
        specification=c.read_json(ROOT/'audited_inputs.json')
        c.require(platform.python_version()=='3.8.20','Runtime checks require the audited Python 3.8.20 environment.')
        for name,wanted in specification['packages'].items():
            c.require(importlib.metadata.version(name).split('+')[0]==wanted,'Runtime dependency differs: '+name)
        for script in ['prepare_ethereum.py','audit_bwgnn.py','train_ethereum.py']:
            subprocess.run([sys.executable,str(ROOT/script),'--self-check'],cwd=str(ROOT),check=True)
        receipt['checks'].append('Actual preprocessing, original-loop instrumentation and three-model CPU self-checks')
    else:receipt['runtime_model_checks']='not requested; saved scores verified only'
    c.write_json(args.output,receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
