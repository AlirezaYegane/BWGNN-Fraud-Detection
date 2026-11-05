"""Software fixtures only. These are not scientific or human/LLM study results."""
import contextlib,csv,hashlib,io,json,sys,tempfile,unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import analyse_explanation_study as explanation
import run_sampling_check as sampling

class WorkflowTests(unittest.TestCase):
    def test_empty_and_duplicate_judge_responses_rejected(self):
        pack=json.loads((ROOT/'explanation_study/BLINDED_CASES.json').read_text());ids=[r['item_id'] for r in pack['explanations']]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'response.json';p.write_text('{"ratings":[]}')
            with self.assertRaises(ValueError):explanation.read_response(p,ids)
            rows=[{'item_id':i,'rating':'supported','reason':'Synthetic test only.'} for i in ids]
            rows[-1]['item_id']=rows[0]['item_id'];p.write_text(json.dumps({'ratings':rows}))
            with self.assertRaises(ValueError):explanation.read_response(p,ids)

    def test_agreement_matches_item_ids_not_presentation_order(self):
        a={'A':'supported','B':'partly_supported','C':'speculative'};b=dict(reversed(list(a.items())))
        result=explanation.compare(a,b,['A','B','C'])
        self.assertEqual(result['exact_agreement'],1);self.assertEqual(result['cohen_kappa_unweighted'],1)

    def test_complete_synthetic_ratings_pipeline(self):
        study=ROOT/'explanation_study';pack=json.loads((study/'BLINDED_CASES.json').read_text())
        manifest=json.loads((study/'study_manifest.json').read_text());ids=sorted(r['item_id'] for r in pack['explanations'])
        values={item:explanation.RATINGS[i%3] for i,item in enumerate(ids)}
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);responses=[]
            for run in [1,2,3]:
                p=td/('run'+str(run)+'.json');p.write_text(json.dumps({'ratings':[{'item_id':i,'rating':values[i],'reason':'Software fixture only.'} for i in reversed(ids)]}));responses.append(p)
            human=td/'human.csv'
            with human.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['item_id','rating','reason','reviewer_id','completed_date']);w.writeheader()
                for item in manifest['human_item_ids']:w.writerow({'item_id':item,'rating':values[item],'reason':'Synthetic fixture.','reviewer_id':'TEST','completed_date':'test'})
            meta=td/'metadata.csv'
            with meta.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['run','provider','model_name_from_UI','run_date','fresh_chat']);w.writeheader()
                for run in [1,2,3]:w.writerow({'run':run,'provider':'Synthetic fixture','model_name_from_UI':'No actual LLM','run_date':'test','fresh_chat':'yes'})
            result=explanation.analyse(study,responses,human,meta)
            self.assertEqual(result['human_items'],12);self.assertEqual(result['majority_tie_items'],[])
            self.assertEqual(result['human_vs_unique_majority']['exact_agreement'],1)

    def test_sampling_failure_resume_and_fixed_protocol(self):
        c=sampling.c;spec=c.read_json(ROOT/'audited_inputs.json')
        baseline=[]
        for seed in sampling.MODEL_SEEDS:
            for model in sampling.MODELS:
                baseline.append({'construction_seed':0,'model_seed':seed,'model':model,**{k:.8 for k in c.METRICS}})
        calls=[];fail=[True]
        with tempfile.TemporaryDirectory() as td:
            project=Path(td)/'project';project.mkdir();core=Path(td)/'core';core.mkdir();compact=Path(td)/'compact';compact.mkdir()
            for name in list(spec['sources_sha256_normalized_lf'])+['prepare_ethereum.py']:
                origin=ROOT/name
                (project/name).write_bytes(origin.read_bytes() if origin.exists() else b'fixture')
            audit=Path(td)/'audit.json';c.write_json(audit,{'compact_directory':str(compact)})
            c.write_json(core/'protocol.json',{'audited_specification':spec})
            original_sha=c.sha
            def sha(path,normalize=False):return sampling.AUDIT_SHA if Path(path)==audit else original_sha(path,normalize)
            def quality(path):return {'dataset_sha256':original_sha(path),'construction_seed':next(int(x.split('_')[1]) for x in Path(path).parts if x in ['sample_1','sample_2'])}
            def worker(args,project,log):
                if '--self-check' in args:return
                get=lambda flag:args[args.index(flag)+1]
                output=Path(get('--output'));output.mkdir(parents=True)
                if args[0]=='prepare_ethereum.py':
                    folder=output/'ethereum_pilot_fixture';folder.mkdir();(folder/'dataset.npz').write_bytes(b'fixture sample');return
                job=(next(x for x in output.parts if x in ['sample_1','sample_2']),int(get('--seeds')));calls.append(job)
                if fail[0] and len(calls)==3:raise RuntimeError('Intentional software fixture interruption')
                folder=output/'ethereum_training_fixture';folder.mkdir();c.write_json(folder/'training_manifest.json',{'dataset_sha256':original_sha(Path(get('--dataset')))})
            def training(run,seed):return [dict(r) for r in baseline if r['model_seed']==seed]
            args=Namespace(project=str(project),core=str(core),audit=str(audit),threads=16,resume=None,preflight_only=True)
            with patch.object(c,'sha',sha),patch.object(c,'validate_inputs',lambda *a:None),patch.object(c,'run_worker',worker), \
                 patch.object(sampling,'reference_rows',lambda p:(baseline,{'audited_specification':spec})), \
                 patch.object(sampling,'check_dataset',quality),patch.object(sampling,'training_result',training),patch.object(sampling,'check_predictions',lambda *a:None):
                with contextlib.redirect_stdout(io.StringIO()):sampling.execute(args)
                self.assertFalse((project/'assessment3_runs').exists());args.preflight_only=False
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Intentional'):sampling.execute(args)
                root=next((project/'assessment3_runs').iterdir());state=c.read_json(root/'batch_state.json')
                self.assertEqual(state['verified_new_fits'],6);self.assertFalse((root/'running.lock').exists())
                with ZipFile(root/'COMP8240_A3_Sampling_Evidence.zip') as z:self.assertIsNone(z.testzip())
                args.resume=str(root);fail[0]=False
                with contextlib.redirect_stdout(io.StringIO()):sampling.execute(args)
                state=c.read_json(root/'batch_state.json');self.assertEqual(state['status'],'complete');self.assertEqual(state['verified_new_fits'],18)
                self.assertEqual(len(calls),7);self.assertEqual(calls.count(calls[2]),2)
                with (root/'sampling_summary.csv').open(newline='') as f:summary=list(csv.DictReader(f))
                self.assertEqual(len(summary),9);self.assertTrue(all(int(r['n_model_seeds'])==3 for r in summary))
                args.threads=8
                with self.assertRaisesRegex(RuntimeError,'Resume configuration differs'):sampling.execute(args)

if __name__=='__main__':unittest.main()
