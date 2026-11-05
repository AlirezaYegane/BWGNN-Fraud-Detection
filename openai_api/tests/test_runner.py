"""Offline software fixtures only; no live OpenAI requests or fabricated study results."""
import ast,contextlib,io,json,os,ssl,sys,tempfile,unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_openai_judge as runner

class RunnerTests(unittest.TestCase):
    def setUp(self):self.items,self.evidence,self.prompts,self.hashes=runner.load_study(ROOT/'study')
    def fixture(self,model=runner.DEFAULT_MODEL):
        rows=[{'item_id':i,'rating':runner.LABELS[n%3],
          'evidence_ids':[sorted(self.evidence[self.items[i]['case_id']])[0]],'reason':'Synthetic software test only.'} for n,i in enumerate(sorted(self.items))]
        return {'model':model,'id':'fixture-only','usage':{'total_tokens':1},'status':'completed','error':None,
          'output':[{'type':'reasoning','summary':[]},{'type':'message','role':'assistant','status':'completed',
                     'content':[{'type':'output_text','text':json.dumps({'ratings':rows})}]}]}
    def args(self,td):return Namespace(study=ROOT/'study',model=runner.DEFAULT_MODEL,max_output_tokens=8192,timeout=240,dry_run=False,resume=None,output=Path(td),previous_gemini=None)
    def test_frozen_inputs_and_python38(self):
        self.assertEqual(len(self.items),24);self.assertEqual(len(self.evidence),8)
        ast.parse((ROOT/'run_openai_judge.py').read_text(),feature_version=8)
    def test_duplicate_ids_wrong_case_and_incomplete_finish_rejected(self):
        response=self.fixture();data=runner.extract(response);data['ratings'][1]['item_id']=data['ratings'][0]['item_id']
        with self.assertRaises(RuntimeError):runner.validate(data,self.items,self.evidence)
        data=runner.extract(self.fixture());data['ratings'][0]['evidence_ids']=['C00.E1']
        with self.assertRaises(RuntimeError):runner.validate(data,self.items,self.evidence)
        response['status']='incomplete';response['incomplete_details']={'reason':'max_output_tokens'}
        with self.assertRaises(RuntimeError):runner.extract(response)
    def test_schema_and_fresh_request(self):
        body=runner.request_body(runner.DEFAULT_MODEL,self.prompts[1],8192,runner.schema(self.items,self.evidence))
        self.assertEqual(len(body['input']),1);self.assertEqual(body['input'][0]['content'],self.prompts[1])
        self.assertNotIn('temperature',body);self.assertNotIn('tools',body)
        self.assertNotIn('previous_response_id',body)
        self.assertFalse(body['store']);self.assertEqual(body['model'],runner.DEFAULT_MODEL)
        self.assertEqual(body['reasoning']['effort'],'medium')
        self.assertTrue(body['text']['format']['strict'])
    def test_three_calls_resume_without_repeating_and_secret_absent(self):
        calls=[]
        def fake_http(method,path,key,payload=None,timeout=180):
            calls.append(method)
            return ({'id':runner.DEFAULT_MODEL} if method=='GET' else self.fixture()),{}
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'OPENAI_API_KEY':'NOT_A_REAL_KEY'}),patch.object(runner,'http',fake_http),patch.object(runner,'configure_tls'):
            a=self.args(td)
            with contextlib.redirect_stdout(io.StringIO()):runner.execute(a)
            self.assertEqual(calls.count('POST'),3);root=next(Path(td).iterdir())
            self.assertEqual(runner.read(root/'judge_analysis.json')['verified_ratings'],72)
            a.resume=root
            with contextlib.redirect_stdout(io.StringIO()):runner.execute(a)
            self.assertEqual(calls.count('POST'),3)
            with ZipFile(root/'COMP8240_A3_OpenAI_Judge_Evidence.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertTrue(all(b'NOT_A_REAL_KEY' not in z.read(n) for n in z.namelist()))
    def test_no_auto_retry_after_uncertain_network_failure(self):
        calls=[]
        def fake_http(method,path,key,payload=None,timeout=180):
            if method=='GET':return {'id':runner.DEFAULT_MODEL},{}
            calls.append('POST');raise TimeoutError('Synthetic timeout')
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'OPENAI_API_KEY':'NOT_A_REAL_KEY'}),patch.object(runner,'http',fake_http),patch.object(runner,'configure_tls'):
            a=self.args(td)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):runner.execute(a)
            root=next(Path(td).iterdir());a.resume=root
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):runner.execute(a)
            self.assertEqual(len(calls),1);self.assertTrue((root/'COMP8240_A3_OpenAI_Judge_Evidence.zip').exists())
    def test_agreement_by_identity(self):
        a={'A':'supported','B':'speculative','C':'partly_supported'}
        self.assertEqual(runner.agreement(a,dict(reversed(list(a.items()))))['cohen_kappa_unweighted'],1)

    def test_tls_uses_explicit_ca_with_verification(self):
        with tempfile.TemporaryDirectory() as td,patch.object(runner.ssl,'create_default_context',wraps=ssl.create_default_context) as factory,contextlib.redirect_stdout(io.StringIO()):
            runner.configure_tls(Path(td))
            self.assertIn('cafile',factory.call_args.kwargs)
            self.assertTrue(runner.read(Path(td)/'tls_metadata.json')['certificate_and_hostname_verification'])

    def test_preserves_gemini_failures_and_rejects_prior_response(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'openai';root.mkdir();previous=Path(td)/'gemini';previous.mkdir()
            protocol={'frozen_input_sha256':self.hashes}
            runner.write(root/'protocol.json',protocol);runner.write(previous/'protocol.json',protocol)
            runner.write(previous/'batch_state.json',{'status':'failed','error':'Synthetic HTTP 503 UNAVAILABLE'})
            runner.preserve_previous(previous,root)
            evidence=runner.read(root/'previous_provider.json')
            self.assertEqual(evidence['received_judgments'],0)
            self.assertEqual((root/'previous_gemini'/'batch_state.json').read_bytes(),(previous/'batch_state.json').read_bytes())
            run=previous/'run_1';run.mkdir();runner.write(run/'raw_response.json',{'synthetic':True})
            with self.assertRaises(RuntimeError):runner.preserve_previous(previous,root)

    def test_refusal_is_not_a_judgment(self):
        response=self.fixture()
        response['output'][1]['content']=[{'type':'refusal','refusal':'Synthetic refusal fixture'}]
        with self.assertRaises(RuntimeError):runner.extract(response)

    def test_wrong_returned_snapshot_saved_and_not_replaced(self):
        calls=[]
        def fake_http(method,path,key,payload=None,timeout=240):
            if method=='GET':return {'id':runner.DEFAULT_MODEL},{}
            calls.append(path);return self.fixture(model='unexpected-fixture-model'),{}
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'OPENAI_API_KEY':'NOT_A_REAL_KEY'}),patch.object(runner,'http',fake_http),patch.object(runner,'configure_tls'):
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):runner.execute(self.args(td))
            self.assertEqual(calls,['/responses'])
            root=next(Path(td).iterdir())
            self.assertTrue((root/'run_1'/'raw_response.json').exists())
            self.assertFalse((root/'run_1'/'verified.json').exists())

if __name__=='__main__':unittest.main()
