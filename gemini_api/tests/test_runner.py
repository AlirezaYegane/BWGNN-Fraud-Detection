"""Offline software fixtures only; no live Gemini requests or fabricated study results."""
import ast,contextlib,io,json,os,sys,tempfile,unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_gemini_judge as runner

class RunnerTests(unittest.TestCase):
    def setUp(self):self.items,self.evidence,self.prompts,self.hashes=runner.load_study(ROOT/'study')
    def fixture(self,model='fixture-version'):
        rows=[{'item_id':i,'rating':runner.LABELS[n%3],
          'evidence_ids':[sorted(self.evidence[self.items[i]['case_id']])[0]],'reason':'Synthetic software test only.'} for n,i in enumerate(sorted(self.items))]
        return {'modelVersion':model,'responseId':'fixture-only','usageMetadata':{'totalTokenCount':1},
          'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({'ratings':rows})}]}}]}
    def args(self,td):return Namespace(study=ROOT/'study',model=runner.DEFAULT_MODEL,max_output_tokens=8192,timeout=180,dry_run=False,resume=None,output=Path(td))
    def test_frozen_inputs_and_python38(self):
        self.assertEqual(len(self.items),24);self.assertEqual(len(self.evidence),8)
        ast.parse((ROOT/'run_gemini_judge.py').read_text(),feature_version=(3,8))
    def test_duplicate_ids_wrong_case_and_incomplete_finish_rejected(self):
        response=self.fixture();data=runner.extract(response);data['ratings'][1]['item_id']=data['ratings'][0]['item_id']
        with self.assertRaises(RuntimeError):runner.validate(data,self.items,self.evidence)
        data=runner.extract(self.fixture());data['ratings'][0]['evidence_ids']=['C00.E1']
        with self.assertRaises(RuntimeError):runner.validate(data,self.items,self.evidence)
        response['candidates'][0]['finishReason']='MAX_TOKENS'
        with self.assertRaises(RuntimeError):runner.extract(response)
    def test_schema_and_fresh_request(self):
        body=runner.request_body(runner.DEFAULT_MODEL,self.prompts[1],8192,runner.schema(self.items,self.evidence))
        self.assertEqual(len(body['contents']),1);self.assertEqual(body['contents'][0]['parts'][0]['text'],self.prompts[1])
        self.assertNotIn('temperature',body['generationConfig']);self.assertNotIn('tools',body)
        self.assertNotIn('previous_interaction_id',body)
    def test_three_calls_resume_without_repeating_and_secret_absent(self):
        calls=[]
        def fake_http(method,path,key,payload=None,timeout=180):
            calls.append(method)
            return ({'supportedGenerationMethods':['generateContent']} if method=='GET' else self.fixture()),{}
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'GEMINI_API_KEY':'NOT_A_REAL_KEY'}),patch.object(runner,'http',fake_http):
            a=self.args(td)
            with contextlib.redirect_stdout(io.StringIO()):runner.execute(a)
            self.assertEqual(calls.count('POST'),3);root=next(Path(td).iterdir())
            self.assertEqual(runner.read(root/'judge_analysis.json')['verified_ratings'],72)
            a.resume=root
            with contextlib.redirect_stdout(io.StringIO()):runner.execute(a)
            self.assertEqual(calls.count('POST'),3)
            with ZipFile(root/'COMP8240_A3_Gemini_Judge_Evidence.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertTrue(all(b'NOT_A_REAL_KEY' not in z.read(n) for n in z.namelist()))
    def test_no_auto_retry_after_uncertain_network_failure(self):
        calls=[]
        def fake_http(method,path,key,payload=None,timeout=180):
            if method=='GET':return {'supportedGenerationMethods':['generateContent']},{}
            calls.append('POST');raise TimeoutError('Synthetic timeout')
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,{'GEMINI_API_KEY':'NOT_A_REAL_KEY'}),patch.object(runner,'http',fake_http):
            a=self.args(td)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):runner.execute(a)
            root=next(Path(td).iterdir());a.resume=root
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):runner.execute(a)
            self.assertEqual(len(calls),1);self.assertTrue((root/'COMP8240_A3_Gemini_Judge_Evidence.zip').exists())
    def test_agreement_by_identity(self):
        a={'A':'supported','B':'speculative','C':'partly_supported'}
        self.assertEqual(runner.agreement(a,dict(reversed(list(a.items()))))['cohen_kappa_unweighted'],1)

if __name__=='__main__':unittest.main()
