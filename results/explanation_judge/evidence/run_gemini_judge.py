"""Three independent Gemini REST judgments using the frozen COMP8240 study.

Python 3.8+, standard library only. No SDK installation or graph-model training.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from zipfile import ZIP_DEFLATED,ZipFile

BASE_URL='https://generativelanguage.googleapis.com/v1beta'
DEFAULT_MODEL='gemini-3.6-flash'
LABELS=['supported','partly_supported','speculative']
HERE=Path(__file__).resolve().parent

def require(condition,message):
    if not condition:raise RuntimeError(message)
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def clean(text,key):return str(text).replace(key,'[REDACTED_API_KEY]') if key else str(text)

def load_study(study):
    manifest=read(study/'study_manifest.json');pack=read(study/'BLINDED_CASES.json')
    immutable=['BLINDED_CASES.json']+['JUDGE_RUN_{}.md'.format(n) for n in [1,2,3]]
    for name in immutable:require(sha(study/name)==manifest['frozen_input_sha256'][name],'Frozen input changed: '+name)
    require(len(pack['cases'])==8 and len(pack['explanations'])==24,'Incorrect study size.')
    items={r['item_id']:r for r in pack['explanations']};require(len(items)==24,'Duplicated study IDs.')
    evidence={case['case_id']:{case['case_id']+'.'+name for name in case['evidence']} for case in pack['cases']}
    prompts={n:(study/('JUDGE_RUN_{}.md'.format(n))).read_text(encoding='utf-8-sig') for n in [1,2,3]}
    for prompt in prompts.values():
        payload=json.loads(prompt.split('INPUT DATA\n',1)[1])
        require({r['item_id'] for r in payload['explanations']}==set(items),'Prompt item inventory differs.')
        require(all(set(case)=={'case_id','evidence'} for case in payload['cases']),'Unexpected case identity/label fields.')
    return items,evidence,prompts,{name:sha(study/name) for name in immutable}

def schema(items,evidence):
    return {'type':'object','properties':{'ratings':{'type':'array','minItems':24,'maxItems':24,
      'items':{'type':'object','properties':{
       'item_id':{'type':'string','enum':sorted(items)},
       'rating':{'type':'string','enum':LABELS},
       'evidence_ids':{'type':'array','minItems':1,'items':{'type':'string','enum':sorted(set().union(*evidence.values()))}},
       'reason':{'type':'string'}},'required':['item_id','rating','evidence_ids','reason'],'additionalProperties':False}}},
       'required':['ratings'],'additionalProperties':False}

def request_body(model,prompt,output_tokens,output_schema):
    return {'contents':[{'role':'user','parts':[{'text':prompt}]}],
      'generationConfig':{'maxOutputTokens':output_tokens,
       'responseFormat':{'text':{'mimeType':'APPLICATION_JSON','schema':output_schema}}}}

def http(method,path,key,payload=None,timeout=180):
    # Authentication is in a header, never a URL or recorded request payload.
    body=None if payload is None else json.dumps(payload,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(BASE_URL+path,data=body,method=method,
         headers={'x-goog-api-key':key,'Content-Type':'application/json','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        raw=response.read().decode('utf-8')
        headers={k:clean(v,key) for k,v in response.headers.items() if k.lower() in {'date','x-request-id','x-goog-request-id','x-cloud-trace-context'}}
    return json.loads(clean(raw,key)),headers

def extract(response):
    candidates=response.get('candidates',[])
    require(len(candidates)==1,'Expected one candidate; inspect saved raw response and promptFeedback.')
    candidate=candidates[0]
    require(candidate.get('finishReason')=='STOP','Generation did not finish normally: '+str(candidate.get('finishReason')))
    text=''.join(part.get('text','') for part in candidate.get('content',{}).get('parts',[]) if not part.get('thought',False))
    require(bool(text.strip()),'Empty model response.')
    return json.loads(text)

def validate(data,items,evidence):
    require(isinstance(data,dict) and set(data)=={'ratings'},'Unexpected output fields.')
    rows=data['ratings'];require(isinstance(rows,list) and len(rows)==24,'Expected exactly 24 ratings.')
    ids=[row.get('item_id') for row in rows]
    require(len(set(ids))==24 and set(ids)==set(items),'Missing, duplicated or unexpected item IDs.')
    for row in rows:
        require(set(row)=={'item_id','rating','evidence_ids','reason'},'Incorrect rating schema.')
        require(row['rating'] in LABELS and isinstance(row['reason'],str) and bool(row['reason'].strip()),'Invalid rating/reason.')
        allowed=evidence[items[row['item_id']]['case_id']]
        require(isinstance(row['evidence_ids'],list) and bool(row['evidence_ids']) and all(x in allowed for x in row['evidence_ids']),'Evidence citation belongs to the wrong case or is missing.')
    return {row['item_id']:row['rating'] for row in rows}

def agreement(a,b):
    ids=sorted(a);require(set(ids)==set(b),'Agreement item sets differ.')
    n=len(ids);observed=sum(a[i]==b[i] for i in ids)/n
    ca,cb=Counter(a.values()),Counter(b.values())
    expected=sum(ca[k]*cb[k] for k in LABELS)/(n*n)
    return {'n':n,'exact_agreement':observed,'cohen_kappa_unweighted':(observed-expected)/(1-expected) if expected<1 else None}

def summarize(root,items,evidence):
    results={};all_rows=[];metadata=[]
    for run in [1,2,3]:
        folder=root/('run_'+str(run))
        if not (folder/'verified.json').exists():continue
        record=read(folder/'verified.json');data=read(folder/'ratings.json')
        require(sha(folder/'ratings.json')==record['ratings_sha256'] and sha(folder/'raw_response.json')==record['raw_response_sha256']
                and sha(folder/'request.json')==record['request_sha256'],'A completed input/output changed.')
        results[run]=validate(data,items,evidence);metadata.append(record)
        for row in data['ratings']:all_rows.append({'run':run,'item_id':row['item_id'],'case_id':items[row['item_id']]['case_id'],
          'rating':row['rating'],'evidence_ids':'|'.join(row['evidence_ids']),'reason':row['reason']})
    if all_rows:
        with (root/'all_ratings.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    report={'status':'complete' if len(results)==3 else 'partial','verified_runs':len(results),'verified_ratings':len(all_rows),
      'per_run_proportions':{str(run):{label:sum(x==label for x in values.values())/24 for label in LABELS} for run,values in results.items()},
      'pairwise_agreement':{'{}_vs_{}'.format(a,b):agreement(results[a],results[b]) for a,b in combinations(sorted(results),2)},
      'human_comparison':'Not computed: the earlier 12-item draft was AI-assisted, not an independent human reference.',
      'limits':['One LLM, three independent single-turn requests with different frozen presentation orders; not three independent models.',
       'Constructed descriptive texts and unsupported-claim controls; not BWGNN-native or causal explanations.',
       'Repeated judgments and within-case variants are dependent; exploratory support/consistency evaluation only.'],
      'run_metadata':metadata}
    write(root/'judge_analysis.json',report);return report

def bundle(root,key):
    path=root/'COMP8240_A3_Gemini_Judge_Evidence.zip';tmp=root/'evidence.tmp'
    with ZipFile(tmp,'w',ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file in [path,tmp] or file.suffix in {'.lock','.zip','.tmp'}:continue
            require(not key or key.encode('utf-8') not in file.read_bytes(),'A secret was found in an evidence file; export stopped.')
            archive.write(file,file.relative_to(root))
    os.replace(tmp,path);print('EVIDENCE ZIP: '+str(path),flush=True);return path

def execute(args):
    study=Path(args.study).resolve();items,evidence,prompts,fingerprints=load_study(study)
    require(bool(re.fullmatch(r'[a-z0-9][a-z0-9.-]+',args.model)),'Invalid model ID.')
    require(1024<=args.max_output_tokens<=16384,'Output-token limit must be between 1024 and 16384.')
    protocol={'protocol_version':2,'transport':'Gemini REST generateContent, independent single-turn requests',
      'model_requested':args.model,'endpoint':BASE_URL+'/models/'+args.model+':generateContent','runs':[1,2,3],
      'items_per_run':24,'frozen_input_sha256':fingerprints,'max_output_tokens':args.max_output_tokens,
      'temperature':'not set; Gemini 3.6+ does not support controlling temperature',
      'seed':'not set','conversation_history':False,'tools_enabled':False,'structured_output_schema':schema(items,evidence),
      'runner_sha256':sha(Path(__file__)),'change_from_prepared_method':'API execution replaces manual fresh-chat transport. Frozen item text, rubric and run order remain unchanged. Structured output constraints added before execution.',
      'reference_status':'Earlier 12-item review draft generated with AI assistance; no independent human agreement is claimed.'}
    if args.dry_run:
        print(json.dumps({'status':'offline_preflight_passed','generation_requests_planned':3,'ratings_planned':72,
              'model':args.model,'frozen_inputs_verified':True,'paid_requests_made':0},indent=2));return
    key=os.environ.get('GEMINI_API_KEY','').strip();require(bool(key),'GEMINI_API_KEY is missing. Use the PowerShell launcher.')
    root=Path(args.resume).resolve() if args.resume else Path(args.output).resolve()/('gemini_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    root.mkdir(parents=True,exist_ok=bool(args.resume))
    if args.resume:require(read(root/'protocol.json')==protocol,'Resume configuration or source differs; no request sent.')
    else:
        write(root/'protocol.json',protocol);snapshot=root/'frozen_inputs';snapshot.mkdir()
        for name in fingerprints:shutil.copy2(study/name,snapshot/name)
        shutil.copy2(study/'study_manifest.json',snapshot/'study_manifest.json');shutil.copy2(Path(__file__),root/'run_gemini_judge.py')
    lock=root/'running.lock';fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    print('BATCH DIRECTORY: '+str(root),flush=True)
    print('MODEL: '+args.model+' | 3 generation requests | 24 items each',flush=True)
    state={'status':'running','model_requested':args.model,'started_utc':now()}
    try:
        write(root/'batch_state.json',state)
        if not (root/'model_metadata.json').exists():
            model_meta,headers=http('GET','/models/'+args.model,key,timeout=args.timeout)
            write(root/'model_metadata.json',model_meta)
            supported=model_meta.get('supportedGenerationMethods',[])
            require(not supported or 'generateContent' in supported,'Selected model does not support generateContent.')
        for run in [1,2,3]:
            folder=root/('run_'+str(run));folder.mkdir(exist_ok=True)
            if (folder/'verified.json').exists():
                summarize(root,items,evidence);print('RUN {}/3: existing output verified; skipped'.format(run),flush=True);continue
            require(not (folder/'attempt.json').exists(),'A prior unverified attempt exists for run {}. Send its evidence before retrying; no replacement judgment was requested.'.format(run))
            body=request_body(args.model,prompts[run],args.max_output_tokens,protocol['structured_output_schema'])
            write(folder/'request.json',body)
            attempt={'run':run,'submitted_utc':now(),'status':'submitted','request_sha256':sha(folder/'request.json'),
              'prompt_sha256':fingerprints['JUDGE_RUN_{}.md'.format(run)]}
            write(folder/'attempt.json',attempt);print('RUN {}/3: requesting judgment...'.format(run),flush=True)
            started=time.monotonic()
            response,headers=http('POST','/models/'+args.model+':generateContent',key,body,args.timeout)
            write(folder/'raw_response.json',response);write(folder/'safe_response_headers.json',headers)
            ratings=extract(response);validate(ratings,items,evidence);write(folder/'ratings.json',ratings)
            record={'run':run,'status':'verified','provider':'Google Gemini API','model_requested':args.model,
              'model_version_returned':response.get('modelVersion'),'response_id':response.get('responseId'),
              'usage_metadata':response.get('usageMetadata',{}),'submitted_utc':attempt['submitted_utc'],'completed_utc':now(),
              'elapsed_seconds':time.monotonic()-started,'fresh_single_turn_request':True,
              'finish_reason':response['candidates'][0].get('finishReason'),'request_sha256':sha(folder/'request.json'),
              'raw_response_sha256':sha(folder/'raw_response.json'),'ratings_sha256':sha(folder/'ratings.json')}
            write(folder/'verified.json',record);attempt['status']='verified';write(folder/'attempt.json',attempt)
            versions=[r['model_version_returned'] for r in summarize(root,items,evidence)['run_metadata'] if r['model_version_returned']]
            require(len(set(versions))<=1,'API model version changed between runs; retain all outputs and inspect before analysis.')
            print('RUN {}/3 VERIFIED: 24 ratings'.format(run),flush=True)
        report=summarize(root,items,evidence);require(report['verified_ratings']==72,'Incomplete judgment batch.')
        versions={r['model_version_returned'] for r in report['run_metadata'] if r['model_version_returned']}
        require(len(versions)<=1,'API model version changed; retain outputs for inspection.')
        state.update(status='complete',verified_runs=3,verified_ratings=72,completed_utc=now());write(root/'batch_state.json',state)
        bundle(root,key);print('GEMINI JUDGE COMPLETE: 3 runs, 72 verified ratings.',flush=True)
    except BaseException as error:
        detail=clean(str(error),key)
        if isinstance(error,urllib.error.HTTPError):
            detail+=' | '+clean(error.read().decode('utf-8',errors='replace'),key)
        state.update(status='failed',error=detail,failed_utc=now());write(root/'batch_state.json',state)
        summarize(root,items,evidence);bundle(root,key)
        print('STOPPED: '+detail,flush=True);raise RuntimeError('Judge batch stopped; send the printed evidence ZIP.') from None
    finally:lock.unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=HERE/'study')
    p.add_argument('--output',type=Path,default=HERE/'gemini_judge_runs');p.add_argument('--model',default=DEFAULT_MODEL)
    p.add_argument('--max-output-tokens',type=int,default=8192);p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--resume',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args();execute(a)
if __name__=='__main__':main()
