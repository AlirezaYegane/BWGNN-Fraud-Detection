"""Analyse actual blinded judge/human ratings; reject blank or incomplete templates."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from itertools import combinations
from sklearn.metrics import cohen_kappa_score

RATINGS=['speculative','partly_supported','supported']
def read_response(path,expected):
    text=Path(path).read_text(encoding='utf-8-sig').strip()
    if text.startswith('```'):
        lines=text.splitlines();text='\n'.join(lines[1:-1])
    data=json.loads(text);rows=data['ratings']
    if len(rows)!=len(expected):raise ValueError('Expected 24 ratings, not a blank or partial response: '+str(path))
    ids=[r['item_id'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('Missing, duplicated or unexpected item IDs.')
    for r in rows:
        if r['rating'] not in RATINGS or not str(r.get('reason','')).strip():raise ValueError('Invalid rating or missing reason.')
    return {r['item_id']:r['rating'] for r in rows}

def compare(a,b,ids):
    x=[a[i] for i in ids];y=[b[i] for i in ids]
    k=float(cohen_kappa_score(x,y,labels=RATINGS))
    return {'n':len(ids),'exact_agreement':sum(u==v for u,v in zip(x,y))/len(ids),'cohen_kappa_unweighted':k if math.isfinite(k) else None}

def analyse(study,responses,human,metadata):
    casepack=json.loads((study/'BLINDED_CASES.json').read_text());manifest=json.loads((study/'study_manifest.json').read_text())
    for name in ['BLINDED_CASES.json']+['JUDGE_RUN_{}.md'.format(i) for i in [1,2,3]]:
        if hashlib.sha256((study/name).read_bytes()).hexdigest()!=manifest['frozen_input_sha256'][name]:raise ValueError('Frozen study input changed: '+name)
    ids=sorted(r['item_id'] for r in casepack['explanations']);judges=[read_response(p,ids) for p in responses]
    with Path(human).open(encoding='utf-8-sig',newline='') as f:hr=list(csv.DictReader(f))
    wanted=set(manifest['human_item_ids'])
    if len(hr)!=12 or len({r['item_id'] for r in hr})!=12 or {r['item_id'] for r in hr}!=wanted:raise ValueError('Expected the fixed 12 human item IDs.')
    if not all(r['rating'] in RATINGS and r['reason'].strip() and r['reviewer_id'].strip() and r['completed_date'].strip() for r in hr):raise ValueError('Human ratings or metadata are blank/incomplete.')
    human_map={r['item_id']:r['rating'] for r in hr}
    with Path(metadata).open(encoding='utf-8-sig',newline='') as f:meta=list(csv.DictReader(f))
    if len(meta)!=3 or {r['run'] for r in meta}!={'1','2','3'}:raise ValueError('Expected metadata for three runs.')
    if not all(r['provider'].strip() and r['model_name_from_UI'].strip() and r['run_date'].strip() and r['fresh_chat'].strip() for r in meta):raise ValueError('Missing actual model/run metadata.')
    if not all(r['fresh_chat'].strip().lower() in {'yes','true','y','1'} for r in meta):raise ValueError('Each run must be a fresh chat under the frozen protocol.')
    if len({(r['provider'].strip(),r['model_name_from_UI'].strip()) for r in meta})!=1:raise ValueError('The fixed design specifies the same model in all three runs.')
    result={'status':'complete','case_count':8,'explanations':24,'judge_runs':3,'human_items':12,
      'judge_metadata':meta,'rating_proportions':[{rating:sum(v==rating for v in judge.values())/24 for rating in RATINGS} for judge in judges],
      'judge_pairwise_agreement':{'{}_vs_{}'.format(i+1,j+1):compare(judges[i],judges[j],ids) for i,j in combinations(range(3),2)},
      'human_vs_judge':{'run_'+str(i+1):compare(human_map,judge,sorted(wanted)) for i,judge in enumerate(judges)},
      'limitations':['Constructed rule-written texts/controls, not native BWGNN explanations or causal explanations.',
       'One author human reviewer; no multi-expert annotation claim.',
       'Same LLM in fresh chats; repeated ratings are not independent models.',
       'Items within a case and templated controls are dependent; exploratory agreement only.']}
    majority={};ties=[]
    for item in sorted(wanted):
        counts={r:sum(j[item]==r for j in judges) for r in RATINGS};maximum=max(counts.values())
        winners=[r for r,v in counts.items() if v==maximum]
        if len(winners)==1:majority[item]=winners[0]
        else:ties.append(item)
    result['majority_tie_items']=ties
    result['human_vs_unique_majority']=compare(human_map,majority,sorted(majority)) if majority else None
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent/'explanation_study')
    p.add_argument('--responses',type=Path,nargs=3,required=True);p.add_argument('--human',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyse(a.study,a.responses,a.human,a.metadata);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('EXPLANATION ANALYSIS COMPLETE: actual ratings verified.')
if __name__=='__main__':main()
