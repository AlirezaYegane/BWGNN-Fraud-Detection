"""Independent checks on a prepared Ethereum sample; no Torch/DGL required."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_dataset(path):
    path=Path(path)
    manifest=json.loads((path.parent/'pipeline_manifest.json').read_text(encoding='utf-8-sig'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=manifest['dataset_sha256']:raise RuntimeError('Prepared dataset hash differs.')
    with np.load(path,allow_pickle=False) as d:
        y=d['labels'];n=len(y);u=d['source'].astype(np.int64);v=d['target'].astype(np.int64)
        masks=[d[k+'_mask'] for k in ['train','val','test']]
        assert 0<n<=50000 and d['features'].shape==(n,18)
        assert len(np.unique(d['global_node_id']))==n and np.all(np.diff(d['global_node_id'])>0)
        assert np.all((u>=0)&(u<n)&(v>=0)&(v<n)&(u!=v))
        keys=np.sort(u*n+v);assert len(np.unique(keys))==len(keys)
        assert np.array_equal(keys,np.sort(v*n+u))
        coverage=sum(x.astype(np.int8) for x in masks)
        assert np.all(coverage<=1) and np.array_equal(coverage==1,y>=0)
        assert all(set(np.unique(y[mask]))=={0,1} for mask in masks)
        scaler=StandardScaler().fit(d['raw_features'][masks[0]])
        assert np.allclose(scaler.mean_,d['standardizer_mean'],rtol=0,atol=1e-10)
        assert np.allclose(scaler.scale_,d['standardizer_scale'],rtol=0,atol=1e-10)
        assert np.allclose(scaler.transform(d['raw_features']).astype(np.float32),d['features'],rtol=0,atol=1e-6)
        assert np.isfinite(d['features']).all() and int(np.sum(y==1))==1165 and int(np.sum(y==0))==4660
        assert all(int(mask.sum())==manifest['split'][key]['total'] for mask,key in zip(masks,['train','val','test']))
        return {'dataset_sha256':digest,'nodes':n,'stored_edges':len(u),'features':18,'supervised_roots':int(np.sum(y>=0)),
          'train':int(masks[0].sum()),'val':int(masks[1].sum()),'test':int(masks[2].sum()),
          'construction_seed':manifest['sampling']['seed'],'status':'passed'}
