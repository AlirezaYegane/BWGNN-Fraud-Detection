"""Obtain the three original files at the audited upstream commit, with hash checks."""
import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

COMMIT='de0631f039bbd19c1890b483cc01f1007f596af7'
EXPECTED={'main.py':'1642e10e5c799ebd303bb20d66a991a55363276b35855996f5615eba0fd1ea13',
 'dataset.py':'229b5e5101c63879264802f909e64396a49be694a8bb4815dfea201f9b78d1fb',
 'BWGNN.py':'0ac45b1104f80fa17c50cd5e3bf9aa6241e008e6b57fdab9366ec69f80cb2f84'}

def fingerprint(content):return hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-project',type=Path,help='Reuse the already audited original project; no network needed.')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    for name,wanted in EXPECTED.items():
        target=root/name
        if target.exists():
            if fingerprint(target.read_bytes())!=wanted:raise RuntimeError('Existing original differs; it was not overwritten: '+name)
            print('VERIFIED '+name);continue
        if args.from_project:content=(args.from_project/name).read_bytes()
        else:
            url='https://raw.githubusercontent.com/squareRoot3/Rethinking-Anomaly-Detection/'+COMMIT+'/'+name
            with urllib.request.urlopen(url,timeout=60) as response:content=response.read()
        if fingerprint(content)!=wanted:raise RuntimeError('Upstream fingerprint mismatch: '+name)
        # Preserve original bytes; never modify model/training semantics.
        target.write_bytes(content);print('INSTALLED AND VERIFIED '+name)
    print('UPSTREAM BOOTSTRAP PASS')

if __name__=='__main__':main()
