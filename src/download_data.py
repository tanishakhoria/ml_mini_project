"""Download the NSL-KDD train/test files into ./data (run once)."""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA, exist_ok=True)
for f in ["KDDTrain+.txt", "KDDTest+.txt"]:
    dest = os.path.join(DATA, f)
    if not os.path.exists(dest):
        print("downloading", f)
        urllib.request.urlretrieve(BASE + f, dest)
print("done")
