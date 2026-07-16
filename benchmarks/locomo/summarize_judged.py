import json, sys
from collections import Counter, defaultdict
path=sys.argv[1]
rows=[json.loads(l) for l in open(path)]
by=defaultdict(list)
for r in rows: by[str(r['category'])].append(r)
out={"by_category":{},"overall":{}}
tl=Counter(); tf=0.0
for c,items in sorted(by.items()):
    lab=Counter(x['judge_label'] for x in items); f1=sum(x['f1'] for x in items)/len(items)
    out["by_category"][c]={"n":len(items),"strict":round(lab.get('correct',0)/len(items),3),
        "lax":round((lab.get('correct',0)+lab.get('partial',0))/len(items),3),"f1":round(f1,3)}
    tl.update(lab); tf+=sum(x['f1'] for x in items)
n=len(rows)
out["overall"]={"strict":round(tl.get('correct',0)/n,3),
    "lax":round((tl.get('correct',0)+tl.get('partial',0))/n,3),"f1":round(tf/n,3)}
print(json.dumps(out))
