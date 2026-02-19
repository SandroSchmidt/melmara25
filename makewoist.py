#!/usr/bin/env python3

import math
import json
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------

RACES = [
    ("mara","Marathon",42.195),
    ("half","Half",21.0975),
    ("tenkm","10km",10.0),
    ("fivekm","5km",5.0),
    ("twofive","3km",3.0)
]

COLS = {
    "mara":(0,1,2),
    "half":(4,5,6),
    "tenkm":(8,9,10),
    "fivekm":(12,13,14),
    "twofive":(16,17,18)
}

# -------------- HELPERS -----------------

def tsec(x):
    if pd.isna(x): return np.nan
    if hasattr(x,"hour"):
        return x.hour*3600+x.minute*60+x.second
    try:
        h,m,s=str(x).split(":")
        return int(h)*3600+int(m)*60+float(s)
    except:
        return np.nan

def hms(s):
    s=int(s)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

# -------------- LOAD --------------------

df=pd.read_excel("Zeiten Melbourne latest.xlsx",header=None)

all_woist={}

for key,name,km in RACES:

    c0,c1,_=COLS[key]

    g=df.iloc[:,c0].map(tsec).to_numpy()
    n=df.iloc[:,c1].map(tsec).to_numpy()

    mask=np.isfinite(g)&np.isfinite(n)&(n>0)
    g=g[mask]
    n=n[mask]

    delay=np.abs(g-n)
    # -------- STATS --------

    print("\n",name)
    print("runners:",len(n))
    print("min delay:",hms(delay.min()))
    print("max delay:",hms(delay.max()))

    speed = km*1000/g
    print("fastest km/h:",speed.max()*3.6)
    print("slowest km/h:",speed.min()*3.6)

    # -------- FASTEST / SLOWEST RUNNER --------

    fastest_idx = np.argmax(speed)
    slowest_idx = np.argmin(speed)

    print("\nFastest runner:")
    print("  Brutto:", hms(n[fastest_idx]))
    print("  Netto:", hms(g[fastest_idx]))
    print("  speed km/h:", speed[fastest_idx]*3.6)

    print("\nSlowest runner:")
    print("  Brutto:", hms(n[slowest_idx]))
    print("  Netto:", hms(g[slowest_idx]))
    print("  speed km/h:", speed[slowest_idx]*3.6)

    # -------- WOIST --------

    dist=km*1000
    seg=int(math.ceil(dist/25))
    buckets=seg+1

    finish=delay+g
    tmax=int(math.ceil(finish.max()/60))

    woist=[]

    for minute in range(tmax+1):
        t=minute*60

        waiting=(t<delay).sum()

        active=(t>=delay)&(t<delay+g)
        pos=(t-delay[active])*speed[active]
        pos=np.clip(pos,0,dist-1e-6)

        segi=(pos//25).astype(int)+1

        arr=np.zeros(buckets,int)
        arr[0]=waiting
        for s in segi:
            arr[s]+=1

        woist.append(arr.tolist())

    all_woist[key]=woist

# optional speichern
json.dump(all_woist,open("woist.json","w"))
print("\nwoist.json written")
