import json,sys,pathlib,collections as co,numpy as np,pandas as pd
from sklearn.metrics import f1_score,accuracy_score
from scipy.stats import binomtest
R=pathlib.Path(sys.argv[1]); C=pathlib.Path(sys.argv[2]); OUT=pathlib.Path(sys.argv[3])
LAB={"smsa":["positive","neutral","negative"],"emot":["sadness","anger","love","fear","happy"]}
HARD={"smsa":"neutral","emot":"fear"}
rd=lambda f:[json.loads(l) for l in open(f,encoding="utf8") if l.strip()]
SA=pd.DataFrame([r for f in (R/"single_agent").glob("*.jsonl") for r in rd(f)])
def maj(cols,idx):
    out=[]
    for v in zip(*cols):
        c=co.Counter(v); top=max(c.values()); out.append(next(p for p in v if c[p]==top))
    return pd.Series(out,index=idx)
def mcn(g,a,b):
    ca,cb=(a==g).values,(b==g).values; x=int((ca&~cb).sum()); y=int((~ca&cb).sum())
    return x,y,(binomtest(x,x+y,.5).pvalue if x+y else 1.0)
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*p[i])); adj[i]=run
    return adj
def boot(g,a,b,L,n=1000,seed=2026):
    rng=np.random.default_rng(seed); G,A,B=g.values,a.values,b.values; idx=np.arange(len(G)); d=[]
    for _ in range(n):
        s=rng.choice(idx,len(idx)); d.append(f1_score(G[s],A[s],labels=L,average="macro",zero_division=0)-f1_score(G[s],B[s],labels=L,average="macro",zero_division=0))
    return np.percentile(d,[2.5,97.5])
res={}
for t,L in LAB.items():
    g=pd.read_csv(C/f"{t}_test.csv").set_index("id").label; idx=g.index
    F=lambda p: f1_score(g,p.reindex(idx),labels=L,average="macro",zero_division=0)
    PC=lambda p: dict(zip(L,f1_score(g,p.reindex(idx),labels=L,average=None,zero_division=0)))
    # H3
    h3r=[pd.read_csv(R/"baselines"/f"B3_indobert_{t}_seed{s}.csv").set_index("id").pred.reindex(idx) for s in [7,77,777]]
    H3=idx[sum((p!=g).astype(int) for p in h3r)>=2]
    runs={}  # method -> list of Series (per run)
    cost={}
    for st in ["B1_tfidf_logreg","B1_tfidf_linsvm","B2_frozen"]:
        runs[st]=[pd.read_csv(R/"baselines"/f"{st}_{t}.csv").set_index("id").pred.reindex(idx)]
        j=json.load(open(R/"baselines"/f"{st}_{t}.json")); cost[st]=dict(min=(j.get("infer_s_per_1000") or ((j.get("classifier_infer_s_per_1000") or 0)+(j.get("encode_infer_s_per_1000") or 0)))/60,tok=0)
    runs["B3_indobert"]=[pd.read_csv(R/"baselines"/f"B3_indobert_{t}_seed{s}.csv").set_index("id").pred.reindex(idx) for s in [13,21,42,87,100]]
    cost["B3_indobert"]=dict(min=np.mean([json.load(open(R/"baselines"/f"B3_indobert_{t}_seed{s}.json"))["infer_s_per_1000"] for s in [13,21,42,87,100]])/60,tok=0,
                             train_min=np.mean([json.load(open(R/"baselines"/f"B3_indobert_{t}_seed{s}.json"))["train_s"] for s in [13,21,42,87,100]])/60)
    for (m,s),gg in SA[SA.task==t].groupby(["model","method"]):
        k=f"{s}|{m}"; runs[k]=[x.set_index("id").pred.reindex(idx) for _,x in gg.groupby(["paraphrase","seed"])]
        cost[k]=dict(min=gg.latency_s.mean()*1000/4/60,tok=(gg.prompt_tokens+gg.completion_tokens).mean()*1000,out=gg.completion_tokens.mean())
    dyn={}
    for comp in ["homo-qwen","homo-llama","homo-gemma","hetero"]:
        D=pd.DataFrame(rd(R/"debate"/f"{t}_{comp}.jsonl"))
        load=lambda c: D[c] if c in D else 0
        for col,tag in [("pred_D0","D0"),("pred_vote","vote"),("pred_judge","judge")]:
            runs[f"{tag}|{comp}"]=[x.set_index("id")[col].reindex(idx) for _,x in D.groupby("seed")]
        lat=(D.latency_s-load("load_s").fillna(0) if "load_s" in D else D.latency_s)
        d0l=(D.d0_latency_s-load("d0_load_s").fillna(0) if "d0_load_s" in D else D.d0_latency_s)
        jl=(D.judge_latency_s-load("judge_load_s").fillna(0) if "judge_load_s" in D else D.judge_latency_s)
        cost[f"D0|{comp}"]=dict(min=d0l.mean()*1000/4/60,tok=(D.d0_prompt_tokens+D.d0_completion_tokens).mean()*1000)
        cost[f"vote|{comp}"]=dict(min=lat.mean()*1000/4/60,tok=(D.prompt_tokens+D.completion_tokens).mean()*1000)
        cost[f"judge|{comp}"]=dict(min=(lat.mean()+jl.mean())*1000/4/60,tok=(D.prompt_tokens+D.completion_tokens+D.judge_prompt_tokens+D.judge_completion_tokens).mean()*1000)
        # dynamics
        dd=dict(seeds=sorted(D.seed.unique().tolist()),unanimous=float((D.n_rounds==0).mean()),n=len(D))
        for tag,col in [("vote","pred_vote"),("judge","pred_judge")]:
            ch=D[D.pred_D0!=D[col]]; gold=g.reindex(ch.id).values
            dd[f"{tag}_fixed"]=int((ch[col].values==gold).sum()); dd[f"{tag}_broken"]=int((ch.pred_D0.values==gold).sum()); dd[f"{tag}_changed"]=len(ch)
        dis=D[D.n_rounds>0]; agc=co.Counter(); gl=co.Counter()
        for _,r in dis.iterrows():
            gold=g[r["id"]]; gl[gold]+=1
            r0=[a for a in r["rounds"][0]]; la=[a for a in r["rounds"][-1]]
            for a0,a1 in zip(r0,la):
                fam=a0["agent"].rstrip("012")
                agc[(fam,"r0_right")]+=a0["label"]==gold; agc[(fam,"last_right")]+=a1["label"]==gold
                agc[(fam,"right_to_wrong")]+=(a0["label"]==gold)&(a1["label"]!=gold); agc[(fam,"wrong_to_right")]+=(a0["label"]!=gold)&(a1["label"]==gold)
        dd["disagreements"]=len(dis); dd["gold_in_disagreements"]=dict(gl); dd["agents"]={f"{k[0]}:{k[1]}":v for k,v in agc.items()}
        dyn[comp]=dd
    tab={}
    for k,rs in runs.items():
        f1s=[F(p) for p in rs]; pcs=[PC(p) for p in rs]
        mj=maj(rs,idx)
        tab[k]=dict(runs=len(rs),f1=np.mean(f1s),sd=(np.std(f1s,ddof=1) if len(rs)>1 else 0.0),f1_maj=F(mj),
                    acc=np.mean([accuracy_score(g,p) for p in rs]),
                    **{f"f1_{c}":np.mean([pc[c] for pc in pcs]) for c in L},
                    h3_acc=float((mj.loc[H3]==g.loc[H3]).mean()),**cost.get(k.split("|")[0]+"|"+k.split("|")[1] if "|" in k else k,{}))
    T=pd.DataFrame(tab).T
    # paired tests on pooled majorities
    M={k:maj(v,idx) for k,v in runs.items()}
    best_single="S2_few_shot|gemma"
    fam1=[("vote|hetero","D0|hetero"),("judge|hetero","D0|hetero"),("vote|homo-llama","D0|homo-llama"),("judge|homo-llama","D0|homo-llama"),
          ("vote|hetero",best_single),("judge|hetero",best_single)]
    fam2=[("vote|hetero","B3_indobert"),("judge|hetero","B3_indobert"),("vote|hetero","B2_frozen"),("vote|hetero","B1_tfidf_linsvm")]
    tests={}
    for name,fam in [("fam1",fam1),("fam2",fam2)]:
        rows=[]
        for a,b in fam:
            x,y,p=mcn(g,M[a],M[b]); lo,hi=boot(g,M[a],M[b],L)
            rows.append(dict(A=a,B=b,A_right=x,B_right=y,p=p,dF1=F(M[a])-F(M[b]),lo=lo,hi=hi))
        df=pd.DataFrame(rows); df["p_holm"]=holm(df.p); tests[name]=df
    # per-seed vote-D0
    perseed={}
    for comp in ["hetero","homo-llama"]:
        perseed[comp]=[(F(v)-F(d)) for v,d in zip(runs[f"vote|{comp}"],runs[f"D0|{comp}"])]
        perseed[comp+"_judge"]=[(F(v)-F(d)) for v,d in zip(runs[f"judge|{comp}"],runs[f"D0|{comp}"])]
    res[t]=dict(H3=len(H3),H3_neutral=int((g.loc[H3]=="neutral").sum()) if t=="smsa" else None)
    pd.set_option("display.width",260); pd.set_option("display.max_columns",30)
    print(f"\n################ {t}  H3={len(H3)}")
    print(T.astype(float).round(4).to_string())
    for k,v in tests.items(): print(k); print(v.round(4).to_string())
    print("per-seed dF1 (vs D0):",{k:np.round(v,4).tolist() for k,v in perseed.items()})
    print(json.dumps(dyn,indent=0,default=str)[:6000])
    T.to_csv(OUT/f"T_{t}.csv"); 
    for k,v in tests.items(): v.to_csv(OUT/f"{k}_{t}.csv",index=False)
    json.dump(dyn,open(OUT/f"dyn_{t}.json","w"),default=str,indent=1)
