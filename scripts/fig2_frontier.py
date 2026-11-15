import sys,pandas as pd,numpy as np,matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A=sys.argv[1]; out=sys.argv[2]
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":8})
fig,axs=plt.subplots(1,2,figsize=(7.0,3.0),dpi=300,sharey=False)
G={"Supervised baseline":("#2a78d6","o"),"Single-agent LLM":("#eb6834","s"),"Multi-agent debate":("#1baf7a","^")}
NM={"B1_tfidf_linsvm":"TF-IDF+SVM","B1_tfidf_logreg":"TF-IDF+LR","B2_frozen":"Frozen emb.","B3_indobert":"IndoBERT"}
for ax,t,title in zip(axs,["smsa","emot"],["(a) SmSA","(b) EmoT"]):
    T=pd.read_csv(f"{A}/T_{t}.csv",index_col=0)
    T["grp"]=["Supervised baseline" if k.startswith("B") else "Multi-agent debate" if k.split("|")[0] in("D0","vote","judge") else "Single-agent LLM" for k in T.index]
    T["x"]=T["min"].clip(lower=1e-3)
    for g,(c,m) in G.items():
        d=T[T.grp==g]; ax.scatter(d.x,d.f1,s=22,c=c,marker=m,edgecolors="white",linewidths=0.6,label=g,zorder=3)
    # pareto frontier
    s=T.sort_values("x"); best=-1; pts=[]
    for k,r in s.iterrows():
        if r.f1>best: pts.append((r.x,r.f1,k)); best=r.f1
    ax.step([p[0] for p in pts]+[300],[p[1] for p in pts]+[pts[-1][1]],where="post",color="#8a8a85",lw=1,ls="--",zorder=2,label="Frontier")
    OFF={"smsa":{"B3_indobert":(4,3),"S2_few_shot|gemma":(-14,5),"D0|hetero":(5,-2),"vote|hetero":(5,-1),"judge|hetero":(5,-3),"D0|homo-llama":(-40,2),"vote|homo-llama":(5,-3),"B1_tfidf_linsvm":(4,3),"B2_frozen":(4,3)},
         "emot":{"B3_indobert":(4,3),"S2_few_shot|gemma":(-48,3),"D0|hetero":(5,-7),"vote|hetero":(6,-6),"judge|hetero":(6,2),"D0|homo-llama":(5,-2),"vote|homo-llama":(5,2),"B1_tfidf_logreg":(4,3),"B2_frozen":(4,3)}}[t]
    lab={"B3_indobert":"IndoBERT","S2_few_shot|gemma":"S2 Gemma","vote|hetero":"Hetero vote","judge|hetero":"Hetero judge","D0|hetero":"Hetero D0","vote|homo-llama":"Llama vote","D0|homo-llama":"Llama D0","B1_tfidf_linsvm":"TF-IDF+SVM","B1_tfidf_logreg":"TF-IDF+LR","B2_frozen":"Frozen emb."}
    for k,o in OFF.items():
        r=T.loc[k]; ax.annotate(lab[k],(r.x,r.f1),xytext=o,textcoords="offset points",fontsize=6.5,color="#333")
    ax.set_xscale("log"); ax.set_xlim(2e-3,300); ax.set_title(title,fontsize=9)
    ax.set_xlabel("Inference time per 1,000 samples (minutes, log scale)"); ax.grid(alpha=.25,lw=.5)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
axs[0].set_ylabel("Macro-F1 (mean over runs)")
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=7,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=(0,.07,1,1)); fig.savefig(out,dpi=300,bbox_inches="tight")
