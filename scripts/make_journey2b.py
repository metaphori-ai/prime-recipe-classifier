"""Regenerate journey-2b.svg (day-2 summary flow). Edit the box() calls, run: python3 scripts/make_journey2b.py"""
from pathlib import Path
F="-apple-system,Inter,Helvetica,Arial,sans-serif"
PAL={'gray':('#F1EFE8','#5F5E5A','#2C2C2A','#5F5E5A'),'purple':('#EEEDFE','#534AB7','#26215C','#534AB7'),
     'green':('#E1F5EE','#0F6E56','#04342C','#0F6E56'),'orange':('#FAECE7','#993C1D','#4A1B0C','#993C1D')}
W=780; BW,BH=220,60; X={'L':30,'C':280,'R':530}; CX={c:X[c]+BW/2 for c in X}
ROW=lambda r:30+90*r
out=[]
def box(col,r,title,sub,kind,dashed=False,bold=False):
    x,y=X[col],ROW(r); fill,stroke,tc,sc=PAL[kind]
    dash=' stroke-dasharray="5 4"' if dashed else ''; sw='1.2' if bold else '0.5'
    out.append(f'<g><rect fill="{fill}" stroke="{stroke}" x="{x}" y="{y}" width="{BW}" height="{BH}" rx="8" stroke-width="{sw}"{dash}/>'
      f'<text style="font:500 14px {F};fill:{tc}" x="{CX[col]}" y="{y+20}" text-anchor="middle" dominant-baseline="central">{title}</text>'
      f'<text style="font:400 12px {F};fill:{sc}" x="{CX[col]}" y="{y+39}" text-anchor="middle" dominant-baseline="central">{sub}</text></g>')
A='stroke="#888780" stroke-width="1.5"'
def down(col,r):
    x=CX[col]; out.append(f'<line x1="{x}" y1="{ROW(r)+BH}" x2="{x}" y2="{ROW(r+1)}" {A} marker-end="url(#arrow)"/>')
def elbow(c0,r,c1):
    x0,x1=CX[c0],CX[c1]; y0=ROW(r)+BH; yb=y0+15; y1=ROW(r+1)
    out.append(f'<path d="M{x0} {y0} L{x0} {yb} L{x1} {yb}" fill="none" {A}/><line x1="{x1}" y1="{yb}" x2="{x1}" y2="{y1}" {A} marker-end="url(#arrow)"/>')
def label(col,r,text,dx=8):
    out.append(f'<text style="font:400 11px {F};fill:#888780" x="{CX[col]+dx}" y="{ROW(r)+BH+8}" dominant-baseline="central">{text}</text>')

box('C',0,'Labels are the ceiling','SFT found 5/32 influences, RL 0/32','orange')
elbow('C',0,'L'); down('C',0); elbow('C',0,'R'); label('C',0,'teacher bake-off on gold',dx=14)
box('L',1,'Qwen3.6-27B (local)','0.808 · thinking +0.018 at 24× cost','gray')
box('C',1,'DeepSeek · Qwen3.6-35B','0.784 · 0.761 (the old silver)','gray')
box('R',1,'Sonnet 5.5  ✓','0.843 · 18/32 influences','purple',bold=True)
down('L',1); down('R',1)
box('L',2,'Thinking trap','2k budget: 34% truncated → biased','orange')
box('C',2,'Gold audit','11 rows · Contemporary = form','gray')
box('R',2,'Relabel 7,704','98% prompt-cached · 0 errors','gray')
elbow('R',2,'C')
box('C',3,'dataset_sonnet','5,066 train · influences 6.6% vs 0.6%','gray')
elbow('C',3,'L'); elbow('C',3,'R'); label('C',3,'two SFT stacks, same data',dx=14)
box('L',4,'SFT on Mac · LoRA r8','mlx-lm-lora · 5.6M params · 95 min','green')
box('R',4,'SFT on pod · full FT','prime-rl · 1.9B params · 33 min','green')
down('L',4); down('R',4); label('R',4,'DCP → convert (patched) → HF',dx=-200)
box('L',5,'B-mac = ?','eval next · LoRA vs full FT','gray',dashed=True)
box('C',5,'Decision: T = 1.2','entropy 0.07 — where arm C died','orange')
box('R',5,'B″ = 0.821','labels beat params (4B was 0.794)','green',bold=True)
elbow('R',5,'C')
box('C',6,'Arm D · GRPO from B″','80 steps · 2 h 12 m · KL 0','purple')
down('C',6)
box('L',7,'Rare is enough','influences 3 → 11 from a 0.7% seed','orange')
box('C',7,'D = 0.857','origin .767 · contains .967','green',bold=True)
box('R',7,'Depth unchanged','bonus too small → reward v3','orange')
down('C',7)
box('C',8,'Next','reward v3 · SVML-think · ship','gray',dashed=True)
H=ROW(8)+BH+30
svg=(f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" xmlns="http://www.w3.org/2000/svg"><rect width="{W}" height="{H}" fill="#FFFFFF"/>'
 '<title>Recipe-classify journey, day 2 (summary)</title><desc>Label ceiling → teacher bake-off → Sonnet relabel → two SFT stacks → GRPO from SFT → 0.857, with the decisions and traps along the way.</desc>'
 '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#888780"/></marker></defs>\n'
 +'\n'.join(out)+'\n</svg>\n')
Path(__file__).resolve().parent.parent.joinpath('journey-2b.svg').write_text(svg); print(W,H)
