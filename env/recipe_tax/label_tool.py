"""Gold labeling tool: local server + single-page UI.  Writes data/gold/gold.jsonl (one line per labeled id).

  cd env && uv run --with pyyaml python -m recipe_tax.label_tool            # http://127.0.0.1:8765
  ... --blind 50                                                              # blind relabel of 50 random labeled ids → gold_relabel.jsonl

Consistency features: picker emits only valid taxonomy paths; convention comments from taxonomy.yaml shown per node;
labeling policies on screen; Jev pre-fill with probabilities; 3-state contains (present / absent / uncertain);
keyboard: Enter = save & next, S = skip, F = flag, / = focus search.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .taxonomy import Taxonomy
from .jev import YES, NO

ROOT = Path(__file__).resolve().parents[2]
CAND = ROOT / "data" / "gold" / "candidates.jsonl"
JEV  = ROOT / "data" / "silver" / "contains_jev_candidates.jsonl"
PREFILL = ROOT / "data" / "gold" / "origin_prefill.jsonl"
GOLD = ROOT / "data" / "gold" / "gold.jsonl"
RELABEL = ROOT / "data" / "gold" / "gold_relabel.jsonl"

def read_jsonl(p: Path) -> list[dict]:
    if not p.exists(): return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def write_jsonl(p: Path, rows: list[dict]):
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)

def node_comments(yaml_path: Path) -> dict[str, str]:
    """name -> trailing comment, for convention notes like 'Hyderabadi filed under Deccan'."""
    out = {}
    for line in open(yaml_path, encoding="utf-8"):
        m = re.match(r"^\s*([^#:_][^:]*?):.*?#\s*(.+)$", line)
        if m: out[m.group(1).strip()] = m.group(2).strip()
    return out

TRAP_CLASS = {"nuts": "tree_nuts", "not_meat": None}   # yaml key -> contains class (None = informational only)

def trap_hits(tax: Taxonomy, ingredients: list[str]) -> dict[str, list[str]]:
    """class -> list of trap keywords found in the ingredient lines (case-insensitive substring)."""
    text = " \n ".join(ingredients).lower()
    out: dict[str, list[str]] = {}
    for key, words in tax.traps.items():
        cls = TRAP_CLASS.get(key, key)
        if cls is None: continue
        hits = [w for w in words if w.lower() in text]
        if hits: out[cls] = hits
    # soy sauce is both gluten and soy; make that explicit even if the yaml lists it under one key
    if "soy sauce" in text or "shoyu" in text:
        out.setdefault("gluten", []); out.setdefault("soy", [])
        for c in ("gluten", "soy"):
            if "soy sauce" not in out[c]: out[c].append("soy sauce")
    if "tamari" in text and "gluten" in out and out["gluten"] == ["soy sauce"] and "soy sauce" not in text:
        out.pop("gluten")
    return out

class State:
    def __init__(self, blind: int = 0):
        self.tax = Taxonomy.load()
        self.comments = node_comments(ROOT / "taxonomy.yaml")
        self.cands = read_jsonl(CAND)
        self.jev = {r["id"]: r for r in read_jsonl(JEV)}
        self.prefill = {r["id"]: r for r in read_jsonl(PREFILL)}
        self.blind = blind
        self.out = RELABEL if blind else GOLD
        self.gold = {r["id"]: r for r in read_jsonl(self.out)}
        if blind:
            labeled = [r["id"] for r in read_jsonl(GOLD)]
            random.seed(7); ids = set(random.sample(labeled, min(blind, len(labeled))))
            self.cands = [c for c in self.cands if c["id"] in ids]

    def payload(self) -> dict:
        nodes = [{"path": list(p), "countries": list(n.countries), "leaf": n.is_leaf,
                  "note": self.comments.get(p[-1], "")} for p, n in self.tax.nodes.items()]
        recs = []
        for c in self.cands:
            j = self.jev.get(c["id"], {})
            recs.append({"id": c["id"], "title": c.get("title"), "source": c.get("source"), "reason": c.get("selection_reason", ""),
                         "ingredients": c.get("ingredients", []), "instructions": (c.get("instructions") or "")[:3000],
                         "existing_tags": c.get("existing_tags", []), "jev": j.get("probs", {}), "traps": trap_hits(self.tax, c.get("ingredients", [])),
                         "prefill": (None if self.blind else self.prefill.get(c["id"])),
                         "gold": None if self.blind else self.gold.get(c["id"])})
        return {"taxonomy_version": self.tax.version, "classes": [c for c in self.tax.contains_classes if c != "incomplete"],
                "nodes": nodes, "records": recs, "blind": bool(self.blind), "out": str(self.out.relative_to(ROOT)), "yes": YES, "no": NO}

    def save(self, rec: dict) -> dict:
        rec["labeled_by"] = "human"; rec["labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec["taxonomy_version"] = self.tax.version
        self.gold[rec["id"]] = rec
        write_jsonl(self.out, list(self.gold.values()))
        return {"ok": True, "n": len(self.gold)}

STATE: State

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/": self._send(200, HTML.encode(), "text/html")
        elif self.path == "/api/state": self._send(200, STATE.payload())
        else: self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path == "/api/save":
            n = int(self.headers.get("Content-Length", 0)); rec = json.loads(self.rfile.read(n))
            self._send(200, STATE.save(rec))
        else: self._send(404, {"error": "not found"})

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Gold labeling</title>
<style>
:root{--bg:#141210;--panel:#1d1a17;--line:#3a352f;--fg:#e9e4da;--mut:#9a9387;--acc:#d9a65a;--ok:#6fbf8a;--warn:#e0a24a;--bad:#d96a5a;--blue:#7aa6d9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
header .prog{color:var(--mut)}header select,header input{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
main{display:grid;grid-template-columns:1fr 1fr;gap:0;height:calc(100vh - 52px)}
section{overflow:auto;padding:16px 20px}section.left{border-right:1px solid var(--line)}
h1{font-size:20px;margin:0 0 4px}.meta{color:var(--mut);font-size:12px;margin-bottom:12px}.meta a{color:var(--blue)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 8px;margin:2px;font-size:12px;color:var(--mut)}
ul.ing{padding-left:18px;margin:6px 0 14px}ul.ing li{margin:2px 0}ul.ing li.hdr{list-style:none;margin-left:-18px;color:var(--acc);font-weight:500}
details summary{cursor:pointer;color:var(--mut)}pre.instr{white-space:pre-wrap;color:#cfc8bc;font:13px/1.5 inherit}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:18px 0 8px}
.policy{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:12.5px;color:#cfc8bc}
.policy b{color:var(--acc)}
.pick{position:relative}.pick input{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-size:14px}
.pick .list{position:absolute;left:0;right:0;top:100%;max-height:300px;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:0 0 8px 8px;z-index:4}
.pick .list div{padding:6px 10px;cursor:pointer;border-bottom:1px solid #2a2622}.pick .list div:hover,.pick .list div.sel{background:#2a2521}
.pick .list small{color:var(--mut)}.pick .list .note{color:var(--warn)}
.chosen{display:flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:6px 0}
.chosen .path{flex:1}.chosen .role{font-size:11px;color:var(--mut);text-transform:uppercase}.chosen button{background:none;border:0;color:var(--mut);cursor:pointer;font-size:16px}
.chosen .note{color:var(--warn);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.cls{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 8px}
.cls .name{font-size:13px}.cls .bar{height:4px;background:#2a2622;border-radius:2px;margin-top:4px;overflow:hidden}.cls .bar i{display:block;height:100%;background:var(--blue)}
.cls .p{font-size:11px;color:var(--mut)}.cls .trap{font-size:11px;color:var(--warn);margin-top:2px}.cls.hit{border-color:var(--warn)}.tri{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.tri button{background:none;border:0;color:var(--mut);padding:3px 7px;cursor:pointer;font-size:12px}.tri button.on-yes{background:var(--ok);color:#0e1a12}.tri button.on-no{background:#2a2622;color:var(--fg)}.tri button.on-unc{background:var(--warn);color:#1a1206}
textarea{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:8px;font:13px inherit;min-height:56px}
.actions{display:flex;gap:8px;margin:16px 0 40px;position:sticky;bottom:0;background:var(--bg);padding:10px 0}
.actions button{padding:9px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--fg);cursor:pointer;font-size:14px}
.actions button.primary{background:var(--acc);color:#1a1206;border-color:var(--acc);font-weight:500}
.actions .kbd{color:var(--mut);font-size:12px;align-self:center}
.flag{color:var(--bad)}.status{color:var(--ok);font-size:12px;margin-left:auto}
.blind{background:var(--bad);color:#fff;border-radius:6px;padding:2px 8px;font-size:12px}
.conf{font-size:11px;border-radius:10px;padding:1px 8px;margin-left:8px;letter-spacing:0;text-transform:lowercase;font-weight:500}
.conf.high{background:var(--ok);color:#0e1a12}.conf.medium{background:var(--warn);color:#1a1206}.conf.low{background:var(--bad);color:#fff}
</style></head><body>
<header>
  <strong>Gold labeling</strong><span id="blind"></span>
  <span class="prog" id="prog"></span>
  <select id="filter"><option value="unlabeled">unlabeled</option><option value="all">all</option><option value="hard">hard</option><option value="ordinary">ordinary</option><option value="flagged">flagged</option><option value="labeled">labeled</option></select>
  <input id="jump" placeholder="jump to # or title" style="width:220px">
  <span class="status" id="status"></span>
</header>
<main>
<section class="left">
  <h1 id="title"></h1>
  <div class="meta"><span id="reason"></span> · <a id="src" target="_blank">source</a> · <span id="idx"></span></div>
  <div id="tags"></div>
  <h2>Ingredients</h2><ul class="ing" id="ing"></ul>
  <details><summary>Method</summary><pre class="instr" id="instr"></pre></details>
  <h2>Policies</h2>
  <div class="policy">
    <b>Origin.</b> Spine = tradition, not country. Primary = the dish's <i>form/technique</i>; influence = borrowed <i>flavor system</i> (max 2). Follow the file's conventions (shown in orange on nodes). Stop at the deepest node the recipe <i>supports</i> — don't guess a region from one ingredient. Pan-national restaurant idiom → national node. No tradition → Unplaced › Foundational (building block) or Contemporary (modern, no parent).<br>
    <b>Contains.</b> Positive facts only. Optional / garnish / "A or B" alternatives <b>count as present</b>. Explicit vegan/vegetarian/X-free qualifiers <b>negate</b> the class. Missing quantities ≠ incomplete. "Serve with naan" is not an ingredient. Can't tell from text → <b>uncertain</b>, not a guess. Trust the ingredient list over the title for origin. For contains, an ingredient named in the <i>title</i> counts even if the list dropped it (Food.com truncation). <b>Soy sauce = gluten + soy</b> unless tamari / gluten-free.
  </div>
</section>
<section class="right">
  <h2>Origin — primary <span id="pfconf"></span></h2>
  <div class="policy" id="pfnote" style="display:none;margin-bottom:8px"></div>
  <div id="primary"></div>
  <div class="pick"><input id="pq" placeholder="search tradition / region  (type: hyderabadi, punjabi, basque, foundational …)"><div class="list" id="plist" hidden></div></div>
  <h2>Origin — influences (optional, max 2)</h2>
  <div id="infl"></div>
  <div class="pick"><input id="iq" placeholder="add an influence"><div class="list" id="ilist" hidden></div></div>
  <h2>Contains <span style="color:var(--mut);font-weight:400;text-transform:none;letter-spacing:0">— Jev pre-fill shown; click to override. yes / no / ?</span></h2>
  <div class="grid" id="classes"></div>
  <h2>Notes (optional)</h2><textarea id="notes" placeholder="why this origin; anything ambiguous"></textarea>
  <div class="actions">
    <button class="primary" id="save">Save &amp; next ⏎</button>
    <button id="skip">Skip (S)</button>
    <button id="flagb">Flag (F)</button>
    <button id="prev">◀ prev</button>
    <span class="kbd">/ = search</span>
  </div>
</section>
</main>
<script>
let S, recs=[], view=[], i=0, cur={}, hist=[];
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
fetch('/api/state').then(r=>r.json()).then(d=>{S=d;recs=d.records;if(d.blind)$('blind').innerHTML='<span class="blind">BLIND RELABEL → '+esc(d.out)+'</span>';applyFilter();});
function applyFilter(){hist=[];const f=$('filter').value;view=recs.map((r,k)=>k).filter(k=>{const r=recs[k];const lab=!!r.gold;const fl=r.gold&&r.gold.flag;
 if(f==='unlabeled')return !lab;if(f==='labeled')return lab;if(f==='flagged')return fl;if(f==='hard')return r.reason.startsWith('hard');if(f==='ordinary')return r.reason.startsWith('ordinary');return true;});
 i=0;show();}
$('filter').onchange=applyFilter;
function goTo(gidx){if(!view.includes(gidx)){$('filter').value='all';view=recs.map((r,k)=>k);} hist.push(view[i]); i=view.indexOf(gidx); show();}
$('jump').onkeydown=e=>{if(e.key!=='Enter')return;const q=$('jump').value.trim().toLowerCase();let k=parseInt(q,10);if(!isNaN(k)&&k>=1&&k<=recs.length){goTo(k-1);}else{const f=recs.findIndex(r=>(r.title||'').toLowerCase().includes(q));if(f>=0)goTo(f);}$('jump').value='';};
function show(){if(!view.length){$('title').textContent='Nothing in this filter.';$('prog').textContent=recs.filter(r=>r.gold).length+' / '+recs.length+' labeled';return;}
 const r=recs[view[i]];$('prog').textContent=recs.filter(r=>r.gold).length+' / '+recs.length+' labeled · #'+(view[i]+1)+' of '+recs.length+' · '+(i+1)+'/'+view.length+' in view';
 $('title').textContent=r.title||'(untitled)';$('reason').textContent=r.reason;$('src').href=r.source||'#';$('idx').textContent='#'+(view[i]+1)+' · '+r.id;
 $('tags').innerHTML=(r.existing_tags||[]).map(t=>'<span class="tag">'+esc(t)+'</span>').join('');
 $('ing').innerHTML=(r.ingredients||[]).map(l=>/:\s*$/.test(l)?'<li class="hdr">'+esc(l)+'</li>':'<li>'+esc(l)+'</li>').join('');
 $('instr').textContent=r.instructions||'';
 const g=r.gold;cur={primary:null,infl:[],contains:{},notes:'',flag:false};
 if(g){const p=(g.labels.origin||[]).find(o=>o.role==='primary');cur.primary=p?p.path:null;cur.infl=(g.labels.origin||[]).filter(o=>o.role==='influence').map(o=>o.path);
   for(const c of S.classes)cur.contains[c]=(g.labels.contains||[]).includes(c)?'yes':((g.labels.contains_uncertain||[]).includes(c)?'unc':'no');cur.notes=g.notes||'';cur.flag=!!g.flag;}
 else{for(const c of S.classes){const p=r.jev[c];cur.contains[c]=p==null?'no':(p>=S.yes?'yes':(p>S.no?'unc':'no'));}
   if(r.prefill){const pf=r.prefill.labels.origin;const pp=pf.find(o=>o.role==='primary');cur.primary=pp?pp.path:null;cur.infl=pf.filter(o=>o.role==='influence').map(o=>o.path);}}
 const pf=r.prefill;$('pfconf').innerHTML=pf?'<span class="conf '+pf.confidence+'">pre-fill: '+pf.confidence+'</span>':'';
 const pn=$('pfnote');if(pf){pn.style.display='';pn.innerHTML='<b>Claude:</b> '+esc(pf.rationale)+(g?' <span style="color:var(--mut)">(your saved label shown below)</span>':'');}else{pn.style.display='none';}
 $('notes').value=cur.notes;renderOrigin();renderClasses(r);$('status').textContent='';}
function nodeLabel(n){return n.path.join(' › ')+(n.countries.length?' <small>['+n.countries.join(', ')+']</small>':'')+(n.note?' <span class="note">— '+esc(n.note)+'</span>':'');}
function findNode(path){return S.nodes.find(n=>n.path.join('/')===path.join('/'));}
function renderOrigin(){$('primary').innerHTML=cur.primary?chosen(cur.primary,'primary',()=>{cur.primary=null;renderOrigin();}):'<div class="chosen" style="color:var(--mut)">none chosen — search below, or type "unknown"</div>';
 $('infl').innerHTML=cur.infl.map((p,k)=>chosen(p,'influence',()=>{cur.infl.splice(k,1);renderOrigin();})).join('');}
function chosen(path,role,onx){const n=findNode(path);const id='x'+Math.random().toString(36).slice(2);setTimeout(()=>{const b=document.getElementById(id);if(b)b.onclick=onx;},0);
 return '<div class="chosen"><span class="role">'+role+'</span><span class="path">'+(n?nodeLabel(n):esc(path.join(' › ')))+'</span><button id="'+id+'" title="remove">✕</button></div>';}
function picker(inp,list,onpick){let sel=0,items=[];
 const render=()=>{const q=inp.value.trim().toLowerCase();if(!q){list.hidden=true;return;}
  items=S.nodes.filter(n=>n.path.join(' ').toLowerCase().includes(q)).slice(0,40);
  if('unknown'.startsWith(q)||q==='unknown')items.unshift({path:['unknown'],countries:[],note:'origin not determinable from text'});
  sel=0;list.innerHTML=items.map((n,k)=>'<div data-k="'+k+'" class="'+(k===sel?'sel':'')+'">'+nodeLabel(n)+'</div>').join('');list.hidden=!items.length;
  [...list.children].forEach(el=>el.onclick=()=>{onpick(items[+el.dataset.k]);inp.value='';list.hidden=true;});};
 inp.oninput=render;inp.onfocus=render;inp.onblur=()=>setTimeout(()=>list.hidden=true,150);
 inp.onkeydown=e=>{if(list.hidden)return;if(e.key==='ArrowDown'){sel=Math.min(items.length-1,sel+1);}else if(e.key==='ArrowUp'){sel=Math.max(0,sel-1);}else if(e.key==='Enter'){e.preventDefault();e.stopPropagation();if(items[sel]){onpick(items[sel]);inp.value='';list.hidden=true;}return;}else if(e.key==='Escape'){list.hidden=true;return;}else return;
  e.preventDefault();[...list.children].forEach((el,k)=>el.classList.toggle('sel',k===sel));};}
picker($('pq'),$('plist'),n=>{cur.primary=n.path;renderOrigin();});
picker($('iq'),$('ilist'),n=>{if(cur.infl.length<2&&n.path[0]!=='unknown'){cur.infl.push(n.path);renderOrigin();}});
function renderClasses(r){$('classes').innerHTML=S.classes.map(c=>{const p=r.jev[c];const v=cur.contains[c];
 const th=(r.traps||{})[c];return '<div class="cls'+(th?' hit':'')+'"><div><div class="name">'+c+'</div><div class="bar"><i style="width:'+Math.round((p||0)*100)+'%"></i></div><div class="p">'+(p==null?'no jev':'jev '+p.toFixed(2))+'</div>'+(th?'<div class="trap">trap: '+esc(th.join(', '))+'</div>':'')+'</div>'+
 '<div class="tri" data-c="'+c+'"><button data-v="yes" class="'+(v==='yes'?'on-yes':'')+'">yes</button><button data-v="no" class="'+(v==='no'?'on-no':'')+'">no</button><button data-v="unc" class="'+(v==='unc'?'on-unc':'')+'">?</button></div></div>';}).join('');
 document.querySelectorAll('.tri button').forEach(b=>b.onclick=()=>{cur.contains[b.parentElement.dataset.c]=b.dataset.v;renderClasses(r);});}
async function save(advance){const r=recs[view[i]];if(!cur.primary){$('status').textContent='pick a primary origin (or "unknown")';$('status').style.color='var(--bad)';$('pq').focus();return;}
 const origin=[{path:cur.primary,role:'primary'},...cur.infl.map(p=>({path:p,role:'influence'}))];
 const contains=S.classes.filter(c=>cur.contains[c]==='yes'),unc=S.classes.filter(c=>cur.contains[c]==='unc');
 const rec={id:r.id,source:r.source,title:r.title,labels:{origin,contains,contains_uncertain:unc},notes:$('notes').value.trim(),flag:cur.flag,selection_reason:r.reason};
 const res=await fetch('/api/save',{method:'POST',body:JSON.stringify(rec)}).then(x=>x.json());
 r.gold=rec;$('status').style.color='var(--ok)';$('status').textContent='saved ('+res.n+')';
 if(advance){hist.push(view[i]);if($('filter').value==='unlabeled'){view.splice(i,1);if(i>=view.length)i=Math.max(0,view.length-1);}else i=Math.min(view.length-1,i+1);show();}}
$('save').onclick=()=>save(true);$('skip').onclick=()=>{hist.push(view[i]);i=Math.min(view.length-1,i+1);show();};
$('prev').onclick=()=>{if(!hist.length){i=Math.max(0,i-1);show();return;}const g=hist.pop();if(!view.includes(g)){$('filter').value='all';view=recs.map((r,k)=>k);}i=view.indexOf(g);show();};
$('flagb').onclick=()=>{cur.flag=!cur.flag;$('flagb').textContent=cur.flag?'Flagged ✓ (F)':'Flag (F)';$('flagb').classList.toggle('flag',cur.flag);};
document.addEventListener('keydown',e=>{const t=e.target.tagName;if(t==='INPUT'||t==='TEXTAREA'){if(e.key==='Enter'&&t==='TEXTAREA'&&!e.shiftKey){e.preventDefault();save(true);}return;}
 if(e.key==='Enter'){e.preventDefault();save(true);}else if(e.key==='s'||e.key==='S'){$('skip').click();}else if(e.key==='f'||e.key==='F'){$('flagb').click();}else if(e.key==='/'){e.preventDefault();$('pq').focus();}});
</script></body></html>"""

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--blind", type=int, default=0); ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    global STATE; STATE = State(blind=a.blind)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"{len(STATE.cands)} candidates · {len(STATE.jev)} with Jev pre-fill · {len(STATE.prefill)} with origin pre-fill · {len(STATE.gold)} already labeled → {STATE.out.relative_to(ROOT)}\n{url}  (Ctrl-C to stop)")
    if not a.no_browser: webbrowser.open(url)
    try: ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    except KeyboardInterrupt: print("\nstopped"); return 0

if __name__ == "__main__":
    sys.exit(main())
