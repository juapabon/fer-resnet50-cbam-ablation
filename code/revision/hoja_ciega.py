"""Punto 3: hoja de revisión ciega para el segundo anotador (120 de los 360 pares, estratificados por muestra y banda).

Genera un HTML autónomo (imágenes incrustadas) que se abre sin conexión, guarda el avance en el navegador y exporta
un CSV con los veredictos. No muestra etiquetas, distancias ni los veredictos del primer anotador; el orden es aleatorio.
Salidas: resultados_frontiers/revision_ciega_120.html y revision_ciega_120_clave.csv (id de la hoja -> id del par)
"""
import base64, io, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

OUT = Path.home() / "TESIS/Articulos Cientifico /resultados_frontiers"
R = pd.read_csv(OUT / "revision_duplicados_muestra.csv", keep_default_na=False)
rng = np.random.RandomState(120)
pick = []
strata = list(R.groupby(["sample", "band"]).indices.values())   # 18 bandas de 20 pares
quota = [7] * 12 + [6] * 6                                         # 12*7 + 6*6 = 120
rng.shuffle(quota)
for idx, k in zip(strata, quota):
    pick += list(rng.choice(idx, k, replace=False))
pick = list(rng.permutation(pick))
key = pd.DataFrame({"hoja_id": range(1, len(pick) + 1), "par_id": R.id.values[pick]})
key.to_csv(OUT / "revision_ciega_120_clave.csv", index=False)

def b64(path):
    im = Image.open(path).convert("L").resize((192, 192), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()

items = [dict(id=int(h), a=b64(R.test_path.values[i]), b=b64(R.ref_path.values[i])) for h, i in zip(key.hoja_id, pick)]
html = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisión ciega de duplicados</title><style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deded8;--acc:#2b6cb0}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--fg:#ecece8;--mut:#a3a39c;--card:#252523;--line:#3a3a37;--acc:#7fb0e8}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.imgs{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}.imgs img{width:192px;height:192px;image-rendering:pixelated;border-radius:6px}
.btns{display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.top{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input{font:inherit;padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.mut{color:var(--mut)}</style></head><body><main>
<h1>Revisión ciega de posibles duplicados (FER2013)</h1>
<p>Cada tarjeta muestra dos imágenes de 48×48 píxeles ampliadas. Indica si proceden de <b>la misma fotografía de origen</b>.
Cuenta como <b>misma</b> aunque haya recorte, cambio de escala, brillo o contraste, marca de agua o versión en espejo.
Si son dos fotografías distintas (aunque sea la misma persona o una sesión parecida), marca <b>distinta</b>.
Usa <b>dudosa</b> solo si no puedes decidir. Trabaja de forma independiente, sin consultar a nadie.</p>
<p class="mut">El avance se guarda en este navegador. Al terminar, pulsa «Exportar CSV» y envía el fichero.</p>
<div class="top"><label>Anotador: <input id="who" placeholder="nombre"></label><span id="cnt"></span>
<button id="exp">Exportar CSV</button></div><div id="list"></div></main><script>
const ITEMS=__ITEMS__;const K="rev_ciega_120";let st={};try{st=JSON.parse(localStorage.getItem(K)||"{}")}catch(e){}
const who=document.getElementById("who");who.value=st.__who||"";who.oninput=()=>{st.__who=who.value;save()};
function save(){try{localStorage.setItem(K,JSON.stringify(st))}catch(e){}cnt()}
function cnt(){document.getElementById("cnt").textContent=ITEMS.filter(i=>st[i.id]).length+" / "+ITEMS.length+" respondidas"}
const L={same:"Misma",different:"Distinta",uncertain:"Dudosa"};const list=document.getElementById("list");
for(const it of ITEMS){const c=document.createElement("div");c.className="card";
c.innerHTML=`<div class="mut">Par ${it.id}</div><div class="imgs"><img src="data:image/png;base64,${it.a}" alt="imagen A"><img src="data:image/png;base64,${it.b}" alt="imagen B"></div><div class="btns"></div>`;
const b=c.querySelector(".btns");for(const v of ["same","different","uncertain"]){const x=document.createElement("button");x.textContent=L[v];
if(st[it.id]===v)x.className="on";x.onclick=()=>{st[it.id]=v;save();b.querySelectorAll("button").forEach(y=>y.className="");x.className="on"};b.appendChild(x)}
list.appendChild(c)}
document.getElementById("exp").onclick=()=>{const miss=ITEMS.filter(i=>!st[i.id]).length;if(miss&&!confirm(miss+" pares sin responder. ¿Exportar igualmente?"))return;
const rows=["hoja_id,verdict,anotador"].concat(ITEMS.map(i=>`${i.id},${st[i.id]||""},"${(who.value||"").replace(/"/g,"'")}"`));
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));a.download="revision_ciega_120_"+(who.value||"anotador").replace(/\\W+/g,"_")+".csv";a.click()};
cnt();</script></body></html>"""
(OUT / "revision_ciega_120.html").write_text(html.replace("__ITEMS__", json.dumps(items)), encoding="utf8")
print("pares:", len(items), "| por muestra:", R.iloc[pick]["sample"].value_counts().to_dict(),
      "| 'same' del primer anotador:", int((R.verdict.values[pick] == "same").sum()))
