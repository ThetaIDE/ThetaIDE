"""Self-contained snapshot timeline, activation heatmaps and intervention maps."""
from __future__ import annotations

import json
from pathlib import Path

from .api import load_snapshot


def write_report(directory):
    directory = Path(directory)
    snapshots = []
    for path in sorted(directory.glob("snapshot_*.json")):
        meta, arrays = load_snapshot(path)
        details = meta["details"]
        entry = {"transitions": meta["transitions"], "updates": meta["optimizer_updates"],
                 "metrics": meta["metrics"], "details": details, "model": meta["model_state_sha256"],
                 "probes": meta["probe_set_sha256"], "seconds": meta["collection_seconds"]}
        entry["heatmaps"] = {row["module"]: arrays[f"layer_{index}"][:128, :32].tolist()
                             for index, row in enumerate(details.get("layers", [])) if f"layer_{index}" in arrays}
        entry["heatmap_metadata"] = {}
        for visual in meta.get("visuals", {}).values():
            if visual["kind"] == "matrix":
                for row in visual["data"].get("layers", []):
                    key = row["id"] if row["id"] in entry["heatmaps"] else f"{visual['label']} / {row['id']}"
                    entry["heatmaps"][key] = arrays[row["array"]][:128, :32].tolist()
                    entry["heatmap_metadata"][key] = row
        network = next((v["data"]["graph"] for v in meta.get("visuals", {}).values()
                        if v["kind"] == "network_graph"), None)
        if network:
            entry["network"] = network
            entry["network_arrays"] = {key: value.tolist() for key, value in arrays.items()}
        snapshots.append(entry)
    payload = json.dumps(snapshots, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    content = r'''<!doctype html><meta charset="utf-8"><title>Training diagnostics</title>
<style>
body{font:15px system-ui;background:#111827;color:#e5e7eb;margin:32px;max-width:1200px}h1{font-size:25px}
select,input,button{padding:8px;background:#253047;color:inherit;border:1px solid #59657a;border-radius:6px}
input{width:65%}canvas,svg{display:block;background:#192235;border-radius:8px;margin:16px 0;max-width:100%}
pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#b7c3d6}table{border-collapse:collapse}td,th{padding:6px 16px;text-align:left;border-bottom:1px solid #374151}
</style><h1>Training diagnostics</h1><p id="caption"></p>
<label>Snapshot <input id="timeline" type="range" min="0" value="0"></label> <button id="play">Play</button>
<p><label>Metric <select id="metric"></select></label></p><canvas id="trend" width="1000" height="220"></canvas>
<div id="layers"><label>Layer <select id="layer"></select></label><p><small id="heatmapLegend">Probe samples × first 32 recorded features.</small></p><canvas id="heatmap" width="1000" height="360"></canvas></div>
<div id="circuits"><h2>Intervention map</h2><p>Edges show measured total effects on behavior. Orange arcs show tested downstream mediation/rescue in sequential targets. These effects do not identify direct synaptic connections.</p><svg id="graph" width="1000" height="500" viewBox="0 0 1000 500"></svg></div>
<div id="network"><h2>Layered network</h2>
<p><select id="networkMode"><option>Weights</option><option>Activations</option><option>Local contributions</option></select>
<label>Probe <select id="networkState"></select></label></p>
<small id="networkLegend"></small><svg id="networkSvg" width="1100" height="700"></svg></div>
<table id="values"></table><details><summary>Method and provenance</summary><pre id="details"></pre></details>
<script>const snapshots=__DATA__;
const $=id=>document.getElementById(id), timeline=$('timeline');timeline.max=Math.max(0,snapshots.length-1);
function options(select,values){const old=select.value;select.replaceChildren();for(const value of values){const o=document.createElement('option');o.value=value;o.textContent=value;select.append(o)}if(values.includes(old))select.value=old}
options($('metric'),[...new Set(snapshots.flatMap(s=>Object.keys(s.metrics)))]);
function trend(){const c=$('trend'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const key=$('metric').value;
const points=snapshots.map(s=>[s.transitions,s.metrics[key]]).filter(p=>Number.isFinite(p[1]));if(!points.length)return;
const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),lo=Math.min(...ys),hi=Math.max(...ys),xmin=Math.min(...xs),xmax=Math.max(...xs);
ctx.strokeStyle='#67e8f9';ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{const x=50+(p[0]-xmin)/(xmax-xmin||1)*900,y=175-(p[1]-lo)/(hi-lo||1)*135;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle='#e5e7eb';ctx.font='13px system-ui';ctx.fillText(key,15,20);ctx.fillText('Environment transitions: '+xmin+' → '+xmax,50,205);ctx.fillText(lo.toPrecision(4)+' → '+hi.toPrecision(4),750,20)}
function draw(){if(!snapshots.length)return;const s=snapshots[Number(timeline.value)];$('caption').textContent='Transitions '+s.transitions+' · optimizer updates '+s.updates+' · collection '+s.seconds.toFixed(3)+'s';
options($('layer'),Object.keys(s.heatmaps));$('layers').hidden=!Object.keys(s.heatmaps).length;const values=s.heatmaps[$('layer').value];
if(values){const c=$('heatmap'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const info=s.heatmap_metadata[$('layer').value]||{},scale=info.value_range?Math.max(...info.value_range.map(Math.abs)):Math.max(1e-9,...values.flat().map(Math.abs));$('heatmapLegend').textContent=(info.value_semantics||'Activation')+'; '+(info.unit||'model units')+'; first 128 probes × first 32 columns; '+(info.value_range?'fixed range '+info.value_range.join(' → '):'per-snapshot scale ±'+scale.toPrecision(4))+'; column IDs '+(info.feature_ids||[]).slice(0,32).join(', ');values.forEach((row,y)=>row.forEach((v,x)=>{const t=Math.min(1,Math.abs(v)/scale);ctx.fillStyle=v<0?'rgb('+Math.round(30+220*t)+',70,90)':'rgb(30,'+Math.round(70+170*t)+','+Math.round(90+150*t)+')';ctx.fillRect(x*c.width/row.length,y*c.height/values.length,c.width/row.length+1,c.height/values.length+1)}))}
const graph=s.details.graph;$('circuits').hidden=!graph;const svg=$('graph');svg.replaceChildren();
if(graph){const ns='http://www.w3.org/2000/svg',units=graph.nodes.filter(n=>n.id!=='behavior'),max=Math.max(1e-9,...graph.edges.map(e=>e.effect));function shape(tag,attrs,text){const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text)el.textContent=text;svg.append(el)}
units.forEach((n,i)=>{const y=25+i*450/Math.max(1,units.length-1);shape('line',{x1:350,y1:y,x2:800,y2:250,stroke:n.selected?'#67e8f9':'#64748b','stroke-width':1+6*(n.effect??n.heldout_effect??0)/max});shape('text',{x:10,y:y+4,fill:n.selected?'#67e8f9':'#cbd5e1','font-size':12},n.id+' · '+(n.effect??n.heldout_effect??0).toPrecision(3));});for(const edge of graph.edges.filter(e=>e.kind==='measured_path_mediation')){const a=units.findIndex(n=>n.id===edge.source),b=units.findIndex(n=>n.id===edge.target);if(a>=0&&b>=0){const y1=25+a*450/Math.max(1,units.length-1),y2=25+b*450/Math.max(1,units.length-1);shape('path',{d:'M 365 '+y1+' Q 610 '+((y1+y2)/2)+' 365 '+y2,fill:'none',stroke:'#fbbf24','stroke-width':2,'stroke-dasharray':'6 3'});const title=document.createElementNS(ns,'title');title.textContent=edge.source+' → '+edge.target+'; mediation '+edge.effect+'; rescue '+edge.rescue_effect;svg.lastChild.append(title)}}shape('circle',{cx:820,cy:250,r:12,fill:'#fbbf24'});shape('text',{x:840,y:255,fill:'#e5e7eb','font-size':12},'behavior')}
drawNetwork(s);
const table=$('values');table.replaceChildren();for(const [key,value] of Object.entries(s.metrics)){const row=document.createElement('tr');for(const text of [key,Number(value).toPrecision(5)]){const cell=document.createElement('td');cell.textContent=text;row.append(cell)}table.append(row)}
$('details').textContent=JSON.stringify({model_state_sha256:s.model,probe_set_sha256:s.probes,...s.details},null,2);trend()}
function drawNetwork(s){$('network').hidden=!s.network;if(!s.network)return;
const g=s.network,a=s.network_arrays,svg=$('networkSvg'),ns='http://www.w3.org/2000/svg';svg.replaceChildren();
options($('networkState'),a[g.sample_ids].map(String));const state=Math.max(0,a[g.sample_ids].map(String).indexOf($('networkState').value)),mode=$('networkMode').value;
function shape(tag,attrs,content,title){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(content)el.textContent=content;if(title){const t=document.createElementNS(ns,'title');t.textContent=title;el.append(t)}svg.append(el);return el}
const positions={},layers=Object.fromEntries(g.layers.map(l=>[l.id,l])),shown=32;
const position=(l,u)=>[80+g.layers.indexOf(l)*210,80+u*24];
for(const l of g.layers)for(let u=0;u<Math.min(shown,l.units);u++)positions[l.id+':'+u]=position(l,u);
const edges=[];for(const c of g.connections){if(c.kind==='normalization'){const x=position(layers[c.source],0)[0],z=position(layers[c.target],0)[0];shape('line',{x1:x,y1:45,x2:z,y2:45,stroke:'#a78bfa','stroke-dasharray':'5 4'},null,c.dependency);continue}
for(const [src,dst] of c.edges){const from=positions[c.source+':'+src],to=positions[c.target+':'+dst];if(!from||!to)continue;const w=c.weights?a[c.weights][dst][src]:null,v=w===null?null:mode==='Local contributions'?a[c.source_values][state][src]*w:w;edges.push({from,to,w,v,c,src,dst})}}
const scale=Math.max(1e-12,...edges.map(e=>Math.abs(e.v||0)));
for(const e of edges)shape('line',{x1:e.from[0],y1:e.from[1],x2:e.to[0],y2:e.to[1],stroke:e.v===null?'#64748b':e.v<0?'#fb7185':'#38bdf8','stroke-width':.5+3*Math.abs(e.v||0)/scale,opacity:.2+.8*Math.abs(e.v||0)/scale,'stroke-dasharray':e.w===null?'5 3':'none'},null,e.c.source+':'+e.src+' → '+e.c.target+':'+e.dst+'; weight '+e.w+'; displayed '+e.v);
for(const l of g.layers){const vals=a[l.activation][state],max=Math.max(1e-12,...vals.map(Math.abs));shape('text',{x:position(l,0)[0]-45,y:20,fill:'#e5e7eb','font-size':11},l.label+' ('+Math.min(shown,l.units)+'/'+l.total_units+')');for(let u=0;u<Math.min(shown,l.units);u++){const [x,y]=position(l,u);shape('circle',{cx:x,cy:y,r:6,fill:mode==='Activations'?(vals[u]<0?'#fb7185':'#38bdf8'):'#334155',opacity:mode==='Activations'?.2+.8*Math.abs(vals[u])/max:1,stroke:'#cbd5e1'},null,l.id+':'+u+'; activation '+vals[u]+'; bias '+(l.bias?a[l.bias][u]:'none')+'; '+l.operation)}}
svg.setAttribute('viewBox','0 0 '+(100+g.layers.length*210)+' '+(120+Math.min(shown,Math.max(...g.layers.map(l=>l.units)))*24));
$('networkLegend').textContent=mode+' · probe ID '+a[g.sample_ids][state]+' · blue positive / red negative, edge magnitude scale '+scale.toPrecision(4)+' (per snapshot); dashed gray = operations, purple = coupled normalization. Display first 32 saved units; '+JSON.stringify(g.coverage)+'. '+g.semantics}
$('networkMode').onchange=draw;$('networkState').onchange=draw;
timeline.oninput=draw;$('layer').onchange=draw;$('metric').onchange=trend;let timer=null;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play'}else{timer=setInterval(()=>{timeline.value=(Number(timeline.value)+1)%snapshots.length;draw()},900);$('play').textContent='Pause'}};draw();
</script>'''
    path = directory / "index.html"
    temporary = directory / ".index.html.tmp"
    temporary.write_text(content.replace("__DATA__", payload), encoding="utf-8")
    temporary.replace(path)
    return path
