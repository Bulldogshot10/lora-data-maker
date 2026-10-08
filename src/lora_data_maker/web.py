from __future__ import annotations

import json
import mimetypes
import re
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .captions import CaptionConfig, caption_manifest
from .config import load_config, write_default_config
from .dataset import DatasetProject
from .export import export_manifest, safe_component
from .manifest import find_image_by_source_url, register_image
from .pack import pack_dataset
from .paths import ProjectPaths, manifest_path
from .processing import ProcessConfig, YOLOXNanoPersonDetector, YuNetFaceDetector, clean_manifest, process_manifest, validate_image
from .reconcile import reconcile_raw
from .scrapers import SCRAPER_BACKENDS, ScraperError, ScraperManager
from .taggers import build_tagger, clean_tag_manifest, resolve_tagger_preset, tag_manifest

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>LoRA Data Maker</title>
<style>
:root{
  color-scheme:dark;
  --bg:#0b0c0f;
  --panel:#121419;
  --panel-2:#171920;
  --line:#292d36;
  --text:#f4f5f7;
  --muted:#8f96a3;
  --accent:#9b87ff;
  --accent-2:#b4a6ff;
  --ok:#71d7a2;
  --bad:#ff7c87;
  --radius:18px;
}
*{box-sizing:border-box}
html{background:var(--bg)}
body{
  margin:0;
  background:
    radial-gradient(900px 420px at 50% -180px,rgba(155,135,255,.12),transparent 70%),
    var(--bg);
  color:var(--text);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
button,input,select{font:inherit}
button{cursor:pointer}
.shell{max-width:1120px;margin:0 auto;padding:24px 16px 48px}
.header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:12px}
.logo{
  width:34px;height:34px;border-radius:10px;
  display:grid;place-items:center;
  background:linear-gradient(145deg,#b9adff,#7f69ef);
  color:#0a0a0d;font-weight:900;
  box-shadow:0 8px 28px rgba(155,135,255,.18);
}
.title{font-size:17px;font-weight:760;letter-spacing:-.02em}
.subtitle{font-size:12px;color:var(--muted)}
.header-actions{display:flex;gap:8px}
.card{
  background:rgba(18,20,25,.92);
  border:1px solid var(--line);
  border-radius:var(--radius);
  padding:16px;
  margin-bottom:12px;
  backdrop-filter:blur(12px);
}
.card-title{font-size:13px;font-weight:700;margin-bottom:12px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
label{display:block;color:var(--muted);font-size:11px;margin:0 0 6px 2px}
input,select{
  width:100%;height:40px;padding:0 11px;
  color:var(--text);background:#0d0f13;border:1px solid var(--line);border-radius:11px;
  outline:none;
}
input:focus,select:focus{border-color:#6255a5;box-shadow:0 0 0 3px rgba(155,135,255,.08)}
.row{display:flex;align-items:center;gap:10px}
.check{
  display:flex;align-items:center;gap:8px;
  height:40px;padding:0 10px;
  color:var(--text);background:#0d0f13;border:1px solid var(--line);border-radius:11px;
}
.check input{width:auto;height:auto;accent-color:var(--accent)}
button{
  min-height:40px;padding:0 13px;border-radius:11px;
  color:#0b0b0e;background:var(--accent);border:1px solid var(--accent);
  font-weight:720;
}
button:hover{background:var(--accent-2)}
button.secondary{color:var(--text);background:#101217;border-color:var(--line)}
button.secondary:hover{border-color:#444a56;background:#151820}
button:disabled{opacity:.42;cursor:not-allowed}
.save{margin-top:12px}
.range{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center}
.range input{padding:0}
.range-value{min-width:40px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.drop{
  border:1px dashed #3c424e;border-radius:14px;padding:22px;
  text-align:center;background:#0d0f13;transition:.15s;
}
.drop.drag{border-color:var(--accent);background:rgba(155,135,255,.07)}
.drop strong{display:block;font-size:14px;margin-bottom:4px}
.drop p{margin:0 0 13px;color:var(--muted);font-size:12px}
#files{display:none}
.inline-actions{display:flex;justify-content:center;gap:8px;flex-wrap:wrap}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.stat{
  background:#0d0f13;border:1px solid var(--line);border-radius:12px;padding:10px;
}
.stat b{display:block;font-size:18px;line-height:1.1;font-variant-numeric:tabular-nums}
.stat span{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.status{
  display:flex;align-items:flex-start;gap:9px;margin-top:12px;
  color:var(--muted);min-height:20px;
}
.progress-panel{margin-top:12px}
.progress-head{display:flex;justify-content:space-between;gap:12px;font-size:11px;color:var(--muted);margin-bottom:7px}
.progress-value{font-variant-numeric:tabular-nums}
.progress-track{height:8px;overflow:hidden;border-radius:999px;background:#0a0b0e;border:1px solid var(--line)}
.progress-fill{height:100%;width:0;border-radius:999px;background:var(--accent);transition:width .2s ease}
.progress-fill.done{background:var(--ok)}
.progress-fill.bad{background:var(--bad)}
.progress-fill.indeterminate{width:35%;animation:progress-indeterminate 1.15s ease-in-out infinite}
@keyframes progress-indeterminate{
  0%{transform:translateX(-120%)}
  50%{transform:translateX(150%)}
  100%{transform:translateX(300%)}
}
.dot{width:8px;height:8px;border-radius:999px;background:#666;margin-top:5px;flex:0 0 auto}
.dot.running{background:var(--accent);box-shadow:0 0 12px rgba(155,135,255,.7)}
.dot.ok{background:var(--ok)}
.dot.bad{background:var(--bad)}
.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.step{
  min-height:74px;padding:11px;
  text-align:left;color:var(--text);
  background:#0d0f13;border-color:var(--line);
}
.step .n{font-size:10px;color:var(--muted);margin-bottom:4px}
.step .name{font-size:12px;font-weight:700}
.step .hint{display:block;font-size:10px;color:var(--muted);margin-top:2px}
.step.primary{grid-column:span 2;background:linear-gradient(145deg,rgba(155,135,255,.18),rgba(155,135,255,.07));border-color:#4f456e}
.step.clean{grid-column:span 2}
.output{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:12px;background:#0d0f13;border:1px solid var(--line);border-radius:13px;
}
.output-name{font-weight:700}
.output-meta{font-size:11px;color:var(--muted)}
a.download{
  display:inline-flex;align-items:center;justify-content:center;
  min-height:38px;padding:0 12px;border-radius:10px;
  color:#0b0b0e;background:var(--ok);font-weight:760;text-decoration:none;
}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:7px}
.tile{min-width:0}
.tile img{display:block;width:100%;aspect-ratio:1;object-fit:cover;border-radius:11px;background:#08090c;border:1px solid var(--line)}
.tile .meta{padding:5px 2px 0}
.tile .path{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:10px}
.tile .tag{font-size:9px;color:var(--muted)}
.empty{padding:22px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:13px}
.note{color:var(--muted);font-size:11px;margin:10px 2px 0}
.danger{color:var(--bad)}
@media(max-width:820px){.stats{grid-template-columns:repeat(3,1fr)}.pipeline{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){
  .shell{padding:16px 12px 34px}
  .header{align-items:flex-start}
  .grid-2,.grid-3{grid-template-columns:1fr}
  .stats{grid-template-columns:repeat(2,1fr)}
  .pipeline{grid-template-columns:1fr 1fr}
  .step.primary,.step.clean{grid-column:span 2}
  .output{align-items:flex-start;flex-direction:column}
  .header-actions button{padding:0 10px}
}
</style>
</head>
<body>
<div class="shell">
  <header class="header">
    <div class="brand">
      <div class="logo">L</div>
      <div>
        <div class="title">LoRA Data Maker</div>
        <div class="subtitle">local dataset pipeline · localhost</div>
      </div>
    </div>
    <div class="header-actions">
      <button class="secondary" onclick="newProject()">new dataset</button>
    </div>
  </header>

  <section class="card">
    <div class="card-title">project</div>
    <div class="grid-2">
      <div><label>name</label><input id="name"></div>
      <div><label>subject</label><input id="subject"></div>
      <div><label>trigger</label><input id="trigger"></div>
      <div><label>tagger</label><select id="preset"><option value="sweetspot">wd14 · sweetspot</option><option value="potato">wd14 · potato</option><option value="joytag">joytag</option></select></div>
    </div>
    <div class="grid-2" style="margin-top:12px">
      <div>
        <label>resize</label>
        <select id="mode"><option value="cover">cover</option><option value="fit">fit</option><option value="stretch">stretch</option></select>
      </div>
      <div class="row">
        <div style="flex:1"><label>width</label><input id="width" type="number" min="1"></div>
        <div style="flex:1"><label>height</label><input id="height" type="number" min="1"></div>
      </div>
    </div>
    <div class="grid-3" style="margin-top:12px">
      <div><label>jpeg quality</label><div class="range"><input id="quality" type="range" min="60" max="100"><span class="range-value" id="qv"></span></div></div>
      <div><label>face crop</label><label class="check"><input id="face_crop" type="checkbox"> enable</label></div>
      <div><label>face padding</label><input id="face_padding" type="number" min="0.1" step="0.05"></div>
    </div>
    <div class="grid-3" style="margin-top:12px">
      <div><label>person quality gate</label><label class="check"><input id="person_check" type="checkbox"> enable</label></div>
      <div><label>max people</label><input id="max_persons" type="number" min="1" step="1"></div>
      <div><label>person confidence</label><input id="person_threshold" type="number" min="0.01" max="1" step="0.01"></div>
    </div>
    <div class="grid-2" style="margin-top:12px">
      <div><label>min subject area</label><input id="min_subject_area" type="number" min="0" max="1" step="0.01"></div>
      <div><label>min subject visibility</label><input id="min_subject_visibility" type="number" min="0" max="1" step="0.01"></div>
    </div>
    <div class="row save"><button onclick="saveSettings()">save settings</button><span class="note">settings are written back to config.toml</span></div>
  </section>


  <section class="card">
    <div class="card-title">scraper</div>
    <div class="grid-2">
      <div>
        <label>backend</label>
        <select id="scraper_backend">
          <option value="duckduckgo">duckduckgo</option>
          <option value="google">google</option>
          <option value="chromium">chromium</option>
          <option value="playwright-bing">playwright · bing</option>
          <option value="playwright-google">playwright · google</option>
          <option value="direct">direct URLs</option>
        </select>
      </div>
      <div>
        <label>count</label>
        <input id="scrape_count" type="number" min="1" step="1">
      </div>
      <div style="grid-column:span 2">
        <label>query</label>
        <input id="scrape_query" placeholder="julia russian woman portrait">
      </div>
    </div>
    <div class="row save">
      <button onclick="run('scrape')">scrape images</button>
      <span class="note">downloads land in <code>raw/</code> and get added to the manifest.</span>
    </div>
  </section>

  <section class="card">
    <div class="card-title">input</div>
    <div class="drop" id="drop">
      <input id="files" type="file" accept="image/*" multiple>
      <strong>drop images here</strong>
      <p>or pick files from your gallery / file manager</p>
      <div class="inline-actions">
        <button class="secondary" onclick="document.getElementById('files').click()">choose images</button>
        <button onclick="upload()">upload selected</button>
      </div>
    </div>
  </section>

  <section class="card">
    <div class="card-title">dataset status</div>
    <div class="stats" id="stats"></div>
    <div class="status"><span class="dot" id="dot"></span><span id="status">ready</span></div>
    <div class="progress-panel">
      <div class="progress-head"><span id="progress-label">idle</span><span class="progress-value" id="progress-value">0%</span></div>
      <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
  </section>

  <section class="card">
    <div class="card-title">pipeline</div>
    <div class="pipeline">
      <button class="step primary" onclick="run('prepare')"><div class="n">ALL</div><div class="name">prepare dataset</div><span class="hint">validate → process → tag → clean → caption → export → ZIP</span></button>
      <button class="step" onclick="run('validate')"><div class="n">01</div><div class="name">validate</div><span class="hint">scan raw images</span></button>
      <button class="step" onclick="run('process')"><div class="n">02</div><div class="name">stage / resize</div><span class="hint">make training images</span></button>
      <button class="step" onclick="run('tag')"><div class="n">03</div><div class="name">tag</div><span class="hint">automatic image tags</span></button>
      <button class="step" onclick="run('clean-tags')"><div class="n">04</div><div class="name">clean tags</div><span class="hint">apply tag rules</span></button>
      <button class="step" onclick="run('caption')"><div class="n">05</div><div class="name">caption</div><span class="hint">write .txt captions</span></button>
      <button class="step" onclick="run('export')"><div class="n">06</div><div class="name">export</div><span class="hint">Kohya-ready folder</span></button>
      <button class="step" onclick="run('pack')"><div class="n">07</div><div class="name">make ZIP</div><span class="hint">final archive</span></button>
      <button class="step clean" onclick="run('clean')"><div class="n">OPTIONAL</div><div class="name">image clean / dedup</div><span class="hint">manual quality + duplicate pass</span></button>
    </div>
    <div id="download" style="margin-top:12px"></div>
  </section>

  <section class="card">
    <div class="card-title">gallery</div>
    <div class="gallery" id="gallery"></div>
  </section>
</div>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));
async function api(url,opt={}){
  const r=await fetch(url,opt);
  const d=await r.json().catch(()=>({detail:r.statusText}));
  if(!r.ok)throw Error(d.detail||"request failed");
  return d;
}
function applySettings(s){
  $("name").value=s.project.name;
  $("subject").value=s.project.subject;
  $("trigger").value=s.project.trigger;
  $("preset").value=s.tagger.preset;
  $("mode").value=s.processing.mode;
  $("width").value=s.processing.width;
  $("height").value=s.processing.height;
  $("quality").value=s.processing.quality;
  $("qv").textContent=s.processing.quality;
  $("scraper_backend").value=s.scraper.backend;
  $("scrape_query").value=s.scraper.query;
  $("scrape_count").value=s.scraper.count;
  $("face_crop").checked=!!s.processing.face_crop;
  $("face_padding").value=s.processing.face_padding;
  $("person_check").checked=!!s.quality.person_check;
  $("max_persons").value=s.quality.max_persons;
  $("person_threshold").value=s.quality.person_threshold;
  $("min_subject_area").value=s.quality.min_subject_area;
  $("min_subject_visibility").value=s.quality.min_subject_visibility;
}
function renderStats(counts){
  const keys=["discovered","validated","processed","tagged","captioned","exported"];
  $("stats").innerHTML=keys.map(k=>'<div class="stat"><b>'+esc(counts[k]??0)+'</b><span>'+esc(k)+'</span></div>').join("");
}
function renderProgress(progress,status){
  const p=progress||{};
  const current=Number(p.current||0);
  const total=Number(p.total||0);
  const percent=total>0 ? Math.max(0,Math.min(100,Number(p.percent ?? (current/total*100)))) : 0;
  const fill=$("progress-fill");
  const label=$("progress-label");
  const value=$("progress-value");
  label.textContent=p.label||status||"idle";
  value.textContent=total>0 ? Math.round(percent)+"%" : (status==="running" ? "…" : "");
  fill.className="progress-fill";
  if(status==="done") fill.classList.add("done");
  if(status==="failed") fill.classList.add("bad");
  if(status==="running" && total<=0) fill.classList.add("indeterminate");
  fill.style.width=total>0 ? percent+"%" : (status==="running" ? "35%" : "0%");
}
function renderGallery(images){
  if(!images.length){
    $("gallery").innerHTML='<div class="empty">no images yet. upload some raw images above.</div>';
    return;
  }
  $("gallery").innerHTML=images.map(x=>{
    const src="/media/"+encodeURIComponent(x.path);
    return '<div class="tile"><img loading="lazy" src="'+src+'" alt="'+esc(x.path)+'"><div class="meta"><div class="path" title="'+esc(x.path)+'">'+esc(x.path)+'</div><div class="tag">'+esc(x.status)+'</div></div></div>';
  }).join("");
}
function renderDownload(name){
  $("download").innerHTML=name
    ? '<div class="output"><div><div class="output-name">'+esc(name)+'</div><div class="output-meta">final dataset archive</div></div><a class="download" href="/api/download">download ZIP</a></div>'
    : "";
}
async function refresh(){
  try{
    const s=await api("/api/state");
    applySettings(s);
    renderStats(s.counts);
    renderGallery(s.images);
    renderDownload(s.archive);
  }catch(e){setStatus(e.message,"bad")}
}
async function saveSettings(){
  try{
    await api("/api/settings",{
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({
        name:$("name").value.trim(),
        subject:$("subject").value.trim(),
        trigger:$("trigger").value.trim(),
        preset:$("preset").value,
        scraper_backend:$("scraper_backend").value,
        scrape_query:$("scrape_query").value.trim(),
        scrape_count:+$("scrape_count").value,
        mode:$("mode").value,
        width:+$("width").value,
        height:+$("height").value,
        quality:+$("quality").value,
        face_crop:$("face_crop").checked,
        face_padding:+$("face_padding").value,
        person_check:$("person_check").checked,
        max_persons:+$("max_persons").value,
        person_threshold:+$("person_threshold").value,
        min_subject_area:+$("min_subject_area").value,
        min_subject_visibility:+$("min_subject_visibility").value
      })
    });
    setStatus("settings saved","ok");
    refresh();
    return true;
  }catch(e){setStatus(e.message,"bad");return false}
}
async function upload(){
  const fs=$("files").files;
  if(!fs.length)return setStatus("select images first","bad");
  const f=new FormData();
  for(const x of fs)f.append("files",x);
  try{
    setStatus("uploading "+fs.length+" images...","running");
    const d=await api("/api/upload",{method:"POST",body:f});
    setStatus("uploaded "+d.uploaded+", skipped "+d.skipped,"ok");
    $("files").value="";
    refresh();
  }catch(e){setStatus(e.message,"bad")}
}
async function run(stage){
  try{
    if(!(await saveSettings())) return;
    const d=await api("/api/run/"+stage,{method:"POST"});
    setStatus(d.message,"running");
    renderProgress(d.progress,"running");
    poll(d.job_id);
  }catch(e){setStatus(e.message,"bad")}
}
async function poll(id){
  try{
    const d=await api("/api/jobs/"+id);
    const cls=d.status==="failed"?"bad":d.status==="done"?"ok":"running";
    setStatus(d.message||d.status,cls);
    renderProgress(d.progress,d.status);
    if(d.status==="running"){setTimeout(()=>poll(id),900)}
    else{refresh()}
  }catch(e){setStatus(e.message,"bad")}
}
function setStatus(t,c=""){
  $("status").textContent=t;
  $("dot").className="dot "+c;
}
async function newProject(){
  const name=prompt("new dataset folder name");
  if(!name)return;
  const subject=prompt("subject",name)||name;
  const trigger=prompt("trigger",subject)||subject;
  try{
    const d=await api("/api/project/new",{
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({name,subject,trigger})
    });
    setStatus("switched to "+d.name,"ok");
    await refresh();
  }catch(e){setStatus(e.message,"bad")}
}
const drop=$("drop");
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("drag")}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("drag")}));
drop.addEventListener("drop",e=>{$("files").files=e.dataTransfer.files});
$("quality").oninput=()=>{$("qv").textContent=$("quality").value};
refresh();
</script>
</body>
</html>
"""

class JobStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}

    def start(self, label: str, fn) -> str:
        with self.lock:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise RuntimeError("another job is already running")
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "status": "running",
                "message": label,
                "progress": {
                    "current": 0,
                    "total": 0,
                    "percent": 0,
                    "label": label,
                },
            }
        threading.Thread(target=self._worker, args=(job_id, fn), daemon=True).start()
        return job_id

    def update_progress(self, job_id: str, current: int, total: int, label: str) -> None:
        current = max(0, int(current))
        total = max(0, int(total))
        percent = round((current / total) * 100, 1) if total else 0
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            job["message"] = label
            job["progress"] = {
                "current": current,
                "total": total,
                "percent": percent,
                "label": label,
            }

    def _worker(self, job_id: str, fn) -> None:
        def progress(current: int, total: int, label: str) -> None:
            self.update_progress(job_id, current, total, label)

        try:
            result = fn(progress)
            with self.lock:
                self.jobs[job_id].update({"status": "done", "message": result or "done"})
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update({"status": "failed", "message": str(exc)})

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.jobs.get(job_id, {"status": "missing", "message": "job not found"}))


def create_app(root: Path | str = ".") -> FastAPI:
    app = FastAPI(title="LoRA Data Maker WebUI")
    project_root = Path(root).expanduser().resolve()
    jobs = JobStore()

    def paths() -> ProjectPaths:
        config = load_config(project_root / "config.toml")
        return ProjectPaths.from_root(project_root, config.get("paths", {}))

    def project() -> DatasetProject:
        return DatasetProject.open(paths())

    def set_config(section: str, values: dict[str, Any]) -> None:
        config_path = project_root / "config.toml"
        lines = config_path.read_text(encoding="utf-8").splitlines()
        start = next((i for i, line in enumerate(lines) if line.strip() == "[" + section + "]"), None)
        if start is None:
            lines.extend(["", "[" + section + "]"])
            start = len(lines) - 1
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if lines[i].strip().startswith("[") and lines[i].strip().endswith("]"):
                end = i
                break
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (int, float)):
                rendered = str(value)
            else:
                rendered = json.dumps(str(value))
            pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=")
            found = next((i for i in range(start + 1, end) if pattern.match(lines[i])), None)
            if found is None:
                lines.insert(end, key + " = " + rendered)
                end += 1
            else:
                lines[found] = key + " = " + rendered
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/state")
    def state():
        p = project()
        cfg = load_config(project_root / "config.toml")
        processing = cfg.get("processing", {})
        tagger = cfg.get("tagger", {})
        return {
            "project": {"name": p.manifest.project_name, "subject": p.manifest.subject, "trigger": p.manifest.trigger},
            "processing": {"mode": str(processing.get("resize_mode", "cover")), "width": int(processing.get("target_width", 512)), "height": int(processing.get("target_height", 512)), "quality": int(processing.get("jpeg_quality", 95)), "face_crop": bool(processing.get("face_crop", False)), "face_padding": float(processing.get("face_padding", 1.75))},
            "tagger": {"backend": str(tagger.get("backend", "wd14")), "preset": str(tagger.get("preset", "sweetspot"))},
            "scraper": {
                "backend": str(cfg.get("scraper", {}).get("backend", "duckduckgo")),
                "query": str(cfg.get("scraper", {}).get("query", p.manifest.subject)),
                "count": int(cfg.get("scraper", {}).get("count", 300)),
                "max_scrolls": int(cfg.get("scraper", {}).get("max_scrolls", 30)),
            },
            "quality": {
                "person_check": bool(cfg.get("quality", {}).get("person_check", True)),
                "person_threshold": float(cfg.get("quality", {}).get("person_threshold", 0.35)),
                "max_persons": int(cfg.get("quality", {}).get("max_persons", 1)),
                "min_subject_area": float(cfg.get("quality", {}).get("min_subject_area", 0.12)),
                "min_subject_visibility": float(cfg.get("quality", {}).get("min_subject_visibility", 0.35)),
            },
            "counts": p.counts(),
            "images": [{"id": x.image_id, "path": x.processed_path or x.path, "status": x.status} for x in sorted(p.manifest.images.values(), key=lambda x: x.path) if x.status != "rejected"],
            "archive": Path(str(p.manifest.source.get("last_pack_archive", ""))).name if p.manifest.source.get("last_pack_archive") else None,
        }

    @app.post("/api/settings")
    def settings(payload: dict[str, Any]):
        name, subject = str(payload.get("name", "")).strip(), str(payload.get("subject", "")).strip()
        trigger = str(payload.get("trigger", "")).strip() or subject
        mode, preset = str(payload.get("mode", "cover")).lower(), str(payload.get("preset", "sweetspot"))
        current_config = load_config(project_root / "config.toml")
        current_scraper = current_config.get("scraper", {})
        if not isinstance(current_scraper, dict):
            current_scraper = {}
        scraper_backend = str(
            payload.get("scraper_backend", current_scraper.get("backend", "duckduckgo"))
        ).lower()
        scrape_query = str(
            payload.get("scrape_query", current_scraper.get("query", subject))
        ).strip() or subject
        scrape_count = int(
            payload.get("scrape_count", current_scraper.get("count", 300))
        )
        try:
            resolved_preset = resolve_tagger_preset(preset)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        width, height, quality = int(payload.get("width", 512)), int(payload.get("height", 512)), int(payload.get("quality", 95))
        face_crop = bool(payload.get("face_crop", False))
        face_padding = float(payload.get("face_padding", 1.75))
        person_check = bool(payload.get("person_check", True))
        max_persons = int(payload.get("max_persons", 1))
        person_threshold = float(payload.get("person_threshold", 0.35))
        min_subject_area = float(payload.get("min_subject_area", 0.12))
        min_subject_visibility = float(payload.get("min_subject_visibility", 0.35))
        if (
            not name
            or not subject
            or mode not in {"cover", "fit", "stretch"}
            or width < 1
            or height < 1
            or scraper_backend not in SCRAPER_BACKENDS
            or not scrape_query
            or scrape_count < 1
            or not 1 <= quality <= 100
            or face_padding <= 0
            or max_persons < 1
            or not 0 < person_threshold <= 1
            or not 0 <= min_subject_area <= 1
            or not 0 <= min_subject_visibility <= 1
        ):
            raise HTTPException(400, "invalid settings")
        set_config("project", {"name": name, "subject": subject, "trigger": trigger})
        set_config("processing", {"resize_mode": mode, "target_width": width, "target_height": height, "jpeg_quality": quality, "face_crop": face_crop, "face_padding": face_padding})
        set_config("tagger", {"backend": "joytag" if resolved_preset.kind == "joytag" else "wd14", "preset": preset})
        set_config(
            "scraper",
            {
                "backend": scraper_backend,
                "query": scrape_query,
                "count": scrape_count,
            },
        )
        set_config(
            "quality",
            {
                "person_check": person_check,
                "person_threshold": person_threshold,
                "max_persons": max_persons,
                "min_subject_area": min_subject_area,
                "min_subject_visibility": min_subject_visibility,
            },
        )
        p = project()
        p.manifest.project_name, p.manifest.subject, p.manifest.trigger = name, subject, trigger
        p.save()
        return {"ok": True}

    @app.post("/api/upload")
    async def upload(files: Annotated[list[UploadFile], File()]):
        p, ps = project(), paths()
        ps.raw.mkdir(parents=True, exist_ok=True)
        allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        uploaded = skipped = 0
        existing = {x.path for x in p.manifest.images.values()}
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in allowed:
                skipped += 1
                continue
            base = safe_component(Path(file.filename or "image").stem) or "image"
            target = ps.raw / (base + suffix)
            counter = 1
            while target.exists():
                target = ps.raw / (base + "_" + str(counter) + suffix)
                counter += 1
            with target.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    out.write(chunk)
            rel = manifest_path(project_root, target)
            if rel in existing:
                target.unlink(missing_ok=True); skipped += 1; continue
            register_image(p.manifest, path=rel, status="discovered")
            existing.add(rel); uploaded += 1
        p.save()
        return {"uploaded": uploaded, "skipped": skipped}

    def make_job(stage: str):
        def run_stage(progress: Callable[[int, int, str], None]) -> str:
            p, ps, cfg = project(), paths(), load_config(project_root / "config.toml")
            if stage == "scrape":
                sc = cfg.get("scraper", {})
                if not isinstance(sc, dict):
                    raise RuntimeError("config section [scraper] must be a table")
                backend = str(sc.get("backend", "duckduckgo"))
                query = str(sc.get("query", p.manifest.subject)).strip()
                count = int(sc.get("count", 300))
                if backend not in SCRAPER_BACKENDS:
                    raise RuntimeError("unknown scraper backend: " + backend)
                if not query:
                    raise RuntimeError("scrape query is required")
                if count < 1:
                    raise RuntimeError("scrape count must be greater than zero")

                timeout = float(sc.get("timeout", 20))
                retries = int(sc.get("retries", 3))
                delay = float(sc.get("delay", 1.0))
                max_scrolls = int(sc.get("max_scrolls", 30))
                manager = ScraperManager(
                    backend=backend,
                    output_dir=ps.raw,
                    cache_dir=ps.cache,
                    timeout=timeout,
                    retries=retries,
                    delay=delay,
                    max_scrolls=max_scrolls,
                )
                known_urls = {
                    record.source_url
                    for record in p.manifest.images.values()
                    if record.source_url
                }

                progress(0, count, f"searching with {backend}...")
                downloads, failures = manager.run(
                    query,
                    count=count,
                    known_urls=known_urls,
                    progress=progress,
                )
                search_error = manager.last_search_error

                added_records = 0
                for result in downloads:
                    relative_path = manifest_path(project_root, result.path)
                    record = find_image_by_source_url(p.manifest, result.url)
                    if record is None:
                        record = register_image(
                            p.manifest,
                            path=relative_path,
                            source_url=result.url,
                            source_query=query,
                            status="downloaded",
                        )
                        added_records += 1
                    record.sha256 = result.sha256
                    record.file_size = result.file_size
                    record.mime_type = result.content_type

                p.manifest.source.update(
                    {
                        "backend": backend,
                        "query": query,
                        "last_scrape_requested": count,
                        "last_scrape_downloaded": len(downloads),
                        "last_scrape_added_records": added_records,
                        "last_scrape_failed": len(failures),
                    }
                )
                p.save()

                if len(downloads) >= count:
                    return f"scrape complete: downloaded {len(downloads)}/{count}"
                if search_error and not downloads:
                    raise RuntimeError(
                        f"scrape failed: downloaded 0/{count}; source error: {search_error}"
                    )
                if not downloads and known_urls and not failures:
                    return "no new images downloaded"
                if downloads:
                    if search_error:
                        return (
                            f"scrape complete with source error: downloaded {len(downloads)}/{count}; "
                            f"skipped {len(failures)} failed candidates; source: {search_error}"
                        )
                    return (
                        f"scrape complete with skips: downloaded {len(downloads)}/{count}; "
                        f"skipped {len(failures)} failed candidates"
                    )
                raise RuntimeError(
                    f"scrape failed: downloaded 0/{count}; failed candidates: {len(failures)}"
                )

            if stage == "validate":
                v = cfg.get("validation", {})
                reconcile_raw(p.manifest, root=project_root, raw_dir=ps.raw)
                existing = {x.path: x for x in p.manifest.images.values()}
                from .images import iter_image_files
                valid = rejected = skipped = 0
                for image_path in iter_image_files(ps.raw):
                    rel = manifest_path(project_root, image_path)
                    record = existing.get(rel) or register_image(p.manifest, path=rel, status="discovered")
                    if record.status not in {"discovered", "downloaded"}:
                        skipped += 1; continue
                    result = validate_image(image_path, min_width=int(v.get("min_width", 1)), min_height=int(v.get("min_height", 1)), max_width=int(v.get("max_width", 0)) or None, max_height=int(v.get("max_height", 0)) or None, min_file_size=int(v.get("min_file_size", 1)), max_file_size=int(v.get("max_file_size", 0)) or None)
                    record.width, record.height, record.file_size, record.mime_type = result.width, result.height, result.file_size, result.mime_type
                    record.metadata.update(result.metadata)
                    if result.valid: record.set_status("validated"); valid += 1
                    else: record.set_status("rejected", reason=result.reason or "validation failed"); rejected += 1
                p.save()
                return "validated " + str(valid) + ", rejected " + str(rejected) + ", skipped " + str(skipped)
            if stage == "clean":
                c, q, proc = cfg.get("dedup", {}), cfg.get("quality", {}), cfg.get("processing", {})
                person_check = bool(q.get("person_check", True))
                person_detector = None
                if person_check:
                    person_detector = YOLOXNanoPersonDetector(
                        cache_dir=ps.cache / "person",
                        confidence_threshold=float(q.get("person_threshold", 0.35)),
                    )
                s = clean_manifest(
                    p.manifest,
                    root=project_root,
                    exact=bool(c.get("exact", True)),
                    perceptual=bool(c.get("perceptual", True)),
                    perceptual_distance=int(c.get("perceptual_distance", 6)),
                    min_width=int(q.get("min_width", 512)),
                    min_height=int(q.get("min_height", 512)),
                    blur_threshold=float(q.get("blur_threshold", proc.get("blur_threshold", 0.0))),
                    max_aspect_ratio=float(q.get("max_aspect_ratio", 0.0)),
                    person_detector=person_detector,
                    max_persons=int(q.get("max_persons", 1)),
                    min_subject_area=float(q.get("min_subject_area", 0.12)),
                    min_subject_visibility=float(q.get("min_subject_visibility", 0.35)),
                )
                p.save()
                return "cleaned " + str(s.kept) + "; exact dup " + str(s.exact_duplicates) + "; perceptual dup " + str(s.perceptual_duplicates) + "; quality rejected " + str(s.quality_rejected)
            if stage == "process":
                pc = cfg.get("processing", {})
                face_crop = bool(pc.get("face_crop", False))
                face_padding = float(pc.get("face_padding", 1.75))
                face_detector = None
                if face_crop:
                    face_detector = YuNetFaceDetector(cache_dir=ps.cache / "face")
                settings = ProcessConfig(target_width=int(pc.get("target_width", 512)), target_height=int(pc.get("target_height", 512)), resize_mode=str(pc.get("resize_mode", "cover")), jpeg_quality=int(pc.get("jpeg_quality", 95)), face_crop=face_crop, face_padding=face_padding)
                done, failed, skipped = process_manifest(p.manifest, root=project_root, output_dir=ps.work, config=settings, face_detector=face_detector)
                p.save()
                return "staged " + str(done) + ", failed " + str(failed) + ", skipped " + str(skipped)
            if stage == "tag":
                tc = cfg.get("tagger", {})
                selected = resolve_tagger_preset(str(tc.get("preset", "sweetspot")))
                cache_root = (
                    Path.home()
                    / ".cache"
                    / "lora-data-maker"
                    / "taggers"
                    / ("joytag" if selected.kind == "joytag" else "wd14")
                )
                tagger = build_tagger(
                    tc,
                    cache_root=cache_root,
                    preset=selected.name,
                )
                s = tag_manifest(p.manifest, root=project_root, tagger=tagger)
                p.save()
                return "tagged " + str(s.tagged) + ", skipped " + str(s.skipped)
            if stage == "clean-tags":
                tc = cfg.get("tag_rules", {})
                s = clean_tag_manifest(p.manifest, remove_tags=tc.get("remove", []))
                p.save()
                return "cleaned tags for " + str(s.cleaned)
            if stage == "caption":
                cc = cfg.get("caption", {})
                settings = CaptionConfig(separator=str(cc.get("separator", ", ")), subject_position=str(cc.get("subject_position", "first")), trigger_from_subject=bool(cc.get("trigger_from_subject", True)))
                s = caption_manifest(p.manifest, root=project_root, output_dir=ps.work / "captions", config=settings)
                p.save()
                return "captioned " + str(s.captioned)
            if stage == "export":
                ec = cfg.get("export", {})
                s = export_manifest(p.manifest, root=project_root, output_dir=ps.output, repeats=int(ec.get("repeats", 1)), folder_name=str(ec.get("folder_name", "") or "") or None)
                p.manifest.source["last_export_dir"] = manifest_path(project_root, s.output_dir); p.save()
                if s.failed: raise RuntimeError("export failed for " + str(s.failed) + " images")
                return "exported " + str(s.exported)
            if stage == "pack":
                last = p.manifest.source.get("last_export_dir")
                if not isinstance(last, str) or not last: raise RuntimeError("export the dataset first")
                archive = ps.output / (safe_component(p.manifest.project_name) + ".zip")
                s = pack_dataset(root=project_root, dataset_dir=project_root / last, archive_path=archive, manifest_path=p.manifest_path)
                p.manifest.source.update({"last_pack_archive": manifest_path(project_root, archive), "last_pack_sha256": s.sha256}); p.save()
                return "ZIP ready: " + archive.name
            if stage == "prepare":
                for part in ("validate", "process", "tag", "clean-tags", "caption", "export", "pack"):
                    make_job(part)(progress)
                return "dataset ready"
            raise RuntimeError("unknown stage: " + stage)
        return run_stage

    @app.post("/api/run/{stage}")
    def run(stage: str):
        allowed = {"scrape", "prepare", "validate", "clean", "process", "tag", "clean-tags", "caption", "export", "pack"}
        if stage not in allowed: raise HTTPException(404, "unknown stage")
        try: job_id = jobs.start("running " + stage + "...", make_job(stage))
        except RuntimeError as exc: raise HTTPException(409, str(exc))
        return {"job_id": job_id, "message": "running " + stage + "..."}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        data = jobs.get(job_id)
        if data.get("status") == "done" and project().manifest.source.get("last_pack_archive"):
            data["download"] = "/api/download"
        return data

    @app.get("/api/download")
    def download():
        rel = project().manifest.source.get("last_pack_archive")
        if not isinstance(rel, str) or not rel: raise HTTPException(404, "no ZIP available")
        target = (project_root / rel).resolve()
        try: target.relative_to(project_root)
        except ValueError: raise HTTPException(400, "invalid archive path")
        if not target.is_file(): raise HTTPException(404, "ZIP not found")
        return FileResponse(target, filename=target.name, media_type="application/zip")

    @app.get("/media/{relative:path}")
    def media(relative: str):
        target = (project_root / relative).resolve()
        try: target.relative_to(project_root)
        except ValueError: raise HTTPException(403, "invalid media path")
        if not target.is_file(): raise HTTPException(404, "image not found")
        return FileResponse(target, media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    @app.post("/api/project/new")
    def new_project(payload: dict[str, Any]):
        nonlocal project_root
        name = safe_component(str(payload.get("name", "")).strip())
        subject = str(payload.get("subject", "")).strip() or name
        trigger = str(payload.get("trigger", "")).strip() or subject
        if not name: raise HTTPException(400, "project name required")
        target = project_root.parent / name
        if target.exists(): raise HTTPException(409, "project already exists")
        target.mkdir(parents=True)
        new_paths = ProjectPaths.from_root(target); new_paths.create_directories()
        write_default_config(new_paths.config, name=name, subject=subject, trigger=trigger)
        DatasetProject.create(new_paths)
        project_root = target.resolve()
        return {"ok": True, "path": target.as_posix(), "name": name}

    return app


def serve(root: Path | str = ".", host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn
    uvicorn.run(create_app(root), host=host, port=port, log_level="info")
