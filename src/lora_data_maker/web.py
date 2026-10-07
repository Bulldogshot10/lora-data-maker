from __future__ import annotations

import json
import mimetypes
import re
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .captions import CaptionConfig, caption_manifest
from .config import load_config, write_default_config
from .dataset import DatasetProject
from .export import export_manifest, safe_component
from .manifest import register_image
from .pack import pack_dataset
from .paths import ProjectPaths, manifest_path
from .processing import ProcessConfig, clean_manifest, process_manifest, validate_image
from .reconcile import reconcile_raw
from .taggers import build_tagger, clean_tag_manifest, tag_manifest

HTML = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>LoRA Data Maker</title>
<style>
:root{color-scheme:dark;--bg:#101014;--panel:#18181f;--line:#30303a;--text:#f2f2f5;--muted:#a8a8b3;--accent:#8b6cff;--ok:#69d39a;--bad:#ff7474}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,sans-serif}
main{max-width:1050px;margin:auto;padding:18px}.top{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:14px}
h1{font-size:24px;margin:0}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px;margin-bottom:12px}
label{display:block;color:var(--muted);font-size:13px;margin:9px 0 5px}
input,select,button{font:inherit;border-radius:10px;border:1px solid var(--line);padding:10px;background:#111118;color:var(--text)}
input,select{width:100%}button{cursor:pointer;background:var(--accent);border-color:var(--accent);font-weight:650}
button.secondary{background:transparent;border-color:var(--line)}button:disabled{opacity:.45;cursor:not-allowed}
.actions{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.actions button{min-height:46px}
.drop{border:2px dashed #454552;border-radius:14px;padding:22px;text-align:center}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(115px,1fr));gap:7px}.gallery img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:9px;background:#0b0b0f}
.stats{display:flex;flex-wrap:wrap;gap:8px}.pill{padding:7px 10px;background:#22222b;border-radius:999px;color:var(--muted)}
#status{white-space:pre-wrap;color:var(--muted);min-height:22px}.ok{color:var(--ok)!important}.bad{color:var(--bad)!important}
.row{display:flex;gap:8px;align-items:center}.row>*{flex:1}
@media(max-width:700px){.grid{grid-template-columns:1fr}.actions{grid-template-columns:1fr 1fr}.top{align-items:flex-start}}
</style></head><body><main>
<div class="top"><div><h1>LoRA Data Maker</h1><div class="muted">localhost · local dataset pipeline</div></div><button class="secondary" onclick="newProject()">new dataset</button></div>

<section class="panel"><div class="grid">
<div><label>project name</label><input id="name"></div>
<div><label>subject</label><input id="subject"></div>
<div><label>trigger</label><input id="trigger"></div>
<div><label>tagger</label><select id="preset"><option value="sweetspot">sweetspot</option><option value="potato">potato</option></select></div>
</div><div class="grid">
<div><label>resize mode</label><select id="mode"><option>cover</option><option>fit</option><option>stretch</option></select></div>
<div class="row"><div><label>width</label><input id="width" type="number" min="1"></div><div><label>height</label><input id="height" type="number" min="1"></div></div>
</div><label>jpeg quality: <span id="qv"></span></label><input id="quality" type="range" min="60" max="100">
<button class="secondary" onclick="saveSettings()">save settings</button></section>

<section class="panel"><div class="drop">
<input id="files" type="file" accept="image/*" multiple>
<p class="muted">pilih foto dari gallery/file manager. upload langsung ke raw/</p>
<button onclick="upload()">upload selected</button></div></section>

<section class="panel"><div class="stats" id="stats"></div><p id="status">ready</p></section>

<section class="panel"><div class="actions">
<button onclick="run('prepare')">prepare dataset</button>
<button class="secondary" onclick="run('validate')">validate</button>
<button class="secondary" onclick="run('process')">stage / resize</button>
<button class="secondary" onclick="run('tag')">tag</button>
<button class="secondary" onclick="run('clean-tags')">clean tags</button>
<button class="secondary" onclick="run('caption')">caption .txt</button>
<button class="secondary" onclick="run('export')">export Kohya</button>
<button class="secondary" onclick="run('pack')">make ZIP</button>
<button class="secondary" onclick="run('clean')">image clean</button>
</div>
<p class="muted">prepare = validate → stage → tag → clean tags → caption → export → ZIP. image cleaning tetap manual/optional.</p>
<div id="download"></div></section>

<section class="panel"><h3>gallery</h3><div class="gallery" id="gallery"></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
async function api(url,opt={}){const r=await fetch(url,opt);const d=await r.json().catch(()=>({detail:r.statusText}));if(!r.ok)throw Error(d.detail||"request failed");return d}
function settings(s){$("name").value=s.project.name;$("subject").value=s.project.subject;$("trigger").value=s.project.trigger;$("preset").value=s.tagger.preset;$("mode").value=s.processing.mode;$("width").value=s.processing.width;$("height").value=s.processing.height;$("quality").value=s.processing.quality;$("qv").textContent=s.processing.quality}
$("quality").oninput=()=>{$("qv").textContent=$("quality").value}
async function refresh(){const s=await api("/api/state");settings(s);$("stats").innerHTML=Object.entries(s.counts).map(([k,v])=>'<span class="pill">'+k+': '+v+'</span>').join("");$("gallery").innerHTML=s.images.map(x=>'<img loading="lazy" src="/media/'+encodeURIComponent(x.path)+'">').join("")}
async function saveSettings(){await api("/api/settings",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({name:$("name").value,subject:$("subject").value,trigger:$("trigger").value,preset:$("preset").value,mode:$("mode").value,width:+$("width").value,height:+$("height").value,quality:+$("quality").value})});setStatus("settings saved","ok");refresh()}
async function upload(){const fs=$("files").files;if(!fs.length)return setStatus("select images first","bad");const f=new FormData();for(const x of fs)f.append("files",x);setStatus("uploading "+fs.length+" images...");const d=await api("/api/upload",{method:"POST",body:f});setStatus("uploaded "+d.uploaded+", skipped "+d.skipped,"ok");$("files").value="";refresh()}
async function run(stage){try{await saveSettings();const d=await api("/api/run/"+stage,{method:"POST"});setStatus(d.message);poll(d.job_id)}catch(e){setStatus(e.message,"bad")}}
async function poll(id){const d=await api("/api/jobs/"+id);setStatus(d.message||d.status,d.status==="failed"?"bad":d.status==="done"?"ok":"");if(d.status==="running"){setTimeout(()=>poll(id),1000)}else{refresh();if(d.download)$("download").innerHTML='<p><a href="'+d.download+'">⬇ download ZIP</a></p>'}}
function setStatus(t,c=""){$("status").textContent=t;$("status").className=c}
async function newProject(){const name=prompt("new dataset folder name");if(!name)return;const subject=prompt("subject",name)||name;const trigger=prompt("trigger",subject)||subject;try{await api("/api/project/new",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({name,subject,trigger})});setStatus("created "+name+". restart WebUI with that project path.");}catch(e){setStatus(e.message,"bad")}}
refresh()
</script></body></html>
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
            self.jobs[job_id] = {"status": "running", "message": label}
        threading.Thread(target=self._worker, args=(job_id, fn), daemon=True).start()
        return job_id

    def _worker(self, job_id: str, fn) -> None:
        try:
            result = fn()
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
            "processing": {"mode": str(processing.get("resize_mode", "cover")), "width": int(processing.get("target_width", 512)), "height": int(processing.get("target_height", 512)), "quality": int(processing.get("jpeg_quality", 95))},
            "tagger": {"preset": str(tagger.get("preset", "sweetspot"))},
            "counts": p.counts(),
            "images": [{"id": x.image_id, "path": x.processed_path or x.path} for x in sorted(p.manifest.images.values(), key=lambda x: x.path) if x.status != "rejected"],
        }

    @app.post("/api/settings")
    def settings(payload: dict[str, Any]):
        name, subject = str(payload.get("name", "")).strip(), str(payload.get("subject", "")).strip()
        trigger = str(payload.get("trigger", "")).strip() or subject
        mode, preset = str(payload.get("mode", "cover")).lower(), str(payload.get("preset", "sweetspot"))
        width, height, quality = int(payload.get("width", 512)), int(payload.get("height", 512)), int(payload.get("quality", 95))
        if not name or not subject or mode not in {"cover", "fit", "stretch"} or width < 1 or height < 1 or not 1 <= quality <= 100:
            raise HTTPException(400, "invalid settings")
        set_config("project", {"name": name, "subject": subject, "trigger": trigger})
        set_config("processing", {"resize_mode": mode, "target_width": width, "target_height": height, "jpeg_quality": quality})
        set_config("tagger", {"preset": preset})
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
        def run_stage() -> str:
            p, ps, cfg = project(), paths(), load_config(project_root / "config.toml")
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
                s = clean_manifest(p.manifest, root=project_root, exact=bool(c.get("exact", True)), perceptual=bool(c.get("perceptual", True)), perceptual_distance=int(c.get("perceptual_distance", 6)), min_width=int(q.get("min_width", 512)), min_height=int(q.get("min_height", 512)), blur_threshold=float(q.get("blur_threshold", proc.get("blur_threshold", 0.0))), max_aspect_ratio=float(q.get("max_aspect_ratio", 0.0)))
                p.save()
                return "cleaned " + str(s.kept) + "; exact dup " + str(s.exact_duplicates) + "; perceptual dup " + str(s.perceptual_duplicates)
            if stage == "process":
                pc = cfg.get("processing", {})
                settings = ProcessConfig(target_width=int(pc.get("target_width", 512)), target_height=int(pc.get("target_height", 512)), resize_mode=str(pc.get("resize_mode", "cover")), jpeg_quality=int(pc.get("jpeg_quality", 95)), face_crop=False)
                done, failed, skipped = process_manifest(p.manifest, root=project_root, output_dir=ps.work, config=settings)
                p.save()
                return "staged " + str(done) + ", failed " + str(failed) + ", skipped " + str(skipped)
            if stage == "tag":
                tc = cfg.get("tagger", {})
                tagger = build_tagger(tc, cache_root=Path.home() / ".cache" / "lora-data-maker" / "taggers" / "wd14", preset=str(tc.get("preset", "sweetspot")))
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
                    make_job(part)()
                return "dataset ready"
            raise RuntimeError("unknown stage: " + stage)
        return run_stage

    @app.post("/api/run/{stage}")
    def run(stage: str):
        allowed = {"prepare", "validate", "clean", "process", "tag", "clean-tags", "caption", "export", "pack"}
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
        return {"ok": True, "path": target.as_posix()}

    return app


def serve(root: Path | str = ".", host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn
    uvicorn.run(create_app(root), host=host, port=port, log_level="info")
