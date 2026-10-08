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

from .captions import (
    caption_config_from_dict,
    caption_manifest,
    reset_vlm_captions_for_deterministic,
)
from .config import load_config, write_default_config
from .dataset import DatasetProject
from .export import export_manifest, safe_component
from .manifest import find_image_by_source_url, register_image
from .pack import pack_dataset
from .paths import ProjectPaths, manifest_path
from .processing import ProcessConfig, YOLOXNanoPersonDetector, YuNetFaceDetector, clean_manifest, process_manifest, validate_image
from .processing.text import PPOCRTextDetector
from .reconcile import reconcile_raw
from .scrapers import SCRAPER_BACKENDS, ScraperError, ScraperManager
from .scrapers.base import ScraperCancelledError
from .taggers import build_tagger, clean_tag_manifest, resolve_tagger_preset, tag_manifest

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#090b10">
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<title>LoRA Data Maker</title>
<style>
:root{
  color-scheme:dark;
  --bg:#090b10;
  --bg-2:#0d1016;
  --panel:#11151d;
  --panel-2:#161b24;
  --panel-3:#0d1118;
  --line:#242b36;
  --line-strong:#313a48;
  --text:#f4f6fa;
  --muted:#929baa;
  --muted-2:#687180;
  --accent:#a795ff;
  --accent-2:#c1b7ff;
  --accent-soft:rgba(167,149,255,.13);
  --ok:#75d8a4;
  --ok-soft:rgba(117,216,164,.1);
  --bad:#ff7d89;
  --bad-soft:rgba(255,125,137,.1);
  --warn:#f0c76e;
  --radius:16px;
  --shadow:0 18px 55px rgba(0,0,0,.24);
}
*{box-sizing:border-box}
html{min-width:320px;background:var(--bg);scroll-behavior:smooth}
body{
  margin:0;
  min-height:100vh;
  color:var(--text);
  background:
    radial-gradient(900px 430px at 78% -180px,rgba(167,149,255,.12),transparent 72%),
    radial-gradient(700px 360px at -20% 80%,rgba(88,120,255,.08),transparent 70%),
    var(--bg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
button,input,select{font:inherit}
button{cursor:pointer}
button:disabled{cursor:not-allowed;opacity:.45}
button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{
  outline:2px solid var(--accent);
  outline-offset:2px;
}
button,input,select{border-radius:11px}
.app{
  width:min(1480px,100%);
  margin:0 auto;
  min-height:100vh;
  display:grid;
  grid-template-columns:282px minmax(0,1fr);
}
.sidebar{
  position:sticky;
  top:0;
  align-self:start;
  height:100vh;
  padding:20px 16px;
  border-right:1px solid var(--line);
  background:rgba(12,15,21,.92);
  display:flex;
  flex-direction:column;
  gap:16px;
  overflow:hidden;
}
.brand{display:flex;align-items:center;gap:11px;padding:2px 4px}
.brand-mark{
  width:36px;height:36px;border-radius:11px;
  display:grid;place-items:center;
  font-size:13px;font-weight:900;
  color:#0b0b10;
  background:linear-gradient(145deg,#c7beff,#7d6bf0);
  box-shadow:0 10px 28px rgba(167,149,255,.18);
}
.brand-name{font-size:14px;font-weight:820;letter-spacing:-.02em}
.brand-sub{font-size:10px;color:var(--muted);margin-top:1px}
.nav{display:grid;gap:5px}
.nav button{
  width:100%;min-height:40px;padding:0 11px;
  display:flex;align-items:center;gap:10px;
  color:var(--muted);
  background:transparent;
  border:1px solid transparent;
  font-weight:700;
  text-align:left;
}
.nav button:hover{color:var(--text);background:#11161e;border-color:var(--line)}
.nav button.active{color:var(--text);background:var(--accent-soft);border-color:#40385b}
.nav .nav-key{
  width:23px;height:23px;display:grid;place-items:center;
  border-radius:7px;background:#151a22;border:1px solid var(--line);
  font-size:10px;font-weight:800;color:var(--muted-2);
}
.nav button.active .nav-key{color:var(--text);border-color:#554b79;background:rgba(167,149,255,.1)}
.side-section{min-height:0;display:flex;flex-direction:column;gap:9px}
.side-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.side-title{
  color:var(--muted-2);font-size:10px;font-weight:820;
  text-transform:uppercase;letter-spacing:.11em;
}
.icon-button{
  width:34px;height:34px;min-height:34px;padding:0;
  display:grid;place-items:center;
  color:var(--muted);
  background:#10151c;border:1px solid var(--line);
  font-weight:850;
}
.icon-button:hover{color:var(--text);border-color:var(--line-strong);background:#161b24}
.search{
  width:100%;height:36px;
  padding:0 10px;
  color:var(--text);
  background:#0e131a;
  border:1px solid var(--line);
}
.project-explorer{
  min-height:72px;
  max-height:360px;
  overflow:auto;
  display:grid;
  align-content:start;
  gap:6px;
  padding-right:2px;
}
.project-item{
  display:grid;
  grid-template-columns:minmax(0,1fr) auto;
  gap:8px;
  padding:9px;
  border:1px solid var(--line);
  border-radius:12px;
  background:#0d1117;
}
.project-item.active{border-color:#514671;background:rgba(167,149,255,.08)}
.project-info{min-width:0}
.project-name{
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-size:12px;font-weight:780;
}
.project-name-row{display:flex;align-items:center;gap:6px;min-width:0}
.badge{
  display:inline-flex;align-items:center;justify-content:center;
  min-height:19px;padding:0 6px;
  border-radius:999px;
  font-size:9px;font-weight:820;
  white-space:nowrap;
  border:1px solid var(--line);
}
.badge.active{color:var(--accent-2);background:var(--accent-soft);border-color:#4f466a}
.badge.busy{color:var(--warn);background:rgba(240,199,110,.08);border-color:#564a2c}
.project-meta{
  margin-top:3px;
  color:var(--muted-2);
  font-size:10px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.project-actions{display:flex;align-items:center;gap:5px}
.project-actions button{
  min-height:31px;height:31px;padding:0 8px;
  font-size:10px;font-weight:760;
}
.project-actions .open{
  color:var(--text);
  background:#11161e;border:1px solid var(--line);
}
.project-actions .open:hover{border-color:var(--line-strong);background:#171c25}
.project-actions .delete{
  color:var(--bad);
  background:transparent;
  border:1px solid #4d2c33;
}
.project-actions .delete:hover{background:var(--bad-soft);border-color:#70414a}
.workspace{
  margin-top:auto;
  padding-top:13px;
  border-top:1px solid var(--line);
  color:var(--muted-2);
  font-size:9px;
  line-height:1.55;
  overflow-wrap:anywhere;
}
.side-new{
  width:100%;
  min-height:38px;
  color:#0b0b10;
  background:var(--accent);
  border:1px solid var(--accent);
  font-weight:820;
}
.side-new:hover{background:var(--accent-2)}
.main{min-width:0;padding:0 0 48px}
.topbar{
  position:sticky;top:0;z-index:10;
  min-height:70px;
  padding:14px 28px;
  display:flex;align-items:center;justify-content:space-between;gap:16px;
  border-bottom:1px solid rgba(36,43,54,.88);
  background:rgba(9,11,16,.84);
  backdrop-filter:blur(14px);
}
.topbar-copy{min-width:0}
.eyebrow{
  color:var(--muted-2);font-size:9px;font-weight:850;
  text-transform:uppercase;letter-spacing:.12em;
}
.topbar-project{
  margin-top:2px;
  font-size:15px;font-weight:820;letter-spacing:-.02em;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.topbar-actions{display:flex;align-items:center;gap:8px;flex:0 0 auto}
.dirty{
  display:none;align-items:center;gap:6px;
  color:var(--warn);font-size:10px;font-weight:740;
  white-space:nowrap;
}
.dirty.show{display:flex}
.dirty-dot{width:6px;height:6px;border-radius:50%;background:var(--warn)}
.action-secondary,.action-primary{
  min-height:38px;padding:0 12px;
  font-weight:790;border:1px solid var(--line);
}
.action-secondary{color:var(--text);background:#10151c}
.action-secondary:hover{background:#171c25;border-color:var(--line-strong)}
.action-primary{color:#0b0b10;background:var(--accent);border-color:var(--accent)}
.action-primary:hover{background:var(--accent-2)}
.view{
  display:none;
  width:min(1180px,100%);
  margin:0 auto;
  padding:28px;
}
.view.active{display:block}
.hero{
  padding:28px;
  border:1px solid #2f3540;
  border-radius:22px;
  background:
    linear-gradient(140deg,rgba(167,149,255,.13),rgba(167,149,255,.035) 54%,rgba(10,13,18,.3)),
    var(--panel);
  box-shadow:var(--shadow);
}
.hero-grid{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:24px;align-items:end}
.hero-label{
  display:inline-flex;align-items:center;gap:7px;
  color:var(--accent-2);font-size:10px;font-weight:830;
  text-transform:uppercase;letter-spacing:.12em;
}
.hero-label::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 4px rgba(167,149,255,.08)}
.hero h1{
  margin:9px 0 7px;
  font-size:clamp(28px,4.5vw,44px);
  line-height:1.03;
  letter-spacing:-.045em;
}
.hero p{margin:0;color:var(--muted);max-width:700px}
.hero-actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.big-action{
  min-height:46px;padding:0 16px;
  border-radius:12px;
  color:#0a0a0f;background:var(--accent);border:1px solid var(--accent);
  font-weight:840;
}
.big-action:hover{background:var(--accent-2)}
.layout{display:grid;gap:14px;margin-top:14px}
.layout.two{grid-template-columns:minmax(0,1.35fr) minmax(300px,.8fr)}
.card{
  min-width:0;
  border:1px solid var(--line);
  border-radius:var(--radius);
  background:rgba(17,21,29,.92);
}
.card-pad{padding:16px}
.card-title-row{display:flex;align-items:center;justify-content:space-between;gap:12px}
.card-title{font-size:12px;font-weight:820}
.card-kicker{color:var(--muted-2);font-size:10px}
.stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}
.stat{
  min-width:0;
  padding:13px 12px;
  border:1px solid var(--line);
  border-radius:12px;
  background:var(--panel-3);
}
.stat-value{font-size:21px;font-weight:820;line-height:1.08;font-variant-numeric:tabular-nums}
.stat-label{
  margin-top:5px;color:var(--muted-2);
  font-size:9px;font-weight:760;text-transform:uppercase;letter-spacing:.08em;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.status-row{display:flex;align-items:center;gap:9px;margin-top:14px}
.status-dot{width:8px;height:8px;border-radius:50%;background:#5e6672;flex:0 0 auto}
.status-dot.running{background:var(--accent);box-shadow:0 0 0 5px rgba(167,149,255,.08),0 0 12px rgba(167,149,255,.6)}
.status-dot.ok{background:var(--ok);box-shadow:0 0 0 5px rgba(117,216,164,.07)}
.status-dot.bad{background:var(--bad);box-shadow:0 0 0 5px rgba(255,125,137,.07)}
.status-dot.cancelled{background:var(--warn);box-shadow:0 0 0 5px rgba(240,199,110,.08)}
.status-text{min-width:0;color:var(--muted);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job{
  min-height:136px;
  padding:16px;
  border:1px solid var(--line);
  border-radius:14px;
  background:#0e131a;
}
.job-empty{display:grid;place-items:center;min-height:104px;color:var(--muted-2);text-align:center;font-size:11px}
.job-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}
.job-title{font-size:12px;font-weight:820}
.job-sub{margin-top:3px;color:var(--muted-2);font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job-project{font-size:9px;color:var(--muted-2);text-align:right;white-space:nowrap}
.job-progress{margin-top:15px}
.progress-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:7px;color:var(--muted);font-size:10px}
.progress-value{font-variant-numeric:tabular-nums;font-weight:780}
.progress-track{height:8px;overflow:hidden;border-radius:999px;background:#090c11;border:1px solid var(--line)}
.progress-fill{height:100%;width:0;border-radius:999px;background:var(--accent);transition:width .22s ease}
.progress-fill.done{background:var(--ok)}
.progress-fill.bad{background:var(--bad)}
.progress-fill.cancelled{background:var(--warn)}
.progress-fill.indeterminate{width:34%;animation:indeterminate 1.2s ease-in-out infinite}
@keyframes indeterminate{
  0%{transform:translateX(-140%)}
  50%{transform:translateX(170%)}
  100%{transform:translateX(340%)}
}
.job-footer{display:flex;justify-content:flex-end;margin-top:11px}
.job-footer a,.job-footer button{
  min-height:32px;display:inline-flex;align-items:center;justify-content:center;
  padding:0 9px;border-radius:9px;font-size:10px;font-weight:760;
}
.job-footer a{
  color:var(--text);text-decoration:none;
  background:#151a22;border:1px solid var(--line);
}
.job-footer button{
  color:var(--warn);
  background:rgba(240,199,110,.08);
  border:1px solid #564a2c;
}
.job-footer button:hover{background:rgba(240,199,110,.14);border-color:#75613a}
.pipeline{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}
.step{
  min-height:112px;
  padding:13px;
  display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-start;
  text-align:left;
  color:var(--text);
  background:#0d1219;
  border:1px solid var(--line);
  border-radius:13px;
}
.step:hover{border-color:#394251;background:#121721}
.step.primary{grid-column:span 2;background:linear-gradient(150deg,rgba(167,149,255,.16),rgba(167,149,255,.05));border-color:#4f4669}
.step-number{
  color:var(--muted-2);font-size:9px;font-weight:800;
  text-transform:uppercase;letter-spacing:.1em;
}
.step-name{margin-top:7px;font-size:12px;font-weight:820}
.step-hint{margin-top:4px;color:var(--muted-2);font-size:10px;line-height:1.4}
.section-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;margin-bottom:14px}
.section-head h2{margin:0;font-size:19px;letter-spacing:-.025em}
.section-head p{margin:5px 0 0;color:var(--muted);font-size:11px}
.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.form-grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.field label{
  display:block;margin:0 0 6px 2px;
  color:var(--muted);font-size:10px;font-weight:760;
}
input,select{
  width:100%;height:40px;
  padding:0 11px;
  color:var(--text);
  background:#0d1218;
  border:1px solid var(--line);
}
input:focus,select:focus{border-color:#5b507d;box-shadow:0 0 0 3px rgba(167,149,255,.07)}
.check{
  min-height:40px;padding:0 10px;
  display:flex;align-items:center;gap:8px;
  color:var(--text);
  background:#0d1218;
  border:1px solid var(--line);
}
.check input{width:auto;height:auto;accent-color:var(--accent)}
.range{display:grid;grid-template-columns:minmax(0,1fr) 38px;align-items:center;gap:10px}
.range-value{text-align:right;color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.field-note{margin:6px 2px 0;color:var(--muted-2);font-size:9px;line-height:1.4}
.save-bar{
  position:sticky;bottom:12px;z-index:8;
  display:flex;align-items:center;justify-content:flex-end;gap:9px;
  margin-top:14px;padding:10px 12px;
  border:1px solid #303745;
  border-radius:13px;
  background:rgba(13,17,24,.93);
  backdrop-filter:blur(12px);
}
.save-bar .save-note{margin-right:auto;color:var(--muted-2);font-size:10px}
.collect-grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(300px,.8fr);gap:14px}
.drop{
  min-height:220px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:24px;
  text-align:center;
  border:1px dashed #3d4653;
  border-radius:15px;
  background:#0d1218;
  transition:.16s ease;
}
.drop.drag{border-color:var(--accent);background:rgba(167,149,255,.08)}
.drop-icon{
  width:48px;height:48px;border-radius:14px;
  display:grid;place-items:center;
  background:#151a22;border:1px solid var(--line);
  color:var(--accent-2);font-weight:900;font-size:18px;
}
.drop strong{margin-top:12px;font-size:14px}
.drop p{margin:5px 0 14px;color:var(--muted);font-size:11px}
.drop-actions{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
#files{display:none}
.selection{margin-top:10px;color:var(--muted-2);font-size:10px}
.gallery{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(138px,1fr));
  gap:8px;
}
.tile{min-width:0}
.tile img{
  display:block;width:100%;aspect-ratio:1;object-fit:cover;
  border-radius:11px;border:1px solid var(--line);
  background:#080b10;
  transition:transform .16s ease,border-color .16s ease;
}
.tile:hover img{transform:translateY(-1px);border-color:#3e4654}
.tile-meta{padding:6px 2px 0}
.tile-path{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:9px}
.tile-status{margin-top:1px;color:var(--muted-2);font-size:8px;text-transform:uppercase;letter-spacing:.05em}
.tile-actions{margin-top:6px;display:flex}
.tile-delete{
  width:100%;min-height:29px;padding:0 8px;
  color:var(--bad);background:transparent;border:1px solid #4d2c33;
  border-radius:9px;font-size:9px;font-weight:790;
}
.tile-delete:hover{background:var(--bad-soft);border-color:#70414a}
.empty{
  padding:26px;text-align:center;color:var(--muted-2);
  border:1px dashed var(--line);border-radius:13px;font-size:11px;
}
.details{
  margin-top:14px;
  border:1px solid var(--line);
  border-radius:13px;
  background:#0f141b;
  overflow:hidden;
}
.details summary{
  list-style:none;
  cursor:pointer;
  padding:12px 13px;
  font-size:11px;font-weight:780;
}
.details summary::-webkit-details-marker{display:none}
.details summary::after{content:"+";float:right;color:var(--muted-2)}
.details[open] summary::after{content:"−"}
.details-body{padding:0 13px 13px}
.output{
  margin-top:14px;
  padding:13px;
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  border:1px solid var(--line);
  border-radius:13px;
  background:#0e131a;
}
.output-name{font-size:12px;font-weight:780}
.output-meta{margin-top:3px;color:var(--muted-2);font-size:10px}
.output a{
  min-height:37px;padding:0 11px;
  display:inline-flex;align-items:center;justify-content:center;
  border-radius:10px;
  text-decoration:none;
  color:#0b0b10;background:var(--ok);font-size:10px;font-weight:820;
}
.toast-wrap{
  position:fixed;right:18px;bottom:18px;z-index:100;
  width:min(380px,calc(100vw - 36px));
  display:grid;gap:8px;
  pointer-events:none;
}
.toast{
  pointer-events:auto;
  padding:11px 12px;
  border:1px solid var(--line-strong);
  border-radius:12px;
  background:#121720;
  box-shadow:var(--shadow);
  font-size:11px;
  animation:toast-in .18s ease-out;
}
.toast.ok{border-color:#345847}
.toast.bad{border-color:#5b3037}
@keyframes toast-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
dialog{
  max-width:min(520px,calc(100vw - 28px));
  width:calc(100% - 28px);
  padding:0;
  border:1px solid var(--line-strong);
  border-radius:17px;
  color:var(--text);
  background:#121720;
  box-shadow:0 28px 100px rgba(0,0,0,.48);
}
dialog::backdrop{background:rgba(0,0,0,.62);backdrop-filter:blur(4px)}
.modal{overflow:hidden}
.modal-head{padding:17px 18px;border-bottom:1px solid var(--line)}
.modal-title{font-size:14px;font-weight:820}
.modal-body{padding:17px 18px}
.modal-body p{margin:0;color:var(--muted);font-size:11px;line-height:1.55}
.modal-actions{
  display:flex;justify-content:flex-end;gap:8px;
  padding:12px 18px;border-top:1px solid var(--line);
}
.modal-actions .danger{
  color:#fff;background:#7c313b;border-color:#8e3b46;
}
.modal-actions .danger:hover{background:#94404b}
.preview{
  width:min(900px,calc(100vw - 26px));
  max-width:900px;
}
.preview-body{padding:12px;background:#080b10}
.preview-body img{
  display:block;width:100%;max-height:76vh;object-fit:contain;
  border-radius:10px;background:#05070a;
}
.preview-meta{padding:13px 15px}
.preview-name{font-size:11px;font-weight:760;overflow-wrap:anywhere}
.preview-status{margin-top:3px;color:var(--muted-2);font-size:9px}
.helper{
  display:inline-flex;align-items:center;gap:7px;
  padding:7px 9px;border-radius:9px;
  color:var(--muted);background:#0e131a;border:1px solid var(--line);
  font-size:9px;
}
.helper b{color:var(--text)}
@media(max-width:1120px){
  .app{grid-template-columns:240px minmax(0,1fr)}
  .view{padding:22px}
  .topbar{padding:13px 22px}
  .collect-grid,.layout.two{grid-template-columns:1fr}
}
@media(max-width:900px){
  .app{display:block}
  .sidebar{
    position:relative;
    height:auto;
    padding:12px;
    border-right:0;
    border-bottom:1px solid var(--line);
    overflow:visible;
  }
  .brand{padding:0 2px}
  .nav{display:flex;overflow:auto;gap:5px}
  .nav button{width:auto;min-width:max-content}
  .nav .nav-key{display:none}
  .side-section{display:block}
  .side-head{margin-bottom:8px}
  .project-explorer{max-height:175px;grid-template-columns:repeat(2,minmax(0,1fr))}
  .workspace{display:none}
  .side-new{margin-top:8px}
  .main{padding-bottom:34px}
  .stats{grid-template-columns:repeat(3,minmax(0,1fr))}
  .pipeline{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media(max-width:620px){
  .topbar{padding:11px 13px}
  .topbar-actions .new-top{display:none}
  .view{padding:14px}
  .hero{padding:19px;border-radius:17px}
  .hero-grid{grid-template-columns:1fr;gap:17px}
  .hero-actions{justify-content:flex-start}
  .big-action{width:100%}
  .form-grid,.form-grid.three{grid-template-columns:1fr}
  .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .pipeline{grid-template-columns:1fr}
  .step.primary{grid-column:span 1}
  .project-explorer{grid-template-columns:1fr}
  .output{align-items:flex-start;flex-direction:column}
  .output a{width:100%}
  .save-bar{bottom:8px}
  .save-bar .save-note{display:none}
  .gallery{grid-template-columns:repeat(2,minmax(0,1fr))}
  .section-head{display:block}
  .section-head .helper{margin-top:9px}
}
@media(prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar" aria-label="workspace navigation">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">L</div>
      <div>
        <div class="brand-name">LoRA Data Maker</div>
        <div class="brand-sub">local dataset workspace</div>
      </div>
    </div>

    <nav class="nav" aria-label="primary">
      <button type="button" data-view="overview"><span class="nav-key">01</span><span>overview</span></button>
      <button type="button" data-view="collect"><span class="nav-key">02</span><span>collect</span></button>
      <button type="button" data-view="prepare"><span class="nav-key">03</span><span>prepare</span></button>
      <button type="button" data-view="settings"><span class="nav-key">04</span><span>settings</span></button>
    </nav>

    <section class="side-section">
      <div class="side-head">
        <div class="side-title">dataset explorer</div>
        <button type="button" class="icon-button" id="refresh-projects" aria-label="refresh datasets" title="refresh datasets">↻</button>
      </div>
      <input id="project_filter" class="search" autocomplete="off" placeholder="search datasets..." aria-label="search datasets">
      <div class="project-explorer" id="project_explorer" aria-live="polite"></div>
      <button type="button" class="side-new" id="new-dataset">+ new dataset</button>
    </section>

    <div class="workspace" id="project_workspace"></div>
  </aside>

  <main class="main">
    <header class="topbar">
      <div class="topbar-copy">
        <div class="eyebrow">active dataset</div>
        <div class="topbar-project" id="topbar-project">loading...</div>
      </div>
      <div class="topbar-actions">
        <div class="dirty" id="dirty-badge" aria-live="polite"><span class="dirty-dot"></span><span>unsaved changes</span></div>
        <button type="button" class="action-secondary new-top" id="save-top" data-save>save</button>
        <button type="button" class="action-primary" id="new-top">new dataset</button>
      </div>
    </header>

    <section class="view active" data-view-panel="overview">
      <div class="hero">
        <div class="hero-grid">
          <div>
            <div class="hero-label">dataset workspace</div>
            <h1 id="hero-name">loading...</h1>
            <p id="hero-description">load the active dataset to get started.</p>
          </div>
          <div class="hero-actions">
            <button type="button" class="big-action" data-run-stage="prepare">prepare dataset</button>
          </div>
        </div>
      </div>

      <div class="layout">
        <section class="card card-pad" id="archive-card" hidden>
          <div class="card-title-row">
            <div class="card-title">latest export</div>
            <div class="card-kicker">ready to download</div>
          </div>
          <div class="output">
            <div><div class="output-name" id="archive-name"></div><div class="output-meta">final dataset archive</div></div>
            <a id="archive-link" href="/api/download">download ZIP</a>
          </div>
        </section>

        <section class="card card-pad">
          <div class="card-title-row">
            <div class="card-title">dataset status</div>
            <div class="card-kicker" id="status-context">ready</div>
          </div>
          <div class="stats" id="stats" style="margin-top:12px"></div>
          <div class="status-row">
            <span class="status-dot" id="dot"></span>
            <span class="status-text" id="status" aria-live="polite">ready</span>
          </div>
        </section>

        <section class="card card-pad">
          <div class="card-title-row">
            <div class="card-title">active job</div>
            <div class="card-kicker" id="job-state">idle</div>
          </div>
          <div class="job" style="margin-top:11px">
            <div class="job-empty" id="job-empty">nothing is running right now.</div>
            <div id="job-content" hidden>
              <div class="job-head">
                <div>
                  <div class="job-title" id="job-title"></div>
                  <div class="job-sub" id="job-sub"></div>
                </div>
                <div class="job-project" id="job-project"></div>
              </div>
              <div class="job-progress">
                <div class="progress-head">
                  <span id="progress-label">working...</span>
                  <span class="progress-value" id="progress-value">0%</span>
                </div>
                <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" id="progress-track">
                  <div class="progress-fill" id="progress-fill"></div>
                </div>
              </div>
              <div class="job-footer" id="job-footer"></div>
            </div>
          </div>
        </section>
      </div>

      <section class="card card-pad" style="margin-top:14px">
        <div class="section-head">
          <div>
            <h2>recent images</h2>
            <p>preview the current dataset without leaving the workspace.</p>
          </div>
          <span class="helper" id="gallery-count">0 images</span>
        </div>
        <div class="gallery" id="gallery"></div>
      </section>
    </section>

    <section class="view" data-view-panel="collect">
      <div class="section-head">
        <div>
          <h2>collect images</h2>
          <p>scrape from a source or upload images directly into <b>raw/</b>.</p>
        </div>
        <span class="helper">active: <b id="collect-project">loading...</b></span>
      </div>

      <div class="collect-grid">
        <section class="card card-pad">
          <div class="card-title-row">
            <div class="card-title">scraper</div>
            <div class="card-kicker">downloads → manifest</div>
          </div>
          <div class="form-grid" style="margin-top:13px">
            <div class="field">
              <label for="scraper_backend">backend</label>
              <select id="scraper_backend">
                <option value="duckduckgo">duckduckgo</option>
                <option value="google">google</option>
                <option value="chromium">chromium</option>
                <option value="playwright-bing">playwright · bing</option>
                <option value="playwright-google">playwright · google</option>
                <option value="direct">direct URLs</option>
              </select>
            </div>
            <div class="field">
              <label for="scrape_count">count</label>
              <input id="scrape_count" type="number" min="1" step="1" inputmode="numeric">
            </div>
            <div class="field" style="grid-column:1/-1">
              <label for="scrape_query">query</label>
              <input id="scrape_query" autocomplete="off" placeholder="julia russian woman portrait">
              <div class="field-note">direct URLs uses the query field as the input source expected by the configured scraper.</div>
            </div>
          </div>
          <div class="save-bar">
            <span class="save-note">save scraper settings before starting a run.</span>
            <button type="button" class="action-secondary" data-save>save</button>
            <button type="button" class="action-primary" data-run-stage="scrape">scrape images</button>
          </div>
        </section>

        <section class="card card-pad">
          <div class="card-title-row">
            <div class="card-title">upload</div>
            <div class="card-kicker">local files → raw/</div>
          </div>
          <div class="drop" id="drop" style="margin-top:13px">
            <input id="files" type="file" accept="image/*" multiple>
            <div class="drop-icon" aria-hidden="true">↑</div>
            <strong>drop images here</strong>
            <p>or choose files from your gallery / file manager</p>
            <div class="drop-actions">
              <button type="button" class="action-secondary" id="choose-images">choose images</button>
              <button type="button" class="action-primary" id="upload-button">upload selected</button>
            </div>
            <div class="selection" id="file-summary">no files selected</div>
          </div>
        </section>
      </div>
    </section>

    <section class="view" data-view-panel="prepare">
      <div class="section-head">
        <div>
          <h2>prepare dataset</h2>
          <p>run the full pipeline or execute one stage at a time when you need control.</p>
        </div>
        <span class="helper">one job at a time · project-bound execution</span>
      </div>

      <section class="card card-pad">
        <div class="pipeline">
          <button type="button" class="step primary" data-run-stage="prepare">
            <span class="step-number">all stages</span>
            <span class="step-name">prepare dataset</span>
            <span class="step-hint">validate → clean / dedup → process → caption → export → ZIP when VLM mode is selected</span>
          </button>
          <button type="button" class="step" data-run-stage="validate">
            <span class="step-number">01 · input</span><span class="step-name">validate</span><span class="step-hint">scan and validate raw images</span>
          </button>
          <button type="button" class="step" data-run-stage="clean">
            <span class="step-number">02 · quality</span><span class="step-name">clean / dedup</span><span class="step-hint">remove duplicates, text, and low-quality images</span>
          </button>
          <button type="button" class="step" data-run-stage="process">
            <span class="step-number">03 · image</span><span class="step-name">stage / resize</span><span class="step-hint">make training-ready images</span>
          </button>
          <button type="button" class="step" data-run-stage="tag">
            <span class="step-number">04 · metadata</span><span class="step-name">tag</span><span class="step-hint">deterministic mode only</span>
          </button>
          <button type="button" class="step" data-run-stage="clean-tags">
            <span class="step-number">05 · metadata</span><span class="step-name">clean tags</span><span class="step-hint">deterministic mode only</span>
          </button>
          <button type="button" class="step" data-run-stage="caption">
            <span class="step-number">06 · metadata</span><span class="step-name">caption</span><span class="step-hint">write .txt captions</span>
          </button>
          <button type="button" class="step" data-run-stage="export">
            <span class="step-number">07 · output</span><span class="step-name">export</span><span class="step-hint">create Kohya-ready folder</span>
          </button>
          <button type="button" class="step" data-run-stage="pack">
            <span class="step-number">08 · output</span><span class="step-name">make ZIP</span><span class="step-hint">build final archive</span>
          </button>
        </div>
      </section>

      <section class="card card-pad" style="margin-top:14px">
        <div class="card-title-row">
          <div class="card-title">how the pipeline flows</div>
          <div class="card-kicker">safe to rerun stages</div>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:11px">
          <span class="helper"><b>raw/</b> source images</span>
          <span class="helper"><b>work/</b> processed images + captions</span>
          <span class="helper"><b>output/</b> export + ZIP</span>
          <span class="helper"><b>manifest.json</b> source of truth</span>
        </div>
      </section>
    </section>

    <section class="view" data-view-panel="settings">
      <div class="section-head">
        <div>
          <h2>dataset settings</h2>
          <p>changes are kept local to the active dataset and written back to <b>config.toml</b>.</p>
        </div>
        <span class="helper">changes are saved explicitly</span>
      </div>

      <section class="card card-pad">
        <div class="card-title">project identity</div>
        <div class="form-grid three" style="margin-top:13px">
          <div class="field"><label for="name">name</label><input id="name" autocomplete="off"></div>
          <div class="field"><label for="subject">subject</label><input id="subject" autocomplete="off"></div>
          <div class="field"><label for="trigger">trigger</label><input id="trigger" autocomplete="off"></div>
        </div>
      </section>

      <section class="card card-pad" style="margin-top:14px">
        <div class="card-title">tagger <span class="card-kicker">(deterministic mode only)</span></div>
        <div class="form-grid" style="margin-top:13px">
          <div class="field">
            <label for="preset">preset</label>
            <select id="preset">
              <option value="sweetspot">wd14 · sweetspot</option>
              <option value="potato">wd14 · potato</option>
              <option value="joytag">joytag</option>
            </select>
          </div>
        </div>
      </section>
      <section class="card card-pad" style="margin-top:14px">
        <div class="card-title">caption</div>
        <div class="form-grid three" style="margin-top:13px">
          <div class="field">
            <label for="caption_backend">backend</label>
            <select id="caption_backend">
              <option value="deterministic">deterministic · tags only</option>
              <option value="vlm">local VLM · VLM only</option>
            </select>
          </div>
          <div class="field">
            <label for="caption_chat_format">VLM chat format</label>
            <select id="caption_chat_format">
              <option value="mtmd">mtmd</option>
              <option value="qwen2.5-vl">qwen2.5-vl</option>
              <option value="gemma4">gemma4</option>
            </select>
          </div>
          <div class="field">
            <label for="caption_max_tokens">max caption tokens</label>
            <div class="range">
              <input id="caption_max_tokens" type="range" min="1" max="77" step="1">
              <span class="range-value" id="caption_tokens_value"></span>
            </div>
          </div>
        </div>
        <div class="form-grid" style="margin-top:12px">
          <div class="field">
            <label for="caption_model_path">VLM model GGUF</label>
            <input id="caption_model_path" autocomplete="off" placeholder="/path/to/model.gguf">
          </div>
          <div class="field">
            <label for="caption_mmproj_path">VLM mmproj GGUF</label>
            <input id="caption_mmproj_path" autocomplete="off" placeholder="/path/to/mmproj.gguf">
          </div>
        </div>
        <div class="field-note" style="margin-top:9px">VLM mode skips WD14/JoyTag completely. captions are trigger → pose → clothing → expression → background, packed with the exact openai/clip-vit-large-patch14 tokenizer under the 1–77 token budget.</div>
      </section>

      <section class="card card-pad" style="margin-top:14px">
        <div class="card-title">processing</div>
        <div class="form-grid three" style="margin-top:13px">
          <div class="field">
            <label for="mode">resize</label>
            <select id="mode">
              <option value="cover">cover</option>
              <option value="fit">fit</option>
              <option value="stretch">stretch</option>
            </select>
          </div>
          <div class="field"><label for="width">width</label><input id="width" type="number" min="1" inputmode="numeric"></div>
          <div class="field"><label for="height">height</label><input id="height" type="number" min="1" inputmode="numeric"></div>
        </div>
        <div class="form-grid three" style="margin-top:12px">
          <div class="field">
            <label for="quality">jpeg quality</label>
            <div class="range"><input id="quality" type="range" min="60" max="100"><span class="range-value" id="qv"></span></div>
          </div>
          <div class="field">
            <label>face crop</label>
            <label class="check"><input id="face_crop" type="checkbox"> enable</label>
          </div>
          <div class="field"><label for="face_padding">face padding</label><input id="face_padding" type="number" min="0.1" step="0.05"></div>
        </div>
      </section>

      <details class="details">
        <summary>quality gate & detection</summary>
        <div class="details-body">
          <div class="form-grid three" style="margin-top:13px">
            <div class="field">
              <label>person quality gate</label>
              <label class="check"><input id="person_check" type="checkbox"> enable</label>
              <div class="field-note">YOLOX person-count and subject-visibility filter.</div>
            </div>
            <div class="field"><label for="max_persons">max people</label><input id="max_persons" type="number" min="1" step="1"></div>
            <div class="field"><label for="person_threshold">person confidence</label><input id="person_threshold" type="number" min="0.01" max="1" step="0.01"></div>
          </div>
          <div class="form-grid" style="margin-top:12px">
            <div class="field"><label for="min_subject_area">min subject area</label><input id="min_subject_area" type="number" min="0" max="1" step="0.01"></div>
            <div class="field"><label for="min_subject_visibility">min subject visibility</label><input id="min_subject_visibility" type="number" min="0" max="1" step="0.01"></div>
          </div>
          <div class="form-grid three" style="margin-top:12px">
            <div class="field">
              <label>text quality gate</label>
              <label class="check"><input id="text_check" type="checkbox"> enable</label>
              <div class="field-note">PP-OCRv3 rejects images with detected visible text before processing/export.</div>
            </div>
            <div class="field"><label for="text_confidence">text confidence</label><input id="text_confidence" type="number" min="0.01" max="1" step="0.01"></div>
            <div class="field"><label for="text_min_area_ratio">min text area</label><input id="text_min_area_ratio" type="number" min="0" max="1" step="0.0001"></div>
          </div>
        </div>
      </details>

      <div class="save-bar">
        <span class="save-note">saving resets the unsaved-changes marker.</span>
        <button type="button" class="action-secondary" data-save>save settings</button>
      </div>
    </section>
  </main>
</div>

<div class="toast-wrap" id="toast-wrap" aria-live="polite" aria-atomic="true"></div>

<dialog id="new-modal" class="modal">
  <div class="modal-head"><div class="modal-title">create a new dataset</div></div>
  <form id="new-form" method="dialog">
    <div class="modal-body">
      <div class="form-grid">
        <div class="field" style="grid-column:1/-1"><label for="new-name">folder name</label><input id="new-name" autocomplete="off" required></div>
        <div class="field"><label for="new-subject">subject</label><input id="new-subject" autocomplete="off" required></div>
        <div class="field"><label for="new-trigger">trigger</label><input id="new-trigger" autocomplete="off" required></div>
      </div>
    </div>
    <div class="modal-actions">
      <button type="button" class="action-secondary" id="new-cancel">cancel</button>
      <button type="submit" class="action-primary" id="new-submit">create dataset</button>
    </div>
  </form>
</dialog>

<dialog id="confirm-modal" class="modal">
  <div class="modal-head"><div class="modal-title" id="confirm-title">confirm action</div></div>
  <div class="modal-body"><p id="confirm-body"></p></div>
  <div class="modal-actions">
    <button type="button" class="action-secondary" id="confirm-cancel">cancel</button>
    <button type="button" class="action-primary" id="confirm-accept">confirm</button>
  </div>
</dialog>

<dialog id="preview-modal" class="preview">
  <div class="modal-head"><div class="modal-title">image preview</div></div>
  <div class="preview-body"><img id="preview-image" alt=""></div>
  <div class="preview-meta">
    <div class="preview-name" id="preview-name"></div>
    <div class="preview-status" id="preview-status"></div>
  </div>
</dialog>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));
const appState={
  settingsDirty:false,
  activeProjectPath:"",
  currentJob:null,
  jobId:localStorage.getItem("ldm.jobId")||"",
  pollTimer:null,
  confirmResolve:null,
  toastSeq:0,
  projects:[]
};

async function api(url,opt={}){
  let response;
  try{response=await fetch(url,opt)}
  catch(e){throw Error("cannot reach the local server")}
  let data={};
  try{data=await response.json()}catch(e){data={detail:response.statusText||"request failed"}}
  if(!response.ok)throw Error(data.detail||"request failed");
  return data;
}

function setStatus(message,kind=""){
  $("status").textContent=message;
  $("dot").className="status-dot "+kind;
}
function toast(message,kind=""){
  const el=document.createElement("div");
  el.className="toast "+kind;
  el.textContent=message;
  $("toast-wrap").appendChild(el);
  const id=++appState.toastSeq;
  setTimeout(()=>{
    if(el.isConnected)el.remove();
  },3400);
  return id;
}
function isBusyOnActiveProject(){
  const job=appState.currentJob;
  return !!job && job.status==="running" && !!job.project_path && !!appState.activeProjectPath && job.project_path===appState.activeProjectPath;
}
function updateDirtyUI(){
  $("dirty-badge").classList.toggle("show",appState.settingsDirty);
  document.querySelectorAll("[data-save]").forEach(button=>{button.disabled=!appState.settingsDirty});
}
function updateBusyUI(){
  const busy=isBusyOnActiveProject();
   document.querySelectorAll("[data-run-stage]").forEach(button=>{button.disabled=busy});
  $("upload-button").disabled=busy;
   document.querySelectorAll("[data-image-delete]").forEach(button=>{button.disabled=busy});
  document.querySelectorAll("[data-save]").forEach(button=>{button.disabled=busy||!appState.settingsDirty});
  document.querySelectorAll("[data-setting]").forEach(el=>{el.disabled=busy});
  document.querySelectorAll(".step").forEach(button=>{button.disabled=busy});
  document.querySelectorAll("#scraper_backend,#scrape_query,#scrape_count,#name,#subject,#trigger,#preset,#mode,#width,#height,#quality,#face_crop,#face_padding,#caption_backend,#caption_chat_format,#caption_model_path,#caption_mmproj_path,#caption_max_tokens,#person_check,#max_persons,#person_threshold,#min_subject_area,#min_subject_visibility,#text_check,#text_confidence,#text_min_area_ratio").forEach(el=>{
    if(el)el.disabled=busy;
  });
  const vlmOnly=$("caption_backend").value==="vlm";
  ["tag","clean-tags"].forEach(stage=>{
    document.querySelectorAll('[data-run-stage="'+stage+'"]').forEach(button=>{
      button.disabled=busy||vlmOnly;
      button.title=vlmOnly?"not used with VLM captions":"";
    });
  });
}
function applySettings(s,force=false){
  if(appState.settingsDirty&&!force)return;
  const project=s.project||{name:"",subject:"",trigger:""};
  $("name").value=project.name||"";
  $("subject").value=project.subject||"";
  $("trigger").value=project.trigger||"";
  $("preset").value=s.tagger.preset;
  $("caption_backend").value=s.caption.backend||"deterministic";
  $("caption_chat_format").value=s.caption.chat_format||"mtmd";
  $("caption_model_path").value=s.caption.model_path||"";
  $("caption_mmproj_path").value=s.caption.mmproj_path||"";
  $("caption_max_tokens").value=s.caption.max_tokens;
  $("caption_tokens_value").textContent=s.caption.max_tokens;
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
  $("text_check").checked=!!s.quality.text_check;
  $("text_confidence").value=s.quality.text_confidence;
  $("text_min_area_ratio").value=s.quality.text_min_area_ratio;
  appState.settingsDirty=false;
  updateDirtyUI();
}
function renderStats(counts){
  const keys=["discovered","validated","processed","tagged","captioned","exported"];
  const labels={"discovered":"discovered","validated":"validated","processed":"processed","tagged":"tagged","captioned":"captioned","exported":"exported"};
  $("stats").innerHTML=keys.map(k=>'<div class="stat"><div class="stat-value">'+esc(counts[k]??0)+'</div><div class="stat-label">'+esc(labels[k])+'</div></div>').join("");
}
function renderProjects(projects){
  appState.projects=projects||[];
  const filter=$("project_filter").value.trim().toLowerCase();
  const items=(projects||[]).filter(p=>{
    const hay=(String(p.name||"")+" "+String(p.subject||"")+" "+String(p.path||"")).toLowerCase();
    return !filter||hay.includes(filter);
  });
  if(!items.length){
    $("project_explorer").innerHTML='<div class="empty">no dataset projects found.</div>';
    return;
  }
  $("project_explorer").innerHTML=items.map(p=>{
    const busy=appState.currentJob&&appState.currentJob.status==="running"&&appState.currentJob.project_path===p.path;
    const open=p.active
      ? '<button type="button" class="open" disabled>open</button>'
      : '<button type="button" class="open" data-project-open="'+esc(p.name)+'">open</button>';
    const remove=p.active||busy
      ? '<button type="button" class="delete" disabled title="'+esc(p.active?"open another dataset first":"a job is running for this dataset")+'">'+(busy&&!p.active?"busy":"delete")+'</button>'
      : '<button type="button" class="delete" data-project-delete="'+esc(p.name)+'">delete</button>';
    const badges=(p.active?'<span class="badge active">active</span>':'')+(busy?'<span class="badge busy">busy</span>':'');
    return '<div class="project-item'+(p.active?' active':'')+'"><div class="project-info"><div class="project-name-row"><div class="project-name">'+esc(p.name)+'</div>'+badges+'</div><div class="project-meta">'+esc(p.subject)+' · '+esc(p.image_count)+' images</div></div><div class="project-actions">'+open+remove+'</div></div>';
  }).join("");
}
function renderDownload(name,url,job){
  const actions=[];
  if(job&&job.status==="running"&&job.stage==="scrape"){
    const cancelling=!!job.cancel_requested;
    actions.push('<button type="button" data-job-cancel="'+esc(appState.jobId)+'"'+(cancelling?" disabled":"")+'>'
      +(cancelling?"cancelling…":"cancel scrape")+"</button>");
  }
  if(url)actions.push('<a href="'+esc(url)+'">download ZIP</a>');
  $("job-footer").innerHTML=actions.join("");
  if(name){
    $("job-footer").dataset.archive=name;
  }else{
    delete $("job-footer").dataset.archive;
  }
}
function renderArchive(name){
  const box=$("archive-card");
  if(!name){
    box.hidden=true;
    return;
  }
  box.hidden=false;
  $("archive-name").textContent=name;
  $("archive-link").href="/api/download";
}
function renderJob(job){
  appState.currentJob=job||null;
  if(!job||job.status==="missing"){
    appState.currentJob=null;
    $("job-empty").hidden=false;
    $("job-content").hidden=true;
    $("job-state").textContent="idle";
    renderDownload("");
    updateBusyUI();
    return;
  }
  $("job-empty").hidden=true;
  $("job-content").hidden=false;
  $("job-state").textContent=job.status==="running"?"running":job.status==="done"?"complete":job.status==="cancelled"?"cancelled":"failed";
  $("job-title").textContent=job.message||job.status;
  $("job-sub").textContent=job.progress&&job.progress.label?job.progress.label:"";
  $("job-project").textContent=job.project_path||"";
  const p=job.progress||{};
  const current=Math.max(0,Number(p.current||0));
  const total=Math.max(0,Number(p.total||0));
  const percent=total>0?Math.max(0,Math.min(100,Number(p.percent??(current/total*100)))):0;
  $("progress-label").textContent=p.label||job.message||job.status;
  $("progress-value").textContent=total>0?Math.round(percent)+"%":job.status==="running"?"…":"";
  $("progress-track").setAttribute("aria-valuenow",String(Math.round(percent)));
  const fill=$("progress-fill");
  fill.className="progress-fill";
  if(job.status==="done")fill.classList.add("done");
  if(job.status==="failed")fill.classList.add("bad");
  if(job.status==="cancelled")fill.classList.add("cancelled");
  if(job.status==="running"&&total<=0)fill.classList.add("indeterminate");
  fill.style.width=total>0?percent+"%":job.status==="running"?"34%":"0%";
  renderDownload(job.download?"archive": "",job.download||"",job);
  updateBusyUI();
}
function renderGallery(images){
  const all=images||[];
  const visible=all.slice(-120);
  const busy=isBusyOnActiveProject();
  $("gallery-count").textContent=all.length+" images"+(all.length>120?" · showing latest 120":"");
  if(!visible.length){
    $("gallery").innerHTML='<div class="empty">no images yet. collect or upload some raw images first.</div>';
    return;
  }
  $("gallery").innerHTML=visible.map(x=>{
    const src="/media/"+encodeURIComponent(x.path);
    return '<div class="tile" data-preview-path="'+esc(src)+'" data-preview-name="'+esc(x.path)+'" data-preview-status="'+esc(x.status)+'"><img loading="lazy" src="'+esc(src)+'" alt="'+esc(x.path)+'"><div class="tile-meta"><div class="tile-path" title="'+esc(x.path)+'">'+esc(x.path)+'</div><div class="tile-status">'+esc(x.status)+'</div><div class="tile-actions"><button type="button" class="tile-delete" data-image-delete="'+esc(x.id)+'" data-image-path="'+esc(x.path)+'"'+(busy?" disabled":"")+' aria-label="remove image">remove</button></div></div></div>';
  }).join("");
}
function updateProjectHeader(s){
  const project=s.project||{};
  const hasProject=!!s.project;
  $("topbar-project").textContent=project.name||"no dataset selected";
  $("hero-name").textContent=project.name||"create your first dataset";
  $("hero-description").textContent=hasProject
    ? (project.subject||"")+" · trigger: "+(project.trigger||"")
    : "use + new dataset in the sidebar to get started.";
  $("collect-project").textContent=project.name||"no dataset";
  $("status-context").textContent=hasProject
    ? s.project_count+" dataset"+(s.project_count===1?"":"s")+" in workspace"
    : "no active dataset";
}
function setView(view,push=true){
  const allowed=new Set(["overview","collect","prepare","settings"]);
  if(!allowed.has(view))view="overview";
  document.querySelectorAll("[data-view]").forEach(button=>button.classList.toggle("active",button.dataset.view===view));
  document.querySelectorAll("[data-view-panel]").forEach(panel=>panel.classList.toggle("active",panel.dataset.viewPanel===view));
  if(push){
    history.replaceState(null,"","#"+view);
  }
  window.scrollTo({top:0,behavior:"smooth"});
}
async function refreshProjects(){
  try{
    const d=await api("/api/projects");
    renderProjects(d.projects);
    $("project_workspace").textContent=d.workspace+" · "+d.projects.length+" dataset projects";
    return d.projects;
  }catch(e){
    toast(e.message,"bad");
    setStatus(e.message,"bad");
    return [];
  }
}
async function refresh(forceSettings=false){
  try{
    const s=await api("/api/state");
    updateProjectHeader(s);
    renderStats(s.counts);
    renderGallery(s.images);
    renderProjects(s.projects);
    renderArchive(s.archive);
    $("project_workspace").textContent=s.workspace+" · "+s.project_count+" dataset projects";
    appState.activeProjectPath=((s.projects||[]).find(p=>p.active)||{}).path||"";
    if(forceSettings||!appState.settingsDirty)applySettings(s,true);
    $("status-context").textContent=s.project_count+" dataset"+(s.project_count===1?"":"s")+" in workspace";
    updateBusyUI();
    return s;
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
    return null;
  }
}
async function saveSettings(){
  if(!appState.activeProjectPath){
    toast("create or open a dataset first.","bad");
    return false;
  }
  if(isBusyOnActiveProject()){
    toast("this dataset is busy. wait for the current job to finish.","bad");
    return false;
  }
  const width=Number($("width").value);
  const height=Number($("height").value);
  const quality=Number($("quality").value);
  const captionMaxTokens=Number($("caption_max_tokens").value);
  const facePadding=Number($("face_padding").value);
  const maxPersons=Number($("max_persons").value);
  const personThreshold=Number($("person_threshold").value);
  const minArea=Number($("min_subject_area").value);
  const minVisibility=Number($("min_subject_visibility").value);
  const textConfidence=Number($("text_confidence").value);
  const textMinArea=Number($("text_min_area_ratio").value);
  const scrapeCount=Number($("scrape_count").value);
  if(!Number.isFinite(width)||!Number.isFinite(height)||!Number.isFinite(quality)||!Number.isFinite(captionMaxTokens)||!Number.isFinite(facePadding)||!Number.isFinite(maxPersons)||!Number.isFinite(personThreshold)||!Number.isFinite(minArea)||!Number.isFinite(minVisibility)||!Number.isFinite(textConfidence)||!Number.isFinite(textMinArea)||!Number.isFinite(scrapeCount)){
    toast("please fill every numeric setting with a valid value.","bad");
    return false;
  }
  document.querySelectorAll("[data-save]").forEach(button=>button.disabled=true);
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
        scrape_count:scrapeCount,
        mode:$("mode").value,
        width:width,
        height:height,
        quality:quality,
        caption_backend:$("caption_backend").value,
        caption_chat_format:$("caption_chat_format").value,
        caption_model_path:$("caption_model_path").value.trim(),
        caption_mmproj_path:$("caption_mmproj_path").value.trim(),
        caption_max_tokens:captionMaxTokens,
        face_crop:$("face_crop").checked,
        face_padding:facePadding,
        person_check:$("person_check").checked,
        max_persons:maxPersons,
        person_threshold:personThreshold,
        min_subject_area:minArea,
        min_subject_visibility:minVisibility,
        text_check:$("text_check").checked,
        text_confidence:textConfidence,
        text_min_area_ratio:textMinArea
      })
    });
    appState.settingsDirty=false;
    updateDirtyUI();
    await refresh(true);
    setStatus("settings saved","ok");
    toast("settings saved","ok");
    return true;
  }catch(e){
    updateDirtyUI();
    setStatus(e.message,"bad");
    toast(e.message,"bad");
    return false;
  }
}
function schedulePoll(id){
  if(appState.pollTimer)clearTimeout(appState.pollTimer);
  appState.pollTimer=setTimeout(()=>poll(id),850);
}
async function poll(id){
  if(appState.jobId!==id)return;
  try{
    const d=await api("/api/jobs/"+encodeURIComponent(id));
    renderJob(d);
    const kind=d.status==="failed"?"bad":d.status==="done"?"ok":d.status==="cancelled"?"cancelled":"running";
    setStatus(d.message||d.status,kind);
    if(d.status==="running"){
      schedulePoll(id);
    }else{
      localStorage.removeItem("ldm.jobId");
      appState.jobId="";
      await refresh();
      if(d.status==="done")toast("job complete","ok");
      if(d.status==="failed")toast(d.message||"job failed","bad");
      if(d.status==="cancelled")toast(d.message||"scrape cancelled","ok");
    }
  }catch(e){
    setStatus(e.message,"bad");
    schedulePoll(id);
  }
}
async function ensureSaved(){
  if(!appState.settingsDirty)return true;
  return saveSettings();
}
async function cancelScrape(id){
  if(!id)return;
  try{
    await api("/api/jobs/"+encodeURIComponent(id)+"/cancel",{method:"POST"});
    toast("cancelling scrape...","running");
    await poll(id);
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
    await refresh();
  }
}
async function deleteImage(id,path){
  if(isBusyOnActiveProject())return;
  const accepted=await confirmAction(
    "remove this image?",
    "this removes \""+path+"\" and its derived files from the dataset. this cannot be undone.",
    "remove image",
    true
  );
  if(!accepted)return;
  try{
    await api("/api/images/"+encodeURIComponent(id),{method:"DELETE"});
    toast("removed image","ok");
    await refresh();
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
    await refresh();
  }
}
async function runStage(stage){
  if(!appState.activeProjectPath){
    toast("create or open a dataset first.","bad");
    return;
  }
  if(isBusyOnActiveProject()){
    toast("this dataset is already running a job.","bad");
    return;
  }
  if(!(await ensureSaved()))return;
  try{
    const d=await api("/api/run/"+encodeURIComponent(stage),{method:"POST"});
    appState.jobId=d.job_id;
    localStorage.setItem("ldm.jobId",d.job_id);
    renderJob({status:"running",message:d.message,progress:d.progress||{current:0,total:0,label:d.message},project_path:appState.activeProjectPath});
    setStatus(d.message,"running");
    toast(stage==="prepare"?"dataset preparation started":stage+" started","running");
    poll(d.job_id);
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
    await refresh();
  }
}
function markDirty(){
  if(!appState.activeProjectPath||isBusyOnActiveProject())return;
  appState.settingsDirty=true;
  updateDirtyUI();
}
async function selectProject(name){
  if(appState.settingsDirty){
    const discard=await confirmAction(
      "discard unsaved changes?",
      "switching datasets will discard the changes currently typed in the settings form.",
      "discard & open",
      true
    );
    if(!discard)return;
    appState.settingsDirty=false;
    updateDirtyUI();
  }
  try{
    await api("/api/projects/select",{
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({name:name})
    });
    await refresh(true);
    setStatus("opened "+name,"ok");
    toast("opened "+name,"ok");
    setView("overview");
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
  }
}
async function deleteProject(name){
  const accepted=await confirmAction(
    "delete this dataset?",
    "this permanently removes \""+name+"\" and every file inside it. this cannot be undone.",
    "delete dataset",
    true
  );
  if(!accepted)return;
  try{
    await api("/api/projects/"+encodeURIComponent(name),{method:"DELETE"});
    toast("deleted "+name,"ok");
    await refreshProjects();
    setStatus("dataset deleted","ok");
  }catch(e){
    toast(e.message,"bad");
    setStatus(e.message,"bad");
    await refreshProjects();
  }
}
function openNewDataset(){
  $("new-form").reset();
  $("new-name").value="";
  $("new-subject").value="";
  $("new-trigger").value="";
  $("new-modal").showModal();
  setTimeout(()=>$("new-name").focus(),0);
}
async function createNewDataset(event){
  event.preventDefault();
  if(appState.settingsDirty){
    const discard=await confirmAction(
      "discard unsaved changes?",
      "creating a new dataset will leave the current unsaved settings behind.",
      "discard & continue",
      true
    );
    if(!discard)return;
    appState.settingsDirty=false;
    updateDirtyUI();
  }
  const button=$("new-submit");
  button.disabled=true;
  try{
    const d=await api("/api/project/new",{
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({
        name:$("new-name").value.trim(),
        subject:$("new-subject").value.trim(),
        trigger:$("new-trigger").value.trim()
      })
    });
    $("new-modal").close();
    await refresh(true);
    setView("overview");
    setStatus("created "+d.name,"ok");
    toast("created "+d.name,"ok");
  }catch(e){
    toast(e.message,"bad");
    setStatus(e.message,"bad");
  }finally{
    button.disabled=false;
  }
}
function confirmAction(title,body,acceptLabel="confirm",danger=false){
  return new Promise(resolve=>{
    appState.confirmResolve=resolve;
    $("confirm-title").textContent=title;
    $("confirm-body").textContent=body;
    $("confirm-accept").textContent=acceptLabel;
    $("confirm-accept").className=danger?"action-primary danger":"action-primary";
    $("confirm-modal").showModal();
  });
}
function finishConfirm(value){
  const resolve=appState.confirmResolve;
  appState.confirmResolve=null;
  if($("confirm-modal").open)$("confirm-modal").close();
  if(resolve)resolve(value);
}
async function upload(){
  if(!appState.activeProjectPath){
    toast("create or open a dataset first.","bad");
    return;
  }
  if(isBusyOnActiveProject()){
    toast("this dataset is busy. wait for the current job to finish.","bad");
    return;
  }
  const files=$("files").files;
  if(!files.length){
    toast("choose some images first.","bad");
    setStatus("select images first","bad");
    return;
  }
  const form=new FormData();
  for(const file of files)form.append("files",file);
  try{
    setStatus("uploading "+files.length+" images...","running");
    const d=await api("/api/upload",{method:"POST",body:form});
    $("files").value="";
    $("file-summary").textContent="no files selected";
    await refresh();
    setStatus("uploaded "+d.uploaded+", skipped "+d.skipped,"ok");
    toast("uploaded "+d.uploaded+" images","ok");
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
  }
}
function updateFileSummary(){
  const n=$("files").files.length;
  $("file-summary").textContent=n?n+" file"+(n===1?"":"s")+" selected":"no files selected";
}
function openPreview(tile){
  const path=tile.dataset.previewPath;
  $("preview-image").src=path;
  $("preview-image").alt=tile.dataset.previewName||"";
  $("preview-name").textContent=tile.dataset.previewName||"";
  $("preview-status").textContent=tile.dataset.previewStatus||"";
  $("preview-modal").showModal();
}
document.querySelectorAll("[data-view]").forEach(button=>{
  button.addEventListener("click",()=>setView(button.dataset.view));
});
window.addEventListener("hashchange",()=>setView(location.hash.slice(1)||"overview",false));
$("project_explorer").addEventListener("click",event=>{
  const open=event.target.closest("[data-project-open]");
  const remove=event.target.closest("[data-project-delete]");
  if(open)selectProject(open.dataset.projectOpen);
  if(remove)deleteProject(remove.dataset.projectDelete);
});
$("project_filter").addEventListener("input",()=>renderProjects(appState.projects));
$("refresh-projects").addEventListener("click",refreshProjects);
$("new-dataset").addEventListener("click",openNewDataset);
$("new-top").addEventListener("click",openNewDataset);
$("save-top").addEventListener("click",saveSettings);
document.querySelectorAll("[data-save]").forEach(button=>button.addEventListener("click",saveSettings));
document.querySelectorAll("[data-run-stage]").forEach(button=>{
  button.addEventListener("click",()=>runStage(button.dataset.runStage));
});
$("choose-images").addEventListener("click",()=>$("files").click());
$("upload-button").addEventListener("click",upload);
$("files").addEventListener("change",updateFileSummary);
$("quality").addEventListener("input",()=>{$("qv").textContent=$("quality").value});
$("caption_max_tokens").addEventListener("input",()=>{$("caption_tokens_value").textContent=$("caption_max_tokens").value});
[
  "name","subject","trigger","preset","scraper_backend","scrape_query","scrape_count",
  "mode","width","height","quality","caption_backend","caption_chat_format","caption_model_path","caption_mmproj_path","caption_max_tokens","face_crop","face_padding","person_check",
  "max_persons","person_threshold","min_subject_area","min_subject_visibility",
  "text_check","text_confidence","text_min_area_ratio"
].forEach(id=>{
  $(id).addEventListener("input",markDirty);
  $(id).addEventListener("change",markDirty);
});
$("job-footer").addEventListener("click",event=>{
  const cancel=event.target.closest("[data-job-cancel]");
  if(cancel)cancelScrape(cancel.dataset.jobCancel);
});
$("gallery").addEventListener("click",event=>{
  const remove=event.target.closest("[data-image-delete]");
  if(remove){
    deleteImage(remove.dataset.imageDelete, remove.dataset.imagePath||"image");
    return;
  }
  const tile=event.target.closest("[data-preview-path]");
  if(tile)openPreview(tile);
});
const drop=$("drop");
["dragenter","dragover"].forEach(type=>drop.addEventListener(type,event=>{
  event.preventDefault();
  if(!isBusyOnActiveProject())drop.classList.add("drag");
}));
["dragleave","drop"].forEach(type=>drop.addEventListener(type,event=>{
  event.preventDefault();
  drop.classList.remove("drag");
}));
drop.addEventListener("drop",event=>{
  if(isBusyOnActiveProject())return;
  const files=event.dataTransfer&&event.dataTransfer.files;
  if(files&&files.length){
    const transfer=new DataTransfer();
    for(const file of files)transfer.items.add(file);
    $("files").files=transfer.files;
    updateFileSummary();
  }
});
$("new-form").addEventListener("submit",createNewDataset);
$("new-cancel").addEventListener("click",()=>$("new-modal").close());
$("confirm-cancel").addEventListener("click",()=>finishConfirm(false));
$("confirm-accept").addEventListener("click",()=>finishConfirm(true));
$("confirm-modal").addEventListener("cancel",event=>{event.preventDefault();finishConfirm(false)});
$("new-modal").addEventListener("cancel",()=>{$("new-form").reset()});
$("preview-modal").addEventListener("click",event=>{
  if(event.target===$("preview-modal"))$("preview-modal").close();
});
document.addEventListener("keydown",event=>{
  if(event.key==="Escape"&&$("confirm-modal").open)finishConfirm(false);
});
const initialView=location.hash.slice(1)||"overview";
setView(initialView,false);
updateDirtyUI();
refresh().then(()=>{
  if(appState.jobId){
    poll(appState.jobId);
  }else{
    renderJob(null);
  }
});
</script>
</body>
</html>
"""
class JobCancelledError(Exception):
    """Raised when a running WebUI job observes a cancellation request."""


class JobStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}

    def start(
        self,
        label: str,
        fn,
        *,
        project_root: Path | None = None,
        stage: str = "",
    ) -> str:
        with self.lock:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise RuntimeError("another job is already running")
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {
                "status": "running",
                "message": label,
                "stage": stage,
                "cancel_requested": False,
                "_cancel_event": threading.Event(),
                "_project_root": project_root.as_posix() if project_root is not None else None,
                "progress": {
                    "current": 0,
                    "total": 0,
                    "percent": 0,
                    "label": label,
                },
            }
        threading.Thread(target=self._worker, args=(job_id, fn), daemon=True).start()
        return job_id

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError("job not found")
            if job["status"] != "running":
                raise RuntimeError("job is no longer running")
            if job.get("stage") != "scrape":
                raise RuntimeError("only scrape jobs can be cancelled")
            job["_cancel_event"].set()
            job["cancel_requested"] = True
            job["message"] = "cancelling scrape..."
            return dict(job)

    def has_running_project(self, project_root: Path) -> bool:
        target = project_root.expanduser().resolve()
        with self.lock:
            for job in self.jobs.values():
                if job["status"] != "running":
                    continue
                value = job.get("_project_root")
                if value and Path(value).expanduser().resolve() == target:
                    return True
        return False

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

        with self.lock:
            event = self.jobs[job_id]["_cancel_event"]

        try:
            result = fn(progress, event)
            with self.lock:
                self.jobs[job_id].update({"status": "done", "message": result or "done"})
        except JobCancelledError as exc:
            with self.lock:
                self.jobs[job_id].update({"status": "cancelled", "message": str(exc)})
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update({"status": "failed", "message": str(exc)})

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            value = self.jobs.get(job_id)
            if value is None:
                return {"status": "missing", "message": "job not found"}
            result = dict(value)
            result.pop("_cancel_event", None)
            return result


def create_app(root: Path | str = ".") -> FastAPI:
    app = FastAPI(title="LoRA Data Maker WebUI")
    static_root = Path(__file__).resolve().parent / "static"
    configured_root = Path(root).expanduser().resolve()
    if (configured_root / "config.toml").is_file() and (configured_root / "manifest.json").is_file():
        project_root: Path | None = configured_root
        workspace_root = configured_root.parent
    else:
        project_root = None
        workspace_root = configured_root
    jobs = JobStore()
    project_root_lock = threading.Lock()

    def active_root() -> Path | None:
        with project_root_lock:
            return project_root

    def require_active_root() -> Path:
        current = active_root()
        if current is None:
            raise HTTPException(409, "create or open a dataset first")
        return current

    def paths(root_path: Path | None = None) -> ProjectPaths:
        selected_root = root_path if root_path is not None else require_active_root()
        config = load_config(selected_root / "config.toml")
        return ProjectPaths.from_root(selected_root, config.get("paths", {}))

    def project(root_path: Path | None = None) -> DatasetProject:
        return DatasetProject.open(paths(root_path))

    def list_projects() -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        active = active_root()
        if not workspace_root.is_dir():
            return projects
        for candidate in sorted(
            (item for item in workspace_root.iterdir() if item.is_dir()),
            key=lambda item: item.name.casefold(),
        ):
            if not (candidate / "config.toml").is_file() or not (candidate / "manifest.json").is_file():
                continue
            try:
                candidate_project = DatasetProject.open(ProjectPaths.from_root(candidate))
                projects.append(
                    {
                        "name": candidate.name,
                        "subject": candidate_project.manifest.subject,
                        "trigger": candidate_project.manifest.trigger,
                        "image_count": len(candidate_project.manifest.images),
                        "path": candidate.as_posix(),
                        "active": active is not None and candidate.resolve() == active.resolve(),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return projects

    def select_active_root(target: Path) -> None:
        nonlocal project_root
        target = target.expanduser().resolve()
        try:
            target.relative_to(workspace_root)
        except ValueError as exc:
            raise HTTPException(400, "project is outside workspace") from exc
        if target.parent != workspace_root:
            raise HTTPException(400, "project must be a direct workspace child")
        if not (target / "config.toml").is_file() or not (target / "manifest.json").is_file():
            raise HTTPException(404, "dataset project not found")
        with project_root_lock:
            project_root = target

    def set_config(section: str, values: dict[str, Any], root_path: Path | None = None) -> None:
        config_path = (root_path or active_root()) / "config.toml"
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

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(static_root / "favicon.ico", media_type="image/x-icon")

    @app.get("/favicon-16x16.png")
    def favicon_16():
        return FileResponse(static_root / "favicon-16x16.png", media_type="image/png")

    @app.get("/favicon-32x32.png")
    def favicon_32():
        return FileResponse(static_root / "favicon-32x32.png", media_type="image/png")

    @app.get("/api/state")
    def state():
        current_root = active_root()
        project_list = list_projects()
        if current_root is None:
            return {
                "project": None,
                "processing": {
                    "mode": "cover",
                    "width": 512,
                    "height": 512,
                    "quality": 95,
                    "face_crop": False,
                    "face_padding": 1.75,
                },
                "tagger": {"backend": "wd14", "preset": "sweetspot"},
                "caption": {"backend": "deterministic", "max_tokens": 77},
                "scraper": {
                    "backend": "duckduckgo",
                    "query": "",
                    "count": 300,
                    "max_scrolls": 30,
                },
                "quality": {
                    "person_check": False,
                    "person_threshold": 0.35,
                    "max_persons": 1,
                    "min_subject_area": 0.12,
                    "min_subject_visibility": 0.35,
                    "text_check": True,
                    "text_confidence": 0.5,
                    "text_min_area_ratio": 0.0004,
                },
                "counts": {
                    "discovered": 0,
                    "validated": 0,
                    "processed": 0,
                    "tagged": 0,
                    "captioned": 0,
                    "exported": 0,
                },
                "workspace": workspace_root.as_posix(),
                "project_count": len(project_list),
                "projects": project_list,
                "images": [],
                "archive": None,
            }

        p = project(current_root)
        cfg = load_config(current_root / "config.toml")
        processing = cfg.get("processing", {})
        tagger = cfg.get("tagger", {})
        caption = cfg.get("caption", {})
        if not isinstance(caption, dict):
            caption = {}
        return {
            "project": {
                "name": p.manifest.project_name,
                "subject": p.manifest.subject,
                "trigger": p.manifest.trigger,
                "path": current_root.as_posix(),
            },
            "processing": {
                "mode": str(processing.get("resize_mode", "cover")),
                "width": int(processing.get("target_width", 512)),
                "height": int(processing.get("target_height", 512)),
                "quality": int(processing.get("jpeg_quality", 95)),
                "face_crop": bool(processing.get("face_crop", False)),
                "face_padding": float(processing.get("face_padding", 1.75)),
            },
            "tagger": {
                "backend": str(tagger.get("backend", "wd14")),
                "preset": str(tagger.get("preset", "sweetspot")),
            },
            "caption": {
                "backend": str(caption.get("backend", "deterministic")),
                "chat_format": str(caption.get("vlm", {}).get("chat_format", "mtmd"))
                if isinstance(caption.get("vlm"), dict)
                else "mtmd",
                "model_path": str(caption.get("vlm", {}).get("model_path", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "mmproj_path": str(caption.get("vlm", {}).get("mmproj_path", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "max_tokens": int(caption.get("max_tokens", 77)),
            },
            "scraper": {
                "backend": str(cfg.get("scraper", {}).get("backend", "duckduckgo")),
                "query": str(cfg.get("scraper", {}).get("query", p.manifest.subject)),
                "count": int(cfg.get("scraper", {}).get("count", 300)),
                "max_scrolls": int(cfg.get("scraper", {}).get("max_scrolls", 30)),
            },
            "quality": {
                "person_check": bool(cfg.get("quality", {}).get("person_check", False)),
                "person_threshold": float(cfg.get("quality", {}).get("person_threshold", 0.35)),
                "max_persons": int(cfg.get("quality", {}).get("max_persons", 1)),
                "min_subject_area": float(cfg.get("quality", {}).get("min_subject_area", 0.12)),
                "min_subject_visibility": float(cfg.get("quality", {}).get("min_subject_visibility", 0.35)),
                "text_check": bool(cfg.get("quality", {}).get("text_check", True)),
                "text_confidence": float(cfg.get("quality", {}).get("text_confidence", 0.5)),
                "text_min_area_ratio": float(cfg.get("quality", {}).get("text_min_area_ratio", 0.0004)),
            },
            "counts": p.counts(),
            "workspace": workspace_root.as_posix(),
            "project_count": len(project_list),
            "projects": project_list,
            "images": [{"id": x.image_id, "path": x.processed_path or x.path, "status": x.status} for x in sorted(p.manifest.images.values(), key=lambda x: x.path) if x.status != "rejected"],
            "archive": Path(str(p.manifest.source.get("last_pack_archive", ""))).name if p.manifest.source.get("last_pack_archive") else None,
        }

    @app.get("/api/projects")
    def projects():
        return {"workspace": workspace_root.as_posix(), "projects": list_projects()}

    @app.delete("/api/projects/{name}")
    def delete_project(name: str):
        project_name = safe_component(name.strip())
        if not project_name or project_name != name:
            raise HTTPException(400, "invalid project name")
        target = (workspace_root / project_name).resolve()
        if target.parent != workspace_root or not target.is_dir():
            raise HTTPException(404, "dataset project not found")
        if not (target / "config.toml").is_file() or not (target / "manifest.json").is_file():
            raise HTTPException(404, "dataset project not found")
        with project_root_lock:
            if target == project_root:
                raise HTTPException(409, "open another dataset before deleting the active project")
            if jobs.has_running_project(target):
                raise HTTPException(409, "dataset has a running job; wait for it to finish before deleting")
            try:
                import shutil
                shutil.rmtree(target)
            except OSError as exc:
                raise HTTPException(500, f"could not delete dataset: {exc}") from exc
        return {"ok": True, "name": project_name}

    @app.post("/api/projects/select")
    def select_project(payload: dict[str, Any]):
        name = safe_component(str(payload.get("name", "")).strip())
        if not name:
            raise HTTPException(400, "project name required")
        target = workspace_root / name
        select_active_root(target)
        p = project(target)
        return {
            "ok": True,
            "name": p.manifest.project_name,
            "subject": p.manifest.subject,
            "trigger": p.manifest.trigger,
            "path": target.as_posix(),
        }

    @app.post("/api/settings")
    def settings(payload: dict[str, Any]):
        current_root = require_active_root()
        name, subject = str(payload.get("name", "")).strip(), str(payload.get("subject", "")).strip()
        trigger = str(payload.get("trigger", "")).strip() or subject
        mode, preset = str(payload.get("mode", "cover")).lower(), str(payload.get("preset", "sweetspot"))
        current_config = load_config(current_root / "config.toml")
        current_scraper = current_config.get("scraper", {})
        if not isinstance(current_scraper, dict):
            current_scraper = {}
        current_quality = current_config.get("quality", {})
        if not isinstance(current_quality, dict):
            current_quality = {}
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
        person_check = bool(payload.get("person_check", current_quality.get("person_check", False)))
        max_persons = int(payload.get("max_persons", 1))
        person_threshold = float(payload.get("person_threshold", 0.35))
        min_subject_area = float(payload.get("min_subject_area", 0.12))
        min_subject_visibility = float(payload.get("min_subject_visibility", 0.35))
        text_check = bool(payload.get("text_check", current_quality.get("text_check", True)))
        text_confidence = float(payload.get("text_confidence", 0.5))
        text_min_area_ratio = float(payload.get("text_min_area_ratio", 0.0004))
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
            or not 0 < text_confidence <= 1
            or not 0 <= text_min_area_ratio <= 1
        ):
            raise HTTPException(400, "invalid settings")
        set_config("project", {"name": name, "subject": subject, "trigger": trigger}, current_root)
        set_config("processing", {"resize_mode": mode, "target_width": width, "target_height": height, "jpeg_quality": quality, "face_crop": face_crop, "face_padding": face_padding}, current_root)
        set_config("tagger", {"backend": "joytag" if resolved_preset.kind == "joytag" else "wd14", "preset": preset}, current_root)
        current_caption = current_config.get("caption", {})
        if not isinstance(current_caption, dict):
            current_caption = {}
        raw_caption_max_tokens = payload.get(
            "caption_max_tokens",
            current_caption.get("max_tokens", 77),
        )
        try:
            caption_max_tokens = int(raw_caption_max_tokens)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                400,
                "caption max tokens must be an integer between 1 and 77",
            ) from exc
        if not 1 <= caption_max_tokens <= 77:
            raise HTTPException(400, "caption max tokens must be between 1 and 77")
        caption_backend = str(
            payload.get("caption_backend", current_caption.get("backend", "deterministic"))
        ).strip().lower()
        current_vlm = current_caption.get("vlm", {})
        if not isinstance(current_vlm, dict):
            current_vlm = {}
        caption_chat_format = str(
            payload.get("caption_chat_format", current_vlm.get("chat_format", "mtmd"))
        ).strip().lower()
        caption_model_path = str(
            payload.get("caption_model_path", current_vlm.get("model_path", ""))
        ).strip()
        caption_mmproj_path = str(
            payload.get("caption_mmproj_path", current_vlm.get("mmproj_path", ""))
        ).strip()
        if caption_backend not in {"deterministic", "vlm"}:
            raise HTTPException(400, "caption backend must be deterministic or vlm")
        if caption_chat_format not in {"mtmd", "qwen2.5-vl", "gemma4"}:
            raise HTTPException(400, "unsupported VLM chat format")
        if caption_backend == "vlm" and (not caption_model_path or not caption_mmproj_path):
            raise HTTPException(400, "VLM model and mmproj paths are required")
        try:
            caption_config_from_dict(
                {
                    "backend": caption_backend,
                    "max_tokens": caption_max_tokens,
                    "vlm": {
                        "model_path": caption_model_path,
                        "mmproj_path": caption_mmproj_path,
                        "chat_format": caption_chat_format,
                    },
                },
                root=current_root,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        p = project(current_root)
        if caption_backend == "deterministic":
            reset_vlm_captions_for_deterministic(
                p.manifest,
                root=current_root,
            )

        set_config(
            "caption",
            {"backend": caption_backend, "max_tokens": caption_max_tokens},
            current_root,
        )
        set_config(
            "caption.vlm",
            {
                "model_path": caption_model_path,
                "mmproj_path": caption_mmproj_path,
                "chat_format": caption_chat_format,
            },
            current_root,
        )
        set_config(
            "scraper",
            {
                "backend": scraper_backend,
                "query": scrape_query,
                "count": scrape_count,
            },
            current_root,
        )
        set_config(
            "quality",
            {
                "person_check": person_check,
                "person_threshold": person_threshold,
                "max_persons": max_persons,
                "min_subject_area": min_subject_area,
                "min_subject_visibility": min_subject_visibility,
                "text_check": text_check,
                "text_confidence": text_confidence,
                "text_min_area_ratio": text_min_area_ratio,
            },
            current_root,
        )
        p.manifest.project_name, p.manifest.subject, p.manifest.trigger = name, subject, trigger
        p.save()
        return {"ok": True}

    @app.delete("/api/images/{image_id}")
    def delete_image(image_id: str):
        current_root = require_active_root()
        if jobs.has_running_project(current_root):
            raise HTTPException(409, "dataset has a running job; wait for it to finish before editing images")

        p = project(current_root)
        record = p.manifest.images.get(image_id)
        if record is None:
            raise HTTPException(404, "image not found")

        relative_paths: list[str] = []
        for value in (
            record.path,
            record.processed_path,
            record.caption_path,
            record.metadata.get("export_image_path"),
            record.metadata.get("export_caption_path"),
        ):
            if isinstance(value, str) and value.strip():
                relative_paths.append(value)

        targets: list[Path] = []
        seen: set[Path] = set()
        for value in relative_paths:
            target = (current_root / value).resolve()
            try:
                target.relative_to(current_root)
            except ValueError as exc:
                raise HTTPException(400, "invalid image artifact path") from exc
            if target == current_root:
                raise HTTPException(400, "invalid image artifact path")
            if target not in seen:
                seen.add(target)
                targets.append(target)

        try:
            for target in targets:
                if target.is_file():
                    target.unlink()
                elif target.exists():
                    raise OSError(f"image artifact is not a file: {target}")
            del p.manifest.images[image_id]
            p.manifest.touch()
            p.save()
        except OSError as exc:
            raise HTTPException(500, f"could not delete image: {exc}") from exc

        return {
            "ok": True,
            "image_id": image_id,
            "deleted": [manifest_path(current_root, target) for target in targets],
        }

    @app.post("/api/upload")
    async def upload(files: Annotated[list[UploadFile], File()]):
        current_root = require_active_root()
        p, ps = project(current_root), paths(current_root)
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
            rel = manifest_path(current_root, target)
            if rel in existing:
                target.unlink(missing_ok=True); skipped += 1; continue
            register_image(p.manifest, path=rel, status="discovered")
            existing.add(rel); uploaded += 1
        p.save()
        return {"uploaded": uploaded, "skipped": skipped}

    def make_job(stage: str, root_path: Path):
        def run_stage(
            progress: Callable[[int, int, str], None],
            cancel_event: threading.Event,
        ) -> str:
            p, ps, cfg = project(root_path), paths(root_path), load_config(root_path / "config.toml")
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
                    cancel_event=cancel_event,
                )
                search_error = manager.last_search_error

                added_records = 0
                for result in downloads:
                    relative_path = manifest_path(root_path, result.path)
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

                if getattr(manager, "cancelled", False):
                    raise JobCancelledError(
                        f"scrape cancelled: downloaded {len(downloads)}/{count}"
                    )

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
                reconcile_raw(p.manifest, root=root_path, raw_dir=ps.raw)
                existing = {x.path: x for x in p.manifest.images.values()}
                from .images import iter_image_files
                valid = rejected = skipped = 0
                for image_path in iter_image_files(ps.raw):
                    rel = manifest_path(root_path, image_path)
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
                text_detector = None
                if person_check:
                    person_detector = YOLOXNanoPersonDetector(
                        cache_dir=ps.cache / "person",
                        confidence_threshold=float(q.get("person_threshold", 0.35)),
                    )
                if bool(q.get("text_check", True)):
                    text_detector = PPOCRTextDetector(
                        cache_dir=ps.cache / "text",
                        confidence_threshold=float(q.get("text_confidence", 0.5)),
                        min_area_ratio=float(q.get("text_min_area_ratio", 0.0004)),
                    )
                s = clean_manifest(
                    p.manifest,
                    root=root_path,
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
                    text_detector=text_detector,
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
                done, failed, skipped = process_manifest(p.manifest, root=root_path, output_dir=ps.work, config=settings, face_detector=face_detector)
                p.save()
                return "staged " + str(done) + ", failed " + str(failed) + ", skipped " + str(skipped)
            if stage == "tag":
                cc = cfg.get("caption", {})
                caption_backend = (
                    str(cc.get("backend", "deterministic")).strip().lower()
                    if isinstance(cc, dict)
                    else "deterministic"
                )
                if caption_backend == "vlm":
                    return "tag skipped: VLM caption backend is VLM-only"
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
                s = tag_manifest(p.manifest, root=root_path, tagger=tagger)
                p.save()
                return "tagged " + str(s.tagged) + ", skipped " + str(s.skipped)
            if stage == "clean-tags":
                cc = cfg.get("caption", {})
                caption_backend = (
                    str(cc.get("backend", "deterministic")).strip().lower()
                    if isinstance(cc, dict)
                    else "deterministic"
                )
                if caption_backend == "vlm":
                    return "clean-tags skipped: VLM caption backend is VLM-only"
                tc = cfg.get("tag_rules", {})
                s = clean_tag_manifest(p.manifest, remove_tags=tc.get("remove", []))
                p.save()
                return "cleaned tags for " + str(s.cleaned)
            if stage == "caption":
                cc = cfg.get("caption", {})
                if not isinstance(cc, dict):
                    raise RuntimeError("config section [caption] must be a table")
                settings = caption_config_from_dict(cc, root=root_path)
                s = caption_manifest(
                    p.manifest,
                    root=root_path,
                    output_dir=ps.work / "captions",
                    config=settings,
                )
                p.save()
                if s.failed:
                    raise RuntimeError(
                        "caption failed for "
                        + str(s.failed)
                        + " images"
                    )
                return (
                    "captioned "
                    + str(s.captioned)
                    + ", failed "
                    + str(s.failed)
                    + " via "
                    + settings.backend
                )
            if stage == "export":
                ec = cfg.get("export", {})
                s = export_manifest(p.manifest, root=root_path, output_dir=ps.output, repeats=int(ec.get("repeats", 1)), folder_name=str(ec.get("folder_name", "") or "") or None)
                p.manifest.source["last_export_dir"] = manifest_path(root_path, s.output_dir); p.save()
                if s.failed: raise RuntimeError("export failed for " + str(s.failed) + " images")
                return "exported " + str(s.exported)
            if stage == "pack":
                last = p.manifest.source.get("last_export_dir")
                if not isinstance(last, str) or not last: raise RuntimeError("export the dataset first")
                archive = ps.output / (safe_component(p.manifest.project_name) + ".zip")
                s = pack_dataset(root=root_path, dataset_dir=root_path / last, archive_path=archive, manifest_path=p.manifest_path)
                p.manifest.source.update({"last_pack_archive": manifest_path(root_path, archive), "last_pack_sha256": s.sha256}); p.save()
                return "ZIP ready: " + archive.name
            if stage == "prepare":
                cc = cfg.get("caption", {})
                caption_backend = (
                    str(cc.get("backend", "deterministic")).strip().lower()
                    if isinstance(cc, dict)
                    else "deterministic"
                )
                parts = (
                    ("validate", "clean", "process", "caption", "export", "pack")
                    if caption_backend == "vlm"
                    else ("validate", "clean", "process", "tag", "clean-tags", "caption", "export", "pack")
                )
                for part in parts:
                    make_job(part, root_path)(progress, cancel_event)
                return "dataset ready"
            raise RuntimeError("unknown stage: " + stage)
        return run_stage

    @app.post("/api/run/{stage}")
    def run(stage: str):
        allowed = {"scrape", "prepare", "validate", "clean", "process", "tag", "clean-tags", "caption", "export", "pack"}
        if stage not in allowed: raise HTTPException(404, "unknown stage")
        current_root = require_active_root()
        try: job_id = jobs.start(
            "running " + stage + "...",
            make_job(stage, current_root),
            project_root=current_root,
            stage=stage,
        )
        except RuntimeError as exc: raise HTTPException(409, str(exc))
        return {"job_id": job_id, "message": "running " + stage + "..."}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        try:
            data = jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "job not found") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        data.pop("_cancel_event", None)
        data.pop("_project_root", None)
        return data

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        data = jobs.get(job_id)
        job_root_value = data.pop("_project_root", None)
        if job_root_value:
            data["project_path"] = job_root_value
        if data.get("status") == "done" and job_root_value:
            try:
                job_project = project(Path(job_root_value))
                if job_project.manifest.source.get("last_pack_archive"):
                    data["download"] = f"/api/jobs/{job_id}/download"
            except (OSError, ValueError, KeyError, TypeError, FileNotFoundError):
                pass
        return data

    @app.get("/api/jobs/{job_id}/download")
    def job_download(job_id: str):
        data = jobs.get(job_id)
        job_root_value = data.get("_project_root")
        if not job_root_value:
            raise HTTPException(404, "job project not available")
        job_root = Path(job_root_value).expanduser().resolve()
        rel = project(job_root).manifest.source.get("last_pack_archive")
        if not isinstance(rel, str) or not rel:
            raise HTTPException(404, "no ZIP available for this job")
        target = (job_root / rel).resolve()
        try:
            target.relative_to(job_root)
        except ValueError as exc:
            raise HTTPException(400, "invalid archive path") from exc
        if not target.is_file():
            raise HTTPException(404, "ZIP not found")
        return FileResponse(target, filename=target.name, media_type="application/zip")

    @app.get("/api/download")
    def download():
        current_root = require_active_root()
        rel = project(current_root).manifest.source.get("last_pack_archive")
        if not isinstance(rel, str) or not rel: raise HTTPException(404, "no ZIP available")
        target = (current_root / rel).resolve()
        try: target.relative_to(current_root)
        except ValueError: raise HTTPException(400, "invalid archive path")
        if not target.is_file(): raise HTTPException(404, "ZIP not found")
        return FileResponse(target, filename=target.name, media_type="application/zip")

    @app.get("/media/{relative:path}")
    def media(relative: str):
        current_root = require_active_root()
        target = (current_root / relative).resolve()
        try: target.relative_to(current_root)
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
        target = workspace_root / name
        if target.exists(): raise HTTPException(409, "project already exists")
        target.mkdir(parents=True)
        new_paths = ProjectPaths.from_root(target); new_paths.create_directories()
        write_default_config(new_paths.config, name=name, subject=subject, trigger=trigger)
        DatasetProject.create(new_paths)
        select_active_root(target)
        return {"ok": True, "path": target.as_posix(), "name": name}

    return app


def serve(root: Path | str = ".", host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn
    uvicorn.run(create_app(root), host=host, port=port, log_level="info")
