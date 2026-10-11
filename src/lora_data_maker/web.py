from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import date
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .captions import (
    caption_config_from_dict,
    caption_manifest,
    reset_vlm_captions_for_deterministic,
)
from .config import load_config, write_default_config
from .credentials import default_api_key_store_path, load_api_keys, save_api_keys
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


def _api_runtime_settings(runtime: str, base_url: str = "") -> tuple[str, str]:
    selected = runtime.strip().lower()
    if selected == "gemini":
        return (
            "GEMINI_API_KEY",
            (base_url.strip() or "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
        )
    if selected == "openai_compatible":
        return (
            "OPENAI_API_KEY",
            (base_url.strip() or "https://api.openai.com/v1").rstrip("/"),
        )
    raise ValueError("unsupported API VLM runtime")

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<meta name="theme-color" content="#000000">
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<title>LoRA Data Maker</title>
<style>

:root{color-scheme:light;--base:#fff;--ink:#000;--safety:#ff3e00;--muted:#888;--surface:#f5f5f0;--surface-2:#e9e9e3;--term-bg:#08080a;--term-green:#00ff66;--term-amber:#ffaa00;--border:3px solid var(--ink);--shadow:6px 6px 0 var(--ink);--shadow-safety:6px 6px 0 var(--safety);--gap:clamp(10px,1.2vw,18px)}
*{box-sizing:border-box}
html{min-width:320px;scroll-behavior:smooth;background:var(--base);-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh;color:var(--ink);background:var(--base);font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.6;overflow-x:hidden}
body::before{content:none}img{max-width:100%;display:block}a{color:inherit;text-decoration:none}button,input,select,textarea{font:inherit;border-radius:0}button{cursor:pointer;transition:background .12s,color .12s,border-color .12s,transform .12s,box-shadow .12s}button:disabled{cursor:not-allowed;opacity:.45}
button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible,summary:focus-visible,a:focus-visible{outline:2px solid var(--safety);outline-offset:2px}::selection{background:var(--safety);color:#fff}[hidden]{display:none!important}
.axis-header{position:sticky;top:0;z-index:50;width:100%;min-height:54px;display:flex;align-items:center;padding:0 clamp(12px,2.4vw,36px);border-bottom:3px solid var(--safety);background:var(--ink);color:#fff}
.axis-header-inner{width:100%;max-width:1820px;margin:0 auto;min-height:51px;display:flex;align-items:center;gap:clamp(10px,1.4vw,24px)}
.axis-brand{display:flex;align-items:center;gap:8px;min-width:max-content;color:#fff}
.axis-logo-mark{width:35px;height:31px;display:grid;place-items:center;flex:0 0 35px;overflow:hidden;background:#fff;border:1px solid var(--safety)}.axis-logo-mark img{width:33px;height:29px;object-fit:contain}
.axis-brand-copy{display:flex;flex-direction:column;line-height:1}.axis-wordmark{display:block;width:47px;height:18px;overflow:hidden;background:#fff}.axis-wordmark-img{width:100%;height:100%;object-fit:contain}
.axis-brand-copy small{margin-top:4px;color:rgba(255,255,255,.52);font:7px/1.2 "JetBrains Mono",monospace;letter-spacing:.01em}
.axis-header-sep{height:25px;width:1px;background:rgba(255,255,255,.18);flex:0 0 1px}
.axis-product{color:rgba(255,255,255,.43);font-size:8px;line-height:1.45;letter-spacing:.1em;white-space:nowrap}
.axis-nav{display:flex;align-self:stretch;align-items:stretch;margin-left:auto;gap:0;border-left:1px solid rgba(255,255,255,.13)}.axis-nav::before{display:none!important}
.axis-nav button{position:relative;min-width:92px;display:flex;align-items:center;justify-content:center;gap:8px;padding:0 12px;border:0;border-right:1px solid rgba(255,255,255,.13);color:rgba(255,255,255,.58);background:transparent;font-size:9px;letter-spacing:.09em;text-transform:uppercase}
.axis-nav button:hover{color:#fff;background:var(--safety)}.axis-nav button.active{color:#fff;background:rgba(255,255,255,.075)}
.axis-nav button.active::after{content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:var(--safety)}
.axis-nav .nav-key{width:auto;height:auto;min-width:16px;padding:0;border:0;background:transparent;color:rgba(255,255,255,.26);font:9px "JetBrains Mono",monospace}.axis-nav button.active .nav-key{color:var(--safety)}
.axis-header-right{display:flex;align-items:center;justify-content:flex-end;gap:15px;min-width:max-content}
.axis-online{display:flex;align-items:center;gap:6px;color:rgba(255,255,255,.75);font-size:8px;letter-spacing:.1em;white-space:nowrap}.axis-online-dot{height:5px;width:5px;background:var(--term-green);box-shadow:0 0 6px rgba(0,255,102,.6)}
.axis-clock{color:rgba(255,255,255,.5);font-size:8px;letter-spacing:.04em;white-space:nowrap}
.app{width:min(1800px,100%);min-height:calc(100vh - 54px);margin:0 auto;display:grid;grid-template-columns:250px minmax(0,1fr);border-left:1px solid rgba(0,0,0,.08);border-right:1px solid rgba(0,0,0,.08)}
.sidebar{position:sticky;top:54px;align-self:start;height:calc(100vh - 54px);min-height:520px;padding:22px 14px 13px;display:flex;flex-direction:column;gap:17px;overflow:hidden;background:var(--surface);border-right:var(--border)}
.side-section{min-height:0;flex:1;display:flex;flex-direction:column;gap:9px;overflow:hidden}.side-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.side-title{color:#555;font:500 9px/1.5 "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase}
.icon-button{width:28px;height:28px;min-height:28px;padding:0;display:grid;place-items:center;border:1px solid var(--ink);background:#fff;color:var(--ink);font-size:15px}.icon-button:hover{background:var(--safety);color:#fff}
.search{width:100%;height:35px;padding:0 9px;border:1px solid #aaa;background:#fff;color:var(--ink);font-size:10px}.search::placeholder,input::placeholder{color:#999}.search:focus{outline:2px solid var(--safety);outline-offset:0;border-color:var(--safety)}
.project-explorer{flex:1;min-height:100px;max-height:none;overflow:auto;display:grid;align-content:start;gap:7px;padding:2px 5px 6px 1px}
.project-item{display:grid;grid-template-columns:minmax(0,1fr);gap:8px;padding:10px 9px;border:1px solid #c8c8c2;background:#fff;transition:background .1s,box-shadow .1s,transform .1s}
.project-item:hover{border-color:var(--ink);box-shadow:3px 3px 0 var(--ink);transform:translate(-2px,-2px)}.project-item.active{border:2px solid var(--ink);padding:9px 8px;background:#fff;box-shadow:4px 4px 0 var(--safety)}
.project-item.active::before{content:"ACTIVE PROJECT";display:block;margin:-1px 0 1px;color:var(--safety);font-size:7px;letter-spacing:.12em}
.project-info{min-width:0}.project-name-row{display:flex;align-items:center;flex-wrap:wrap;gap:5px}.project-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:"Space Grotesk",sans-serif;font-size:12px;font-weight:700;letter-spacing:-.025em;text-transform:uppercase}
.project-meta{margin-top:3px;color:#777;font-size:8px;overflow-wrap:anywhere}.badge{display:inline-flex;align-items:center;padding:2px 4px;border:1px solid var(--ink);color:var(--ink);font-size:7px;letter-spacing:.04em;text-transform:uppercase}
.badge.active{color:#fff;border-color:var(--ink);background:var(--ink)}.badge.busy{color:#fff;border-color:var(--safety);background:var(--safety)}
.project-actions{display:flex;gap:5px}.project-actions button{min-height:25px;padding:0 7px;border:1px solid var(--ink);color:var(--ink);background:#fff;font-size:8px;letter-spacing:.04em;text-transform:uppercase}
.project-actions button:hover{color:#fff;background:var(--ink)}.project-actions .delete{margin-left:auto;border-color:var(--safety);color:var(--safety)}.project-actions .delete:hover{color:#fff;background:var(--safety)}
.side-new{display:flex;align-items:center;gap:8px;width:100%;min-height:37px;padding:0 10px;border:2px solid var(--ink);background:var(--safety);color:#fff;box-shadow:3px 3px 0 var(--ink);font-size:9px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.side-new:hover{transform:translate(-2px,-2px);box-shadow:5px 5px 0 var(--ink)}.side-new .plus{font-size:17px;line-height:1}.side-new .shortcut{margin-left:auto;color:rgba(255,255,255,.7);font-size:9px}
.side-status{display:flex;align-items:center;gap:8px;padding:10px 4px;border-top:1px solid rgba(0,0,0,.2);border-bottom:1px solid rgba(0,0,0,.2)}
.side-status-icon{width:27px;height:27px;display:grid;place-items:center;border:2px solid var(--ink);background:#fff;color:var(--safety);font-size:13px}.side-status b,.side-status small{display:block}.side-status b{color:var(--ink);font-size:8px;letter-spacing:.08em}.side-status small{margin-top:2px;color:#777;font-size:7px}
.side-status-led{margin-left:auto;width:5px;height:5px;background:var(--term-green);border:1px solid var(--ink)}.workspace{padding:8px 2px 0;color:#777;font-size:8px;line-height:1.6;overflow-wrap:anywhere}
.main{min-width:0;background:#fff}.topbar{position:sticky;top:54px;z-index:20;min-height:69px;padding:12px clamp(16px,2.6vw,38px);display:flex;align-items:center;gap:12px;background:#fff;border-bottom:var(--border)}
.topbar-copy{min-width:0}.eyebrow{color:#777;font-size:8px;letter-spacing:.16em;text-transform:uppercase}.eyebrow-slash{color:var(--safety);padding:0 4px}
.topbar-project{margin-top:2px;max-width:42vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:"Space Grotesk",sans-serif;font-size:clamp(12px,1.4vw,17px);font-weight:700;letter-spacing:-.035em;text-transform:uppercase}.topbar-signal{display:none}
.topbar-actions{display:flex;align-items:center;justify-content:flex-end;gap:7px;margin-left:auto}.dirty{display:flex;align-items:center;gap:6px;color:#666;font-size:8px;white-space:nowrap}.dirty-dot{width:5px;height:5px;background:var(--safety)}
.action-primary,.action-secondary{min-height:33px;padding:0 10px;display:inline-flex;align-items:center;justify-content:center;gap:7px;border:2px solid var(--ink);font-size:9px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap}
.action-primary{color:#fff;background:var(--ink);box-shadow:3px 3px 0 var(--safety)}.action-primary:hover{background:var(--safety);border-color:var(--safety);box-shadow:3px 3px 0 var(--ink)}
.action-secondary{color:var(--ink);background:#fff}.action-secondary:hover{background:var(--surface);box-shadow:3px 3px 0 var(--ink);transform:translate(-1px,-1px)}
.view{display:none;max-width:1580px;margin:0 auto;padding:20px clamp(16px,2.6vw,38px) 50px;animation:axis-enter .22s ease-out}.view.active{display:block}@keyframes axis-enter{from{opacity:.6;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
.hero{position:relative;isolation:isolate;overflow:hidden;padding:clamp(18px,2.2vw,32px);border:var(--border);background:var(--base);box-shadow:var(--shadow-safety)}
.hero::before{content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;opacity:.065;background-image:linear-gradient(var(--ink) 1px,transparent 1px),linear-gradient(90deg,var(--ink) 1px,transparent 1px);background-size:35px 35px}
.hero::after{content:"LDM / 01";position:absolute;top:12px;right:14px;color:rgba(0,0,0,.2);font-size:8px;letter-spacing:.15em}
.hero-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(190px,250px) minmax(125px,155px);align-items:center;gap:clamp(12px,2vw,28px)}
.hero-copy{min-width:0}.hero-label{display:flex;align-items:center;gap:8px;color:var(--safety);font-size:8px;font-weight:600;letter-spacing:.17em;text-transform:uppercase}.hero-label-index{padding:3px 5px;border:1px solid var(--safety);color:var(--safety);font-size:8px}
.hero h1{max-width:100%;margin:15px 0 12px;font-family:"Space Grotesk",sans-serif;font-size:clamp(30px,4.15vw,63px);font-weight:700;line-height:.9;letter-spacing:-.07em;text-transform:uppercase;overflow-wrap:anywhere}
.hero p{max-width:570px;margin:0;color:#555;font-size:10px;line-height:1.8;overflow-wrap:anywhere}
.hero-visual{position:relative;isolation:isolate;min-width:0;min-height:183px;padding:9px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;border:2px solid var(--ink);background:var(--surface);box-shadow:4px 4px 0 var(--ink)}
.hero-visual-top,.hero-visual-bottom{display:flex;align-items:center;justify-content:space-between;gap:4px;color:#777;font-size:7px;letter-spacing:.05em;white-space:nowrap}.hero-visual-code{color:var(--safety);font-weight:600}
.axis-diagram{position:relative;display:grid;grid-template-columns:24px minmax(0,1fr);gap:7px;flex:1;min-height:110px;padding:10px 3px}
.axis-diagram-ruler{display:flex;flex-direction:column;justify-content:space-between;color:#999;font-size:6px;line-height:1}.axis-diagram-chart{position:relative;display:flex;align-items:flex-end;gap:5px;border-left:1px solid #aaa;border-bottom:1px solid #aaa;background:repeating-linear-gradient(to bottom,transparent 0 20px,rgba(0,0,0,.09) 20px 21px)}
.axis-diagram-chart i{flex:1;min-width:4px;background:var(--ink);border-top:3px solid var(--safety)}.axis-diagram-chart i:nth-child(3n){background:var(--safety);border-top-color:var(--ink)}
.axis-diagram-cross{position:absolute;right:2px;top:4px;color:var(--safety);font-size:17px;line-height:1}.axis-diagram-corner{position:absolute;width:9px;height:9px;border-color:var(--safety);border-style:solid}.axis-diagram-corner-a{left:30px;top:5px;border-width:2px 0 0 2px}.axis-diagram-corner-b{right:1px;bottom:7px;border-width:0 2px 2px 0}
.hero-visual-bottom{justify-content:center;gap:6px;color:#555;font-size:6px}.hero-visual-bottom i{color:var(--safety);font-style:normal}
.hero-actions{display:flex;flex-direction:column;align-items:stretch;gap:10px}.big-action{width:100%;min-height:43px;padding:0 10px;display:flex;align-items:center;justify-content:space-between;gap:8px;border:2px solid var(--ink);color:#fff;background:var(--ink);box-shadow:4px 4px 0 var(--safety);text-align:left;font-size:9px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.big-action:hover{transform:translate(-2px,-2px);color:#fff;background:var(--safety);border-color:var(--safety);box-shadow:6px 6px 0 var(--ink)}.button-arrow{font-size:15px}.hero-action-note{display:flex;align-items:center;gap:7px;color:#777;font-size:8px;text-transform:uppercase}.hero-action-led{width:5px;height:5px;flex:0 0 5px;background:var(--safety)}
.axis-ticker{margin:15px 0 14px;overflow:hidden;border-top:2px solid var(--ink);border-bottom:2px solid var(--ink);background:var(--ink);color:#fff}
.axis-ticker-track{width:max-content;min-height:38px;display:flex;align-items:center;gap:22px;padding:0 12px;animation:axis-ticker-scroll 42s linear infinite}.axis-ticker:hover .axis-ticker-track{animation-play-state:paused}
.axis-ticker-track span{font-family:"Space Grotesk",sans-serif;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}.axis-ticker-track b{color:var(--safety);font-size:13px}
@keyframes axis-ticker-scroll{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.layout{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:var(--gap);margin-top:0}.layout>.card[id="archive-card"]{grid-column:1/-1}.layout>.card:nth-child(2){grid-column:span 5}.layout>.card:nth-child(3){grid-column:span 7}
.card{position:relative;isolation:isolate;min-width:0;border:var(--border);background:#fff;box-shadow:var(--shadow)}.card::before{content:"";position:absolute;left:-1px;top:-1px;width:0;height:0;border:0}
.card::after{content:"";position:absolute;right:-1px;bottom:-1px;width:11px;height:11px;border-right:3px solid var(--safety);border-bottom:3px solid var(--safety);pointer-events:none}.card-pad{padding:clamp(13px,1.4vw,20px)}
.card-title-row{display:flex;align-items:center;justify-content:space-between;gap:10px}.card-title{font-family:"Space Grotesk",sans-serif;font-size:11px;font-weight:700;letter-spacing:.02em;text-transform:uppercase}.card-kicker{color:#888;font-size:7px;letter-spacing:.11em;text-transform:uppercase;text-align:right}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0;border:2px solid var(--ink)}.stat{min-width:0;padding:11px 9px;border-right:1px solid rgba(0,0,0,.18);border-bottom:1px solid rgba(0,0,0,.18);background:#fff}
.stat:nth-child(3n){border-right:0}.stat:nth-last-child(-n+3){border-bottom:0}.stat:nth-child(3n+2){background:var(--ink);color:#fff}.stat:nth-child(3n){background:var(--safety);color:#fff}
.stat-value{font-family:"Space Grotesk",sans-serif;font-size:clamp(18px,2.1vw,29px);font-weight:700;line-height:1;letter-spacing:-.06em;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.stat-label{margin-top:6px;color:inherit;opacity:.55;font-size:7px;letter-spacing:.12em;text-transform:uppercase;overflow-wrap:anywhere}
.status-row{display:flex;align-items:center;gap:8px;margin-top:12px;padding-top:10px;border-top:1px solid rgba(0,0,0,.14)}.status-dot{width:6px;height:6px;flex:0 0 6px;background:#aaa}.status-dot.running{background:var(--term-green);border:1px solid var(--ink);box-shadow:0 0 0 2px rgba(0,255,102,.16)}.status-dot.bad{background:var(--safety)}.status-text{color:#555;font-size:9px}
.layout>.card:nth-child(3){background:var(--ink);color:#fff;box-shadow:var(--shadow-safety)}.layout>.card:nth-child(3) .card-kicker{color:rgba(255,255,255,.45)}.layout>.card:nth-child(3) .job{border-color:rgba(255,255,255,.18)}
.job{min-height:112px;margin-top:10px;padding:11px;border:1px solid #ccc;background:var(--surface)}.job-empty{display:flex;min-height:88px;align-items:center;justify-content:center;color:#888;font-size:9px;text-align:center}
.job-head{display:flex;align-items:flex-start;justify-content:space-between;gap:9px}.job-title{font-family:"Space Grotesk",sans-serif;font-size:12px;font-weight:700;overflow-wrap:anywhere}.job-sub{margin-top:4px;color:#777;font-size:9px;overflow-wrap:anywhere}.job-project{max-width:38%;color:#999;font-size:8px;overflow-wrap:anywhere;text-align:right}
.job-progress{margin-top:17px}.progress-head{display:flex;justify-content:space-between;align-items:center;gap:10px;color:#999;font-size:8px}.progress-value{color:var(--safety);font-size:10px;font-weight:600}.progress-track{height:5px;margin-top:8px;overflow:hidden;border:1px solid rgba(0,0,0,.22);background:rgba(128,128,128,.16)}
.progress-fill{height:100%;background:var(--safety);transition:width .2s ease}.progress-fill.done{background:var(--term-green)}.progress-fill.bad{background:var(--safety)}.progress-fill.cancelled{background:var(--term-amber)}.progress-fill.indeterminate{animation:axis-progress 1.4s ease-in-out infinite}
@keyframes axis-progress{0%{transform:translateX(-120%)}100%{transform:translateX(300%)}}
.layout>.card:nth-child(3) .progress-track{border-color:rgba(255,255,255,.18);background:rgba(255,255,255,.1)}.layout>.card:nth-child(3) .job-title{color:#fff}.layout>.card:nth-child(3) .job-sub,.layout>.card:nth-child(3) .progress-head{color:rgba(255,255,255,.5)}.layout>.card:nth-child(3) .progress-value{color:var(--safety)}
.job-footer{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}.job-footer button,.job-footer a{min-height:28px;padding:0 8px;display:inline-flex;align-items:center;justify-content:center;border:1px solid var(--ink);background:#fff;color:var(--ink);font-size:8px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;text-decoration:none}
.layout>.card:nth-child(3) .job-footer button,.layout>.card:nth-child(3) .job-footer a{border-color:#fff;background:transparent;color:#fff}.job-footer a{border-color:var(--safety);background:var(--safety);color:#fff}
.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:27px 0 13px}.section-head h2{margin:0;font-family:"Space Grotesk",sans-serif;font-size:clamp(19px,2.4vw,32px);font-weight:700;line-height:.95;letter-spacing:-.05em;text-transform:uppercase}
.section-head h2::after{content:"";display:inline-block;width:7px;height:7px;margin:0 0 2px 7px;background:var(--safety)}.section-head p{margin:7px 0 0;color:#777;font-size:9px;line-height:1.75}
.helper{display:inline-flex;align-items:center;gap:6px;padding:6px 7px;border:1px solid #aaa;background:var(--surface);color:#666;font-size:8px;line-height:1.5}.helper b{color:var(--ink);font-weight:600}
#prepare-pipeline{margin-top:13px!important}.pipeline{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:10px}
.step{position:relative;grid-column:span 4;min-width:0;min-height:124px;padding:13px;display:flex;flex-direction:column;align-items:flex-start;gap:8px;border:2px solid var(--ink);background:#fff;box-shadow:4px 4px 0 var(--ink);text-align:left;overflow:hidden;transition:transform .12s,box-shadow .12s,background .12s}
.step::after{content:"";position:absolute;right:0;top:0;width:17px;height:17px;border-left:2px solid var(--safety);border-bottom:2px solid var(--safety);transform:translate(9px,-9px) rotate(45deg)}.step:hover{transform:translate(-3px,-3px);box-shadow:7px 7px 0 var(--ink);background:var(--surface)}
.step.primary{grid-column:1/-1;min-height:65px;padding:10px 13px;display:grid;grid-template-columns:140px minmax(0,1fr) minmax(130px,245px);align-items:center;gap:13px;background:var(--safety);border-color:var(--ink);color:#fff;box-shadow:4px 4px 0 var(--ink)}
.step.primary:hover{background:var(--ink);color:#fff;box-shadow:7px 7px 0 var(--safety)}.step.primary .step-number{align-self:center;padding:5px 6px;border:1px solid rgba(255,255,255,.5);color:#fff;text-align:center}.step.primary .step-name{font-size:13px}.step.primary .step-hint{max-width:280px;text-align:right;color:rgba(255,255,255,.73)}
.step-number{color:#888;font-size:8px;letter-spacing:.1em;text-transform:uppercase}.step-name{font-family:"Space Grotesk",sans-serif;font-size:12px;font-weight:700;letter-spacing:-.02em;text-transform:uppercase}.step-hint{color:#777;font-size:8px;line-height:1.6}
#prepare-pipeline .card-title-row{margin-top:17px!important;padding-top:13px;border-top:1px solid #ccc}#prepare-pipeline .card-title-row+div{margin-top:10px!important}
.collect-grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(250px,.85fr);gap:var(--gap);align-items:start}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.form-grid.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.field{display:flex;flex-direction:column;gap:6px;min-width:0}.field label{color:#555;font-size:8px;font-weight:600;letter-spacing:.1em;text-transform:uppercase}.field-note{color:#888;font-size:8px;line-height:1.6}
.field input,.field select,.field textarea,input,select,textarea{min-width:0;max-width:100%;color:var(--ink);background:#fff;border:1px solid #aaa;font-size:10px}.field input,.field select,.field textarea{width:100%;min-height:36px;padding:8px}
.field input:focus,.field select:focus,.field textarea:focus,input:focus,select:focus,textarea:focus{border-color:var(--safety);outline:1px solid var(--safety);outline-offset:0;box-shadow:none}
input[type=checkbox],input[type=radio]{accent-color:var(--safety)}.field input[type=range]{padding:0;min-height:24px;border:0;background:transparent;box-shadow:none}.field input[type=file]{padding:6px}
.check{display:flex;align-items:flex-start;gap:8px;padding:10px;border:1px solid #bbb;background:var(--surface);color:#555;font-size:9px}.check input{margin:3px 0 0;flex:0 0 auto}
.range{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:10px}.range-value{color:var(--safety);font-size:9px}
.save-bar{position:sticky;bottom:8px;z-index:12;display:flex;align-items:center;justify-content:flex-end;gap:7px;margin-top:17px;padding:9px 10px;border:2px solid var(--ink);background:rgba(255,255,255,.96);box-shadow:4px 4px 0 var(--safety)}.save-note{margin-right:auto;color:#777;font-size:8px}
.drop{position:relative;min-height:265px;padding:22px 14px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:2px dashed #999;background:var(--surface);text-align:center}.drop.drag{border-color:var(--safety);background:#fff1eb}
.drop-icon{width:45px;height:45px;display:grid;place-items:center;border:2px solid var(--ink);background:#fff;color:var(--safety);font-size:20px;box-shadow:3px 3px 0 var(--ink)}.drop strong{margin-top:13px;font-family:"Space Grotesk",sans-serif;font-size:12px;font-weight:700;text-transform:uppercase}.drop p{margin:5px 0 16px;color:#888;font-size:9px}
.drop-actions{display:flex;gap:7px;flex-wrap:wrap;justify-content:center}#files{display:none}.selection{margin-top:10px;color:#888;font-size:8px;overflow-wrap:anywhere}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(135px,1fr));gap:12px}.tile{min-width:0;padding:5px;border:2px solid var(--ink);background:#fff;box-shadow:3px 3px 0 var(--ink);transition:transform .12s,box-shadow .12s}
.tile:hover{transform:translate(-2px,-2px);box-shadow:5px 5px 0 var(--safety)}.tile img{display:block;width:100%;aspect-ratio:4/5;object-fit:cover;border:1px solid #bbb;background:var(--surface);filter:grayscale(.18);transition:filter .12s}.tile:hover img{filter:grayscale(0)}
.tile-meta{padding:7px 2px 2px}.tile-path{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--ink);font-size:8px}.tile-status{margin-top:3px;color:#888;font-size:7px;letter-spacing:.06em;text-transform:uppercase}
.tile-actions{margin-top:8px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}.tile-save,.tile-delete{min-width:0;min-height:27px;padding:0 5px;border:1px solid var(--ink);font-size:8px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.tile-save{color:#fff;background:var(--ink)}.tile-save:hover{color:#fff;background:var(--safety);border-color:var(--safety)}.tile-delete{color:var(--safety);background:#fff;border-color:var(--safety)}.tile-delete:hover{color:#fff;background:var(--safety)}
.empty{padding:24px 13px;border:1px dashed #aaa;color:#888;font-size:9px;text-align:center}
.details{margin-top:14px;border:2px solid var(--ink);background:var(--surface);overflow:hidden}.details summary{list-style:none;cursor:pointer;padding:12px;font-size:9px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}.details summary::-webkit-details-marker{display:none}
.details summary::after{content:"+";float:right;color:var(--safety);font-weight:700}.details[open] summary::after{content:"−"}.details-body{padding:0 12px 13px}
.output{margin-top:13px;padding:12px;display:flex;align-items:center;justify-content:space-between;gap:12px;border:2px solid var(--ink);background:var(--surface)}.output-name{font-family:"Space Grotesk",sans-serif;font-size:12px;font-weight:700;overflow-wrap:anywhere}.output-meta{margin-top:3px;color:#777;font-size:8px}
.output a{min-height:33px;padding:0 10px;display:inline-flex;align-items:center;justify-content:center;border:2px solid var(--ink);box-shadow:3px 3px 0 var(--safety);color:#fff;background:var(--ink);font-size:8px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;white-space:nowrap}.output a:hover{background:var(--safety);border-color:var(--safety);box-shadow:3px 3px 0 var(--ink)}
.toast-wrap{position:fixed;right:18px;bottom:18px;z-index:100;width:min(380px,calc(100vw - 36px));display:grid;gap:8px;pointer-events:none}.toast{pointer-events:auto;padding:11px 12px;border:2px solid var(--ink);background:#fff;color:var(--ink);box-shadow:4px 4px 0 var(--safety);font-size:9px;animation:axis-toast .18s ease-out}
.toast.ok{border-color:var(--ink)}.toast.bad{border-color:var(--safety);color:var(--safety)}@keyframes axis-toast{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
dialog{max-width:min(520px,calc(100vw - 28px));width:calc(100% - 28px);padding:0;border:3px solid var(--ink);color:var(--ink);background:#fff;box-shadow:8px 8px 0 var(--safety)}dialog::backdrop{background:rgba(0,0,0,.62);backdrop-filter:grayscale(1)}
.modal{overflow:hidden}.modal-head{padding:16px 18px;border-bottom:2px solid var(--ink)}.modal-title{font-family:"Space Grotesk",sans-serif;font-size:14px;font-weight:700;text-transform:uppercase}.modal-body{padding:17px 18px}.modal-body p{margin:0;color:#666;font-size:10px;line-height:1.6}
.modal-actions{display:flex;justify-content:flex-end;gap:7px;padding:12px 18px;border-top:1px solid #ccc}.modal-actions .danger{color:#fff;background:var(--safety);border-color:var(--safety)}.modal-actions .danger:hover{background:var(--ink);border-color:var(--ink)}
.preview{width:min(900px,calc(100vw - 26px));max-width:900px}.preview-body{padding:12px;background:#08080a}.preview-body img{display:block;width:100%;max-height:76vh;object-fit:contain;background:#08080a}.preview-meta{padding:13px 15px}.preview-name{font-size:9px;font-weight:600;overflow-wrap:anywhere}.preview-status{margin-top:3px;color:#888;font-size:8px}
@media(max-width:1320px){.app{grid-template-columns:224px minmax(0,1fr)}.axis-product{display:none}.axis-header-inner{gap:11px}.axis-nav button{min-width:79px;padding-inline:9px}.axis-header-right{gap:9px}.hero-grid{grid-template-columns:minmax(0,1fr) 190px 130px;gap:12px}.stats{gap:0}.stat{padding:9px 7px}.stat-value{font-size:19px}}
@media(max-width:1080px){.app{grid-template-columns:195px minmax(0,1fr)}.hero-grid{grid-template-columns:minmax(0,1fr) 180px}.hero-visual{grid-column:2;grid-row:1/3}.hero-actions{grid-column:1;grid-row:2;max-width:230px}.layout>.card:nth-child(2),.layout>.card:nth-child(3){grid-column:span 12}.stats{grid-template-columns:repeat(6,minmax(0,1fr))}.stat-value{font-size:17px}.collect-grid{grid-template-columns:1fr}.axis-clock{display:none}}
@media(max-width:820px){
 .axis-header{padding:0 12px}.axis-header-inner{min-height:50px;flex-wrap:wrap;gap:0}.axis-brand{margin-right:auto}.axis-logo-mark{width:31px;height:28px;flex-basis:31px}.axis-logo-mark img{width:29px;height:26px}.axis-wordmark{width:43px;height:16px}
 .axis-header-sep,.axis-product,.axis-header-right{display:none}.axis-nav{flex-basis:100%;order:3;width:100%;height:38px;margin:0 -12px;border-top:1px solid rgba(255,255,255,.12);border-left:0}.axis-nav button{flex:1;min-width:0;padding:0 7px;justify-content:center}
 .app{display:block;min-height:calc(100vh - 88px);border:0}.sidebar{position:relative;top:auto;height:auto;min-height:0;padding:14px 13px 12px;gap:10px;overflow:visible;border-right:0;border-bottom:3px solid var(--ink)}
 .side-section{flex:none;gap:8px;overflow:visible}.project-explorer{flex:none;max-height:125px;min-height:0;grid-template-columns:repeat(2,minmax(0,1fr));overflow:auto}.project-item{padding:8px}.side-status,.workspace{display:none}
 .topbar{position:relative;top:auto;min-height:61px;padding:10px 13px}.topbar-project{max-width:48vw;font-size:13px}.topbar-actions{gap:5px}.topbar-actions .new-top{display:none}
 .action-primary,.action-secondary{min-height:30px;padding:0 8px;font-size:8px}.view{padding:15px 13px 42px}.hero{padding:18px}.hero-grid{grid-template-columns:minmax(0,1fr) 150px;gap:12px}
 .hero-visual{grid-column:2;grid-row:1/3;min-height:145px;padding:7px}.axis-diagram{min-height:84px}.hero-actions{grid-column:1;grid-row:2}.hero h1{font-size:clamp(27px,5.6vw,42px)}
 .stats{grid-template-columns:repeat(3,minmax(0,1fr))}.stat-value{font-size:17px}.section-head{display:block;margin-top:22px}.section-head .helper{margin-top:9px}
 .pipeline{grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}.step{grid-column:span 3;min-height:105px}.step.primary{grid-column:1/-1;grid-template-columns:95px minmax(0,1fr);gap:8px}
 .step.primary .step-hint{grid-column:1/-1;max-width:none;text-align:left;margin-top:-2px}.form-grid,.form-grid.three{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.save-bar{bottom:8px}.gallery{grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.tile-actions{grid-template-columns:1fr}
}
@media(max-width:520px){
 .axis-brand-copy small{font-size:6px}.axis-nav button{font-size:8px;letter-spacing:.035em}.axis-nav .nav-key{font-size:7px;min-width:12px}
 .project-explorer{grid-template-columns:1fr;max-height:105px}.project-item{grid-template-columns:minmax(0,1fr) auto;align-items:center}.project-actions{flex-direction:column}
 .topbar{padding:10px}.topbar-project{max-width:43vw;font-size:11px}.topbar-actions .dirty{display:none}.view{padding:12px 10px 35px}.hero{padding:15px;box-shadow:4px 4px 0 var(--safety)}.hero-grid{grid-template-columns:1fr}
 .hero-visual{display:none}.hero-actions{grid-row:auto}.hero h1{font-size:clamp(29px,9vw,41px)}.hero p{font-size:9px}.axis-ticker{margin:11px 0}.axis-ticker-track{min-height:32px;gap:15px}.axis-ticker-track span{font-size:9px}
 .layout{gap:11px}.layout>.card:nth-child(2),.layout>.card:nth-child(3){grid-column:1/-1}.stats{grid-template-columns:repeat(3,minmax(0,1fr))}.stat{padding:8px 6px}.stat-value{font-size:16px}.stat-label{font-size:6px;letter-spacing:.06em}
 .job-head{flex-direction:column}.job-project{max-width:100%;text-align:left}.pipeline{grid-template-columns:1fr}.step,.step.primary{grid-column:1/-1;min-height:unset}.step.primary{grid-template-columns:75px minmax(0,1fr)}.step.primary .step-hint{font-size:8px}
 .form-grid,.form-grid.three{grid-template-columns:1fr}.save-bar{bottom:6px;padding:7px;gap:5px}.save-bar .save-note{display:none}.save-bar button{flex:1;padding:0 5px}.gallery{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.tile-actions{grid-template-columns:repeat(2,minmax(0,1fr))}
 .drop{min-height:200px}.output{align-items:flex-start;flex-direction:column}.output a{width:100%}
}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}

</style>
</head>
<body>
<header class="axis-header">
  <div class="axis-header-inner">
    <a class="axis-brand" href="#overview" aria-label="LDM — Your data is my data">
      <span class="axis-logo-mark"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHgAAABmBAMAAADylkgKAAAAMFBMVEXZ2dkmJiafn59TU1P+/v4BAQF1dXVWVlYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABK48vUAAAEdUlEQVR42u1Y309bVRz/3HMPyATa0oIySuG2MI2LYGGgZqIcuyZmMRsP+GrCy3z2b/LJTEmMJL5Yy1kxGoMbtNswRNre0iZsCWP9RRQ43PrAj7b00t2ehiwaz0MfTr+f+/2ez/l8f9yrBGG+BPeBshBg6OeZQDH/J3znbrD8mDt3hxsAL9lY9cZP45bBJScin3JDB0A04GNbBnbdKjjXb89Un8L1Uc7EjprsqZr7UfVOgJCsw8TQV7un96UgdL+29sfW1tal91+5H9NKl1v3rIUdLflLsUrGRLL70LHAagyJyQ0zB2IMoXBXl21zYCMUAvXyEtUsnZlAH2TxB0EkgM6oyyW+eXPUQfmgpTOrtC9yFeV94rOvv7t3k/daCfuQAGfY6afgAlbAlKLmfIcAqwsObyb4EWO1VifUiMTAhhlhZBaYnGdm2LJVaqIwWVZQ2XMSoZDhAAAV+Nk8B40JEeKp6meKiAZlXLFv8g+jQMmRcI2sVmeKP1HwJIDpYrTHTvPbMDLsWGE1CWhhiWQ3QIGl22h80StJByiEzVgxOZ1+PtAH0PbM1AKjuPeZ3SSzMPwC11HaAdVX8DyVCBve9ff2CWExSC0P1dXNNy7JgZWnGfJlCZJr9ROlp+SXBJdsBJqsZ9UsJS2LDARZeXgTnv8HS4BZM2CHPLhJbYtmwg7IQiMgUGTBcyAgumRSZUGAtyWpBgi4IuVaFAUIPqfXuAQ4pa2BgPxuzDaMFvGJQw0EGNSM2QYjF94J8j1AALqsGSMbjWCN3SKZZ0dZRZc76Fh/A+p4qJF5dtLcaTI1sbsTs5ZfIj2DezusPG3QodgOS+lnS3iknXSe7ZbGWA/io9WjiudPHhBnBmVjBhAPq5+46HQJu/Ok7Lf7nxy1zO6t7oNf1itNN3s93P1a5TwnBoe0SPa4+pAWcmpNXTF+Y6aCdeH/Meb9Vq3QQPhxEfFnmmkN8zzbIJM7p8ZEFIFpPlcWxiwzlp0VtKpu7Um527euX3ZeWTs+kmF7HSA3vz4eWCP564hHKl5USEsVGKTrh1ed7l93nQCwePUvAL8V+wAgbLuuhbfdlZSQlrOlN9D+gAfGRvnpRegMgEi7hoUnN3zORFoesLzh/kz61sowcHAfgAMQ6TxDuLPALEz6gZV0nk2GvojSIAAY73zFEO4cM6sHk9OrZhLMM0A8ZhALDBApM+mqHdWElYlbE9fypbbnz3N7vRGx9JbZ7EFazT0fXVxu5/TXdCbpqNPojtWyL9UlX1xdLrg/1ymO5PzyMXUIAKtUzrMBAPqcFFihOmCw76TABh+FGBKaFJhm1ZVctk4HpnXowAc8UFMALRNGR5zeeljQuiKI15/GCfhFKuy/B1YP0Cbrt+2lhS3+rVflb+KmiIUKe0EiKfzdK+uXvsyrmmri5VuX9h2BOt22LwnOxalCdblPDgazE8JH5RynhU6wrUp5NvyLoJjmUwPhRjXqQ2GqCCUI1VZmv14Rp9Wt3P0IShBIjkuEHXcetRtvtPG3d+EE8A9ChXnMV5vKRAAAAABJRU5ErkJggg==" alt="LDM briefcase logo"></span>
      <span class="axis-brand-copy"><span class="axis-wordmark"><img class="axis-wordmark-img" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAABEBAMAAADacQIYAAAAMFBMVEWfn59eXl4uLi7Pz8/+/v4BAQHn5+cWFhYAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABoyaNbAAAE00lEQVR42u2Yy08kRRzHP909cGJJI6yJEeLswHoi0CzqZUVnA+cNWeU+Uf8E/x0NiQcPJpvJ7s2wPDwaBhrDSVYs2V1PwDSPE3T3eOhXVXXPC/Riti4zXY9v/+r7/T2qy1jiv2omb6HfQndqJeBk15ofxjxp8O5M9s4Ta0ZApUEDPhnWljWZP2xnbLnBSAK9vwIT9ZXDCsZpOuOqYo6uw3yFCpg/nz9S1h9VR2bqcO6osDs2rFDxLwCMJdhaToY+cNNZwZi6avtVVXqaE/p8APaiORG0CQQp3Go2q6Wt+ujxpvT0XfTzUuOiFv3ukljN8dEiEKxP2dK81vqUA8EGYHwtgOCZZHe0JDIvbf5dwPz1zWcZdLSTDY04gjG4vgQYf7YI+K9sbfPmHSEvCEeB8Ex1Pq+zH71+cPALlCbKWn+4qjxe3Mivx2brUBqSemwA9V0CWblu0FZq4EPgjaYbs0Uq5qG7kFKqAx+nfmMuA4TyjCth9kOIlf19CPBpqrHwXRgsK1jv9QMtLS1tAobknULT8YIf5BmdoFvAU+l5OsGPm6PpKFQX6SfzWZuJY0SOcEfV0awxU75pUp0GPld3FcoqBnY9T4jTc3IPVWhJR0P1l/6stspQ8hIgq+QqOl5i4cFA+SbQYT0JuSwSypKKewChm4fugRRPJduTdTRrWtD1xTVKjphkQeb+SlDGA9wbl10j+XlKKOtoENhFVvf4Bi3NDMp59VJzkARTdMkf+XYfCDclHQVWRKytQl+D2QPfTkZIiBfBJjrW2FI168bEcrsBF6JQj2nwBR4c9QGtjj9ACokYNtbRIKjCdZ6QiJOuXB9kIeESw8Y6XtKKY7/AKtEpWWdZQ3aXcDWdJVKoJyp0qw/HdrWXz8ZVeEsz0VTiIN9EsZskbSjR0XfwYhODHCED5d4iMY45W82rRlbPgx49JFCeJmXuHCDNq5dciVsd3d2cLCImPFbRKoQOeyDazVbbSQxGOtpRspZ1M7OM4HYmI8FLIKLE8DIyynfydvSfVD0lA5iRmUbsOFY5Bx32Bmun0Ka4FgCGC5aHgW/fympzOXXrQNZDcBGrG9ZzXLvdHQ8IRD73fgXYCPazAMlb3R0+iEqg0v87MIstF6Dyv/RJagJhy0nwvOy0UAjdWvPaxrmeyIxNMAK95LblOlxx20IP6gnWA+tMcTH3BoQMADt6qlwALvmwx89/A7sQeqfosyQEHL7PHycKoS/aZMFqEbSVO6HYHaDd4vqwD/g1fczYTN7aC7Td3ujnIsvcEgehvPRJDtruIuI+sJ5YJ30eLejF/48OVteKOieqcD1XdOqGtOTKkWoWWNwqqLXmizHwX0tbTNkeLGviGoVWV5prLxq7eeTj3S8FPC/kLFxVVZTvnuDYBoYajMxVKukuxn98DMYawDuL92Wi4aTKwMajNB2l0T/+FzAYjRhLyW2H0u7ttNa/1Xh5P82b/m+LANvxnUowFpxH/XvfiOw2yVhKb4yUTZ5Nnmpd0kVLclkV3cPg3+VPW+6P7ltKMtE/0fJoGYx8ccCOeuLZbk61dc6S6witQNxzI2iz0aRpOYxUkpHTv4en6y0PwLAZ4fRCuYvz15qAYZuxL3qBF/cnaywnIqRyeMtycGLZbTzktsiM/o+ua/8BO/mLdAr9dMEAAAAASUVORK5CYII=" alt="LDM"></span><small>Your data is my data.</small></span>
    </a>
    <span class="axis-header-sep" aria-hidden="true"></span>
    <span class="axis-product">DATASET<br>OPERATIONS</span>
    <nav class="nav axis-nav" aria-label="primary">
      <button type="button" data-view="overview"><span class="nav-key">01</span><span>overview</span></button>
      <button type="button" data-view="collect"><span class="nav-key">02</span><span>collect</span></button>
      <button type="button" data-view="settings"><span class="nav-key">03</span><span>settings</span></button>
    </nav>
    <div class="axis-header-right">
      <div class="axis-online"><span class="axis-online-dot"></span><span>LOCAL / ONLINE</span></div>
      <div class="axis-clock" id="axis-clock">--:--:-- UTC</div>
    </div>
  </div>
</header>

<div class="app">
  <aside class="sidebar" aria-label="workspace navigation">
    <section class="side-section">
      <div class="side-head">
        <div class="side-title">dataset explorer</div>
        <button type="button" class="icon-button" id="refresh-projects" aria-label="refresh datasets" title="refresh datasets">↻</button>
      </div>
      <input id="project_filter" class="search" autocomplete="off" placeholder="search datasets..." aria-label="search datasets">
      <div class="project-explorer" id="project_explorer" aria-live="polite"></div>
      <button type="button" class="side-new" id="new-dataset"><span class="plus">+</span><span>new dataset</span><span class="shortcut">＋</span></button>
    </section>

    <div class="side-status"><span class="side-status-icon">⌘</span><div><b>PROJECT STORAGE</b><small>workspace-bound execution</small></div><span class="side-status-led"></span></div>
    <div class="workspace" id="project_workspace"></div>
  </aside>

  <main class="main">
    <header class="topbar">
      <div class="topbar-copy">
        <div class="eyebrow">ACTIVE DATASET <span class="eyebrow-slash">/</span> WORKSPACE</div>
        <div class="topbar-project" id="topbar-project">loading...</div>
      </div>
      <div class="topbar-signal" aria-label="dataset studio"><span class="topbar-signal-dot"></span><span>DATA STUDIO</span><span class="topbar-signal-divider"></span><code>LDM / 01</code></div>
      <div class="topbar-actions">
        <div class="dirty" id="dirty-badge" aria-live="polite"><span class="dirty-dot"></span><span>unsaved changes</span></div>
        <button type="button" class="action-secondary new-top" id="save-top" data-save>save</button>
        <button type="button" class="action-primary" id="new-top">new dataset</button>
      </div>
    </header>

    <section class="view active" data-view-panel="overview">
      <div class="hero">
        <div class="hero-grid">
          <div class="hero-copy">
            <div class="hero-label"><span class="hero-label-index">01</span> DATASET WORKSPACE</div>
            <h1 id="hero-name">loading...</h1>
            <p id="hero-description">load the active dataset to get started.</p>
          </div>
          <div class="hero-visual" aria-hidden="true">
            <div class="hero-visual-top"><span>PROCESS MONITOR</span><span class="hero-visual-code">LDM—01</span></div>
            <div class="axis-diagram">
              <div class="axis-diagram-ruler"><span>100</span><span>075</span><span>050</span><span>025</span><span>000</span></div>
              <div class="axis-diagram-chart"><i style="height:34%"></i><i style="height:52%"></i><i style="height:43%"></i><i style="height:78%"></i><i style="height:63%"></i><i style="height:91%"></i><i style="height:70%"></i><i style="height:100%"></i></div>
              <span class="axis-diagram-cross">+</span>
              <span class="axis-diagram-corner axis-diagram-corner-a"></span><span class="axis-diagram-corner axis-diagram-corner-b"></span>
            </div>
            <div class="hero-visual-bottom"><span>RAW / INPUT</span><i>→</i><span>WORK</span><i>→</i><span>EXPORT</span></div>
          </div>
          <div class="hero-actions">
            <button type="button" class="big-action" data-run-stage="prepare"><span>prepare dataset</span><span class="button-arrow">↗</span></button>
            <div class="hero-action-note"><span class="hero-action-led"></span> run the full pipeline</div>
          </div>
        </div>
      </div>

      <div class="axis-ticker" aria-hidden="true">
        <div class="axis-ticker-track">
          <span>DATASET ENGINEERING</span><b>✳</b><span>RAW IMAGE CONTROL</span><b>✳</b><span>DETERMINISTIC PROCESSING</span><b>✳</b><span>VLM CAPTION SYSTEMS</span><b>✳</b><span>KOHYA-READY EXPORT</span><b>✳</b>
          <span>DATASET ENGINEERING</span><b>✳</b><span>RAW IMAGE CONTROL</span><b>✳</b><span>DETERMINISTIC PROCESSING</span><b>✳</b><span>VLM CAPTION SYSTEMS</span><b>✳</b><span>KOHYA-READY EXPORT</span><b>✳</b>
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

      <section class="card card-pad" id="prepare-pipeline" style="margin-top:14px">
        <div class="section-head">
          <div>
            <h2>prepare dataset</h2>
            <p>run the full pipeline or execute one stage at a time when you need control.</p>
          </div>
          <span class="helper">one job at a time · project-bound execution</span>
        </div>
        <div class="pipeline">
          <button type="button" class="step primary" data-run-stage="prepare">
            <span class="step-number">all stages</span>
            <span class="step-name">prepare dataset</span>
            <span class="step-hint">run the stages enabled by your caption and tagger settings</span>
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
        <div class="card-title-row" style="margin-top:17px">
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
          <div class="form-grid" style="margin-top:12px">
            <div class="field">
              <label for="scrape_date_mode">date filter</label>
              <select id="scrape_date_mode">
                <option value="any">any date</option>
                <option value="year">specific year</option>
                <option value="range">custom range</option>
              </select>
              <div class="field-note" id="scrape_date_note">date filters are supported by Playwright Bing.</div>
            </div>
            <div class="field" id="scrape_date_year_field" hidden>
              <label for="scrape_year">year</label>
              <input id="scrape_year" type="number" min="1970" max="9998" step="1" inputmode="numeric" placeholder="2019">
            </div>
            <div class="field" id="scrape_date_from_field" hidden>
              <label for="scrape_date_from">from</label>
              <input id="scrape_date_from" type="date" min="1970-01-01">
            </div>
            <div class="field" id="scrape_date_to_field" hidden>
              <label for="scrape_date_to">to</label>
              <input id="scrape_date_to" type="date" min="1970-01-01">
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
              <option value="off">off · disabled</option>
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
            <label for="caption_runtime">VLM runtime</label>
            <select id="caption_runtime">
              <option value="llama_cpp">llama.cpp · GGUF</option>
              <option value="gemini">Google Gemini API</option>
              <option value="openai_compatible">OpenAI-compatible API</option>
            </select>
          </div>
          <div class="field">
            <label for="caption_chat_format">VLM chat format</label>
            <select id="caption_chat_format">
              <option value="mtmd">mtmd</option>
              <option value="qwen3.5">qwen3.5 · native mtmd</option>
              <option value="qwen2.5-vl">qwen2.5-vl</option>
              <option value="smolvlm">smolvlm · native mtmd</option>
              <option value="gemma4">gemma4</option>
            </select>
          </div>
          <div class="field">
            <label for="caption_max_tokens">max caption tokens · deterministic</label>
            <div class="range">
              <input id="caption_max_tokens" type="range" min="1" max="77" step="1">
              <span class="range-value" id="caption_tokens_value"></span>
            </div>
          </div>
        </div>
        <div class="form-grid" style="margin-top:12px">
          <div class="field">
            <label for="caption_model_path">VLM model path · llama.cpp only</label>
            <div style="display:flex;gap:8px"><input id="caption_model_path" autocomplete="off" placeholder="/path/to/model.gguf"><button type="button" class="action-secondary" id="caption_model_upload">upload</button></div><input id="caption_model_file" type="file" accept=".gguf" hidden><div class="field-note" id="caption_model_status">no model uploaded</div>
          </div>
          <div class="field">
            <label for="caption_mmproj_path">VLM mmproj GGUF · llama.cpp only</label>
            <div style="display:flex;gap:8px"><input id="caption_mmproj_path" autocomplete="off" placeholder="/path/to/mmproj.gguf"><button type="button" class="action-secondary" id="caption_mmproj_upload">upload</button></div><input id="caption_mmproj_file" type="file" accept=".gguf" hidden><div class="field-note" id="caption_mmproj_status">no mmproj uploaded</div>
          </div>
        </div>
        <div class="form-grid three" style="margin-top:12px">
          <div class="field">
            <label for="caption_api_key">API key</label>
            <div style="display:flex;gap:8px"><input id="caption_api_key" type="password" autocomplete="off" placeholder="enter once; blank keeps saved key"><button type="button" class="action-secondary" id="caption_api_key_set">save / reuse</button><button type="button" class="action-secondary" id="caption_api_key_clear">clear</button></div>
            <div class="field-note" id="caption_api_key_status">API keys are saved locally with owner-only file permissions (plaintext, not encrypted), shared across dataset folders and WebUI restarts.</div>
          </div>
          <div class="field">
            <label for="caption_api_model">API model</label>
            <select id="caption_api_model"><option value="">select model…</option></select>
          </div>
          <div class="field">
            <label for="caption_api_base_url">API base URL · OpenAI-compatible only</label>
            <input id="caption_api_base_url" autocomplete="off" placeholder="https://api.openai.com/v1">
          </div>
        </div>
        <div class="field-note" style="margin-top:9px">VLM captioning uses five structured fields: pose, camera angle, clothing, expression, and background. API captions combine camera angle with pose and normalize the result into four phrases; the backend injects the resolved trigger first.</div>
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
  <div class="modal-actions">
    <button type="button" class="action-secondary" id="preview-close">close</button>
    <button type="button" class="action-primary" id="preview-save">save to phone</button>
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
  projects:[],
  apiKeysSet:{gemini:false,openai_compatible:false},
  apiModelCache:{}
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
  document.querySelectorAll("#scraper_backend,#scrape_query,#scrape_count,#scrape_date_mode,#scrape_year,#scrape_date_from,#scrape_date_to,#name,#subject,#trigger,#preset,#mode,#width,#height,#quality,#face_crop,#face_padding,#caption_backend,#caption_runtime,#caption_chat_format,#caption_model_path,#caption_mmproj_path,#caption_api_key,#caption_api_key_set,#caption_api_key_clear,#caption_api_model,#caption_api_base_url,#caption_max_tokens,#person_check,#max_persons,#person_threshold,#min_subject_area,#min_subject_visibility,#text_check,#text_confidence,#text_min_area_ratio,#caption_model_upload,#caption_mmproj_upload,#caption_model_file,#caption_mmproj_file").forEach(el=>{
    if(el)el.disabled=busy;
  });
  const vlmOnly=$("caption_backend").value==="vlm";
  const taggerOff=$("preset").value==="off";
  ["tag","clean-tags"].forEach(stage=>{
    document.querySelectorAll('[data-run-stage="'+stage+'"]').forEach(button=>{
      button.disabled=busy||vlmOnly||taggerOff;
      button.title=vlmOnly?"not used with VLM captions":taggerOff?"tagger disabled":"";
    });
  });
}
function syncCaptionRuntimeUI(){
  const runtime=$("caption_runtime").value;
  const vlmEnabled=$("caption_backend").value==="vlm";
  const local=vlmEnabled&&runtime==="llama_cpp";
  const gemini=vlmEnabled&&runtime==="gemini";
  const api=vlmEnabled&&(gemini||runtime==="openai_compatible");
  $("caption_chat_format").disabled=!local;
  $("caption_mmproj_path").disabled=!local;
  $("caption_mmproj_upload").disabled=!local;
  $("caption_mmproj_file").disabled=!local;
  $("caption_model_path").disabled=!local;
  $("caption_model_upload").disabled=!local;
  $("caption_model_file").disabled=!local;
  $("caption_api_key").disabled=!api;
  $("caption_api_key_set").disabled=!api;
  $("caption_api_key_clear").disabled=!api;
  $("caption_api_model").disabled=!api;
  $("caption_api_base_url").disabled=!api||gemini;

  const showField=(id,visible)=>{
    const el=$(id);
    if(!el)return;
    const field=el.closest(".field");
    if(field)field.style.display=visible?"":"none";
  };
  showField("caption_chat_format",local);
  showField("caption_model_path",local);
  showField("caption_mmproj_path",local);
  showField("caption_api_key",api);
  showField("caption_api_model",api);
  showField("caption_api_base_url",runtime==="openai_compatible");
  showField("caption_max_tokens",runtime!=="gemini"&&runtime!=="openai_compatible");

  if(local){
    $("caption_model_path").placeholder="/path/to/model.gguf";
    $("caption_mmproj_path").placeholder="/path/to/mmproj.gguf";
    $("caption_model_status").textContent=$("caption_model_path").value
      ?"configured: "+$("caption_model_path").value
      :"no model uploaded";
    $("caption_mmproj_status").textContent=$("caption_mmproj_path").value
      ?"configured: "+$("caption_mmproj_path").value
      :"no mmproj uploaded";
    $("caption_api_key_status").textContent="local llama.cpp runtime selected";
  }else if(gemini){
    $("caption_api_key").placeholder="paste Gemini API key";
    $("caption_api_base_url").value="https://generativelanguage.googleapis.com/v1beta";
    $("caption_api_base_url").placeholder="https://generativelanguage.googleapis.com/v1beta";
    $("caption_model_status").textContent="unused by Gemini API";
    $("caption_mmproj_status").textContent="unused by Gemini API";
    $("caption_api_key_status").textContent=appState.apiKeysSet.gemini
      ? "Gemini key is saved locally. leave this field blank to keep using it."
      : "No saved Gemini key yet. enter it once and it will persist across projects and restarts.";
  }else{
    $("caption_api_key").placeholder="paste OpenAI-compatible API key";
    if(!$("caption_api_base_url").value.trim()
      || $("caption_api_base_url").value==="https://generativelanguage.googleapis.com/v1beta"){
      $("caption_api_base_url").value="https://api.openai.com/v1";
    }
    $("caption_model_status").textContent="unused by OpenAI-compatible API";
    $("caption_mmproj_status").textContent="unused by OpenAI-compatible API";
    $("caption_api_key_status").textContent=appState.apiKeysSet.openai_compatible
      ? "OpenAI-compatible key is saved locally. leave this field blank to keep using it."
      : "No saved key for this provider yet. enter it once and it will persist across projects and restarts.";
  }
}
async function loadVlmModels(){
  const runtime=$("caption_runtime").value;
  if(runtime!=="gemini"&&runtime!=="openai_compatible")return;
  // A custom API destination is trusted only after it has been saved in this project.
  if(runtime==="openai_compatible"&&appState.settingsDirty){
    $("caption_api_key_status").textContent="save settings before discovering models for this endpoint.";
    return;
  }
  const select=$("caption_api_model");
  const selected=select.dataset.selectedModel||select.value||"";
  const base=runtime==="openai_compatible"?$("caption_api_base_url").value.trim():"";
  const cacheKey=runtime+"|"+base;
  const fallbackModels=runtime==="gemini"
    ?[
      {id:"gemini-3.5-flash-lite",label:"Gemini 3.5 Flash-Lite"},
      {id:"gemini-3.8-flash",label:"Gemini 3.8 Flash"},
      {id:"gemini-2.5-flash-lite",label:"Gemini 2.5 Flash-Lite (legacy)"}
    ]
    :[];
  const renderModels=(models)=>{
    select.innerHTML="";
    if(!models.length){
      select.innerHTML='<option value="">no models available</option>';
      return;
    }
    for(const model of models){
      const option=document.createElement("option");
      option.value=String(model.id||model.name||"");
      option.textContent=String(model.label||model.id||model.name||"");
      select.appendChild(option);
    }
    const selectedExists=selected&&models.some(m=>String(m.id||m.name||"")===selected);
    if(selected&&!selectedExists){
      const option=document.createElement("option");
      option.value=selected;
      option.textContent=selected+" (saved, not currently listed by API)";
      select.appendChild(option);
    }
    if(selected)select.value=selected;
    select.dataset.selectedModel=select.value;
  };
  const stillCurrent=()=>(
    $("caption_runtime").value===runtime
    && (runtime!=="openai_compatible"||$("caption_api_base_url").value.trim()===base)
  );

  const cached=appState.apiModelCache[cacheKey];
  if(cached&&Date.now()-cached.at<600000){
    renderModels(cached.models);
    select.disabled=false;
    $("caption_api_key_status").textContent=appState.apiKeysSet[runtime]
      ? "API key saved locally. model list reused from cache."
      : "No saved API key for this provider.";
    return;
  }

  if(!appState.apiKeysSet[runtime]){
    renderModels(fallbackModels);
    const env=runtime==="gemini"?"GEMINI_API_KEY":"OPENAI_API_KEY";
    $("caption_api_key_status").textContent=env+" is not set; enter it once and click Save / reuse.";
    syncCaptionRuntimeUI();
    select.disabled=false;
    return;
  }

  // Show known Gemini choices immediately while provider discovery runs.
  renderModels(fallbackModels);
  select.disabled=false;
  $("caption_api_key_status").textContent="key saved locally; refreshing model list in the background…";
  try{
    const qs=new URLSearchParams({runtime:runtime});
    const d=await api("/api/vlm/models?"+qs.toString());
    const models=Array.isArray(d.models)?d.models:[];
    if(!stillCurrent())return;
    if(models.length){
      appState.apiModelCache[cacheKey]={at:Date.now(),models};
      renderModels(models);
    }else{
      renderModels(fallbackModels);
    }
    $("caption_api_key_status").textContent=d.cached
      ? "key saved locally; provider model list loaded from server cache."
      : "key saved locally; provider model list refreshed.";
  }catch(e){
    if(!stillCurrent())return;
    if(fallbackModels.length){
      renderModels(fallbackModels);
      $("caption_api_key_status").textContent="model discovery failed; known Gemini models are available, and the saved key is kept. "+e.message;
    }else{
      $("caption_api_key_status").textContent="model discovery failed; saved key is kept. "+e.message;
    }
  }
}
function syncScrapeDateUI(){
  const supported=$("scraper_backend").value==="playwright-bing";
  if(!supported&&$("scrape_date_mode").value!=="any"){
    $("scrape_date_mode").value="any";
    $("scrape_year").value="";
    $("scrape_date_from").value="";
    $("scrape_date_to").value="";
  }
  const mode=$("scrape_date_mode").value;
  $("scrape_date_mode").disabled=!supported||isBusyOnActiveProject();
  $("scrape_date_year_field").hidden=mode!=="year";
  $("scrape_date_from_field").hidden=mode!=="range";
  $("scrape_date_to_field").hidden=mode!=="range";
  $("scrape_date_note").textContent=supported
    ?"Bing date filtering uses the selected publication date range."
    :"date filters are only available with the Playwright · Bing backend.";
}
function applySettings(s,force=false){
  if(appState.settingsDirty&&!force)return;
  const captionState=s.caption||{};
  const keyState=captionState.api_keys_set||{};
  appState.apiKeysSet={
    gemini:!!(keyState.gemini??(captionState.runtime==="gemini"&&captionState.api_key_set)),
    openai_compatible:!!(keyState.openai_compatible??(captionState.runtime==="openai_compatible"&&captionState.api_key_set))
  };
  const project=s.project||{name:"",subject:"",trigger:""};
  $("name").value=project.name||"";
  $("subject").value=project.subject||"";
  $("trigger").value=project.trigger||"";
  $("preset").value=s.tagger.preset;
  $("caption_backend").value=s.caption.backend||"deterministic";
  $("caption_runtime").value=s.caption.runtime||"llama_cpp";
  $("caption_chat_format").value=s.caption.chat_format||"mtmd";
  $("caption_model_path").value=s.caption.model_path||"";
  $("caption_mmproj_path").value=s.caption.mmproj_path||"";
  $("caption_api_model").dataset.selectedModel=s.caption.api_model||"";
  $("caption_api_model").value=s.caption.api_model||"";
  $("caption_api_model").dataset.selectedModel=s.caption.api_model||"";
  $("caption_api_base_url").value=s.caption.api_base_url||"";
  $("caption_model_status").textContent=s.caption.model_path
    ?"configured: "+s.caption.model_path
    :"no model uploaded";
  $("caption_mmproj_status").textContent=s.caption.mmproj_path?"configured: "+s.caption.mmproj_path:"no mmproj uploaded";
  syncCaptionRuntimeUI();
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
  const scraper=s.scraper||{};
  $("scrape_date_mode").value=Number(scraper.year||0)>0?"year":(scraper.date_from&&scraper.date_to?"range":"any");
  $("scrape_year").value=Number(scraper.year||0)>0?String(scraper.year):"";
  $("scrape_date_from").value=scraper.date_from||"";
  $("scrape_date_to").value=scraper.date_to||"";
  syncScrapeDateUI();
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
  loadVlmModels();
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
    return '<div class="tile" data-preview-path="'+esc(src)+'" data-preview-name="'+esc(x.path)+'" data-preview-status="'+esc(x.status)+'"><img loading="lazy" src="'+esc(src)+'" alt="'+esc(x.path)+'"><div class="tile-meta"><div class="tile-path" title="'+esc(x.path)+'">'+esc(x.path)+'</div><div class="tile-status">'+esc(x.status)+'</div><div class="tile-actions"><button type="button" class="tile-save" data-image-save="'+esc(src)+'" data-image-name="'+esc(x.path)+'" title="save image to phone" aria-label="save image to phone">save</button><button type="button" class="tile-delete" data-image-delete="'+esc(x.id)+'" data-image-path="'+esc(x.path)+'"'+(busy?" disabled":"")+' aria-label="remove image">remove</button></div></div></div>';
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
  if(view==="prepare")view="overview";
  const allowed=new Set(["overview","collect","settings"]);
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
  const scrapeDateMode=$("scrape_date_mode").value;
  const scrapeYearText=$("scrape_year").value.trim();
  const scrapeYear=scrapeYearText?Number(scrapeYearText):0;
  const scrapeDateFrom=$("scrape_date_from").value;
  const scrapeDateTo=$("scrape_date_to").value;
  if(!Number.isFinite(width)||!Number.isFinite(height)||!Number.isFinite(quality)||!Number.isFinite(captionMaxTokens)||!Number.isFinite(facePadding)||!Number.isFinite(maxPersons)||!Number.isFinite(personThreshold)||!Number.isFinite(minArea)||!Number.isFinite(minVisibility)||!Number.isFinite(textConfidence)||!Number.isFinite(textMinArea)||!Number.isFinite(scrapeCount)){
    toast("please fill every numeric setting with a valid value.","bad");
    return false;
  }
  if(scrapeDateMode==="year"&&(!Number.isInteger(scrapeYear)||scrapeYear<1970||scrapeYear>9998)){
    toast("scraper year must be between 1970 and 9998.","bad");
    return false;
  }
  if(scrapeDateMode==="range"&&(!scrapeDateFrom||!scrapeDateTo||scrapeDateFrom>scrapeDateTo)){
    toast("choose a valid date range, with the start on or before the end.","bad");
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
        scrape_date_mode:scrapeDateMode,
        scrape_year:scrapeDateMode==="year"?scrapeYear:0,
        scrape_date_from:scrapeDateMode==="range"?scrapeDateFrom:(scrapeDateMode==="year"?scrapeYear+"-01-01":""),
        scrape_date_to:scrapeDateMode==="range"?scrapeDateTo:(scrapeDateMode==="year"?scrapeYear+"-12-31":""),
        mode:$("mode").value,
        width:width,
        height:height,
        quality:quality,
        caption_backend:$("caption_backend").value,
        caption_runtime:$("caption_runtime").value,
        caption_chat_format:$("caption_chat_format").value,
        caption_model_path:$("caption_model_path").value.trim(),
        caption_mmproj_path:$("caption_mmproj_path").value.trim(),
        caption_api_model:$("caption_api_model").value.trim(),
        caption_api_base_url:$("caption_api_base_url").value.trim(),
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
async function setVlmApiKey(){
  const runtime=$("caption_runtime").value;
  if(runtime!=="gemini"&&runtime!=="openai_compatible")return;
  const key=$("caption_api_key").value.trim();
  if(!key&&!appState.apiKeysSet[runtime]){
    toast("enter this provider's API key once first.","bad");
    return;
  }
  try{
    const d=await api("/api/vlm/api-key",{
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({runtime:runtime,api_key:key})
    });
    $("caption_api_key").value="";
    appState.apiKeysSet[runtime]=!!d.key_set;
    appState.apiModelCache={};
    $("caption_api_key_status").textContent=d.saved
      ? (key?"API key saved locally with owner-only file permissions.":"saved API key kept; nothing needed re-entering.")
      : "no saved API key for this provider.";
    void loadVlmModels();
    toast(key?"API key saved for future sessions":"saved API key reused","ok");
  }catch(e){
    $("caption_api_key_status").textContent=e.message;
    toast(e.message,"bad");
  }
}
async function clearVlmApiKey(){
  const runtime=$("caption_runtime").value;
  if(runtime!=="gemini"&&runtime!=="openai_compatible")return;
  try{
    const d=await api("/api/vlm/api-key",{
      method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({runtime:runtime,clear:true})
    });
    $("caption_api_key").value="";
    appState.apiKeysSet[runtime]=false;
    appState.apiModelCache={};
    $("caption_api_key_status").textContent="API key removed from the local credential file and server process.";
    $("caption_api_model").innerHTML='<option value="">select model…</option>';
    syncCaptionRuntimeUI();
    toast("API key cleared","ok");
  }catch(e){
    toast(e.message,"bad");
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
async function uploadVlmFile(kind,inputId,pathId,statusId){
  if(!appState.activeProjectPath){toast("create or open a dataset first.","bad");return;}
  if(isBusyOnActiveProject()){toast("this dataset is busy. wait for the current job to finish.","bad");return;}
  const input=$(inputId);
  const file=input.files[0];
  if(!file){toast("choose a GGUF file first.","bad");return;}
  if(!file.name.toLowerCase().endsWith(".gguf")){toast("only .gguf files are accepted.","bad");input.value="";return;}
  const form=new FormData();
  form.append("file",file);
  try{
    setStatus("uploading "+file.name+"...","running");
    const d=await api("/api/vlm/upload/"+encodeURIComponent(kind),{method:"POST",body:form});
    $(pathId).value=d.path;
    $(statusId).textContent="uploaded: "+d.path;
    input.value="";
    markDirty();
    updateBusyUI();
    setStatus("uploaded "+file.name,"ok");
    toast("uploaded "+file.name,"ok");
  }catch(e){
    setStatus(e.message,"bad");
    toast(e.message,"bad");
    input.value="";
  }
}
function updateFileSummary(){
  const n=$("files").files.length;
  $("file-summary").textContent=n?n+" file"+(n===1?"":"s")+" selected":"no files selected";
}
function openPreview(tile){
  const path=tile.dataset.previewPath;
  const name=tile.dataset.previewName||"";
  $("preview-image").src=path;
  $("preview-image").alt=name;
  $("preview-name").textContent=name;
  $("preview-status").textContent=tile.dataset.previewStatus||"";
  $("preview-save").dataset.imageSave=path;
  $("preview-save").dataset.imageName=name;
  $("preview-modal").showModal();
}
function saveImageToPhone(button){
  const src=button.dataset.imageSave||"";
  const path=button.dataset.imageName||"image";
  if(!src){toast("image path is missing","bad");return;}
  const filename=path.split("/").pop()||"image";
  const url=new URL(src,window.location.href);
  url.searchParams.set("download","true");
  const link=document.createElement("a");
  link.href=url.toString();
  link.download=filename;
  link.style.display="none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  toast("download started. check Downloads or your gallery app.","ok");
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
$("caption_model_upload").addEventListener("click",()=>$("caption_model_file").click());
$("caption_mmproj_upload").addEventListener("click",()=>$("caption_mmproj_file").click());
$("caption_backend").addEventListener("change",()=>{syncCaptionRuntimeUI();markDirty();});
$("caption_runtime").addEventListener("change",()=>{syncCaptionRuntimeUI();markDirty();loadVlmModels();});
$("caption_api_model").addEventListener("change",()=>{ $("caption_api_model").dataset.selectedModel=$("caption_api_model").value; markDirty(); });
$("caption_api_base_url").addEventListener("change",()=>{markDirty();if($("caption_runtime").value==="openai_compatible")loadVlmModels();});
$("caption_api_key_set").addEventListener("click",setVlmApiKey);
$("caption_api_key_clear").addEventListener("click",clearVlmApiKey);
$("caption_model_file").addEventListener("change",()=>uploadVlmFile("model","caption_model_file","caption_model_path","caption_model_status"));
$("caption_mmproj_file").addEventListener("change",()=>uploadVlmFile("mmproj","caption_mmproj_file","caption_mmproj_path","caption_mmproj_status"));
$("upload-button").addEventListener("click",upload);
$("files").addEventListener("change",updateFileSummary);
$("quality").addEventListener("input",()=>{$("qv").textContent=$("quality").value});
$("caption_max_tokens").addEventListener("input",()=>{$("caption_tokens_value").textContent=$("caption_max_tokens").value});
$("scraper_backend").addEventListener("change",syncScrapeDateUI);
$("scrape_date_mode").addEventListener("change",syncScrapeDateUI);
[
  "name","subject","trigger","preset","scraper_backend","scrape_query","scrape_count",
  "mode","width","height","quality","caption_backend","caption_runtime","caption_chat_format","caption_model_path","caption_mmproj_path","caption_api_model","caption_api_base_url","caption_max_tokens","face_crop","face_padding","person_check",
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
  const save=event.target.closest("[data-image-save]");
  if(save){
    saveImageToPhone(save);
    return;
  }
  const remove=event.target.closest("[data-image-delete]");
  if(remove){
    deleteImage(remove.dataset.imageDelete, remove.dataset.imagePath||"image");
    return;
  }
  const tile=event.target.closest("[data-preview-path]");
  if(tile)openPreview(tile);
});
$("preview-save").addEventListener("click",()=>saveImageToPhone($("preview-save")));
$("preview-close").addEventListener("click",()=>$("preview-modal").close());
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

function updateAxisClock(){
  const clock=$("axis-clock");
  if(!clock)return;
  const now=new Date();
  clock.textContent=String(now.getUTCHours()).padStart(2,"0")+":"+String(now.getUTCMinutes()).padStart(2,"0")+":"+String(now.getUTCSeconds()).padStart(2,"0")+" UTC";
}
updateAxisClock();
setInterval(updateAxisClock,1000);
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


def create_app(
    root: Path | str = ".",
    *,
    api_key_store_path: Path | str | None = None,
) -> FastAPI:
    app = FastAPI(title="LoRA Data Maker WebUI")

    @app.middleware("http")
    async def reject_cross_origin_browser_requests(request: Request, call_next):
        # The local UI has no login. Block cross-site browser requests globally so
        # websites cannot trigger jobs, mutate projects, upload files, or probe
        # provider APIs through the user's loopback service.
        if request.url.path.startswith("/api/"):
            fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
            origin = request.headers.get("origin", "").strip()
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if (
                (fetch_site and fetch_site not in {"same-origin", "none"})
                or (origin and origin.rstrip("/").lower() != expected.rstrip("/").lower())
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "cross-origin browser requests are not allowed"},
                )
        return await call_next(request)

    credential_path = (
        Path(api_key_store_path).expanduser().resolve()
        if api_key_store_path is not None
        else default_api_key_store_path()
    )
    for env_name, saved_key in load_api_keys(credential_path).items():
        if not os.environ.get(env_name, "").strip():
            os.environ[env_name] = saved_key
    model_list_cache: dict[
        tuple[str, str, str], tuple[float, list[dict[str, str]]]
    ] = {}
    model_list_cache_lock = threading.Lock()
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
                "caption": {
                    "backend": "deterministic",
                    "runtime": "llama_cpp",
                    "chat_format": "mtmd",
                    "model_path": "",
                    "mmproj_path": "",
                    "api_key_env": "",
                    "api_model": "",
                    "api_base_url": "",
                    "api_key_set": False,
                    "api_keys_set": {
                        "gemini": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
                        "openai_compatible": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                    },
                    "max_tokens": 77,
                },
                "scraper": {
                    "backend": "duckduckgo",
                    "query": "",
                    "count": 300,
                    "max_scrolls": 30,
                    "year": 0,
                    "date_from": "",
                    "date_to": "",
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
                "backend": "off"
                if str(tagger.get("backend", "wd14")).strip().lower() == "off"
                or str(tagger.get("preset", "sweetspot")).strip().lower() == "off"
                else str(tagger.get("backend", "wd14")),
                "preset": "off"
                if str(tagger.get("backend", "wd14")).strip().lower() == "off"
                or str(tagger.get("preset", "sweetspot")).strip().lower() == "off"
                else str(tagger.get("preset", "sweetspot")),
            },
            "caption": {
                "backend": str(caption.get("backend", "deterministic")),
                "runtime": str(caption.get("vlm", {}).get("runtime", "llama_cpp"))
                if isinstance(caption.get("vlm"), dict)
                else "llama_cpp",
                "chat_format": str(caption.get("vlm", {}).get("chat_format", "mtmd"))
                if isinstance(caption.get("vlm"), dict)
                else "mtmd",
                "model_path": str(caption.get("vlm", {}).get("model_path", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "mmproj_path": str(caption.get("vlm", {}).get("mmproj_path", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "api_key_env": str(caption.get("vlm", {}).get("api_key_env", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "api_model": str(caption.get("vlm", {}).get("api_model", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "api_base_url": str(caption.get("vlm", {}).get("api_base_url", ""))
                if isinstance(caption.get("vlm"), dict)
                else "",
                "api_key_set": (
                    bool(os.environ.get(
                        "GEMINI_API_KEY"
                        if str(caption.get("vlm", {}).get("runtime", "")).strip().lower() == "gemini"
                        else "OPENAI_API_KEY",
                        "",
                    ).strip())
                    if str(caption.get("vlm", {}).get("runtime", "")).strip().lower()
                    in {"gemini", "openai_compatible"}
                    else False
                ),
                "api_keys_set": {
                    "gemini": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
                    "openai_compatible": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
                },
                "max_tokens": int(caption.get("max_tokens", 77)),
            },
            "scraper": {
                "backend": str(cfg.get("scraper", {}).get("backend", "duckduckgo")),
                "query": str(cfg.get("scraper", {}).get("query", p.manifest.subject)),
                "count": int(cfg.get("scraper", {}).get("count", 300)),
                "max_scrolls": int(cfg.get("scraper", {}).get("max_scrolls", 30)),
                "year": int(cfg.get("scraper", {}).get("year", 0) or 0),
                "date_from": str(cfg.get("scraper", {}).get("date_from", "") or ""),
                "date_to": str(cfg.get("scraper", {}).get("date_to", "") or ""),
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

    @app.post("/api/vlm/api-key")
    def set_vlm_api_key(payload: dict[str, Any]):
        runtime = str(payload.get("runtime", "")).strip().lower()
        if runtime not in {"gemini", "openai_compatible"}:
            raise HTTPException(400, "API key can only be set for an API VLM runtime")
        env_name, _ = _api_runtime_settings(runtime)
        saved_keys = load_api_keys(credential_path)
        clear_requested = bool(payload.get("clear", False))
        value = str(payload.get("api_key", "")).strip()

        if clear_requested:
            saved_keys.pop(env_name, None)
            try:
                save_api_keys(credential_path, saved_keys)
            except OSError as exc:
                raise HTTPException(500, f"could not update local credential store: {exc}") from exc
            os.environ.pop(env_name, None)
            with model_list_cache_lock:
                for cache_key in list(model_list_cache):
                    if cache_key[0] == runtime:
                        model_list_cache.pop(cache_key, None)
            return {
                "ok": True,
                "runtime": runtime,
                "env": env_name,
                "key_set": False,
                "saved": False,
                "cleared": True,
            }

        if value:
            saved_keys[env_name] = value
            try:
                save_api_keys(credential_path, saved_keys)
            except OSError as exc:
                raise HTTPException(500, f"could not save API key locally: {exc}") from exc
            os.environ[env_name] = value
            with model_list_cache_lock:
                for cache_key in list(model_list_cache):
                    if cache_key[0] == runtime:
                        model_list_cache.pop(cache_key, None)
            return {
                "ok": True,
                "runtime": runtime,
                "env": env_name,
                "key_set": True,
                "saved": True,
                "cleared": False,
            }

        existing_key = os.environ.get(env_name, "").strip()
        if not existing_key and saved_keys.get(env_name):
            existing_key = saved_keys[env_name]
            os.environ[env_name] = existing_key
        elif existing_key and not saved_keys.get(env_name):
            # If a key was supplied by the shell environment, an explicit
            # Save / reuse click opts it into persistent WebUI storage too.
            saved_keys[env_name] = existing_key
            try:
                save_api_keys(credential_path, saved_keys)
            except OSError as exc:
                raise HTTPException(500, f"could not save API key locally: {exc}") from exc
        return {
            "ok": True,
            "runtime": runtime,
            "env": env_name,
            "key_set": bool(existing_key),
            "saved": bool(saved_keys.get(env_name)),
            "cleared": False,
        }


    @app.get("/api/vlm/models")
    def vlm_models(runtime: str = "gemini"):
        selected = runtime.strip().lower()
        if selected not in {"gemini", "openai_compatible"}:
            raise HTTPException(400, "unsupported API VLM runtime")

        # Never trust a request query parameter as an outbound credential destination.
        # OpenAI-compatible discovery must use the URL explicitly saved in the active
        # project's config, not a URL supplied by whichever site calls this GET route.
        if selected == "openai_compatible":
            current_root = active_root()
            if current_root is None:
                raise HTTPException(
                    409,
                    "create or open a dataset and save the API base URL before model discovery",
                )
            current_config = load_config(current_root / "config.toml")
            caption_config = current_config.get("caption", {})
            vlm_config = caption_config.get("vlm", {}) if isinstance(caption_config, dict) else {}
            saved_base_url = (
                str(vlm_config.get("api_base_url", "")).strip()
                if isinstance(vlm_config, dict)
                else ""
            )
            env_name, api_base = _api_runtime_settings(selected, saved_base_url)
        else:
            env_name, api_base = _api_runtime_settings(selected)

        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise HTTPException(
                400,
                f"{env_name} is not set; enter the API key in WebUI first",
            )

        cache_key = (
            selected,
            api_base,
            hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
        )
        now = time.monotonic()
        with model_list_cache_lock:
            cached = model_list_cache.get(cache_key)
            if cached and now - cached[0] < 600:
                return {
                    "runtime": selected,
                    "env": env_name,
                    "key_set": True,
                    "models": [dict(item) for item in cached[1]],
                    "cached": True,
                }

        try:
            import requests
        except ImportError as exc:
            raise HTTPException(
                500,
                "the requests package is required for API VLM runtimes",
            ) from exc

        try:
            if selected == "gemini":
                models: list[dict[str, str]] = []
                page_token = ""
                for _ in range(3):
                    params = {"pageSize": "100"}
                    if page_token:
                        params["pageToken"] = page_token
                    response = requests.get(
                        f"{api_base}/models",
                        headers={"x-goog-api-key": api_key},
                        params=params,
                        timeout=12,
                    )
                    if not response.ok:
                        detail = response.text.strip()[:500] or f"HTTP {response.status_code}"
                        raise HTTPException(502, f"Gemini model list failed: {detail}")
                    data = response.json()
                    for item in data.get("models", []):
                        if not isinstance(item, dict):
                            continue
                        methods = item.get("supportedGenerationMethods", [])
                        if "generateContent" not in methods:
                            continue
                        name = str(item.get("name", "")).strip()
                        if not name:
                            continue
                        model_id = name.removeprefix("models/")
                        models.append({
                            "id": model_id,
                            "label": str(item.get("displayName") or model_id),
                        })
                    page_token = str(data.get("nextPageToken", "") or "").strip()
                    if not page_token:
                        break
                models.sort(key=lambda item: item["label"].lower())
                with model_list_cache_lock:
                    model_list_cache[cache_key] = (
                        time.monotonic(),
                        [dict(item) for item in models],
                    )
                return {
                    "runtime": selected,
                    "env": env_name,
                    "key_set": True,
                    "models": models,
                    "cached": False,
                }

            models = []
            response = requests.get(
                f"{api_base}/models"
                if api_base.endswith("/v1") or api_base.endswith("/v1/")
                else f"{api_base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=12,
            )
            if not response.ok:
                detail = response.text.strip()[:500] or f"HTTP {response.status_code}"
                raise HTTPException(502, f"OpenAI-compatible model list failed: {detail}")
            data = response.json()
            raw_models = data.get("data", data.get("models", []))
            if not isinstance(raw_models, list):
                raise HTTPException(502, "OpenAI-compatible model list has an invalid format")
            for item in raw_models:
                if isinstance(item, dict):
                    model_id = str(item.get("id") or item.get("name") or "").strip()
                else:
                    model_id = str(item).strip()
                if model_id:
                    models.append({"id": model_id, "label": model_id})
            models.sort(key=lambda item: item["label"].lower())
            with model_list_cache_lock:
                model_list_cache[cache_key] = (
                    time.monotonic(),
                    [dict(item) for item in models],
                )
            return {
                "runtime": selected,
                "env": env_name,
                "key_set": True,
                "models": models,
                "cached": False,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"could not fetch VLM models: {exc}") from exc


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
        current_scrape_year = int(current_scraper.get("year", 0) or 0)
        current_date_from = str(current_scraper.get("date_from", "") or "")
        current_date_to = str(current_scraper.get("date_to", "") or "")
        inferred_date_mode = "year" if current_scrape_year else (
            "range" if current_date_from and current_date_to else "any"
        )
        scrape_date_mode = str(payload.get("scrape_date_mode", inferred_date_mode)).strip().lower()
        try:
            scrape_year = int(payload.get("scrape_year", current_scrape_year) or 0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "scraper year must be an integer") from exc
        scrape_date_from = str(payload.get("scrape_date_from", current_date_from) or "").strip()
        scrape_date_to = str(payload.get("scrape_date_to", current_date_to) or "").strip()
        if scrape_date_mode == "any":
            scrape_year, scrape_date_from, scrape_date_to = 0, "", ""
        elif scrape_date_mode == "year":
            if not 1970 <= scrape_year <= 9998:
                raise HTTPException(400, "scraper year must be between 1970 and 9998")
            scrape_date_from = f"{scrape_year:04d}-01-01"
            scrape_date_to = f"{scrape_year:04d}-12-31"
        elif scrape_date_mode == "range":
            scrape_year = 0
            if not scrape_date_from or not scrape_date_to:
                raise HTTPException(400, "provide both start and end dates")
            try:
                date_start = date.fromisoformat(scrape_date_from)
                date_end = date.fromisoformat(scrape_date_to)
            except ValueError as exc:
                raise HTTPException(400, "dates must use YYYY-MM-DD format") from exc
            if date_start < date(1970, 1, 1) or date_end < date(1970, 1, 1) or date_start > date_end:
                raise HTTPException(400, "choose a valid date range from 1970 onward")
        else:
            raise HTTPException(400, "invalid scraper date filter mode")
        if scrape_date_mode != "any" and scraper_backend != "playwright-bing":
            raise HTTPException(400, "date filters are only supported by the Playwright Bing backend")
        preset = preset.strip().lower()
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
        caption_runtime = str(
            payload.get("caption_runtime", current_vlm.get("runtime", "llama_cpp"))
        ).strip().lower()
        caption_chat_format = str(
            payload.get("caption_chat_format", current_vlm.get("chat_format", "mtmd"))
        ).strip().lower()
        caption_model_path = str(
            payload.get("caption_model_path", current_vlm.get("model_path", ""))
        ).strip()
        caption_mmproj_path = str(
            payload.get("caption_mmproj_path", current_vlm.get("mmproj_path", ""))
        ).strip()
        caption_api_key_env = str(
            payload.get("caption_api_key_env", current_vlm.get("api_key_env", ""))
        ).strip()
        if caption_runtime == "gemini":
            caption_api_key_env = "GEMINI_API_KEY"
        elif caption_runtime == "openai_compatible":
            caption_api_key_env = "OPENAI_API_KEY"
        caption_api_model = str(
            payload.get("caption_api_model", current_vlm.get("api_model", ""))
        ).strip()
        caption_api_base_url = str(
            payload.get("caption_api_base_url", current_vlm.get("api_base_url", ""))
        ).strip()
        if caption_backend not in {"deterministic", "vlm"}:
            raise HTTPException(400, "caption backend must be deterministic or vlm")
        if caption_runtime not in {"llama_cpp", "onnx", "gemini", "openai_compatible"}:
            raise HTTPException(400, "unsupported VLM runtime")
        if caption_runtime == "onnx":
            caption_chat_format = "smolvlm"
        elif caption_chat_format not in {"mtmd", "qwen2.5-vl", "gemma4", "qwen3.5", "smolvlm"}:
            raise HTTPException(400, "unsupported VLM chat format")
        if caption_backend == "vlm" and caption_runtime in {"gemini", "openai_compatible"}:
            if not caption_api_key_env:
                raise HTTPException(400, "API key environment variable is required")
            if not caption_api_model:
                raise HTTPException(400, "API model is required")
        if caption_backend == "vlm" and caption_runtime == "llama_cpp" and not caption_model_path:
            raise HTTPException(400, "VLM model path is required for llama.cpp")
        if caption_backend == "vlm" and caption_runtime == "llama_cpp" and not caption_mmproj_path:
            raise HTTPException(400, "VLM mmproj path is required for llama.cpp")

        tagger_disabled = preset == "off"
        resolved_preset = None
        if not tagger_disabled:
            try:
                resolved_preset = resolve_tagger_preset(preset)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        try:
            caption_config_from_dict(
                {
                    "backend": caption_backend,
                    "max_tokens": caption_max_tokens,
                    "vlm": {
                        "model_path": caption_model_path,
                        "mmproj_path": caption_mmproj_path,
                        "runtime": caption_runtime,
                        "chat_format": caption_chat_format,
                        "api_key_env": caption_api_key_env,
                        "api_model": caption_api_model,
                        "api_base_url": caption_api_base_url,
                    },
                },
                root=current_root,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        set_config(
            "project",
            {"name": name, "subject": subject, "trigger": trigger},
            current_root,
        )
        set_config(
            "processing",
            {
                "resize_mode": mode,
                "target_width": width,
                "target_height": height,
                "jpeg_quality": quality,
                "face_crop": face_crop,
                "face_padding": face_padding,
            },
            current_root,
        )
        set_config(
            "tagger",
            {
                "backend": "off"
                if tagger_disabled
                else ("joytag" if resolved_preset.kind == "joytag" else "wd14"),
                "preset": "off" if tagger_disabled else preset,
            },
            current_root,
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
                "runtime": caption_runtime,
                "chat_format": caption_chat_format,
                "api_key_env": caption_api_key_env,
                "api_model": caption_api_model,
                "api_base_url": caption_api_base_url,
            },
            current_root,
        )
        set_config(
            "scraper",
            {
                "backend": scraper_backend,
                "query": scrape_query,
                "count": scrape_count,
                "year": scrape_year,
                "date_from": scrape_date_from,
                "date_to": scrape_date_to,
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
        p = project(current_root)
        if caption_backend == "deterministic":
            reset_vlm_captions_for_deterministic(
                p.manifest,
                root=current_root,
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

    @app.post("/api/vlm/upload/{kind}")
    async def upload_vlm(kind: str, file: Annotated[UploadFile, File()]):
        current_root = require_active_root()
        if jobs.has_running_project(current_root):
            raise HTTPException(409, "this dataset is busy. wait for the current job to finish")
        if kind not in {"model", "mmproj"}:
            raise HTTPException(404, "unknown VLM file kind")

        filename = Path(file.filename or "").name
        if Path(filename).suffix.lower() != ".gguf":
            raise HTTPException(400, "VLM uploads must be .gguf files")

        models_dir = current_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        base = safe_component(Path(filename).stem) or kind
        target = models_dir / f"{base}.gguf"
        counter = 1
        while target.exists():
            target = models_dir / f"{base}_{counter}.gguf"
            counter += 1

        temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            with temp.open("wb") as out:
                while chunk := await file.read(4 * 1024 * 1024):
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, target)
        except OSError as exc:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(500, f"could not save VLM file: {exc}") from exc
        finally:
            await file.close()

        return {
            "ok": True,
            "kind": kind,
            "path": manifest_path(current_root, target),
            "filename": filename,
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
                date_from = str(sc.get("date_from", "") or "").strip() or None
                date_to = str(sc.get("date_to", "") or "").strip() or None
                year = int(sc.get("year", 0) or 0)
                if year:
                    if not 1970 <= year <= 9998:
                        raise RuntimeError("scraper.year must be between 1970 and 9998")
                    date_from, date_to = f"{year:04d}-01-01", f"{year:04d}-12-31"
                manager = ScraperManager(
                    backend=backend,
                    output_dir=ps.raw,
                    cache_dir=ps.cache,
                    timeout=timeout,
                    retries=retries,
                    delay=delay,
                    max_scrolls=max_scrolls,
                    date_from=date_from,
                    date_to=date_to,
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
                if not isinstance(tc, dict):
                    raise RuntimeError("config section [tagger] must be a table")
                if str(tc.get("backend", "wd14")).strip().lower() == "off" or str(tc.get("preset", "sweetspot")).strip().lower() == "off":
                    return "tag skipped: tagger disabled"
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
                tagger_cfg = cfg.get("tagger", {})
                if isinstance(tagger_cfg, dict) and (
                    str(tagger_cfg.get("backend", "wd14")).strip().lower() == "off"
                    or str(tagger_cfg.get("preset", "sweetspot")).strip().lower() == "off"
                ):
                    return "clean-tags skipped: tagger disabled"
                cc = cfg.get("caption", {})
                if isinstance(cc, dict) and str(cc.get("backend", "deterministic")).strip().lower() == "vlm":
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
                    progress=progress,
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
    def media(relative: str, download: bool = False):
        current_root = require_active_root()
        target = (current_root / relative).resolve()
        try: target.relative_to(current_root)
        except ValueError: raise HTTPException(403, "invalid media path")
        if not target.is_file(): raise HTTPException(404, "image not found")
        return FileResponse(
            target,
            media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream",
            filename=target.name if download else None,
        )

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