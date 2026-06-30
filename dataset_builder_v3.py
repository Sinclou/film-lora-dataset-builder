#!/usr/bin/env python3
"""
Dataset Builder v3 — Film Mode
Pipeline 100% local pour générer des datasets LoRA de style depuis des films.

Workflow :
  1. PySceneDetect → détecte les scènes du film
  2. Extraction d'une frame représentative par scène (centre de la scène)
  3. CLIP scoring → filtrage sémantique selon les catégories choisies
  4. JoyCaption → captions de training quality
  5. Export ZIP au format AI-Toolkit (image.jpg + image.txt)

Hardware cible : NVIDIA RTX 3090 (24 GB VRAM)
Modèles chargés à la demande pour ménager la VRAM (compatible ComfyUI en parallèle si <12 GB libres).
"""

import os, sys, json, base64, threading, webbrowser, tempfile, shutil, gc, time, traceback, subprocess
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify, send_file, Response, send_from_directory

# ── Config ────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
WORK_DIR = Path(tempfile.gettempdir()) / "dataset_builder_v3"
WORK_DIR.mkdir(exist_ok=True)
CACHE_DIR = APP_DIR / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Empêche les modèles HF d'aller polluer le home
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))

# État global des jobs (analyse asynchrone)
JOBS = {}  # job_id -> dict
JOBS_LOCK = threading.Lock()

# Hosts distants config
HOSTS_FILE = APP_DIR / "hosts.json"
MOUNTS_DIR = WORK_DIR / "mounts"
MOUNTS_DIR.mkdir(exist_ok=True)
ACTIVE_MOUNTS = {}  # host_name -> mount_path
MOUNTS_LOCK = threading.Lock()

# 💾 Destination des datasets exportés — configurable via variable d'env ou argument CLI
# Priorité : --output <dossier> > variable d'env DATASET_OUTPUT > dossier par défaut ~/datasets
import argparse as _argparse
_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--output", default=None)
_args, _ = _parser.parse_known_args()
_default_output = Path.home() / "datasets"
LORA_MAKER_DATASETS = Path(
    _args.output or os.environ.get("DATASET_OUTPUT", str(_default_output))
)

# Cache des modèles en mémoire (loaded on demand, unload sur trigger)
MODELS = {
    "clip": None,
    "joycaption": None,
}
MODELS_LOCK = threading.Lock()

app = Flask(__name__)

# ── Lazy imports (modèles ML chargés seulement quand on en a besoin) ──────────
def lazy_import_torch():
    import torch
    return torch

def lazy_import_clip():
    """Charge open_clip + le modèle ViT-L-14 (~900 MB VRAM)."""
    with MODELS_LOCK:
        if MODELS["clip"] is not None:
            return MODELS["clip"]
        import torch
        import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", cache_dir=str(CACHE_DIR)
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model = model.to(device).eval()
        MODELS["clip"] = {
            "model": model, "preprocess": preprocess,
            "tokenizer": tokenizer, "device": device,
        }
        return MODELS["clip"]

def unload_clip():
    with MODELS_LOCK:
        if MODELS["clip"]:
            del MODELS["clip"]
            MODELS["clip"] = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

def lazy_import_joycaption():
    """Charge JoyCaption Beta One (~12 GB VRAM en bf16)."""
    with MODELS_LOCK:
        if MODELS["joycaption"] is not None:
            return MODELS["joycaption"]
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model_id = "fancyfeast/llama-joycaption-beta-one-hf-llava"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=str(CACHE_DIR)
        ).to(device).eval()
        MODELS["joycaption"] = {
            "model": model, "processor": processor, "device": device,
        }
        return MODELS["joycaption"]

def unload_joycaption():
    with MODELS_LOCK:
        if MODELS["joycaption"]:
            del MODELS["joycaption"]
            MODELS["joycaption"] = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

# ── Pipeline ──────────────────────────────────────────────────────────────────
def detect_scenes(video_path, threshold=27.0):
    """Détecte les changements de plan avec PySceneDetect."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]

def extract_frames(video_path, scenes, out_dir, max_frames=None, frames_per_scene=1):
    """Extrait N frames par scène, réparties uniformément dans la scène."""
    import subprocess
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if max_frames and len(scenes) > max_frames:
        # On garde un échantillon réparti uniformément
        step = len(scenes) / max_frames
        scenes = [scenes[int(i * step)] for i in range(max_frames)]
    global_idx = 0
    for i, (start, end) in enumerate(scenes):
        duration = end - start
        margin = duration * 0.1
        eff_start = start + margin
        eff_end = end - margin
        if frames_per_scene == 1 or eff_end <= eff_start:
            timestamps = [(start + end) / 2]
        else:
            timestamps = [
                eff_start + (eff_end - eff_start) * j / (frames_per_scene - 1)
                for j in range(frames_per_scene)
            ]
        for j, ts in enumerate(timestamps):
            out = out_dir / f"frame_{i:04d}_{j:02d}.jpg"
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "3", str(out)
                ], capture_output=True, timeout=30, check=True)
                frames.append({
                    "idx": global_idx, "timestamp": ts, "path": str(out),
                    "scene_start": start, "scene_end": end,
                    "scene_duration": duration,
                    "scene_idx": i, "frame_in_scene": j,
                })
                global_idx += 1
            except subprocess.CalledProcessError:
                continue
    return frames

def assess_quality(frames):
    """Score chaque frame : sharpness + luminosity (anti-blur/anti-dark)."""
    import cv2
    import numpy as np
    for f in frames:
        try:
            img = cv2.imread(f["path"])
            if img is None:
                f["sharpness"] = 0.0
                f["brightness"] = 0.0
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Variance du Laplacien : haute = net, basse = flou
            f["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            f["brightness"] = float(gray.mean())
        except Exception:
            f["sharpness"] = 0.0
            f["brightness"] = 0.0
    return frames

def clip_score(frames, prompts, neg_prompts=None, batch_size=16):
    """Score chaque frame contre les prompts (CLIP)."""
    import torch
    from PIL import Image
    clip = lazy_import_clip()
    model, preprocess, tokenizer, device = (
        clip["model"], clip["preprocess"], clip["tokenizer"], clip["device"]
    )
    # Encode texts
    with torch.no_grad():
        text_pos = tokenizer(prompts).to(device)
        text_pos_feat = model.encode_text(text_pos)
        text_pos_feat = text_pos_feat / text_pos_feat.norm(dim=-1, keepdim=True)
        if neg_prompts:
            text_neg = tokenizer(neg_prompts).to(device)
            text_neg_feat = model.encode_text(text_neg)
            text_neg_feat = text_neg_feat / text_neg_feat.norm(dim=-1, keepdim=True)
        else:
            text_neg_feat = None

    # Encode images en batches
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        imgs = []
        valid_idx = []
        for j, f in enumerate(batch):
            try:
                img = Image.open(f["path"]).convert("RGB")
                imgs.append(preprocess(img))
                valid_idx.append(j)
            except Exception:
                f["clip_score"] = 0.0
        if not imgs:
            continue
        with torch.no_grad():
            img_tensor = torch.stack(imgs).to(device)
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            # Score = max(sim) sur les prompts positifs, moyenne pondérée
            sims = img_feat @ text_pos_feat.T  # (N, P)
            score = sims.max(dim=-1).values  # meilleur match
            if text_neg_feat is not None:
                sims_neg = img_feat @ text_neg_feat.T
                score = score - 0.5 * sims_neg.max(dim=-1).values
            score = score.cpu().tolist()
        for k, j in enumerate(valid_idx):
            batch[j]["clip_score"] = float(score[k])
    return frames

def caption_frame(frame_path, system_prompt=None):
    """Génère un caption JoyCaption pour une image."""
    import torch
    from PIL import Image
    jc = lazy_import_joycaption()
    model, processor, device = jc["model"], jc["processor"], jc["device"]
    img = Image.open(frame_path).convert("RGB")
    if system_prompt is None:
        system_prompt = (
            "You are a captioner for a video LoRA training dataset. "
            "Write ONE rich, precise descriptive sentence (30-50 words) in English. "
            "Include: subject and action, environment, lighting style and colors, "
            "camera angle and framing, cinematic mood. Be specific and visual. "
            "Output only the caption, no quotes, no preamble."
        )
    convo = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Write a training caption for this image."},
    ]
    convo_str = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[convo_str], images=[img], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=180, do_sample=False,
            suppress_tokens=None, use_cache=True,
        )
    out = out[:, inputs["input_ids"].shape[1]:]
    caption = processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return caption

# ── Job runner (analyse en background) ───────────────────────────────────────
def run_analysis_job(job_id, video_path, target_count, categories, quality_filters, caption_now, frames_per_scene=1):
    """Exécute le pipeline complet en background."""
    def update(stage, msg, pct=None, **kw):
        with JOBS_LOCK:
            JOBS[job_id]["stage"] = stage
            JOBS[job_id]["log"].append({"t": time.time(), "msg": msg, "stage": stage})
            if pct is not None:
                JOBS[job_id]["progress"] = pct
            JOBS[job_id].update(kw)
    try:
        update("scenes", "▶ Détection des scènes (PySceneDetect)…", 5)
        scenes = detect_scenes(video_path)
        update("scenes", f"✓ {len(scenes)} scènes détectées", 15)

        update("frames", f"▶ Extraction de {frames_per_scene} frame(s) par scène (max {target_count*4} scènes)…", 20)
        frames_dir = WORK_DIR / job_id / "frames"
        # On extrait 3-4x plus que la cible pour avoir de la marge au filtrage
        max_extract = min(len(scenes), max(target_count * 4, 60))
        frames = extract_frames(video_path, scenes, frames_dir, max_frames=max_extract, frames_per_scene=frames_per_scene)
        update("frames", f"✓ {len(frames)} frames extraites", 35)

        if quality_filters.get("avoid_blur") or quality_filters.get("avoid_dark"):
            update("quality", "▶ Analyse qualité (netteté + luminosité)…", 40)
            frames = assess_quality(frames)
            n_before = len(frames)
            all_frames_scored = list(frames)  # garde une copie pour le rescue
            kept = []
            for f in frames:
                ok = True
                if quality_filters.get("avoid_blur") and f.get("sharpness", 0) < 30:
                    ok = False
                if quality_filters.get("avoid_dark"):
                    b = f.get("brightness", 0)
                    if b < 12 or b > 245:
                        ok = False
                if ok:
                    kept.append(f)
            frames = kept
            update("quality", f"✓ Filtres qualité : {n_before} → {len(frames)} frames", 47)

            # Rescue : si on a moins que (target * 1.5), on repeche les moins pires des recalées
            min_needed = int(target_count * 1.5)
            if len(frames) < min_needed:
                rejected = [f for f in all_frames_scored if f not in frames]
                # Score combiné netteté + distance à la moyenne de luminosité (128)
                rejected.sort(
                    key=lambda f: (f.get("sharpness", 0) - abs(f.get("brightness", 0) - 128) * 0.5),
                    reverse=True,
                )
                rescue_n = min(min_needed - len(frames), len(rejected))
                for f in rejected[:rescue_n]:
                    f["rescued"] = True
                    frames.append(f)
                update("quality", f"⚠ Rescue : +{rescue_n} frames repechées (sharpness/brightness limite) pour atteindre la cible", 49)
            update("quality", f"✓ Pool final : {len(frames)} frames pour scoring", 50)

        if categories:
            update("clip", f"▶ Scoring CLIP contre {len(categories)} catégorie(s)…", 55)
            neg = quality_filters.get("clip_neg", [
                "blurry image, motion blur",
                "subtitle, text overlay, logo, watermark",
                "transition, fade to black",
            ])
            frames = clip_score(frames, categories, neg_prompts=neg)
            # Tri par score CLIP descendant
            frames.sort(key=lambda f: f.get("clip_score", 0), reverse=True)
            update("clip", f"✓ Frames triées par pertinence (top score: {frames[0].get('clip_score', 0):.2f})", 70)
            unload_clip()
        else:
            # Pas de catégorie → on garde l'ordre temporel mais on score quand même pour info
            for f in frames:
                f["clip_score"] = 0.0

        # On garde les top (target_count * 1.5) pour laisser le choix dans l'UI
        keep = min(int(target_count * 1.5), len(frames))
        frames = frames[:keep]
        update("select", f"✓ {len(frames)} frames présélectionnées (cible: {target_count})", 75)

        if caption_now:
            update("caption", f"▶ Captioning JoyCaption ({len(frames)} frames)…", 80)
            unload_clip()
            total = len(frames)
            for i, f in enumerate(frames):
                try:
                    f["caption"] = caption_frame(f["path"])
                except Exception as e:
                    f["caption"] = ""
                    f["caption_error"] = str(e)[:200]
                pct = 80 + int(18 * (i+1) / total)
                update("caption", f"  [{i+1}/{total}] {f['caption'][:60]}…", pct,
                       frames=frames)
            unload_joycaption()
            update("done", f"✓ Analyse complète — {len(frames)} frames prêtes", 100,
                   frames=frames, done=True)
        else:
            # Captioning sera fait à la demande sur les frames retenues
            for f in frames:
                f.setdefault("caption", "")
            update("done", f"✓ Frames extraites (captioning à la demande)", 100,
                   frames=frames, done=True)
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["error"] = f"{type(e).__name__}: {e}"
            JOBS[job_id]["traceback"] = traceback.format_exc()
            JOBS[job_id]["log"].append({
                "t": time.time(), "stage": "error",
                "msg": f"✗ ERREUR: {type(e).__name__}: {e}"
            })
            JOBS[job_id]["done"] = True

# ── Routes API ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Si index.html existe à côté du script, on le sert (nouveau design)
    new_html = APP_DIR / "index.html"
    if new_html.exists():
        return send_from_directory(str(APP_DIR), "index.html")
    return HTML

@app.route("/api/status")
def status():
    import shutil as _sh
    has_gpu = False
    gpu_info = "N/A"
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_info = torch.cuda.get_device_name(0)
            free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
            gpu_info += f" ({free_mem:.1f} GB free)"
    except Exception:
        pass
    return jsonify({
        "gpu": gpu_info,
        "has_gpu": has_gpu,
        "ffmpeg": bool(_sh.which("ffmpeg")),
        "models_loaded": {k: v is not None for k, v in MODELS.items()},
    })

@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    job_id = base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip("=")
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    ext = Path(f.filename).suffix or ".mp4"
    path = job_dir / f"source{ext}"
    f.save(str(path))
    with JOBS_LOCK:
        JOBS[job_id] = {
            "video_path": str(path), "filename": f.filename,
            "log": [], "frames": [], "done": False, "progress": 0,
            "stage": "uploaded",
        }
    return jsonify({"job_id": job_id, "filename": f.filename})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    job_id = data.get("job_id")
    if not job_id or job_id not in JOBS:
        return jsonify({"error": "Job ID invalide"}), 400
    target_count = int(data.get("target_count", 30))
    categories = data.get("categories", [])
    quality_filters = data.get("quality_filters", {})
    caption_now = bool(data.get("caption_now", True))
    frames_per_scene = max(1, min(10, int(data.get("frames_per_scene", 1))))
    video_path = JOBS[job_id]["video_path"]
    th = threading.Thread(
        target=run_analysis_job,
        args=(job_id, video_path, target_count, categories, quality_filters, caption_now, frames_per_scene),
        daemon=True,
    )
    th.start()
    return jsonify({"ok": True})

@app.route("/api/job/<job_id>")
def get_job(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    with JOBS_LOCK:
        j = dict(JOBS[job_id])
    # Ne renvoie pas le video_path (sensible)
    j.pop("video_path", None)
    return jsonify(j)

@app.route("/api/frame-image/<job_id>/<int:idx>")
def serve_frame(job_id, idx):
    if job_id not in JOBS:
        return "Not found", 404
    with JOBS_LOCK:
        frames = JOBS[job_id].get("frames", [])
    for f in frames:
        if f["idx"] == idx:
            return send_file(f["path"])
    return "Not found", 404

@app.route("/api/recaption", methods=["POST"])
def recaption():
    """Re-caption une frame spécifique (utile si l'utilisateur veut retenter)."""
    data = request.json or {}
    job_id = data.get("job_id")
    idx = int(data.get("idx", -1))
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    with JOBS_LOCK:
        frames = JOBS[job_id].get("frames", [])
    frame = next((f for f in frames if f["idx"] == idx), None)
    if not frame:
        return jsonify({"error": "Frame not found"}), 404
    try:
        caption = caption_frame(frame["path"])
        frame["caption"] = caption
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/caption-batch", methods=["POST"])
def caption_batch():
    """Lance le captioning JoyCaption sur une liste d'indices (background job).
    Permet de captionner les frames sélectionnées après triage manuel."""
    data = request.json or {}
    job_id = data.get("job_id")
    indices = data.get("indices", [])
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    if not indices:
        return jsonify({"error": "No indices provided"}), 400

    def do_batch_caption():
        with JOBS_LOCK:
            frames = JOBS[job_id].get("frames", [])
        targets = [f for f in frames if f["idx"] in set(indices)]
        total = len(targets)

        def update(msg, pct=None):
            with JOBS_LOCK:
                JOBS[job_id]["caption_progress"] = pct
                JOBS[job_id]["caption_msg"] = msg
                JOBS[job_id]["log"].append({"t": time.time(), "stage": "caption", "msg": msg})

        with JOBS_LOCK:
            JOBS[job_id]["captioning"] = True
        update(f"▶ Captioning JoyCaption ({total} frames)…", 0)
        try:
            for i, f in enumerate(targets):
                if f.get("caption"):
                    # Skip si déjà captionnée
                    continue
                try:
                    f["caption"] = caption_frame(f["path"])
                except Exception as e:
                    f["caption"] = ""
                    f["caption_error"] = str(e)[:200]
                pct = int(100 * (i+1) / total)
                preview = (f.get("caption") or "")[:60]
                update(f"  [{i+1}/{total}] {preview}…", pct)
            unload_joycaption()
            update(f"✓ Captioning terminé : {total} frames", 100)
        except Exception as e:
            update(f"✗ Erreur captioning : {e}")
        finally:
            with JOBS_LOCK:
                JOBS[job_id]["captioning"] = False

    threading.Thread(target=do_batch_caption, daemon=True).start()
    return jsonify({"ok": True, "started": True, "count": len(indices)})

@app.route("/api/caption-status/<job_id>")
def caption_status(job_id):
    """Retourne l'avancement du captioning batch."""
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    with JOBS_LOCK:
        j = JOBS[job_id]
        captioning = j.get("captioning", False)
        progress = j.get("caption_progress")
        msg = j.get("caption_msg", "")
        frames = j.get("frames", [])
    return jsonify({
        "captioning": captioning,
        "progress": progress,
        "msg": msg,
        "frames": [{"idx": f["idx"], "caption": f.get("caption", "")} for f in frames],
    })

@app.route("/api/update-caption", methods=["POST"])
def update_caption():
    """Édition manuelle d'un caption."""
    data = request.json or {}
    job_id = data.get("job_id")
    idx = int(data.get("idx", -1))
    caption = (data.get("caption") or "").strip()
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    with JOBS_LOCK:
        for f in JOBS[job_id].get("frames", []):
            if f["idx"] == idx:
                f["caption"] = caption
                return jsonify({"ok": True})
    return jsonify({"error": "Frame not found"}), 404

@app.route("/api/export-to-lora-maker", methods=["POST"])
def export_to_lora_maker():
    """Exporte directement dans /media/sinclu/RENDERMAT/LORA MAKER/datasets/<name>/
    sans passer par le download navigateur. Pratique pour gros datasets.
    Ajoute aussi un trigger_word optionnel au début de chaque .txt."""
    data = request.json or {}
    job_id = data.get("job_id")
    selected_idx = set(data.get("selected", []))
    name = (data.get("name") or "").strip()
    trigger_word = (data.get("trigger_word") or "").strip()
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    if not selected_idx:
        return jsonify({"error": "Aucune frame sélectionnée"}), 400
    if not LORA_MAKER_DATASETS.parent.exists():
        return jsonify({"error": f"RENDERMAT/LORA MAKER non monté : {LORA_MAKER_DATASETS}"}), 400
    with JOBS_LOCK:
        frames = JOBS[job_id].get("frames", [])
        filename = JOBS[job_id].get("filename", "dataset")
    if not name:
        name = Path(filename).stem.replace(" ", "_")[:40]
    # Sanitize
    import re
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    target_dir = LORA_MAKER_DATASETS / name
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = [f for f in frames if f["idx"] in selected_idx]
    base = name
    written = 0
    for i, f in enumerate(selected):
        out_jpg = target_dir / f"{base}_{i:03d}.jpg"
        out_txt = target_dir / f"{base}_{i:03d}.txt"
        try:
            shutil.copy2(f["path"], out_jpg)
            caption = f.get("caption", "").strip()
            if trigger_word and caption and not caption.lower().startswith(trigger_word.lower() + ","):
                caption = f"{trigger_word}, {caption}"
            elif trigger_word and not caption:
                caption = trigger_word
            out_txt.write_text(caption)
            written += 1
        except Exception as e:
            return jsonify({"error": f"Erreur sur frame {i}: {e}"}), 500
    # Metadata
    meta = {
        "source": filename,
        "builder": "dataset-builder-v3",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": written,
        "trigger_word": trigger_word,
        "frames": [{
            "name": f"{base}_{i:03d}",
            "timestamp_seconds": round(f["timestamp"], 3),
            "clip_score": round(f.get("clip_score", 0), 4),
        } for i, f in enumerate(selected)],
    }
    (target_dir / "dataset_info.json").write_text(json.dumps(meta, indent=2))
    return jsonify({
        "ok": True,
        "target": str(target_dir),
        "written": written,
        "trigger_word": trigger_word,
    })

@app.route("/api/lora-maker-status")
def lora_maker_status():
    """Vérifie si RENDERMAT/LORA MAKER est monté et accessible."""
    available = LORA_MAKER_DATASETS.parent.exists() and os.access(LORA_MAKER_DATASETS.parent, os.W_OK)
    free_gb = 0
    if available:
        try:
            stat = shutil.disk_usage(LORA_MAKER_DATASETS.parent)
            free_gb = stat.free / (1024**3)
        except Exception:
            pass
    return jsonify({
        "available": available,
        "path": str(LORA_MAKER_DATASETS),
        "free_gb": round(free_gb, 1),
    })

@app.route("/api/export", methods=["POST"])
def export():
    """Crée un ZIP avec image.jpg + image.txt pour chaque frame sélectionnée.
    Paramètres optionnels :
      - dataset_name : préfixe des fichiers (ex: 'alien1979_style' → alien1979_style_000.jpg)
      - trigger_word : ajouté en tête de chaque caption (ex: 'alien79_style')
    """
    import zipfile, re
    data = request.json or {}
    job_id = data.get("job_id")
    selected_idx = set(data.get("selected", []))
    dataset_name = (data.get("dataset_name") or "").strip()
    trigger_word = (data.get("trigger_word") or "").strip()
    if job_id not in JOBS:
        return jsonify({"error": "Not found"}), 404
    with JOBS_LOCK:
        frames = JOBS[job_id].get("frames", [])
        filename = JOBS[job_id].get("filename", "dataset")
    # Préfixe : nom dataset si fourni, sinon depuis le filename source
    if dataset_name:
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name)[:50]
    else:
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(filename).stem)[:50]
    selected = [f for f in frames if f["idx"] in selected_idx]
    if not selected:
        return jsonify({"error": "Aucune frame sélectionnée"}), 400

    out_zip = WORK_DIR / job_id / "dataset.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(selected):
            name = f"{base}_{i:03d}"
            zf.write(f["path"], f"{name}.jpg")
            caption = (f.get("caption") or "").strip()
            if trigger_word and caption:
                if not caption.lower().startswith(trigger_word.lower() + ","):
                    caption = f"{trigger_word}, {caption}"
            elif trigger_word and not caption:
                caption = trigger_word
            zf.writestr(f"{name}.txt", caption)
        meta = {
            "source": filename,
            "dataset_name": base,
            "trigger_word": trigger_word,
            "builder": "dataset-builder-v3",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": len(selected),
            "frames": [{
                "name": f"{base}_{i:03d}",
                "timestamp_seconds": round(f["timestamp"], 3),
                "scene_duration": round(f.get("scene_duration", 0), 3),
                "clip_score": round(f.get("clip_score", 0), 4),
                "sharpness": round(f.get("sharpness", 0), 2),
                "brightness": round(f.get("brightness", 0), 2),
                "caption": f.get("caption", ""),
            } for i, f in enumerate(selected)],
        }
        zf.writestr("dataset_info.json", json.dumps(meta, indent=2))
    return send_file(str(out_zip), as_attachment=True,
                     download_name=f"{base}.zip", mimetype="application/zip")

# ===== HOSTS / REMOTE BROWSING =====

def load_hosts_config():
    if not HOSTS_FILE.exists():
        return {"hosts": []}
    try:
        return json.loads(HOSTS_FILE.read_text())
    except Exception:
        return {"hosts": []}

def get_host(name):
    for h in load_hosts_config().get("hosts", []):
        if h.get("name") == name:
            return h
    return None

def ssh_command(host_cfg, remote_cmd):
    """Construit un ssh shellescape pour exécuter remote_cmd sur host_cfg."""
    key = os.path.expanduser(host_cfg.get("key", "~/.ssh/id_ed25519"))
    return [
        "ssh", "-i", key,
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        f"{host_cfg['user']}@{host_cfg['host']}",
        remote_cmd,
    ]

def check_host_alive(host_cfg):
    if host_cfg.get("type") == "local":
        return True
    try:
        out = subprocess.run(
            ssh_command(host_cfg, "echo ok"),
            capture_output=True, timeout=8
        )
        return out.returncode == 0 and b"ok" in out.stdout
    except Exception:
        return False

def browse_path(host_cfg, path):
    """Liste les dossiers/fichiers vidéo à un chemin donné (local ou SSH)."""
    video_exts = {".mkv", ".mp4", ".mov", ".avi", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv", ".flv"}
    if host_cfg.get("type") == "local":
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return {"error": f"Path not found: {path}"}
        folders, videos = [], []
        try:
            for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if child.name.startswith("."):
                    continue
                try:
                    if child.is_dir():
                        folders.append({"name": child.name, "path": str(child)})
                    elif child.is_file() and child.suffix.lower() in video_exts:
                        videos.append({
                            "name": child.name,
                            "path": str(child),
                            "size": child.stat().st_size,
                        })
                except PermissionError:
                    continue
        except PermissionError:
            return {"error": "Permission denied"}
        return {"path": path, "folders": folders, "videos": videos}
    else:
        # SSH browse via une commande shell unique qui sort du JSON-friendly
        # Format : TYPE\tNAME\tFULLPATH\tSIZE
        path_esc = path.replace("'", "'\\''")
        exts_find = " -o ".join([f"-iname '*{e}'" for e in video_exts])
        cmd = (
            f"cd '{path_esc}' 2>/dev/null && "
            "for f in */; do [ -d \"$f\" ] && printf 'D\\t%s\\t%s/%s\\t0\\n' \"${f%/}\" \"$PWD\" \"${f%/}\"; done; "
            f"find . -maxdepth 1 -type f \\( {exts_find} \\) -printf 'F\\t%f\\t%p\\t%s\\n' 2>/dev/null "
            "| sed 's|\\t\\./|\\t'\"$PWD\"'/|'"
        )
        try:
            out = subprocess.run(
                ssh_command(host_cfg, cmd),
                capture_output=True, text=True, timeout=20
            )
        except Exception as e:
            return {"error": f"SSH failed: {e}"}
        if out.returncode != 0:
            err = out.stderr[:200] if out.stderr else "unknown SSH error"
            return {"error": err}
        folders, videos = [], []
        for line in out.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            kind, name, full, size = parts[0], parts[1], parts[2], parts[3]
            if name.startswith("."):
                continue
            if kind == "D":
                folders.append({"name": name, "path": full})
            elif kind == "F":
                try:
                    sz = int(size)
                except Exception:
                    sz = 0
                videos.append({"name": name, "path": full, "size": sz})
        folders.sort(key=lambda x: x["name"].lower())
        videos.sort(key=lambda x: x["name"].lower())
        return {"path": path, "folders": folders, "videos": videos}

def ensure_mount(host_cfg):
    """Monte le host racine via sshfs si pas déjà monté. Renvoie le mountpoint local."""
    if host_cfg.get("type") == "local":
        return None
    name = host_cfg["name"]
    with MOUNTS_LOCK:
        if name in ACTIVE_MOUNTS:
            mp = ACTIVE_MOUNTS[name]
            if Path(mp).is_mount():
                return mp
        # Vérif sshfs dispo
        if not shutil.which("sshfs"):
            raise RuntimeError("sshfs non installé. Lance: sudo apt install -y sshfs")

        mp = MOUNTS_DIR / name
        # Nettoyage des mounts zombies d'une session précédente
        if mp.exists():
            # Tente un démontage forcé au cas où le mount serait orphelin
            for tool in ("fusermount3", "fusermount"):
                try:
                    subprocess.run([tool, "-uz", str(mp)], capture_output=True, timeout=5)
                except Exception:
                    pass
            # Si le dossier appartient à root (ancien mount), on le recrée
            try:
                st = mp.stat()
                if st.st_uid != os.getuid():
                    # On ne peut pas le supprimer en user, on essaie via une commande
                    subprocess.run(["rm", "-rf", str(mp)], capture_output=True, timeout=5)
            except Exception:
                pass
        mp.mkdir(exist_ok=True, parents=True)

        key = os.path.expanduser(host_cfg.get("key", "~/.ssh/id_ed25519"))
        cmd = [
            "sshfs",
            f"{host_cfg['user']}@{host_cfg['host']}:/",
            str(mp),
            "-o", f"IdentityFile={key}",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ro",
            "-o", "reconnect",
            "-o", "ServerAliveInterval=15",
            "-o", "default_permissions",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=15, check=True)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode()[:300] if e.stderr else ""
            raise RuntimeError(f"sshfs mount failed: {err}")
        ACTIVE_MOUNTS[name] = str(mp)
        return str(mp)

def remote_to_local_path(host_cfg, remote_path):
    """Convertit un chemin distant en chemin local accessible via sshfs."""
    if host_cfg.get("type") == "local":
        return remote_path
    mp = ensure_mount(host_cfg)
    # remote_path est absolu sur la machine distante : /media/sinclou/Films/...
    # mp est le mount du / distant : /tmp/.../mounts/macpro
    return mp + remote_path

def unmount_host(host_name):
    with MOUNTS_LOCK:
        mp = ACTIVE_MOUNTS.pop(host_name, None)
    if mp:
        for tool in ("fusermount3", "fusermount"):
            try:
                subprocess.run([tool, "-uz", mp], capture_output=True, timeout=10)
                break
            except Exception:
                continue

# Cleanup automatique à l'extinction
import atexit
def cleanup_all_mounts():
    for name in list(ACTIVE_MOUNTS.keys()):
        unmount_host(name)
atexit.register(cleanup_all_mounts)

@app.route("/api/hosts")
def list_hosts():
    cfg = load_hosts_config()
    result = []
    for h in cfg.get("hosts", []):
        alive = check_host_alive(h)
        result.append({
            "name": h.get("name"),
            "label": h.get("label"),
            "type": h.get("type"),
            "host": h.get("host"),
            "user": h.get("user"),
            "roots": h.get("roots", []),
            "alive": alive,
        })
    return jsonify({
        "hosts": result,
        "sshfs_available": bool(shutil.which("sshfs")),
    })

@app.route("/api/hosts/<name>/browse")
def browse_host(name):
    h = get_host(name)
    if not h:
        return jsonify({"error": "Host inconnu"}), 404
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "path manquant"}), 400
    result = browse_path(h, path)
    return jsonify(result)

@app.route("/api/select-remote-file", methods=["POST"])
def select_remote_file():
    """Sélectionne un fichier sur un host distant : monte le host (sshfs)
    et crée un job_id qui pointe sur le mountpath local."""
    data = request.json or {}
    host_name = data.get("host")
    remote_path = data.get("path")
    if not host_name or not remote_path:
        return jsonify({"error": "host + path requis"}), 400
    h = get_host(host_name)
    if not h:
        return jsonify({"error": "Host inconnu"}), 404
    try:
        local_access = remote_to_local_path(h, remote_path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    if not Path(local_access).exists():
        return jsonify({"error": f"Fichier introuvable après mount: {local_access}"}), 404
    # Créer un job_id
    job_id = base64.urlsafe_b64encode(os.urandom(6)).decode().rstrip("=")
    filename = Path(remote_path).name
    with JOBS_LOCK:
        JOBS[job_id] = {
            "video_path": local_access,
            "remote_path": remote_path,
            "host_name": host_name,
            "filename": filename,
            "log": [], "frames": [], "done": False, "progress": 0,
            "stage": "selected",
        }
    return jsonify({
        "job_id": job_id,
        "filename": filename,
        "local_access": local_access,
    })

@app.route("/api/unload-models", methods=["POST"])
def unload_models():
    unload_clip()
    unload_joycaption()
    return jsonify({"ok": True})

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset Builder v3 — Film Mode</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060606;--bg1:#0c0c0c;--bg2:#111;--bg3:#181818;
  --border:#1f1f1f;--amber:#d4a853;--amber2:#f0c870;
  --green:#4e9e6a;--red:#c0503a;--muted:#444;
  --text:#e8dcc8;--text2:#999;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);overflow:hidden}
header{position:fixed;top:0;left:0;right:0;z-index:100;height:52px;background:var(--bg1);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;gap:20px}
.logo{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:12px;font-weight:700;letter-spacing:.15em;color:var(--amber)}
.logo-dot{width:7px;height:7px;border-radius:50%;background:var(--amber);box-shadow:0 0 8px var(--amber);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.logo-sub{color:var(--muted);font-size:10px;letter-spacing:.1em}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.pill{font-family:var(--mono);font-size:10px;letter-spacing:.08em;padding:4px 10px;border-radius:20px;border:1px solid;white-space:nowrap}
.pill-ok{border-color:var(--green);color:var(--green)}
.pill-none{border-color:var(--muted);color:var(--muted)}
.pill-warn{border-color:var(--amber);color:var(--amber)}
.btn-key{font-family:var(--mono);font-size:9px;letter-spacing:.1em;padding:4px 10px;border-radius:3px;border:1px solid var(--muted);background:transparent;color:var(--muted);cursor:pointer}
.btn-key:hover{border-color:var(--amber);color:var(--amber)}

#app{margin-top:52px;height:calc(100vh - 52px);display:flex;overflow:hidden}

/* PANEL CONFIG (left) */
#config{width:340px;flex-shrink:0;background:var(--bg1);border-right:1px solid var(--border);padding:24px;overflow-y:auto;display:flex;flex-direction:column;gap:20px}
.cfg-sec{display:flex;flex-direction:column;gap:10px}
.cfg-lbl{font-family:var(--mono);font-size:10px;letter-spacing:.15em;color:var(--muted);text-transform:uppercase}
#dropzone{border:1.5px dashed var(--border);border-radius:6px;padding:24px;text-align:center;cursor:pointer;transition:all .15s}
#dropzone:hover{border-color:var(--amber)}
#dropzone.has-file{border-color:var(--green);border-style:solid}
.dz-icon{font-size:32px;line-height:1;margin-bottom:8px}
.dz-text{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.05em}
.dz-file{font-family:var(--mono);font-size:11px;color:var(--green);margin-top:6px;word-break:break-all}
.num-input{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:8px 10px;color:var(--text);font-family:var(--mono);font-size:13px;outline:none}
.num-input:focus{border-color:var(--amber)}
.cat-row{display:flex;gap:6px}
.cat-input{flex:1;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:7px 10px;color:var(--text);font-family:var(--mono);font-size:11px;outline:none}
.cat-input:focus{border-color:var(--amber)}
.cat-rm{padding:0 10px;background:transparent;border:1px solid var(--border);border-radius:4px;color:var(--muted);cursor:pointer;font-family:var(--mono)}
.cat-rm:hover{border-color:var(--red);color:var(--red)}
.cat-add{padding:7px;background:transparent;border:1px dashed var(--border);border-radius:4px;color:var(--muted);font-family:var(--mono);font-size:10px;cursor:pointer;text-align:center}
.cat-add:hover{border-color:var(--amber);color:var(--amber)}
.preset-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.preset-btn{padding:4px 9px;background:var(--bg);border:1px solid var(--border);border-radius:3px;color:var(--text2);font-family:var(--mono);font-size:9px;cursor:pointer}
.preset-btn:hover{border-color:var(--amber);color:var(--amber)}
.check-row{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--text)}
.check-row input[type=checkbox]{accent-color:var(--amber)}
#btn-analyze{margin-top:auto;padding:14px;background:var(--amber);border:none;border-radius:5px;color:#0d0d0d;font-family:var(--mono);font-size:12px;font-weight:700;letter-spacing:.12em;cursor:pointer}
#btn-analyze:hover{background:var(--amber2)}
#btn-analyze:disabled{background:var(--muted);cursor:not-allowed}

/* CENTER */
#center{flex:1;display:flex;flex-direction:column;min-width:0;background:var(--bg2);overflow:hidden}
#workspace{flex:1;overflow-y:auto;padding:20px}

/* Welcome */
#welcome{height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px;color:var(--muted);font-family:var(--mono)}
#welcome .big{font-size:64px;margin-bottom:20px;opacity:.4}
#welcome .t{font-size:14px;letter-spacing:.1em;color:var(--text);margin-bottom:12px}
#welcome .d{font-size:11px;line-height:1.8;max-width:520px}
.kbd{background:var(--bg);border:1px solid var(--border);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:10px}

/* Progress */
#progress-panel{display:none;padding:24px;background:var(--bg1);border-bottom:1px solid var(--border)}
#progress-panel.show{display:block}
.pb-wrap{width:100%;height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;margin-bottom:14px}
.pb-bar{height:100%;background:linear-gradient(90deg,var(--amber),var(--amber2));transition:width .3s}
.pb-msg{font-family:var(--mono);font-size:10px;color:var(--text);margin-bottom:8px}
#progress-log{max-height:120px;overflow-y:auto;font-family:var(--mono);font-size:10px;line-height:1.7;color:var(--muted);background:#020202;padding:10px 14px;border-radius:4px;border:1px solid var(--border)}
.log-stage{color:var(--amber);font-weight:600;text-transform:uppercase;font-size:9px;letter-spacing:.1em;margin-right:8px}

/* Grid */
.grid-hdr{padding:14px 20px;background:var(--bg1);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}
.grid-stats{font-family:var(--mono);font-size:11px;color:var(--text2)}
.grid-stats b{color:var(--amber)}
.grid-actions{display:flex;gap:8px}
.gb{padding:7px 14px;background:transparent;border:1px solid var(--border);border-radius:3px;color:var(--text2);font-family:var(--mono);font-size:10px;cursor:pointer;letter-spacing:.06em}
.gb:hover{border-color:var(--amber);color:var(--amber)}
.gb-p{background:var(--amber);border-color:var(--amber);color:#0d0d0d;font-weight:600}
.gb-p:hover{background:var(--amber2);border-color:var(--amber2);color:#0d0d0d}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;padding:20px}
.card{background:var(--bg1);border:1.5px solid var(--border);border-radius:6px;overflow:hidden;cursor:pointer;transition:all .15s;display:flex;flex-direction:column}
.card.sel{border-color:var(--amber);box-shadow:0 0 12px rgba(212,168,83,.3)}
.card-img-wrap{position:relative;background:#000;aspect-ratio:16/9;overflow:hidden}
.card-img{width:100%;height:100%;object-fit:cover;display:block}
.card-check{position:absolute;top:6px;left:6px;width:24px;height:24px;background:rgba(0,0,0,.7);border:1.5px solid rgba(255,255,255,.4);border-radius:4px;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:14px;color:transparent}
.card.sel .card-check{background:var(--amber);border-color:var(--amber);color:#0d0d0d}
.card-score{position:absolute;top:6px;right:6px;background:rgba(0,0,0,.75);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:9px;color:var(--amber)}
.card-time{position:absolute;bottom:6px;right:6px;background:rgba(0,0,0,.75);padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:9px;color:var(--text2)}
.card-body{padding:10px;flex:1;display:flex;flex-direction:column;gap:6px}
.card-caption{font-family:var(--mono);font-size:10px;line-height:1.5;color:var(--text);min-height:50px;max-height:90px;overflow-y:auto}
.card-caption.empty{color:var(--muted);font-style:italic}
.card-actions{display:flex;gap:6px;margin-top:auto}
.ca-btn{flex:1;padding:5px;background:transparent;border:1px solid var(--border);border-radius:3px;color:var(--muted);font-family:var(--mono);font-size:9px;cursor:pointer}
.ca-btn:hover{border-color:var(--amber);color:var(--amber)}

/* Edit modal */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:500;display:none;align-items:center;justify-content:center}
.overlay.show{display:flex}
.modal{background:var(--bg1);border:1px solid var(--border);border-radius:8px;width:680px;max-height:80vh;overflow:hidden;display:flex;flex-direction:column}
.mhd{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.mt{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--amber);letter-spacing:.1em}
.mx{background:none;border:none;color:var(--muted);cursor:pointer;font-size:18px}
.mbody{padding:20px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}
#edit-img{width:100%;border-radius:4px;background:#000;max-height:360px;object-fit:contain}
#edit-ta{width:100%;height:120px;resize:vertical;background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:12px;color:var(--text);font-family:var(--mono);font-size:12px;line-height:1.7;outline:none}
#edit-ta:focus{border-color:var(--amber)}
.mft{padding:14px 20px;border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;gap:10px}
.btn-p{padding:10px 20px;background:var(--amber);border:none;border-radius:4px;color:#0d0d0d;font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.08em;cursor:pointer}
.btn-s{padding:9px 16px;background:transparent;border:1px solid var(--border);border-radius:4px;color:var(--text2);font-family:var(--mono);font-size:10px;cursor:pointer}
.btn-s:hover{border-color:var(--amber);color:var(--amber)}

::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    DATASET BUILDER v3
    <span class="logo-sub">FILM MODE · LOCAL</span>
  </div>
  <div class="hdr-right">
    <div id="gpu-pill" class="pill pill-none">GPU ···</div>
    <div id="ff-pill" class="pill pill-none">ffmpeg ···</div>
    <button class="btn-key" onclick="unloadModels()" title="Libérer la VRAM occupée par les modèles">⏏ Décharger VRAM</button>
  </div>
</header>

<div id="app">
  <aside id="config">
    <div class="cfg-sec">
      <div class="cfg-lbl">1 · Film source</div>
      <div id="dropzone" onclick="document.getElementById('finput').click()" ondragover="event.preventDefault()" ondrop="onDrop(event)">
        <div class="dz-icon">🎬</div>
        <div class="dz-text">Glisse un film<br>ou clique pour choisir</div>
        <div class="dz-file" id="dz-file"></div>
      </div>
      <input type="file" id="finput" accept="video/*" style="display:none" onchange="onFile(event)">
    </div>

    <div class="cfg-sec">
      <div class="cfg-lbl">2 · Nombre d'images cible</div>
      <input type="number" class="num-input" id="target-count" value="30" min="5" max="200" step="5">
    </div>

    <div class="cfg-sec">
      <div class="cfg-lbl">3 · Frames par scène</div>
      <select class="num-input" id="frames-per-scene" style="width:100%;padding:6px 8px;cursor:pointer">
        <option value="1">1 — rapide (centre de scène)</option>
        <option value="2">2 — début + fin</option>
        <option value="3" selected>3 — début / milieu / fin</option>
        <option value="5">5 — panel complet</option>
        <option value="10">10 — très dense</option>
      </select>
      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:4px">
        Plus de frames par scène = meilleur panel mais analyse plus longue.
      </div>
    </div>

    <div class="cfg-sec">
      <div class="cfg-lbl">4 · Catégories (CLIP filter)</div>
      <div id="cat-list"></div>
      <button class="cat-add" onclick="addCategory()">+ Ajouter une catégorie</button>
      <div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:4px">Presets rapides :</div>
      <div class="preset-row">
        <button class="preset-btn" onclick="addPreset(['street scene at night','crowded street'])">rue</button>
        <button class="preset-btn" onclick="addPreset(['close-up of a face','portrait of a person'])">portrait</button>
        <button class="preset-btn" onclick="addPreset(['wide landscape','epic vista'])">paysage</button>
        <button class="preset-btn" onclick="addPreset(['interior scene','indoor room'])">intérieur</button>
        <button class="preset-btn" onclick="addPreset(['neon lights','colorful lighting'])">néon</button>
        <button class="preset-btn" onclick="addPreset(['two people talking','conversation scene'])">dialogue</button>
        <button class="preset-btn" onclick="addPreset(['action scene','dynamic motion'])">action</button>
        <button class="preset-btn" onclick="addPreset(['atmospheric mood','cinematic lighting'])">atmosphère</button>
      </div>
    </div>

    <div class="cfg-sec">
      <div class="cfg-lbl">4 · Filtres qualité</div>
      <label class="check-row"><input type="checkbox" id="avoid-blur" checked> Éviter les frames floues</label>
      <label class="check-row"><input type="checkbox" id="avoid-dark" checked> Éviter trop sombres/claires</label>
      <label class="check-row"><input type="checkbox" id="caption-now" checked> Captionner immédiatement (JoyCaption)</label>
    </div>

    <button id="btn-analyze" onclick="startAnalysis()" disabled>▶ ANALYSER LE FILM</button>
  </aside>

  <div id="center">
    <div id="progress-panel">
      <div class="pb-msg" id="pb-msg">Préparation…</div>
      <div class="pb-wrap"><div class="pb-bar" id="pb-bar" style="width:0%"></div></div>
      <div id="progress-log"></div>
    </div>
    <div id="workspace">
      <div id="welcome">
        <div class="big">🎞</div>
        <div class="t">DATASET BUILDER v3 — FILM MODE</div>
        <div class="d">
          Pipeline 100% local : <b>PySceneDetect</b> → <b>CLIP</b> → <b>JoyCaption</b><br><br>
          1. Glisse un film à gauche<br>
          2. Choisis combien d'images tu veux extraire<br>
          3. Ajoute des catégories pour que CLIP cible les bonnes scènes<br>
          4. Lance l'analyse — ~5-10 min sur RTX 3090<br>
          5. Sélectionne tes frames finales dans la grille<br>
          6. Export ZIP prêt pour <b>AI-Toolkit</b> (format LoRA)
        </div>
      </div>
    </div>
  </div>
</div>

<div class="overlay" id="edit-overlay">
  <div class="modal">
    <div class="mhd">
      <span class="mt">ÉDITION CAPTION</span>
      <button class="mx" onclick="closeEdit()">✕</button>
    </div>
    <div class="mbody">
      <img id="edit-img" src="">
      <textarea id="edit-ta" placeholder="Caption…"></textarea>
      <div style="font-family:var(--mono);font-size:10px;color:var(--muted)" id="edit-info"></div>
    </div>
    <div class="mft">
      <button class="btn-s" onclick="recaptionFrame()">🔄 Re-caption (JoyCaption)</button>
      <div style="display:flex;gap:8px">
        <button class="btn-s" onclick="closeEdit()">Annuler</button>
        <button class="btn-p" onclick="saveEdit()">✓ Enregistrer</button>
      </div>
    </div>
  </div>
</div>

<script>
let state = {
  job_id: null,
  filename: null,
  categories: [],
  frames: [],
  selected: new Set(),
  poll_timer: null,
  edit_idx: null,
};

const $ = id => document.getElementById(id);

async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    $('gpu-pill').className = 'pill ' + (d.has_gpu ? 'pill-ok' : 'pill-warn');
    $('gpu-pill').textContent = d.has_gpu ? d.gpu : 'GPU ✗';
    $('ff-pill').className = 'pill ' + (d.ffmpeg ? 'pill-ok' : 'pill-warn');
    $('ff-pill').textContent = d.ffmpeg ? 'ffmpeg ✓' : 'ffmpeg ✗';
  } catch(e) { console.error(e); }
}
setInterval(checkStatus, 5000);
checkStatus();

// ── File handling ──
function onFile(e) { if (e.target.files[0]) uploadFile(e.target.files[0]); }
function onDrop(e) {
  e.preventDefault();
  const f = [...e.dataTransfer.files].find(x => x.type.startsWith('video/'));
  if (f) uploadFile(f);
}
async function uploadFile(file) {
  $('dz-file').textContent = '⏳ Upload…';
  const fd = new FormData(); fd.append('file', file);
  try {
    const r = await fetch('/api/upload-video', { method:'POST', body:fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    state.job_id = d.job_id;
    state.filename = d.filename;
    $('dz-file').textContent = '✓ ' + d.filename;
    $('dropzone').classList.add('has-file');
    $('btn-analyze').disabled = false;
  } catch(e) {
    $('dz-file').textContent = '✗ ' + e.message;
  }
}

// ── Categories ──
function renderCategories() {
  const list = $('cat-list');
  list.innerHTML = '';
  state.categories.forEach((cat, i) => {
    const row = document.createElement('div');
    row.className = 'cat-row';
    row.innerHTML = `<input class="cat-input" value="${cat.replace(/"/g, '&quot;')}" onchange="updateCat(${i}, this.value)"><button class="cat-rm" onclick="rmCat(${i})">✕</button>`;
    list.appendChild(row);
  });
}
function addCategory() { state.categories.push(''); renderCategories(); setTimeout(() => { const inputs = document.querySelectorAll('.cat-input'); if (inputs.length) inputs[inputs.length-1].focus(); }, 10); }
function rmCat(i) { state.categories.splice(i, 1); renderCategories(); }
function updateCat(i, v) { state.categories[i] = v; }
function addPreset(prompts) {
  for (const p of prompts) if (!state.categories.includes(p)) state.categories.push(p);
  renderCategories();
}

// ── Analysis ──
async function startAnalysis() {
  if (!state.job_id) return;
  const cats = state.categories.filter(c => c.trim());
  const payload = {
    job_id: state.job_id,
    target_count: parseInt($('target-count').value || 30),
      frames_per_scene: parseInt($('frames-per-scene').value || 1),
    categories: cats,
    quality_filters: {
      avoid_blur: $('avoid-blur').checked,
      avoid_dark: $('avoid-dark').checked,
    },
    caption_now: $('caption-now').checked,
  };
  $('btn-analyze').disabled = true;
  $('progress-panel').classList.add('show');
  $('progress-log').innerHTML = '';
  $('workspace').innerHTML = '<div id="welcome"><div class="big">⏳</div><div class="t">ANALYSE EN COURS</div><div class="d">Cela peut prendre quelques minutes selon la longueur du film…</div></div>';
  try {
    const r = await fetch('/api/analyze', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    pollJob();
  } catch(e) {
    alert('Erreur: ' + e.message);
    $('btn-analyze').disabled = false;
  }
}

async function pollJob() {
  try {
    const r = await fetch('/api/job/' + state.job_id);
    const d = await r.json();
    $('pb-bar').style.width = (d.progress || 0) + '%';
    $('pb-msg').textContent = (d.stage || '').toUpperCase() + ' · ' + (d.progress || 0) + '%';
    // Refresh log
    const logEl = $('progress-log');
    const lastLogs = (d.log || []).slice(-12);
    logEl.innerHTML = lastLogs.map(l => `<div><span class="log-stage">${l.stage||''}</span>${l.msg||''}</div>`).join('');
    logEl.scrollTop = logEl.scrollHeight;
    if (d.error) {
      $('pb-msg').textContent = '✗ ' + d.error;
      $('btn-analyze').disabled = false;
      return;
    }
    if (d.done) {
      state.frames = d.frames || [];
      // Sélection auto des top N (cible)
      const target = parseInt($('target-count').value || 30);
      state.selected = new Set(state.frames.slice(0, target).map(f => f.idx));
      renderGrid();
      $('btn-analyze').disabled = false;
      return;
    }
    state.poll_timer = setTimeout(pollJob, 1500);
  } catch(e) {
    console.error(e);
    state.poll_timer = setTimeout(pollJob, 3000);
  }
}

function renderGrid() {
  const ws = $('workspace');
  ws.innerHTML = '';
  const hdr = document.createElement('div');
  hdr.className = 'grid-hdr';
  hdr.innerHTML = `
    <div class="grid-stats">
      <b id="sel-cnt">${state.selected.size}</b> / ${state.frames.length} sélectionnées
      <span style="color:var(--muted);margin-left:14px">cible: ${$('target-count').value}</span>
    </div>
    <div class="grid-actions">
      <button class="gb" onclick="selectAll()">Tout sélectionner</button>
      <button class="gb" onclick="selectNone()">Tout désélectionner</button>
      <button class="gb" onclick="selectTop()">Top ${$('target-count').value}</button>
      <button class="gb gb-p" onclick="exportDataset()">▶ EXPORT</button>
    </div>
  `;
  ws.appendChild(hdr);
  const grid = document.createElement('div');
  grid.id = 'grid';
  for (const f of state.frames) {
    const sel = state.selected.has(f.idx);
    const card = document.createElement('div');
    card.className = 'card' + (sel ? ' sel' : '');
    card.dataset.idx = f.idx;
    const score = f.clip_score !== undefined ? f.clip_score.toFixed(2) : '';
    const time = f.timestamp ? `${Math.floor(f.timestamp/60)}:${String(Math.floor(f.timestamp%60)).padStart(2,'0')}` : '';
    const cap = (f.caption || '').trim();
    card.innerHTML = `
      <div class="card-img-wrap">
        <img class="card-img" src="/api/frame-image/${state.job_id}/${f.idx}" loading="lazy">
        <div class="card-check">✓</div>
        ${score ? `<div class="card-score">CLIP ${score}</div>` : ''}
        ${time ? `<div class="card-time">${time}</div>` : ''}
      </div>
      <div class="card-body">
        <div class="card-caption ${cap ? '' : 'empty'}">${cap || '(pas de caption)'}</div>
        <div class="card-actions">
          <button class="ca-btn" onclick="event.stopPropagation();openEdit(${f.idx})">✎ Éditer</button>
          <button class="ca-btn" onclick="event.stopPropagation();recaptionFrameIdx(${f.idx})">🔄 Re-cap</button>
        </div>
      </div>
    `;
    card.onclick = () => toggleSelect(f.idx);
    grid.appendChild(card);
  }
  ws.appendChild(grid);
}

function toggleSelect(idx) {
  if (state.selected.has(idx)) state.selected.delete(idx);
  else state.selected.add(idx);
  document.querySelector(`.card[data-idx="${idx}"]`)?.classList.toggle('sel');
  const el = $('sel-cnt'); if (el) el.textContent = state.selected.size;
}
function selectAll() { state.selected = new Set(state.frames.map(f => f.idx)); renderGrid(); }
function selectNone() { state.selected = new Set(); renderGrid(); }
function selectTop() {
  const n = parseInt($('target-count').value || 30);
  state.selected = new Set(state.frames.slice(0, n).map(f => f.idx));
  renderGrid();
}

// ── Edit modal ──
function openEdit(idx) {
  const f = state.frames.find(x => x.idx === idx);
  if (!f) return;
  state.edit_idx = idx;
  $('edit-img').src = `/api/frame-image/${state.job_id}/${idx}`;
  $('edit-ta').value = f.caption || '';
  $('edit-info').textContent = `Frame ${idx} · CLIP score: ${(f.clip_score || 0).toFixed(3)} · Sharpness: ${(f.sharpness || 0).toFixed(0)} · Brightness: ${(f.brightness || 0).toFixed(0)}`;
  $('edit-overlay').classList.add('show');
}
function closeEdit() { $('edit-overlay').classList.remove('show'); state.edit_idx = null; }
async function saveEdit() {
  if (state.edit_idx === null) return;
  const cap = $('edit-ta').value;
  await fetch('/api/update-caption', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ job_id: state.job_id, idx: state.edit_idx, caption: cap })
  });
  const f = state.frames.find(x => x.idx === state.edit_idx);
  if (f) f.caption = cap;
  closeEdit(); renderGrid();
}
async function recaptionFrame() {
  if (state.edit_idx === null) return;
  $('edit-ta').value = '⏳ Génération…';
  try {
    const r = await fetch('/api/recaption', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ job_id: state.job_id, idx: state.edit_idx })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    $('edit-ta').value = d.caption;
  } catch(e) { $('edit-ta').value = '✗ ' + e.message; }
}
async function recaptionFrameIdx(idx) {
  const f = state.frames.find(x => x.idx === idx);
  if (!f) return;
  const oldCap = f.caption;
  f.caption = '⏳…'; renderGrid();
  try {
    const r = await fetch('/api/recaption', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ job_id: state.job_id, idx })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    f.caption = d.caption;
  } catch(e) { f.caption = oldCap; alert('Erreur: ' + e.message); }
  renderGrid();
}

async function exportDataset() {
  if (state.selected.size === 0) { alert('Aucune frame sélectionnée'); return; }
  const r = await fetch('/api/export', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ job_id: state.job_id, selected: [...state.selected] })
  });
  if (!r.ok) { const d = await r.json(); alert('Erreur: ' + d.error); return; }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = (state.filename || 'dataset').replace(/\.[^.]+$/, '') + '_dataset.zip';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function unloadModels() {
  await fetch('/api/unload-models', { method:'POST' });
  checkStatus();
}
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _parser.add_argument("--port", type=int, default=7862)
    _parser.add_argument("--host", default="127.0.0.1",
                         help="0.0.0.0 pour accès réseau local")
    _parser.add_argument("--no-browser", action="store_true",
                         help="Ne pas ouvrir le navigateur automatiquement")
    _args, _ = _parser.parse_known_args()

    port = _args.port
    host = _args.host
    url = f"http://localhost:{port}"
    output_display = str(LORA_MAKER_DATASETS)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Dataset Builder v3 — Film Mode                          ║
╠══════════════════════════════════════════════════════════╣
║  Interface : {url:<44}║
║  Pipeline  : PySceneDetect → CLIP → JoyCaption           ║
║  Cache     : {str(CACHE_DIR)[:44]:<44}║
║  Export    : {output_display[:44]:<44}║
║                                                          ║
║  Premier run = télécharge ~15 GB de modèles (HF)         ║
║  Options   : --port N  --output /dossier  --no-browser   ║
╚══════════════════════════════════════════════════════════╝
""")
    if not _args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=False, threaded=True)
