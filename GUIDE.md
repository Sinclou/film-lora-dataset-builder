# 📖 Guide d'utilisation — Dataset Builder v3 (Film Mode)

> Pipeline 100% local pour créer des datasets LoRA de style depuis des films.
> Cible : LTX-Video 2.3, AI-Toolkit, RTX 3090 24 GB VRAM.

---

## 🎯 À qui ça sert

Tu veux entraîner un **LoRA de style** pour LTX-Video à partir d'un (ou plusieurs) film(s) :
- Esthétique cinéma années 70
- Néon noir / cyberpunk
- Style d'un réalisateur (Wong Kar-wai, Tarkovski, Refn…)
- Palette / grain / lumière particuliers
- N'importe quel "look" cohérent qui se voit sur des images fixes

**Cet outil sort un dataset prêt pour AI-Toolkit** : pour chaque image extraite, tu auras :
- `image_001.jpg` — la frame
- `image_001.txt` — sa caption descriptive en anglais

C'est exactement ce que **AI-Toolkit (Ostris)** attend pour entraîner un LoRA LTX 2.3 en mode image (`frames=1`).

---

## 🚀 Lancement

Double-clic sur **`~/Bureau/Dataset-Builder-v3-Film.sh`**

→ Un terminal s'ouvre, Flask démarre, le navigateur ouvre **http://localhost:7862**.

> **Premier run :** ~15 GB de modèles HF se téléchargent automatiquement quand tu lances la première analyse (JoyCaption Beta + CLIP ViT-L). Compte 10-15 min de download la première fois selon ta connexion. **C'est une fois pour toutes**, les modèles sont ensuite en cache local.

---

## 📋 Pipeline interne (ce qui se passe quand tu lances "ANALYSER")

```
┌────────────────────┐
│ 1. PYSCENEDETECT   │ Détecte les changements de plan (~30s pour un film 90 min)
└──────────┬─────────┘ → Sortie : 200-400 scènes typiquement
           ↓
┌────────────────────┐
│ 2. EXTRACTION      │ ffmpeg sort N frames par scène (configurable 1-10)
└──────────┬─────────┘ → On garde ~4× la cible selon le nombre de frames/scène
           ↓
┌────────────────────┐
│ 3. FILTRES QUALITÉ │ OpenCV : variance Laplacien (netteté) + luminosité
└──────────┬─────────┘ → Vire flou, transitions, noir total, surex
           ↓
┌────────────────────┐
│ 4. CLIP SCORING    │ ViT-L-14 score chaque frame vs tes catégories
└──────────┬─────────┘ → "street scene at night" → top 50 frames ranked
           ↓
┌────────────────────┐
│ 5. JOYCAPTION      │ VLM dédié au captioning training Diffusion
└──────────┬─────────┘ → Caption training par frame (30-50 mots EN)
           ↓
┌────────────────────┐
│ 6. EXPORT ZIP      │ jpg + txt + dataset_info.json
└────────────────────┘ → Prêt pour AI-Toolkit
```

---

## ✅ La bonne marche à suivre (pas à pas)

### Étape 1 — Choisis ton film source

**Bons candidats :**
- Films au style **visuellement cohérent du début à la fin** (Refn, Lynch, Wong Kar-wai, Villeneuve)
- Durée ≥ 30 min (besoin de variété de scènes)
- Qualité ≥ 720p (sinon les captions seront molles)

**Mauvais candidats :**
- Documentaires avec interviews (style change tout le temps)
- Films avec beaucoup d'archives mixées
- Compressions trop fortes (mosaïque, blocky)

> 💡 **Astuce** : si tu veux un style très défini, prends plusieurs films du **même chef-op** ou **réalisateur**, lance le pipeline sur chacun, puis fusionne les ZIP exports avant training.

### Étape 2 — Définis ton nombre d'images cible

Le slider va jusqu'à **400 images**. Le bon nombre dépend de ton usage et du nombre de frames par scène choisi.

| Type de LoRA | 1 frame/scène | 3 frames/scène | 5 frames/scène |
|---|---|---|---|
| Style visuel pur (palette, lumière) | **25-40** | **75-120** | **125-200** |
| Style + composition spécifique | **40-80** | **120-240** | **200-400** |
| Style très subtil / multi-aspects | **80-150** | **240-400** | **400** |

> ⚠️ **Attention avec les frames par scène multiples.** Si tu mets 3 frames/scène et que tu vises seulement 30 images, l'outil ne va explorer que ~10 scènes — très peu de diversité. Multiplie ta cible par le nombre de frames/scène pour couvrir autant de séquences qu'en mode 1 frame.
>
> Exemple : 30 images × 3 frames/scène = vise **90 images** pour une couverture équivalente.

### Étape 3 — Définis tes catégories (CLIP filter) — ÉTAPE LA PLUS IMPORTANTE

**C'est là que tu décides ce que CLIP va chercher dans le film.** Sans catégories, l'outil prendra des frames au hasard ; avec, il cible.

**Stratégie A — Variété équilibrée** (recommandé pour démarrer)
```
• atmospheric street scene
• close-up of a face
• wide landscape shot
• interior scene with mood lighting
• two people in conversation
```
→ CLIP va piocher dans **plusieurs registres** = dataset diversifié.

**Stratégie B — Style très focus** (tu sais ce que tu veux)
```
• neon-lit street at night
• rain on windows
• cinematic close-up portrait
• warm tungsten lighting
```
→ CLIP va piocher **uniquement les scènes qui matchent**. Plus risqué : si le film n'a pas assez de ces moments, tu auras des frames de qualité moyenne.

**Stratégie C — Pas de catégorie** (laisser l'outil sampler uniformément)
→ Utile si le film est **homogène** stylistiquement de bout en bout. Tu auras une frame par segment temporel.

**Règles d'écriture des catégories :**
- ✅ En **anglais** (CLIP est entraîné dessus)
- ✅ Phrases descriptives courtes (4-8 mots)
- ✅ Décris l'image, pas l'émotion (sauf si visuel : "moody dark scene" OK)
- ❌ Pas de noms propres (CLIP ne connaît pas ton réalisateur)
- ❌ Pas trop spécifique au film ("the protagonist running") — CLIP ne sait pas qui est protagonist

**Presets disponibles** (boutons rapides dans l'UI) : rue, portrait, paysage, intérieur, néon, dialogue, action, atmosphère.

### Étape 4 — Filtres qualité

- ☑ **Éviter floues** : recommandé. Élimine les transitions, motion blur extrême.
- ☑ **Éviter trop sombres/claires** : recommandé. Vire les fondus au noir, les flashes.
- ☑ **Captionner immédiatement** : laisser coché si tu veux tout en un coup. Si décoché, JoyCaption ne sera appelé qu'à la demande (utile pour gros datasets où tu veux d'abord présélectionner visuellement avant de payer le coût compute).

### Étape 4b — Taille de sortie (résolution du ZIP exporté)

Ce paramètre s'applique **uniquement à l'export** — la prévisualisation dans la grille reste toujours en haute résolution. Tu peux le changer sans relancer l'analyse.

| Option | Résolution | Usage |
|---|---|---|
| **Originale** | Taille native du film | Si tu veux contrôler le resize toi-même |
| **LTX 2.3** *(défaut)* | 768×512 (paysage) / 512×768 (portrait) | Training LTX-Video 2.3 avec AI-Toolkit |
| **1024×1024 crop centré** | Carré recadré depuis le centre | SDXL, Flux, Illustrious, ou LoRA carré |
| **1920×1080 letterbox** | Full HD, bandes noires si besoin | Préservation qualité, usage hors training |

> 💡 Le mode LTX 2.3 détecte automatiquement l'orientation (paysage ou portrait) et choisit la bonne résolution.

### Étape 5 — Lance l'analyse

Clique **▶ ANALYSER LE FILM**.

**Timing typique sur ta 3090 pour un film de 90 min :**
| Étape | Durée |
|---|---|
| PySceneDetect | ~30-60s |
| Extraction frames ffmpeg | ~1 min |
| Filtres qualité | ~5s |
| CLIP scoring (~120 frames) | ~30s |
| JoyCaption captioning (~45 frames) | ~3-5 min |
| **Total** | **~5-7 min** |

Le panneau de progression t'indique où on en est.

### Étape 6 — Triage dans la grille

Une fois fini, **la grille s'affiche** avec :
- Vignette de chaque frame
- Score CLIP (plus c'est haut, mieux ça matche tes catégories)
- Timestamp dans le film
- Caption générée
- État coché/décoché (les top N sont auto-cochés selon ta cible)

**Ce que tu fais :**
1. **Parcours la grille visuellement** — décoche les frames qui ne te plaisent pas
2. **Coche-en d'autres** si tu vois des frames intéressantes qui n'étaient pas dans le top auto
3. **Édite les captions** qui sont à côté de la plaque (clic sur ✎)
4. **Re-caption** une frame si la première tentative de JoyCaption est bizarre (clic sur 🔄)

> 💡 Un caption mal foutu plombe l'apprentissage de cette image. Mieux vaut **30 captions bons que 50 captions moyens.**

**Les boutons rapides :**
- "Tout sélectionner" / "Tout désélectionner"
- "Top N" — re-sélectionne les N premiers par score CLIP

### Étape 7 — Export

Clique **▶ EXPORT**. Un ZIP se télécharge avec :
```
NomDuFilm_001.jpg
NomDuFilm_001.txt
NomDuFilm_002.jpg
NomDuFilm_002.txt
...
dataset_info.json   ← métadonnées complètes (timestamps, scores, etc.)
```

→ **Décompresse direct dans le dossier dataset de AI-Toolkit.**

---

## 🔧 Gestion de la VRAM (important sur 3090)

**JoyCaption mange ~12 GB**, CLIP ~1 GB. Ta 3090 a 24 GB.

**Si ComfyUI tourne en parallèle :**
- ComfyUI prend ~12 GB → conflit possible avec JoyCaption
- **Solutions :**
  1. Ferme ComfyUI avant l'analyse (le plus simple)
  2. OU clique sur **"⏏ Décharger VRAM"** entre deux analyses pour libérer
  3. OU laisse ComfyUI tourner et accepte un swap CPU plus lent

**Bonne pratique :** lance le builder, fais ton analyse, ferme l'app (le venv libère la VRAM). Tu peux ensuite relancer ComfyUI sans problème.

---

## 📦 Workflow complet jusqu'au LoRA

```
1. Builder v3                  → Dataset ZIP (jpg + txt)
   ↓
2. Décompresse dans AI-Toolkit/datasets/mon_style/
   ↓
3. AI-Toolkit config :
   - model: LTX-Video 2.3 (13B)
   - frames: 1                  ← MODE IMAGE
   - resolution: 512x512 ou 768x512
   - rank: 32 (style) ou 16 (rapide)
   - steps: 1500-2500
   - learning_rate: 1e-4
   - low VRAM tricks: ON (layer offload, cached latents)
   ↓
4. Train sur la 3090           → ~3-6h pour 2000 steps
   ↓
5. .safetensors LoRA           → tu plug dans ComfyUI
   ↓
6. Test : "in the style of <mon-mot-cle>"
```

---

## 🎓 Trucs de pro

### Pour un style cohérent
- Vise des **scènes de même registre** (toutes des extérieurs nuit, ou toutes des intérieurs jour)
- Évite de mélanger des scènes très contrastées (jour ensoleillé + nuit néon → le LoRA va se perdre)

### Pour les captions
- Re-lis les captions de JoyCaption après l'export — il est bon mais pas infaillible
- **Ajoute manuellement** un mot-clé unique en début de caption (ex: "wkw_style") sur TOUS les fichiers .txt → c'est ton trigger word pour invoquer le LoRA
- Ex de caption finale : `wkw_style, a close-up portrait bathed in warm tungsten light, shallow depth of field, rich amber tones, cinematic 35mm look`

### Pour multi-films (même style)
1. Lance le builder sur chaque film séparément
2. Fusionne les ZIP extracts dans un seul dossier
3. Renumérote ou laisse tel quel (les noms sont uniques)
4. Train sur le dataset combiné

### Pour itérer
- Lance une 1ʳᵉ analyse, regarde la grille → ajuste tes catégories si le résultat ne te plaît pas
- Relance avec d'autres prompts CLIP
- Les modèles restent en cache, donc c'est rapide

---

## 🐛 Dépannage

| Problème | Cause probable | Solution |
|---|---|---|
| **Crash CUDA OOM** | ComfyUI prend la VRAM | Ferme ComfyUI, clique "⏏ Décharger VRAM" |
| **Téléchargement modèles bloqué** | HF rate limit ou réseau | Patience, ça reprend. Re-lance si vraiment bloqué |
| **Frames toutes floues** | Vidéo source compressée | Désactive le filtre flou, vérifie la qualité source |
| **Captions vides/courtes** | JoyCaption a planté | Re-caption à la main via le bouton 🔄 |
| **Aucune frame ne match** | Catégories CLIP trop spécifiques | Élargis les catégories, ou supprime-les |
| **Le terminal se ferme tout seul** | Crash Python | Lance via terminal manuellement : `cd ~/Outils/dataset-builder-v3 && ./venv/bin/python dataset_builder_v3.py` |

---

## 📁 Où sont les fichiers

```
~/Outils/dataset-builder-v3/
├── dataset_builder_v3.py     ← l'app
├── venv/                     ← python + libs (5.4 GB)
├── model_cache/              ← modèles HF téléchargés (~15 GB après 1er run)
└── GUIDE.md                  ← ce fichier

~/Bureau/Dataset-Builder-v3-Film.sh   ← launcher (double-clic)

/tmp/dataset_builder_v3/<job_id>/      ← workdir temporaire (auto-clean au reboot)
```

---

## 🎬 Récap visuel — checklist avant de lancer

- [ ] Film source choisi (cohérent stylistiquement, ≥ 30 min, ≥ 720p)
- [ ] ComfyUI fermé (ou prêt à libérer la VRAM)
- [ ] Nombre cible défini (30 en 1 frame/scène, multiplier si plus de frames/scène)
- [ ] 3-5 catégories CLIP en anglais ajoutées
- [ ] Filtres qualité activés
- [ ] Captionning immédiat coché
- [ ] **▶ ANALYSER**

Bon entraînement ! 🎨💽

---

*Dataset Builder v3 — Created June 2026 by Deckie pour Mathieu*
