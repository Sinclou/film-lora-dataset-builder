# 📖 Dataset Builder v3 — Film Mode

> Pipeline 100% local pour créer des datasets LoRA de style depuis des films.  
> Conçu pour LTX-Video / AI-Toolkit. Fonctionne sur toute machine Linux avec GPU NVIDIA.

---

## 🎯 À quoi ça sert

Tu veux entraîner un **LoRA de style** à partir d'un film :
- Esthétique cinéma années 70
- Néon noir / cyberpunk
- Style d'un réalisateur (Wong Kar-wai, Tarkovski, Refn…)
- Palette / grain / lumière particuliers

**L'outil produit un dataset prêt pour AI-Toolkit** : pour chaque image extraite, un fichier `.jpg` + un fichier `.txt` avec sa caption en anglais — exactement ce qu'AI-Toolkit attend pour un LoRA LTX-Video en mode image (`frames=1`).

---

## 🖥️ Prérequis

| Composant | Minimum | Recommandé |
|---|---|---|
| GPU NVIDIA | 12 GB VRAM | 24 GB (RTX 3090 / 4090) |
| RAM | 16 GB | 32 GB |
| Stockage libre | 20 GB | 50 GB+ |
| Python | 3.10 | 3.11 |
| OS | Ubuntu 22.04+ | Ubuntu 22.04 / 24.04 |

**Dépendances système :**
```bash
sudo apt install ffmpeg python3-venv python3-pip
```

> ⚠️ **Carte AMD / Intel Arc / CPU seul** : JoyCaption nécessite CUDA. La partie CLIP peut tourner sur CPU mais sera lente. Pour l'instant, GPU NVIDIA requis.

---

## 🚀 Installation

```bash
# 1. Clone ou copie le dossier
git clone <url-du-repo>
cd dataset-builder-public

# 2. Crée un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installe les dépendances
pip install -r requirements.txt
```

> **Premier lancement :** ~15 GB de modèles se téléchargent automatiquement depuis Hugging Face (JoyCaption Beta One + CLIP ViT-L). Compte 10-30 min selon ta connexion. C'est **une seule fois** — ils sont ensuite en cache dans `./model_cache/`.

---

## ▶️ Lancement

```bash
# Lancement simple (output par défaut : ~/datasets)
source venv/bin/activate
python dataset_builder_v3.py

# Avec un dossier d'export personnalisé
python dataset_builder_v3.py --output /mon/dossier/datasets

# Ou via variable d'environnement
DATASET_OUTPUT=/mon/dossier/datasets python dataset_builder_v3.py
```

Le terminal affiche l'URL — ouvre **http://localhost:7862** dans ton navigateur.

---

## 📋 Pipeline interne

```
┌────────────────────┐
│ 1. PYSCENEDETECT   │ Détecte les changements de plan (~30s pour 90 min)
└──────────┬─────────┘ → 200-400 scènes typiquement
           ↓
┌────────────────────┐
│ 2. EXTRACTION      │ ffmpeg extrait N frames par scène (configurable)
└──────────┬─────────┘ → On garde ~4× la cible
           ↓
┌────────────────────┐
│ 3. FILTRES QUALITÉ │ Vire flou, transitions, noir total, surex
└──────────┬─────────┘
           ↓
┌────────────────────┐
│ 4. CLIP SCORING    │ ViT-L-14 score chaque frame vs tes catégories
└──────────┬─────────┘ → Frames classées par pertinence sémantique
           ↓
┌────────────────────┐
│ 5. JOYCAPTION      │ VLM dédié au captioning pour Diffusion training
└──────────┬─────────┘ → Caption en anglais par frame (30-50 mots)
           ↓
┌────────────────────┐
│ 6. EXPORT ZIP      │ jpg + txt + dataset_info.json
└────────────────────┘ → Prêt pour AI-Toolkit
```

---

## ✅ Utilisation pas à pas

### Étape 1 — Choisis ton film source

**Bons candidats :**
- Films au style **visuellement cohérent** du début à la fin
- Durée ≥ 30 min (besoin de variété de scènes)
- Qualité ≥ 720p

**Mauvais candidats :**
- Documentaires avec interviews (style change trop souvent)
- Films d'archives mixées
- Sources très compressées (mosaïque, blocky)

> 💡 Pour un style très défini, combine plusieurs films du **même chef-opérateur ou réalisateur** et fusionne les exports avant training.

### Étape 2 — Nombre d'images cible

| Type de LoRA | Images recommandées |
|---|---|
| Style visuel pur (palette, lumière) | **20-30** |
| Style + composition spécifique | **30-50** |
| Style très subtil / nuancé | **50-80** |

> 25 images parfaitement choisies > 100 images médiocres. En cas de doute, vise **30**.

### Étape 3 — Frames par scène

Nouveau paramètre : combien de frames extraire **par scène détectée**.

| Valeur | Usage |
|---|---|
| 1 | Rapide — prend le centre de chaque scène |
| 3 | Recommandé — début / milieu / fin de scène |
| 5 | Panel complet — bonne couverture de chaque plan |
| 10 | Très dense — utile pour scènes longues et statiques |

### Étape 4 — Catégories CLIP (étape clé)

C'est ce qui pilote la sélection sémantique des frames.

**Stratégie A — Variété équilibrée** (recommandé pour démarrer)
```
atmospheric street scene
close-up of a face
wide landscape shot
interior scene with mood lighting
two people in conversation
```

**Stratégie B — Style très ciblé**
```
neon-lit street at night
rain on windows
cinematic close-up portrait
warm tungsten lighting
```

**Stratégie C — Aucune catégorie** → sampling temporel uniforme (utile si le film est homogène de bout en bout)

**Règles :**
- ✅ En **anglais** (CLIP est entraîné dessus)
- ✅ Phrases descriptives courtes (4-8 mots)
- ❌ Pas de noms propres
- ❌ Pas d'émotions abstraites (sauf si elles se voient : "moody dark scene" OK)

### Étape 5 — Filtres qualité

- **Éviter floues** : recommandé — élimine transitions et motion blur
- **Éviter trop sombres/claires** : recommandé — vire fondus au noir et flashes
- **Captionner immédiatement** : laisse coché pour tout générer en une passe. Décoche si tu veux d'abord présélectionner visuellement avant de lancer JoyCaption

### Étape 6 — Analyse

Clique **▶ ANALYSER LE FILM**. Le panneau de progression t'indique où on en est.

**Timing indicatif sur RTX 3090 pour un film de 90 min :**

| Étape | Durée |
|---|---|
| PySceneDetect | ~30-60s |
| Extraction frames | ~1-2 min |
| Filtres qualité | ~5s |
| CLIP scoring (~120 frames) | ~30s |
| JoyCaption (~45 frames) | ~3-5 min |
| **Total** | **~5-8 min** |

### Étape 7 — Triage dans la grille

Une fois fini, la grille affiche chaque frame avec son score CLIP, son timestamp, et sa caption.

1. **Parcours visuellement** — décoche ce qui ne te convient pas
2. **Édite les captions** si JoyCaption est à côté (clic sur ✎)
3. **Re-caption** une frame si besoin (clic sur 🔄)

> Un caption inexact nuit à l'apprentissage. Mieux vaut 30 bons que 50 approximatifs.

### Étape 8 — Export

Clique **▶ EXPORT**. Tu obtiens un ZIP :

```
NomDuFilm_001.jpg
NomDuFilm_001.txt
NomDuFilm_002.jpg
NomDuFilm_002.txt
...
dataset_info.json
```

→ Décompresse directement dans ton dossier AI-Toolkit.

---

## 🔧 Gestion de la VRAM

JoyCaption charge ~12 GB en mémoire GPU. Si une autre app GPU tourne en parallèle (ComfyUI, autre inférence) :
- Ferme-la avant l'analyse, **ou**
- Clique sur **"⏏ Décharger VRAM"** entre deux analyses pour libérer

---

## 📦 Workflow complet jusqu'au LoRA

```
1. Dataset Builder v3  → ZIP (jpg + txt)
   ↓
2. Décompresse dans AI-Toolkit/datasets/mon_style/
   ↓
3. Config AI-Toolkit :
   model: LTX-Video 2.3 (13B)
   frames: 1             ← mode image
   resolution: 512x512 ou 768x512
   rank: 32 (style) ou 16 (rapide)
   steps: 1500-2500
   learning_rate: 1e-4
   ↓
4. Training ~3-6h sur 3090 (2000 steps)
   ↓
5. .safetensors → plug dans ComfyUI ou autre
   ↓
6. Test : "in the style of <ton-trigger-word>"
```

---

## 🎓 Conseils

**Pour un style cohérent**  
Vise des scènes de même registre (toutes extérieurs nuit, ou toutes intérieurs jour). Mélanger des atmosphères très différentes dilue le LoRA.

**Pour les captions**  
Ajoute un **trigger word unique** au début de chaque `.txt` (ex: `monstyle_`) sur tous les fichiers après export. C'est ce mot qui invoque le LoRA en génération.

**Pour multi-films**  
Lance le builder sur chaque film séparément, fusionne les ZIP dans un dossier commun, puis entraîne sur le dataset combiné.

---

## 🐛 Dépannage

| Problème | Cause probable | Solution |
|---|---|---|
| **Crash CUDA OOM** | Une autre app occupe la VRAM | Ferme-la, clique "⏏ Décharger VRAM" |
| **Téléchargement bloqué** | Rate limit HF ou réseau | Relance l'app, ça reprend |
| **Frames toutes floues** | Source trop compressée | Désactive le filtre flou, vérifie la qualité source |
| **Captions vides** | JoyCaption a planté | Re-caption via le bouton 🔄 |
| **Aucune frame ne match** | Catégories CLIP trop spécifiques | Élargis les catégories ou supprime-les |
| **Page blanche au démarrage** | Port 7862 déjà utilisé | Lance avec `--port 7863` |

---

## 📁 Structure du projet

```
dataset-builder-public/
├── dataset_builder_v3.py   ← application Flask
├── index.html              ← interface web
├── requirements.txt        ← dépendances Python
├── GUIDE.md                ← ce fichier
└── model_cache/            ← créé au 1er run (~15 GB)
```

**Dossier de travail temporaire** : `/tmp/dataset_builder_v3/` (nettoyé au reboot)  
**Dossier d'export** : `~/datasets/` par défaut (configurable via `--output`)

---

## ✅ Checklist avant de lancer

- [ ] GPU NVIDIA avec ≥ 12 GB VRAM disponible
- [ ] ffmpeg installé (`ffmpeg -version`)
- [ ] Venv activé (`source venv/bin/activate`)
- [ ] Film source choisi (cohérent, ≥ 30 min, ≥ 720p)
- [ ] Nombre cible défini (commence par 30)
- [ ] 3-5 catégories CLIP en anglais
- [ ] Filtres qualité activés
- [ ] **▶ ANALYSER**

---

*Dataset Builder v3 — open source, contributions bienvenues.*
