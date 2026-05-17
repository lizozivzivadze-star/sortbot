# SortBot — Vision Sorting System (Web)

A browser-playable port of the Python/pygame SortBot vision sorting game.  
**[▶ Play it live](https://YOUR-USERNAME.github.io/sortbot)**

---

## What is SortBot?

SortBot simulates an industrial computer-vision sorting pipeline. Synthetic CIFAR-10–style images roll down a conveyor belt, get scanned by a vision system, and are routed into bins by a sort arm. The game makes the **accuracy vs. speed trade-off** interactive and visible.

| Bin | Classes |
|-----|---------|
| ♻️ Recycle | airplane, automobile |
| 🌿 Organic | bird, cat, deer, dog, frog, horse |
| 🔩 Metal | ship, truck |
| ❌ Reject | frog, deer (simulated defects) |

---

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / unpause |
| `1`–`5` | Set belt speed |
| `Q` / `ESC` | Quit session |

---

## Configuration (menu)

| Setting | Options | Effect |
|---------|---------|--------|
| **Classifier** | COLOR · HOG · ML | Accuracy vs latency profile |
| **Belt Speed** | 1–5 | Faster = harder to classify in time |
| **Noise Level** | OFF–4 | Degrades classification accuracy |
| **Batch Size** | 20–80 | Images per session |

### Classifier comparison

| Classifier | Accuracy | Avg Latency |
|------------|----------|-------------|
| Color histogram | ~72% | 8–18 ms |
| HOG + rules | ~83% | 22–40 ms |
| MobileNetV2 (ML) | ~94% | 55–95 ms |

Crank speed to 5 and watch accuracy drop even with ML — the pipeline can't keep up.

---

## Deploying your own copy

### One-time setup

```bash
git clone https://github.com/YOUR-USERNAME/sortbot.git
cd sortbot
```

### Enable GitHub Pages

1. Go to **Settings → Pages**
2. Set **Source** to `GitHub Actions`
3. Push to `main` — the workflow deploys automatically

The live URL will be: `https://YOUR-USERNAME.github.io/sortbot`

---

## Project structure

```
sortbot/
├── index.html           # Entire game — single self-contained file
├── .github/
│   └── workflows/
│       └── deploy.yml   # GitHub Pages deployment
└── README.md
```

The web version is a zero-dependency single HTML file. No build step, no npm, no backend.

---

## Original Python version

The original desktop version (pygame + OpenCV + optional TensorFlow) lives in `python/` and runs with:

```bash
pip install -r python/requirements.txt
python python/main.py --clf hog --speed 3
```

---

## Architecture (web port)

```
index.html
  ├── VisionPipeline     JS port of core/pipeline.py
  │     Simulates preprocessing log, feature extraction,
  │     colour/HOG/ML classifiers, noise injection
  │
  ├── SortBotGame        JS port of game/conveyor.py
  │     Canvas 2D rendering, belt animation,
  │     scan line, sort arm, bins, score, streak
  │
  └── makeSyntheticImage JS port of core/dataset._load_synthetic
        Procedurally generated coloured shapes per class
```
