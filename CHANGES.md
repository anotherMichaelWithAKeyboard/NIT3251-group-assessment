# Changes Made

Record of everything changed while getting the project running on Windows. Useful for the
report's required "challenges faced and how they were overcome" section, and so `mtaha05`
knows what moved.

---

## 1. Environment setup (one-off, outside the repo)

| Change | Detail |
|---|---|
| Cloned repo | into `C:\Users\Mikae\uni\NIT3251` |
| Created virtualenv | `.venv\` — Python 3.13.5, pip upgraded to 26.2.1 |
| Installed requirements | all of `requirements.txt` — TensorFlow 2.20.0, numpy 2.4.2, etc. |
| Registered Jupyter kernel | "Python 3.13 (NIT3251 .venv)" |
| Installed Graphviz | 16.1.0 via `winget` → `C:\Program Files\Graphviz` |
| Updated user PATH | appended `C:\Program Files\Graphviz\bin` so `plot_model` works |

**Note:** TensorFlow reports no GPU. Native-Windows GPU support was dropped after TF 2.10,
so training is CPU-only. This turned out to be fine — about 3–5 min/epoch.

---

## 2. Fixes to `image_captioning.ipynb`

Seven edits across seven cells. Each was verified against the real dataset.

### 2.1 Import — added `load_model`

```diff
- from tensorflow.keras.models import Model
+ from tensorflow.keras.models import Model, load_model
```

**Why:** needed to reload a saved model instead of re-training. Without it the resume line
raises `NameError`.

### 2.2 Cell 4 — feature extraction is now cached and batched

**Was:** unconditionally re-extracted all 8,091 images on every run, one `predict` call per
image, reusing the name `model` for the VGG16 extractor.

**Now:** checks for `features.pkl` first and loads it if present; otherwise extracts in
batches of 32 and writes the cache once the loop finishes.

| | Before | After |
|---|---|---|
| Cache hit | n/a — always re-extracted | **0.2 s** |
| Extraction (first run) | 310 ms/image → **~42 min** | 60 ms/image → **~8 min** |

Three separate problems fixed:

- **No cache check.** The cached features existed but were never used.
- **No batching.** A `predict` call on 32 images costs barely more than on 1, because
  per-call overhead dominates. That is a 5.2× speedup, measured.
- **`model` was clobbered.** The extractor is now named `vgg`. Previously, re-running that
  cell after building the captioning model left `model` pointing at VGG16, so `model.fit`
  would try to train VGG16 and fail confusingly.

Also moved the pickle write *inside* the extraction branch, after the loop completes. This
is what caused the earlier 111-image `features.pkl`: extraction was interrupted and a
separate cell dumped the partial dictionary. An interrupted run now leaves no file at all.

### 2.3 Cell 5 — disabled the blind re-dump

**Was:** `pickle.dump(features, open(..., 'wb'))` ran unconditionally.

**Now:** commented out, with a note. It only prints the count.

**Why:** it overwrote `features.pkl` with whatever was in memory. Harmless on a cache hit,
destructive after a partial extraction. Saving is handled by cell 4.

### 2.4 Cell 8 — removed `tqdm` from the mapping loop

```diff
- for line in tqdm(captions_doc.split('\n')):
+ for line in captions_doc.split('\n'):
```

**The single biggest time saving in the notebook: ~30 minutes → 38 milliseconds.**

`from tqdm.notebook import tqdm` is the ipywidgets progress bar. Every iteration sends a
comm message over ZMQ to VS Code to redraw it, costing roughly 44 ms. Across 40,456
iterations of microsecond-scale work, the progress bar was about **47,000× more expensive
than the computation it was measuring**.

Measured: the loop body is 38 ms. The whole preprocessing chain — read, map, clean,
tokenise, `max_length` — is **0.55 s**.

The other two `tqdm` loops were left alone deliberately. Feature extraction and BLEU
evaluation have slow iterations, so tqdm throttles itself and the bar is genuinely useful.
The pathology only appears when iterations are faster than the redraw.

### 2.5 Cell 22 — `data_generator` yields a tuple

```diff
- yield [X1, X2], y
+ yield (X1, X2), y
```

**Why:** training could not start at all. Keras 3 routes generators through
`tf.data.Dataset.from_generator`, which infers a signature from the first yield. In
`tf.data` a **tuple** means "several structured inputs" while a **list** is read as a single
tensor, so Keras tried to build one `TypeSpec` from two differently shaped arrays:

```
TypeError: `output_signature` must contain objects that are subclass of `tf.TypeSpec`
           but found <class 'list'> which is not.
```

The code predates Keras 3, which accepted lists. Verified both directions against the real
data: list → `TypeError`, tuple → `fit()` succeeds.

Worth noting for the Q&A: the traceback pointed at the `model.fit` cell, but the defect was
in the generator two cells earlier. `fit` is only where the bad yield gets consumed.

### 2.6 Cell 25 — save is live, broken load removed

**Was:**

```python
#model.save(WORKING_DIR+'best_model.h5')   # commented out
model.load_weights('best_model.h5')        # live, and the file does not exist
```

**Now:**

```python
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.keras')
model.save(MODEL_PATH)
print('saved', MODEL_PATH)
# model = load_model(MODEL_PATH)   # uncomment to resume instead of re-training
```

**Why:** after a one-hour training run, the cell as written either threw a file-not-found
or silently replaced the trained weights. Nothing in the notebook wrote a checkpoint, so
there was no file to load. `.keras` is the current Keras 3 format; `.h5` is legacy.

### 2.7 Cell 34 - the `download.jpg` demo cell now has a fallback

**Was:** `image_path = 'download.jpg'` - a file the original tutorial author had locally
and never committed, so the cell always raised:

```
FileNotFoundError: [Errno 2] No such file or directory: 'download.jpg'
```

**Now:** `IMAGE_PATH` is checked with `os.path.exists` first. If the file is missing the
cell falls back to `test[0]` - a held-out image the model never trained on - so it runs
either way and prints which image it used.

To caption your own photo, drop any `.jpg` in the project root and set `IMAGE_PATH` to it.

Nikon `.NEF` raw files also work, which is not obvious: NEF is a TIFF container, so
Pillow's TIFF plugin opens it. Note it exposes only the **160x120 embedded preview**
(one IFD; the full-resolution sensor data sits in a Nikon-proprietary sub-IFD Pillow
does not read), so `target_size=(224, 224)` upscales a thumbnail. Captions from a raw
file are therefore from a lower-resolution version of the image than a JPEG export
would give.

Also added `plt.axis('off')` and wrapped the final call in `print(...)`, since the bare
`predict_caption(...)` expression only renders when it is the last line of a cell.

---

## 3. New files

| File | Purpose |
|---|---|
| `PLAN.md` | Project plan — phases, measured compute budget, risk register, 25-question Q&A bank |
| `draft.ipynb` | Clean 41-cell rebuild: all the above fixes **plus** the `clean()` fix, beam search, BLEU evaluation and error analysis. Verified end to end, zero errors |
| `CHANGES.md` | This file |
| `.vscode/settings.json` | Pins the Python interpreter to `.venv` |

---

## 4. `.gitignore` additions

```
.venv/                      # ~1.5 GB virtualenv

*.keras                     # trained models
history.csv                 # regenerated by one notebook run
bleu_results.csv
loss_curve.png
decoder_comparison.png
error_analysis.png
caption_examples.png
model.png
```

---

## 5. Measured impact

| Fix | Before | After |
|---|---|---|
| `tqdm` on mapping loop | ~30 min | **38 ms** |
| Feature extraction (cache hit) | ~25–42 min | **0.2 s** |
| Feature extraction (first run) | ~42 min | **~8 min** |
| `data_generator` tuple | `TypeError`, could not train | trains |
| Sparse labels (`draft.ipynb` only) | ~54 MB labels/batch | **0.01 MB** |
| `clean()` fix (`draft.ipynb` only) | 798 junk vocab tokens | **0** |

Getting from a fresh kernel to the training cell went from roughly **55 minutes to about
3 seconds**.

---

## 6. Still outstanding

**`clean()` in `image_captioning.ipynb` is still broken.** It uses `str.replace` with regex
patterns:

```python
caption = caption.replace('[^A-Za-z]', '')   # literal match - does nothing
caption = caption.replace('\s+', ' ')        # literal match - does nothing
```

`str.replace` does literal substring replacement, so these search for the literal text
`[^A-Za-z]`, never find it, and strip nothing. Punctuation survives into the vocabulary:
**798 of 9,149 entries are punctuation-contaminated duplicates** such as `"a` beside `a`,
each getting its own embedding row and splitting the training signal for that word.

The fix is `re.sub(r'[^a-z ]', '', caption)` and `re.sub(r'\s+', ' ', caption)`, which is
already applied in `draft.ipynb`.

**Fix it before training, not after.** It changes `vocab_size` from 9,149 to 8,768, which
resizes the Embedding and output Dense layers and invalidates any trained weights.

Also not yet done: beam search, BLEU evaluation, hyperparameter tuning, error analysis and
the report — all in `draft.ipynb` or scoped in `PLAN.md`.

---

## 7. Note on line endings

`git diff` warns that `.gitignore` and `image_captioning.ipynb` have LF endings that Git
will convert to CRLF on checkout. Pre-existing, not caused by these edits — but if the two
of you are on different platforms, a `.gitattributes` containing
`*.ipynb text eol=lf` will stop the notebook churning line endings between you.
