# NIT3251 Group Assessment: Automatic Image Captioning with CNN and LSTM

Topic 14. An encoder/decoder image captioning model built on a pretrained VGG16 feature
extractor and an LSTM decoder, trained on Flickr8k. The extended challenge is beam search
decoding, evaluated against greedy decoding with BLEU.

| Member | Student ID |
|---|---|
| Muhammad Taha | s8128767 |
| Perneil Calimlim | s8076141 |
| Mikaere McCarthy | s8090942 |

---

## 1. Results

Evaluated on 810 held out test images that the model never trained on.

| Metric | Greedy | Beam search (k = 3) |
|---|---:|---:|
| BLEU-1 | 0.4896 | **0.5455** |
| BLEU-2 | 0.3280 | **0.3724** |
| BLEU-3 | 0.2120 | **0.2476** |
| BLEU-4 | 0.1373 | **0.1620** |

Beam search wins on all four metrics. BLEU-4 improves by **18.0% relative**.

Training stopped at epoch 8 with the best validation loss of **3.2633 at epoch 5**. Early
stopping restored the epoch 5 weights, which are what `best_model.keras` contains.

---

## 2. What is in here

| File | What it is |
|---|---|
| `image_captioning.ipynb` | **The main notebook.** Runs end to end: features, preprocessing, training, both decoders, BLEU |
| `draft.ipynb` | Clean rebuild of the same pipeline, used as a working copy while fixing the notebook |
| `hyperparameter_tuning.py` | Standalone tuning script, three controlled runs. See section 6 |
| `captions.txt` | 40,455 captions, 8,091 images, five each |
| `best_model.keras` | Trained captioning model, 5,574,133 parameters |
| `tokenizer.pkl` | Fitted vocabulary. **Must match the model**, do not regenerate on its own |
| `caption_config.json` | Vocabulary size (7,669), max caption length (38), CNN used, random seed |
| `dataset_split.json` | The exact image IDs in each of the train, validation and test sets |
| `training_validation_loss.csv/.png` | Per epoch loss, and the loss curve figure |
| `bleu_comparison.csv/.png` | The BLEU table above, and the comparison figure |
| `model.png` | The architecture diagram, drawn by `plot_model` |
| `requirements.txt` | Pinned dependencies |
| `CHANGES.md` | Full record of every fix made getting this running, with before and after measurements |
| `DSC_0875.NEF` | A raw photo used by the last notebook cell to caption an image from outside Flickr8k |

---

## 3. Setup

```bash
git clone https://github.com/anotherMichaelWithAKeyboard/NIT3251-group-assessment.git
cd NIT3251-group-assessment

python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Built against **Python 3.13.5** and **TensorFlow 2.20.0**.

`plot_model` needs Graphviz installed as a system package, not just the Python wrapper:

```bash
winget install graphviz
```

Then add `C:\Program Files\Graphviz\bin` to PATH. If you skip this, everything still runs,
you just cannot regenerate the architecture diagram.

---

## 4. The dataset is not committed

`Images/` holds 8,091 JPEGs and is gitignored. Download Flickr8k separately and extract the
images into `Images/` in the project root:

<https://www.kaggle.com/datasets/adityajn105/flickr8k>

`captions.txt` **is** committed, so you only need the image files. The notebook asserts that
captions and features line up and will warn you if any images are missing.

---

## 5. Running it

Open `image_captioning.ipynb` and run the cells in order. Two things are worth knowing
before you start.

**Feature extraction takes about 8 minutes the first time.** VGG16 is frozen, so its output
for a given image never changes. The notebook extracts all 8,091 feature vectors once,
caches them to `features.pkl` (133 MB) and loads from that cache on every later run, which
takes under a second. `features.pkl` is gitignored because it is over GitHub's 100 MB per
file limit.

**Training takes roughly 30 minutes.** About 3.5 minutes per epoch across 203 steps at a
batch size of 32, stopping around epoch 8. To skip it, uncomment the `load_model` line and
reuse the committed `best_model.keras`.

Decoding the full test set with both strategies takes about **42 minutes**, since beam
search runs a forward pass per candidate per step.

A fixed seed of 42 is used for the shuffle, the split and the generator, so the split is
reproducible.

---

## 6. Hyperparameter tuning

```bash
python hyperparameter_tuning.py
```

Three runs, changing one variable at a time against the baseline, with everything else held
fixed:

| Arm | LSTM units | Dropout |
|---|---:|---:|
| A | 256 | 0.4 |
| B | 512 | 0.4 |
| C | 256 | 0.2 |

Each arm trains to early stopping and is then scored with greedy decoding on a fixed
250 image slice of the **validation** set. Validation rather than test, because these
scores choose a configuration, and selecting on the test set would make the final test
figures optimistic. The test set is decoded once, in the notebook, after every decision
about the model has been made, and that is where the headline BLEU figures in section 1
come from.

`rescore_arms_on_validation.py` re-scores saved arm checkpoints without retraining.

Writes `hyperparameter_results.csv` and `hyperparameter_results.png`. Results are saved
after every arm, so an interrupted run still leaves usable output for the arms that
finished. Budget about **2 hours** on CPU.

---

## 7. Constraints worth knowing

**Everything here is CPU only.** TensorFlow dropped native Windows GPU support after 2.10,
so TF 2.20 reports no GPU on this setup. That is why the timings above look the way they do.

**No image augmentation was applied.** Features are extracted once from a frozen VGG16 and
cached, so augmenting the images would mean re-extracting features every epoch and giving
up the cache entirely. This was a deliberate trade off to keep the focus on caption
generation and decoding.

**The tokenizer and the model are a matched pair.** The vocabulary size sets the Embedding
and output Dense layer shapes, so regenerating `tokenizer.pkl` on its own invalidates
`best_model.keras`. If you change the preprocessing, retrain.

---

## 8. Problems hit along the way

Short version, for anyone picking this up cold. `CHANGES.md` has the full account, with
the measurements behind each one.

**Training could not start at all.** The data generator yielded `[X1, X2], y`. Keras 3
routes generators through `tf.data.Dataset.from_generator`, where a tuple means several
structured inputs but a list is read as a single tensor, so it could not build an output
signature from two differently shaped arrays. Changing it to `(X1, X2), y` fixed it. The
traceback pointed at `model.fit`, but the fault was in the generator two cells earlier.

**Caption cleaning silently did nothing.** It used `str.replace` with regex patterns, which
does literal substring replacement, so it searched for the literal text `[^A-Za-z]`, never
found it and stripped nothing. Punctuation survived into the vocabulary, leaving 798
contaminated duplicate tokens such as `"a` sitting alongside `a`, each getting its own
embedding row and splitting the training signal. Fixed with `re.sub`. This has to be fixed
**before** training, because it changes the vocabulary size and therefore the layer shapes.

**A progress bar cost about 30 minutes.** `tqdm.notebook` on the caption mapping loop sent a
redraw message over ZMQ on every iteration, roughly 44 ms each, across 40,456 iterations of
microsecond scale work. The loop body itself is 38 ms. The bars on feature extraction and
BLEU evaluation were left alone, because those iterations are genuinely slow and tqdm
throttles itself.

**Feature extraction was 5 times slower than it needed to be.** It ran one `predict` call
per image, where per call overhead dominates the actual computation. Batching 32 images at a
time took it from 310 ms per image to 60 ms per image. An earlier interrupted run had also
left a partial 111 image `features.pkl` behind, so the cache write now happens only after
the extraction loop completes.
