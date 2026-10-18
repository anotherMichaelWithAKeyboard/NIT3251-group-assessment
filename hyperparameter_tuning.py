"""
Hyperparameter tuning for the VGG16 + LSTM captioning model (report section 4.5).

Three controlled runs, one variable changed at a time against the baseline, matching
Phase 3 of PLAN.md:

    A  baseline      LSTM 256, dropout 0.4
    B  wider LSTM    LSTM 512, dropout 0.4
    C  less dropout  LSTM 256, dropout 0.2

Everything else is held fixed: the same 80/10/10 image-level split (seed 42), the same
tokenizer fitted on training captions only, Adam, early stopping on validation loss with
patience 3, and a maximum of 20 epochs.

Each arm records its best validation loss, the epoch that produced it, wall-clock training
time, and a BLEU score measured with greedy decoding on a fixed subset of the test set.
A subset is used because full greedy+beam decoding over all 810 test images takes about
42 minutes; the arms only need to be ranked against each other, and the headline BLEU
figures in the report still come from the full-test-set evaluation in the notebook.

Results are written to hyperparameter_results.csv and hyperparameter_results.png.

Run from the project root:
    python hyperparameter_tuning.py
"""

import json
import os
import pickle
import re
import time
from math import ceil

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Input, add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from make_tuning_figure import write_figure

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# How many held-out images to decode when scoring each arm. 250 is enough to separate
# the arms while keeping each evaluation to roughly four minutes on CPU.
BLEU_SUBSET = 250

EPOCHS = 20
BATCH_SIZE = 32
SEED = 42

# The three arms. Only `units` and `dropout` vary.
ARMS = [
    {"name": "A: baseline (LSTM 256, dropout 0.4)", "units": 256, "dropout": 0.4},
    {"name": "B: wider LSTM (LSTM 512, dropout 0.4)", "units": 512, "dropout": 0.4},
    {"name": "C: less dropout (LSTM 256, dropout 0.2)", "units": 256, "dropout": 0.2},
]


def clean(mapping):
    """Lowercase, strip non-alphabetic characters, and add the sequence boundary tokens.

    Identical to the notebook's clean() so the tuning runs stay comparable to the
    baseline reported in section 4.2.
    """
    for captions in mapping.values():
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = re.sub(r"[^a-z\s]", " ", caption)
            caption = re.sub(r"\s+", " ", caption).strip()
            captions[i] = "startseq " + caption + " endseq"


def load_data():
    """Load cached VGG16 features and cleaned captions, then rebuild the 80/10/10 split."""
    with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    with open(os.path.join(BASE_DIR, "captions.txt"), "r") as f:
        next(f)
        captions_doc = f.read()

    mapping = {}
    for line in captions_doc.split("\n"):
        if len(line) < 2:
            continue
        tokens = line.split(",")
        image_id = tokens[0].split(".")[0]
        mapping.setdefault(image_id, []).append(" ".join(tokens[1:]))

    clean(mapping)

    # Same ordering and same seed as the notebook, so the arms train and are scored on
    # exactly the images the reported baseline used.
    image_ids = sorted(set(mapping.keys()) & set(features.keys()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(image_ids)

    train_end = int(len(image_ids) * 0.80)
    val_end = int(len(image_ids) * 0.90)
    train = image_ids[:train_end]
    val = image_ids[train_end:val_end]
    test = image_ids[val_end:]

    train_captions = [c for image_id in train for c in mapping[image_id]]

    tokenizer = Tokenizer(oov_token="unk")
    tokenizer.fit_on_texts(train_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in train_captions)

    return features, mapping, train, val, test, tokenizer, vocab_size, max_length


def data_generator(keys, mapping, features, tokenizer, max_length, batch_size,
                   shuffle=False, seed=SEED):
    """Yield ((image features, partial caption), next word) batches.

    Labels are integer word IDs rather than one-hot vectors. The loss is identical --
    sparse_categorical_crossentropy on an index equals categorical_crossentropy on its
    one-hot -- but a one-hot batch of this vocabulary costs about 54 MB against 0.01 MB,
    and avoiding that allocation is a large speedup across three full training runs.
    """
    keys = list(keys)
    rng = np.random.default_rng(seed)

    while True:
        if shuffle:
            rng.shuffle(keys)

        X1, X2, y = [], [], []
        images_in_batch = 0

        for image_id in keys:
            image_features = np.asarray(features[image_id]).reshape(-1)

            for caption in mapping[image_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
                    X1.append(image_features)
                    X2.append(in_seq)
                    y.append(seq[i])

            images_in_batch += 1

            if images_in_batch == batch_size:
                yield ((np.asarray(X1, dtype=np.float32),
                        np.asarray(X2, dtype=np.int32)),
                       np.asarray(y, dtype=np.int32))
                X1, X2, y = [], [], []
                images_in_batch = 0

        if images_in_batch > 0:
            yield ((np.asarray(X1, dtype=np.float32),
                    np.asarray(X2, dtype=np.int32)),
                   np.asarray(y, dtype=np.int32))


def build_model(vocab_size, max_length, units, dropout):
    """The encoder-decoder from section 2.4, with the LSTM width and dropout rate open."""
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(dropout)(inputs1)
    fe2 = Dense(units, activation="relu")(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(dropout)(se1)
    se3 = LSTM(units)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(units, activation="relu")(decoder1)
    outputs = Dense(vocab_size, activation="softmax")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model


def predict_caption(model, image, tokenizer, max_length):
    """Greedy decoding: take the highest-probability word at every step."""
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = int(np.argmax(model.predict([image, sequence], verbose=0)))
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text


def score_bleu(model, test_ids, mapping, features, tokenizer, max_length):
    """Corpus BLEU-1 to BLEU-4 with greedy decoding over a fixed slice of the test set."""
    references, predictions = [], []

    for image_id in test_ids:
        refs = [[w for w in c.split() if w not in ("startseq", "endseq")]
                for c in mapping[image_id]]
        caption = predict_caption(model, features[image_id], tokenizer, max_length)
        pred = [w for w in caption.split() if w not in ("startseq", "endseq")]
        references.append(refs)
        predictions.append(pred)

    smooth = SmoothingFunction().method1
    scores = {}
    for n in range(1, 5):
        weights = tuple([1 / n] * n + [0] * (4 - n))
        scores[f"BLEU-{n}"] = corpus_bleu(references, predictions,
                                          weights=weights, smoothing_function=smooth)
    return scores


def main():
    print("Loading features and captions...", flush=True)
    (features, mapping, train, val, test,
     tokenizer, vocab_size, max_length) = load_data()

    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"vocab={vocab_size} max_length={max_length}", flush=True)

    steps = ceil(len(train) / BATCH_SIZE)
    val_steps = ceil(len(val) / BATCH_SIZE)
    bleu_ids = test[:BLEU_SUBSET]

    rows = []

    for arm in ARMS:
        print(f"\n{'=' * 70}\n{arm['name']}\n{'=' * 70}", flush=True)

        train_gen = data_generator(train, mapping, features, tokenizer,
                                   max_length, BATCH_SIZE, shuffle=True)
        val_gen = data_generator(val, mapping, features, tokenizer,
                                 max_length, BATCH_SIZE, shuffle=False)

        model = build_model(vocab_size, max_length, arm["units"], arm["dropout"])

        early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min",
                                       restore_best_weights=True, verbose=1)

        started = time.time()
        history = model.fit(
            train_gen,
            steps_per_epoch=steps,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=[early_stopping],
            verbose=2,
        )
        train_seconds = time.time() - started

        val_losses = history.history["val_loss"]
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val_loss = float(min(val_losses))

        print(f"scoring BLEU on {len(bleu_ids)} held-out images...", flush=True)
        bleu_started = time.time()
        scores = score_bleu(model, bleu_ids, mapping, features, tokenizer, max_length)
        bleu_seconds = time.time() - bleu_started

        row = {
            "Configuration": arm["name"],
            "LSTM units": arm["units"],
            "Dropout": arm["dropout"],
            "Epochs run": len(val_losses),
            "Best epoch": best_epoch,
            "Best val loss": round(best_val_loss, 4),
            "Train loss at best": round(float(history.history["loss"][best_epoch - 1]), 4),
            "Train minutes": round(train_seconds / 60, 1),
            "BLEU minutes": round(bleu_seconds / 60, 1),
        }
        row.update({k: round(v, 4) for k, v in scores.items()})
        rows.append(row)

        # Written after every arm so a long run that gets interrupted still leaves
        # usable results for the arms that finished.
        pd.DataFrame(rows).to_csv(
            os.path.join(BASE_DIR, "hyperparameter_results.csv"), index=False)

        model.save(os.path.join(
            BASE_DIR, f"tuning_arm_{arm['units']}u_{str(arm['dropout']).replace('.', '')}d.keras"))

        print(json.dumps(row, indent=2), flush=True)

    df = pd.DataFrame(rows)
    print("\nFinal results\n", df.to_string(index=False), flush=True)

    # Validation loss and BLEU-4 side by side: the arm that generalises best is not
    # always the arm with the lowest loss, and the report should show both.
    write_figure(df, os.path.join(BASE_DIR, "hyperparameter_results.png"))
    print("wrote hyperparameter_results.csv and hyperparameter_results.png", flush=True)


if __name__ == "__main__":
    main()
