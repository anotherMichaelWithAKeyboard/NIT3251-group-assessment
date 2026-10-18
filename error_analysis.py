"""
Error analysis and report figures for the VGG16 + LSTM captioning model
(report sections 4.6, 5.1 and 5.2).

Decodes every one of the 810 held-out test images with greedy decoding, scores each
caption individually with sentence-level BLEU-4, and from that produces:

    error_analysis_results.csv    per-image caption and BLEU-4
    figure6_caption_examples.png  four high-scoring captions with their images
    figure7_error_cases.png       four low-scoring captions with their images

It also prints the aggregate numbers section 5.2 needs (BLEU-4 distribution, scores
bucketed by reference caption length, and the rate of near-duplicate generated captions),
and re-generates the three specific captions quoted in sections 4.6 and 5.1 so those
quotes can be checked against the model currently committed.

Run from the project root, after best_model.keras and tokenizer.pkl are in place:
    python error_analysis.py
"""

import json
import os
import pickle
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The three images quoted in the report, so the quoted captions can be verified against
# whichever model is currently committed.
QUOTED_IN_REPORT = {
    "1001773457_577c3a7d70": "section 4.6, two dogs",
    "1002674143_1b742ab4b8": "section 5.1, girl painting under a rainbow",
    "101669240_b2d3e7f17b": "section 5.1, skier beside displayed paintings",
}


def clean(mapping):
    """Identical to the notebook's clean(), so captions match what the model was trained on."""
    for captions in mapping.values():
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = re.sub(r"[^a-z\s]", " ", caption)
            caption = re.sub(r"\s+", " ", caption).strip()
            captions[i] = "startseq " + caption + " endseq"


def load_everything():
    with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "caption_config.json")) as f:
        config = json.load(f)

    with open(os.path.join(BASE_DIR, "captions.txt"), "r") as f:
        next(f)
        captions_doc = f.read()

    mapping = {}
    for line in captions_doc.split("\n"):
        if len(line) < 2:
            continue
        tokens = line.split(",")
        mapping.setdefault(tokens[0].split(".")[0], []).append(" ".join(tokens[1:]))
    clean(mapping)

    # Use the committed split rather than re-deriving it, so these are exactly the images
    # the reported BLEU figures were measured on.
    with open(os.path.join(BASE_DIR, "dataset_split.json")) as f:
        test = json.load(f)["test"]

    model = load_model(os.path.join(BASE_DIR, "best_model.keras"))
    return features, tokenizer, mapping, test, model, config["max_length"]


def predict_caption(model, image, tokenizer, max_length):
    """Greedy decoding, matching the notebook."""
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


def strip_tokens(caption):
    return [w for w in caption.split() if w not in ("startseq", "endseq")]


def show_example(ax, image_id, predicted, references, score):
    """Draw one image with its generated caption underneath."""
    img = Image.open(os.path.join(BASE_DIR, "Images", image_id + ".jpg"))
    ax.imshow(img)
    ax.axis("off")
    # Wrap by hand: matplotlib titles do not wrap, and these captions are long.
    words = predicted.split()
    lines, current = [], []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > 34:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    ax.set_title("\n".join(lines) + f"\n(BLEU-4 {score:.3f})", fontsize=9)


def main():
    print("Loading model, features and captions...", flush=True)
    features, tokenizer, mapping, test, model, max_length = load_everything()
    print(f"{len(test)} test images, max_length {max_length}", flush=True)

    smooth = SmoothingFunction().method1
    rows = []

    for n, image_id in enumerate(test, 1):
        refs = [strip_tokens(c) for c in mapping[image_id]]
        caption = predict_caption(model, features[image_id], tokenizer, max_length)
        pred = strip_tokens(caption)

        score = sentence_bleu(refs, pred, weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smooth)

        rows.append({
            "image_id": image_id,
            "generated": " ".join(pred),
            "bleu4": score,
            "generated_length": len(pred),
            "mean_reference_length": float(np.mean([len(r) for r in refs])),
        })

        if n % 100 == 0:
            print(f"  {n}/{len(test)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "error_analysis_results.csv"), index=False)

    # ---------------------------------------------------------------- aggregates
    print("\n" + "=" * 70)
    print("BLEU-4 DISTRIBUTION (sentence level, greedy decoding)")
    print("=" * 70)
    print(f"  mean    {df['bleu4'].mean():.4f}")
    print(f"  median  {df['bleu4'].median():.4f}")
    print(f"  worst   {df['bleu4'].min():.4f}")
    print(f"  best    {df['bleu4'].max():.4f}")
    print(f"  below 0.10: {(df['bleu4'] < 0.10).sum()} of {len(df)} "
          f"({100 * (df['bleu4'] < 0.10).mean():.1f}%)")
    print(f"  above 0.30: {(df['bleu4'] > 0.30).sum()} of {len(df)} "
          f"({100 * (df['bleu4'] > 0.30).mean():.1f}%)")

    print("\n" + "=" * 70)
    print("BLEU-4 BY REFERENCE CAPTION LENGTH")
    print("=" * 70)
    buckets = pd.cut(df["mean_reference_length"], [0, 9, 12, 15, 100],
                     labels=["short (<9)", "medium (9-12)", "long (12-15)", "very long (15+)"])
    print(df.groupby(buckets, observed=True)["bleu4"].agg(["count", "mean"]).round(4))

    print("\n" + "=" * 70)
    print("GENERIC CAPTION COLLAPSE")
    print("=" * 70)
    counts = Counter(df["generated"])
    repeated = sum(c for c in counts.values() if c > 1)
    print(f"  {len(counts)} unique captions across {len(df)} images")
    print(f"  {repeated} images ({100 * repeated / len(df):.1f}%) share a caption "
          f"with at least one other image")
    print("  most frequent generated captions:")
    for caption, count in counts.most_common(5):
        print(f"    {count:>3}x  {caption}")

    print("\n" + "=" * 70)
    print("CAPTIONS QUOTED IN THE REPORT (verify against the current model)")
    print("=" * 70)
    for image_id, where in QUOTED_IN_REPORT.items():
        if image_id in features:
            caption = " ".join(strip_tokens(
                predict_caption(model, features[image_id], tokenizer, max_length)))
            in_test = "test set" if image_id in set(test) else "NOT in test set"
            print(f"  {where} ({in_test})")
            print(f"    {caption}")

    # ---------------------------------------------------------------- figures
    ordered = df.sort_values("bleu4", ascending=False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, (_, row) in zip(axes, ordered.head(4).iterrows()):
        show_example(ax, row["image_id"], row["generated"],
                     mapping[row["image_id"]], row["bleu4"])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figure6_caption_examples.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, (_, row) in zip(axes, ordered.tail(4).iterrows()):
        show_example(ax, row["image_id"], row["generated"],
                     mapping[row["image_id"]], row["bleu4"])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "figure7_error_cases.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    print("\nwrote error_analysis_results.csv, figure6_caption_examples.png, "
          "figure7_error_cases.png", flush=True)


if __name__ == "__main__":
    main()
