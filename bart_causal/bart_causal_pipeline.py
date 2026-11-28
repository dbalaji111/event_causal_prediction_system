#!/usr/bin/env python3
"""
BART causal-link model for the causal-chain view (redflag_market_kit/demo/export_causal_chains.py).

The chain view links one event's gold_effect text to a LATER event's gold_cause text. Today that link score is TF-IDF
word overlap (a placeholder). This pipeline trains a BART cross-encoder that reads both texts and says whether event A
plausibly led to event B, then scores every candidate pair so export_causal_chains.py can use it (--bart-scores).

Stages (each is one batch job on the supercomputer; see jobs/):
    pairs   build candidate pairs: for every event A, later events B within 120 days, the closest by TF-IDF plus random
            ones (easy negatives)                                                                    CPU, minutes
    label   label a sample of pairs: "did A plausibly contribute to causing B?"
              --labeler qwen       a local Qwen (or any chat LLM) as the judge; P(Yes) from its next-token logits  GPU
              --labeler heuristic  silver labels from similarity alone (for a dry run; weaker)        CPU
    train   fine-tune facebook/bart-large-mnli as a pair classifier (entailment = link). Time-based split: the last
            20% of source dates are held out                                                          GPU, ~1 h
    score   score every candidate pair with the trained model (or --zero-shot with the base MNLI model)   GPU
    eval    compare BART with TF-IDF and embedding similarity on the held-out labels                  CPU

    python bart_causal_pipeline.py pairs --out runs/r1
    python bart_causal_pipeline.py label --out runs/r1 --labeler qwen --llm /models/Qwen2.5-7B-Instruct --n 6000
    python bart_causal_pipeline.py train --out runs/r1 --model /models/bart-large-mnli
    python bart_causal_pipeline.py score --out runs/r1
    python bart_causal_pipeline.py eval  --out runs/r1
    python redflag_market_kit/demo/export_causal_chains.py --bart-scores runs/r1/bart_link_scores.csv

Labels from an LLM judge are a teacher signal, not ground truth; the eval stage reports agreement on held-out labels,
which measures how well BART reproduces the judge, not whether the links are causal in the market.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "redflag_market_kit" / "demo"))
sys.path.insert(0, str(REPO / "redflag_market_kit"))

SEED = 7
LAG_DAYS = 120


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


# ───────────────────────────── stage 1: candidate pairs ─────────────────────────────
def short(text, n=320):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def stage_pairs(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from export_causal_chains import _headline_or_snippet, load_events
    from redflag_kit.corpus import parse_embedding

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load_events()
    if a.limit:
        df = df.iloc[: a.limit].reset_index(drop=True)
    log(f"{len(df)} events with cause and effect text, {df['Date'].min().date()} to {df['Date'].max().date()}")
    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    vec.fit(pd.concat([df["gold_effect"], df["gold_cause"]], ignore_index=True))
    E, C = vec.transform(df["gold_effect"]), vec.transform(df["gold_cause"])
    emb = np.stack([parse_embedding(x) if parse_embedding(x) is not None else np.zeros(384, np.float32) for x in df["gold_general_embedding"]])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
    text = [short(_headline_or_snippet(h, c), 240) for h, c in zip(df["Headline"], df["Content"])]
    dates = df["Date"].values
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(len(df)):
        lo = np.searchsorted(dates, dates[i], side="right")
        hi = np.searchsorted(dates, dates[i] + np.timedelta64(a.lag_days, "D"), side="right")
        if lo >= hi:
            continue
        sims = (E[i] @ C[lo:hi].T).toarray().ravel()
        top = list(np.argsort(-sims)[: a.top])
        rest = [k for k in range(hi - lo) if k not in set(top)]
        rand = list(rng.choice(rest, size=min(a.random, len(rest)), replace=False)) if rest else []
        for k, kind in [(k, "top") for k in top] + [(k, "random") for k in rand]:
            j = lo + int(k)
            rows.append((df.at[i, "Doc_ID"], df.at[j, "Doc_ID"], df.at[i, "Date"].date(), df.at[j, "Date"].date(),
                         int((dates[j] - dates[i]) / np.timedelta64(1, "D")), text[i], short(df.at[i, "gold_effect"]),
                         text[j], short(df.at[j, "gold_cause"]), float(sims[k]), float(emb[i] @ emb[j]),
                         df.at[i, "assigned_topic_bert"], df.at[j, "assigned_topic_bert"], kind))
    p = pd.DataFrame(rows, columns=["src", "tgt", "src_date", "tgt_date", "days_gap", "src_text", "effect", "tgt_text", "cause",
                                    "tfidf", "emb_cos", "src_topic", "tgt_topic", "kind"])
    p.insert(0, "pair_id", np.arange(len(p)))
    p.to_csv(out / "pairs.csv", index=False)
    log(f"{len(p)} candidate pairs ({(p.kind == 'top').sum()} closest, {(p.kind == 'random').sum()} random) -> {out / 'pairs.csv'}")


# ───────────────────────────── stage 2: labels ─────────────────────────────
JUDGE_SYSTEM = ("You are an expert analyst of commodity markets. You judge whether one news event plausibly helped cause "
                "a later one through the gold market (prices, demand, policy, risk sentiment). A shared topic or similar "
                "wording is NOT enough: there must be a plausible causal path from A's effect to B's cause.")


def judge_prompt(r):
    return (f"Event A ({r.src_date}): {r.src_text}\nEffect of A: {r.effect}\n\n"
            f"Event B ({r.tgt_date}, {r.days_gap} days later): {r.tgt_text}\nCause of B: {r.cause}\n\n"
            "Did event A plausibly contribute to causing event B? Answer with one word: Yes or No.")


def sample_for_labels(p, n):
    if n >= len(p):
        return p
    top, rnd = p[p.kind == "top"], p[p.kind == "random"]
    n_rnd = min(len(rnd), n // 4)                                    # a quarter easy negatives, the rest the hard cases
    return pd.concat([top.sample(n - n_rnd, random_state=SEED), rnd.sample(n_rnd, random_state=SEED)]).sort_values("pair_id")


def stage_label(a):
    out = Path(a.out)
    p = pd.read_csv(out / "pairs.csv")
    s = sample_for_labels(p, a.n)
    if a.labeler == "heuristic":
        t_hi, e_hi = p.loc[p.kind == "top", "tfidf"].quantile(.75), p["emb_cos"].quantile(.75)
        t_lo = p["tfidf"].quantile(.25)
        pos = (s.kind == "top") & (s.tfidf >= t_hi) & (s.emb_cos >= e_hi)
        neg = (s.kind == "random") | (s.tfidf <= t_lo)
        lab = s[pos | neg].assign(label=lambda d: pos[d.index].astype(int), p_yes=lambda d: pos[d.index].astype(float), labeler="heuristic")
    else:
        lab = s.assign(p_yes=qwen_p_yes(s, a), labeler=f"llm:{Path(a.llm).name}")
        lab["label"] = (lab["p_yes"] >= a.threshold).astype(int)
    lab[["pair_id", "label", "p_yes", "labeler"]].to_csv(out / "labels.csv", index=False)
    rate = lab.label.mean()
    log(f"{len(lab)} labelled pairs, {rate:.1%} linked ({lab.labeler.iloc[0]}) -> {out / 'labels.csv'}")
    if rate < 0.02 or rate > 0.98:
        q = lab["p_yes"].quantile([.5, .8, .9]).round(3).tolist()
        log(f"WARNING: almost every pair got the same label. P(Yes) median {q[0]}, 80th pct {q[1]}, 90th pct {q[2]}. "
            "Check the judge model (an instruct/chat model is needed), or set --threshold, before training.")


def qwen_p_yes(s, a):
    """P(Yes) from the judge's next-token logits after the chat prompt: one forward pass per pair, no sampling."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.llm, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import transformers
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}   # the keyword was renamed in 4.56
    model = AutoModelForCausalLM.from_pretrained(a.llm, device_map="auto" if torch.cuda.is_available() else None, **dkw)
    model.eval()
    ids = lambda w: list({tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w)})
    yes, no = ids("Yes"), ids("No")
    texts = [tok.apply_chat_template([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": judge_prompt(r)}],
                                     tokenize=False, add_generation_prompt=True) for r in s.itertuples()]
    out = []
    dev = next(model.parameters()).device
    for b in range(0, len(texts), a.batch):
        enc = tok(texts[b:b + a.batch], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        py, pn = torch.logsumexp(lg[:, yes], 1), torch.logsumexp(lg[:, no], 1)
        out.extend(torch.sigmoid(py - pn).cpu().tolist())
        if (b // a.batch) % 20 == 0:
            log(f"judged {min(b + a.batch, len(texts))}/{len(texts)}")
    return out


# ───────────────────────────── stage 3: train ─────────────────────────────
def pair_text(r):
    return f"{r.src_text} Effect: {r.effect}", f"{r.tgt_text} Cause: {r.cause}"


def link_label_ids(config):
    """For an NLI model, train 'linked' as entailment and 'not linked' as contradiction, keeping its head; else 2 labels."""
    l2i = {k.lower(): v for k, v in (config.label2id or {}).items()}
    if "entailment" in l2i and "contradiction" in l2i:
        return l2i["entailment"], l2i["contradiction"], config.num_labels
    return 1, 0, 2


def load_model(path, num_labels=None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    kw = {} if num_labels is None else {"num_labels": num_labels, "ignore_mismatched_sizes": True}
    return tok, AutoModelForSequenceClassification.from_pretrained(path, **kw)


def batches(df, tok, bs, max_len, shuffle=False):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    for b in range(0, len(idx), bs):
        part = df.iloc[idx[b:b + bs]]
        prem, hyp = zip(*[pair_text(r) for r in part.itertuples()])
        enc = tok(list(prem), list(hyp), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield part, enc


def predict(model, tok, df, bs, max_len, pos_id):
    import torch
    dev = next(model.parameters()).device
    model.eval(); out = []
    use_amp = dev.type == "cuda"
    with torch.no_grad():
        for _, enc in batches(df, tok, bs, max_len):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg = model(**enc).logits.float()
            out.extend(torch.softmax(lg, -1)[:, pos_id].cpu().tolist())
    return np.array(out)


def metrics(y, s):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    y = np.asarray(y); s = np.asarray(s)
    if len(set(y)) < 2:
        return {"n": int(len(y)), "note": "one class only"}
    k = max(1, int(y.sum()))
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "f1@0.5": float(f1_score(y, s >= 0.5)), f"precision@{k}": float(y[np.argsort(-s)[:k]].mean())}


def split_by_time(lab):
    cut = pd.to_datetime(lab["src_date"]).quantile(0.8)
    tr = lab[pd.to_datetime(lab["src_date"]) < cut]; va = lab[pd.to_datetime(lab["src_date"]) >= cut]
    return tr, va, cut


def stage_train(a):
    import torch
    from transformers import get_linear_schedule_with_warmup
    seed_all()
    out = Path(a.out)
    lab = pd.read_csv(out / "labels.csv").merge(pd.read_csv(out / "pairs.csv"), on="pair_id")
    tr, va, cut = split_by_time(lab)
    log(f"train {len(tr)} pairs (source before {cut.date()}), validate {len(va)}; linked {tr.label.mean():.1%} / {va.label.mean():.1%}")
    if tr["label"].nunique() < 2 or va["label"].nunique() < 2:
        sys.exit("Training needs both linked and not-linked pairs in the training and validation sets. "
                 "Re-run the label stage (more pairs with --n, or a different --threshold).")
    tok, model = load_model(a.model)
    pos_id, neg_id, n_lab = link_label_ids(model.config)
    if n_lab != model.config.num_labels:
        tok, model = load_model(a.model, n_lab)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    use_amp = dev.type == "cuda"
    w = torch.zeros(model.config.num_labels, device=dev)
    pr = max(tr.label.mean(), 1e-3)
    w[pos_id], w[neg_id] = 0.5 / pr, 0.5 / max(1 - pr, 1e-3)        # balance linked and not linked
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(tr) / a.batch / a.accum) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run = 0.0
        for k, (part, enc) in enumerate(batches(tr, tok, a.batch, a.max_len, shuffle=True), 1):
            enc = {kk: v.to(dev) for kk, v in enc.items()}
            y = torch.tensor(np.where(part["label"].values == 1, pos_id, neg_id), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                lg = model(**enc).logits
            loss = lossf(lg.float(), y) / a.accum
            loss.backward(); run += loss.item()
            if k % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if k % 200 == 0:
                log(f"epoch {ep} step {k}: loss {run / k * a.accum:.4f}")
        s = predict(model, tok, va, a.batch * 2, a.max_len, pos_id)
        m = metrics(va["label"], s); m.update(epoch=ep, minutes=round((time.time() - t0) / 60, 1))
        hist.append(m); log(f"epoch {ep} validation: {json.dumps(m)}")
        score = m.get("ap", 0)
        if score > best:
            best = score
            model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
            json.dump({"pos_label_id": pos_id, "neg_label_id": neg_id, "base_model": str(a.model), "max_len": a.max_len},
                      open(out / "model" / "link_head.json", "w"))
    json.dump({"history": hist, "best_ap": best, "split_date": str(cut.date()), "n_train": len(tr), "n_val": len(va),
               "args": {k: str(v) for k, v in vars(a).items() if k != "func"}}, open(out / "train_metrics.json", "w"), indent=1)
    log(f"best validation AP {best:.3f}; model -> {out / 'model'}")


# ───────────────────────────── stage 4: score ─────────────────────────────
def stage_score(a):
    import torch
    out = Path(a.out)
    path = a.model if a.zero_shot else str(out / "model")
    tok, model = load_model(path)
    head = out / "model" / "link_head.json"
    pos_id = json.load(open(head))["pos_label_id"] if (not a.zero_shot and head.exists()) else link_label_ids(model.config)[0]
    max_len = json.load(open(head))["max_len"] if (not a.zero_shot and head.exists()) else a.max_len
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    p = pd.read_csv(out / "pairs.csv")
    log(f"scoring {len(p)} pairs with {'zero-shot ' if a.zero_shot else ''}{path}")
    s = predict(model, tok, p, a.batch, max_len, pos_id)
    res = p[["src", "tgt", "days_gap", "tfidf"]].assign(score=np.round(s, 5))
    name = "bart_link_scores_zeroshot.csv" if a.zero_shot else "bart_link_scores.csv"
    res.to_csv(out / name, index=False)
    log(f"-> {out / name}  (mean score {s.mean():.3f}, {(s >= 0.5).mean():.1%} of pairs at or above 0.5)")


# ───────────────────────────── stage 5: eval ─────────────────────────────
def stage_eval(a):
    out = Path(a.out)
    lab = pd.read_csv(out / "labels.csv").merge(pd.read_csv(out / "pairs.csv"), on="pair_id")
    _, va, cut = split_by_time(lab)
    rows = {"TF-IDF (current placeholder)": metrics(va.label, va.tfidf), "Embedding cosine": metrics(va.label, va.emb_cos)}
    for name, f in (("BART zero-shot", "bart_link_scores_zeroshot.csv"), ("BART fine-tuned", "bart_link_scores.csv")):
        if (out / f).exists():
            sc = pd.read_csv(out / f)[["src", "tgt", "score"]]
            m = va.merge(sc, on=["src", "tgt"], how="left")
            rows[name] = metrics(m.label, m.score.fillna(0))
    print(f"\nHeld-out pairs (source on/after {cut.date()}): {len(va)}, linked {va.label.mean():.1%}")
    print(pd.DataFrame(rows).T.round(3).to_string())
    json.dump(rows, open(out / "eval.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("pairs"); s.add_argument("--out", required=True); s.add_argument("--top", type=int, default=20)
    s.add_argument("--random", type=int, default=5); s.add_argument("--lag-days", type=int, default=LAG_DAYS)
    s.add_argument("--limit", type=int, default=0, help="first N events only (quick tests)"); s.set_defaults(func=stage_pairs)
    s = sub.add_parser("label"); s.add_argument("--out", required=True); s.add_argument("--labeler", choices=["qwen", "heuristic"], default="qwen")
    s.add_argument("--llm", help="local path of the judge model, e.g. /models/Qwen2.5-7B-Instruct"); s.add_argument("--n", type=int, default=6000)
    s.add_argument("--batch", type=int, default=16); s.add_argument("--threshold", type=float, default=0.5); s.set_defaults(func=stage_label)
    s = sub.add_parser("train"); s.add_argument("--out", required=True); s.add_argument("--model", default="facebook/bart-large-mnli")
    s.add_argument("--epochs", type=int, default=3); s.add_argument("--batch", type=int, default=16); s.add_argument("--accum", type=int, default=2)
    s.add_argument("--lr", type=float, default=1e-5); s.add_argument("--max-len", type=int, default=256); s.set_defaults(func=stage_train)
    s = sub.add_parser("score"); s.add_argument("--out", required=True); s.add_argument("--model", default="facebook/bart-large-mnli")
    s.add_argument("--zero-shot", action="store_true", help="score with the base MNLI model, no fine-tuning (a baseline)")
    s.add_argument("--batch", type=int, default=64); s.add_argument("--max-len", type=int, default=256); s.set_defaults(func=stage_score)
    s = sub.add_parser("eval"); s.add_argument("--out", required=True); s.set_defaults(func=stage_eval)
    a = ap.parse_args()
    if a.stage == "label" and a.labeler == "qwen" and not a.llm:
        ap.error("--labeler qwen needs --llm /path/to/model")
    a.func(a)


if __name__ == "__main__":
    main()
