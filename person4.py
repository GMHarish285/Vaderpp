import sys
import os
import re
import json
import math
import random
import string
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as NLTK_VADER
from nltk.sentiment.vader import VaderConstants, SentiText
nltk.download('vader_lexicon', quiet=True)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, f1_score
from scipy.stats import spearmanr

def rating_to_label(r):
    if r >= 4:
        return "pos"
    if r <= 2:
        return "neg"
    return "neu"

def label_by_compound(c):
    if c >= 0.05:
        return "pos"
    if c <= -0.05:
        return "neg"
    return "neu"

def tokenize(s):
    return [t.lower() for t in re.findall(r"[A-Za-z']+|[!?.,]", s)]

class CustomVaderConstants(VaderConstants):
    def _init_(self, booster_dict_override=None):
        super()._init_()
        if booster_dict_override:
            self.BOOSTER_DICT = booster_dict_override

class CustomVader(NLTK_VADER):
    def _init_(self, booster_dict_override=None, *args, **kwargs):
        self.constants = CustomVaderConstants(booster_dict_override=booster_dict_override)
        super()._init_(*args, **kwargs)
        self.constants = self._sid.constants = self.constants
    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        words_and_emoticons = sentitext.words_and_emoticons
        is_cap_diff = self.constants.is_cap_diff(words_and_emoticons)
        item_lower = item.lower()
        if item_lower in self.lexicon:
            valence = self.lexicon[item_lower]
            if item.isupper() and is_cap_diff:
                valence += self.constants.C_INCR if valence > 0 else -self.constants.C_INCR
            for start_i in range(0, 3):
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    s = self._scalar_inc_dec_context(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff, offset=(start_i + 1), head_item=item)
                    if s != 0:
                        s = s * (0.95 if start_i == 1 else 1.0)
                        valence += s
            valence = self._never_check(valence, words_and_emoticons, 0, i)
            valence = self._never_check(valence, words_and_emoticons, 1, i)
            valence = self._idioms_check(valence, words_and_emoticons, i)
            if i > 1 and not words_and_emoticons[i - 1].lower() in self.lexicon:
                two_one = f"{words_and_emoticons[i - 2].lower()} {words_and_emoticons[i - 1].lower()}"
                three_two = f"{words_and_emoticons[i - 3].lower()} {words_and_emoticons[i - 2].lower()}" if i > 2 else ""
                if two_one in self.constants.BOOSTER_DICT or three_two in self.constants.BOOSTER_DICT:
                    valence += self.constants.B_DECR
        return valence
    def _scalar_inc_dec_context(self, word, valence, is_cap_diff, offset=1, head_item=None):
        wlower = word.lower()
        base_scalar = 0.0
        if wlower in self.constants.BOOSTER_DICT:
            base_scalar = self.constants.BOOSTER_DICT[wlower]
            if valence < 0:
                base_scalar *= -1
            if word.isupper() and is_cap_diff:
                base_scalar += self.constants.C_INCR if valence > 0 else -self.constants.C_INCR
        head_token = None
        if nlp is not None and head_item is not None:
            try:
                doc = nlp(head_item)
                head_token = doc[0]
            except Exception:
                head_token = None
        ctx_scalar = modifier_scalar(wlower, head_token, offset=offset)
        if ctx_scalar != 0.0:
            scalar = ctx_scalar
            if valence < 0:
                scalar *= -1
        else:
            scalar = base_scalar
        return scalar

from nltk.sentiment.vader import SentimentIntensityAnalyzer
vc = VaderConstants()
sia = SentimentIntensityAnalyzer()   # load VADER analyzer
vader_lex = set(sia.lexicon.keys())  # lexicon words
seed_boosters = set(vc.BOOSTER_DICT.keys())

def base_anchor_valence(word):
    return sia.lexicon.get(word.lower(), 0.0)



def find_candidates(text, window=3):
    toks = tokenize(text)
    anchors = [i for i,t in enumerate(toks) if t in vader_lex]
    cands = []
    for i in anchors:
        for off in range(1, window+1):
            if i-off >= 0:
                cands.append(("L", toks[i-off], toks[i]))
            if i+off < len(toks):
                cands.append(("R", toks[i+off], toks[i]))
    return cands



def extract_training_rows(df, window=2):
    rows = []
    for r in df.itertuples():
        toks = tokenize(r.text)
        for i,t in enumerate(toks):
            if t in vader_lex:
                v = base_anchor_valence(t)
                ctx = []
                for off in range(1, window+1):
                    if i-off >= 0:
                        ctx.append(toks[i-off])
                    if i+off < len(toks):
                        ctx.append(toks[i+off])
                rows.append({"anchor": t, "val": v, "ctx": ctx, "rating": r.rating, "label": r.label})
    return rows

def featurize(rows, cand_set):
    if not rows:
        return np.zeros((0,1)), np.zeros((0,)), []
    X, y = [], []
    for r in rows:
        feats = {}
        feats["base_val"] = r["val"]
        ctxset = set(r["ctx"])
        for m in cand_set:
            feats[f"m::{m}"] = 1 if m in ctxset else 0
        X.append(feats)
        y.append(r["rating"])
    feat_names = sorted(X[0].keys())
    X_mat = np.array([[f[k] for k in feat_names] for f in X], dtype=float)
    return X_mat, np.array(y, dtype=float), feat_names

def pairwise_delta_weight(df, modifier, max_pairs=500, rating_to_vader=0.2):
    pairs = []
    texts = df["text"].tolist()
    ratings = df["rating"].values
    for i, t in enumerate(texts):
        toks = tokenize(t)
        has_mod = modifier in toks
        anchors = [tok for tok in toks if tok in vader_lex]
        if not anchors:
            continue
        for j in range(i+1, len(texts)):
            if abs(len(texts[j]) - len(t)) > 40:
                continue
            toks_j = tokenize(texts[j])
            if not any(a in toks_j for a in anchors):
                continue
            has_mod_j = modifier in toks_j
            if has_mod != has_mod_j:
                d = (ratings[i] - ratings[j]) if has_mod else (ratings[j] - ratings[i])
                pairs.append(d)
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        return 0.0, 0
    delta_rating = float(np.mean(pairs))
    delta_vader = float(np.clip(delta_rating * rating_to_vader, -0.8, 0.8))
    return delta_vader, len(pairs)

distance_decay = {1: 1.0, 2: 0.95, 3: 0.9}
context_overrides = {("critically", "ADJ", "important"): 0.15, ("critically", "ADJ", "ill"): -0.20}

def modifier_scalar(mod, head_token, offset=1):
    base = custom_booster.get(mod, 0.0) if "custom_booster" in globals() else 0.0
    decay = distance_decay.get(offset, 0.85)
    scalar = base * decay
    if head_token is not None and head_token.pos_ in ("ADJ","VERB","ADV","NOUN"):
        key = (mod, head_token.pos_, head_token.lemma_.lower())
        if key in context_overrides:
            scalar = context_overrides[key]
    return scalar

data = [
    {"text": "Insanely good gameplay but barely functional menus.", "rating": 3},
    {"text": "Very smooth experience, extremely fun!", "rating": 5},
    {"text": "Somewhat disappointing and kinda buggy.", "rating": 2},
    {"text": "Critically important update, massively improved stability.", "rating": 5},
    {"text": "Critically ill-designed UI; barely usable.", "rating": 1},
    {"text": "Good, but not great.", "rating": 3},
    {"text": "Insanely OP character, ridiculously unbalanced.", "rating": 2},
    {"text": "Slightly better than before.", "rating": 3}
]
df = pd.DataFrame(data)
df["label"] = df["rating"].apply(rating_to_label)

sia_base = NLTK_VADER()
df_scores = df["text"].apply(sia_base.polarity_scores)
df["compound"] = df_scores.apply(lambda d: d["compound"])
df["vader_label"] = df["compound"].apply(label_by_compound)

cand_counts = Counter()
pair_examples = defaultdict(list)
for row in df.itertuples():
    cands = find_candidates(row.text)
    for side, m, anchor in cands:
        if not re.match(r"^[a-z']+$", m):
            continue
        if m in vader_lex:
            continue
        cand_counts[m] += 1
        if len(pair_examples[m]) < 3:
            pair_examples[m].append((row.text, anchor, side))
top_candidates = [w for w,c in cand_counts.most_common(200) if w not in seed_boosters]
K = min(50, len(top_candidates))
cand_set = set(top_candidates[:K]) | set(list(seed_boosters)[:50])

train_rows = extract_training_rows(df)
X, y, feat_names = featurize(train_rows, cand_set)
learned_weights = {}
if X.shape[0] > 0:
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    coefs = dict(zip(feat_names, model.coef_))
    rating_to_vader = 0.2
    for k,v in coefs.items():
        if k.startswith("m::"):
            m = k.split("::",1)[1]
            learned_weights[m] = float(np.clip(v * rating_to_vader, -0.8, 0.8))

pair_stats = {}
for m in list(cand_set)[:50]:
    w, n = pairwise_delta_weight(df, m)
    if n >= 5:
        pair_stats[m] = (w, n)

final_weights = {}
for m, (w, n) in pair_stats.items():
    final_weights[m] = w
for m, w in learned_weights.items():
    if m not in final_weights:
        final_weights[m] = w
for m in list(final_weights.keys()):
    if abs(final_weights[m]) < 0.03:
        final_weights.pop(m)

custom_booster = dict(vc.BOOSTER_DICT)
for m, w in final_weights.items():
    custom_booster[m] = w

try:
    custom_sia = CustomVader(booster_dict_override=custom_booster)
    runtime_sia = custom_sia
except Exception:
    runtime_sia = NLTK_VADER()

def emit_json(sentence, analyzer):
    s = analyzer.polarity_scores(sentence)
    return {"sentence": sentence, "score": {"neg": round(s["neg"], 3), "neu": round(s["neu"], 3), "pos": round(s["pos"], 3), "compound": round(s["compound"], 2)}}

if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "The food was absolutely fantastic!"
    print(json.dumps(emit_json(sentence, runtime_sia), ensure_ascii=False))

def analyze(sentence):
    return emit_json(sentence, runtime_sia)['score']
