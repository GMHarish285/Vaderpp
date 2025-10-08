import re
import string
import math
from collections import Counter, defaultdict
import itertools
import numpy as np
from gensim.models import Word2Vec 
from gensim.downloader import load as gensim_load 
from datasets import load_dataset

# 1. Data Loading

def load_corpus(domain="amazon", sample_size=5000):
    """
    Load a domain-specific corpus.
    For demonstration: Amazon polarity reviews (Electronics).
    """
    print("Loading domain corpus...")
    dataset = load_dataset("amazon_polarity", split=f"train[:{sample_size}]")
    texts = [x["content"] for x in dataset]
    return texts

# 2. Preprocessing

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_corpus(texts):
    preprocessed = [preprocess_text(t) for t in texts]
    tokenized = [t.split() for t in preprocessed]
    return tokenized

# 3. PMI-based Lexicon

def build_pmi_lexicon(tokenized_texts, positive_seed, negative_seed, window_size=5):
    co_occur_counts = defaultdict(lambda: defaultdict(int))
    word_counts = Counter(itertools.chain(*tokenized_texts))
    total_windows = 0

    for sentence in tokenized_texts:
        for i, target_word in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            window = sentence[start:end]
            for context_word in window:
                if context_word != target_word:
                    co_occur_counts[target_word][context_word] += 1
                    total_windows += 1

    def pmi_score(word, seed_words):
        scores = []
        for seed in seed_words:
            co = co_occur_counts[word].get(seed, 0)
            if co == 0:
                continue
            p_word = word_counts[word] / total_windows
            p_seed = word_counts[seed] / total_windows
            pmi = math.log2(co / (p_word * p_seed))
            scores.append(pmi)
        return np.mean(scores) if scores else 0.0

    lexicon = {}
    for word in word_counts:
        pos_score = pmi_score(word, positive_seed)
        neg_score = pmi_score(word, negative_seed)
        lexicon[word] = pos_score - neg_score
    return lexicon

# 4. Word2Vec Embedding Lexicon

def build_embedding_lexicon(tokenized_texts, positive_seed, negative_seed):
    print("Training Word2Vec on domain corpus...")
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=5, workers=4)
    pos_vec = np.mean([w2v_model.wv[w] for w in positive_seed if w in w2v_model.wv], axis=0)
    neg_vec = np.mean([w2v_model.wv[w] for w in negative_seed if w in w2v_model.wv], axis=0)
    sentiment_axis = pos_vec - neg_vec

    lexicon = {}
    for word in w2v_model.wv.index_to_key:
        vec = w2v_model.wv[word]
        score = np.dot(vec, sentiment_axis) / (np.linalg.norm(vec) * np.linalg.norm(sentiment_axis))
        lexicon[word] = float(score)
    return lexicon

# 5. FastText Lexicon

def build_fasttext_lexicon(word_counts, positive_seed, negative_seed):
    print("Loading pretrained FastText vectors...")
    fasttext_model = gensim_load("fasttext-wiki-news-subwords-300")
    fast_pos_vec = np.mean([fasttext_model[w] for w in positive_seed if w in fasttext_model], axis=0)
    fast_neg_vec = np.mean([fasttext_model[w] for w in negative_seed if w in fasttext_model], axis=0)

    lexicon = {}
    for word in word_counts:
        if word in fasttext_model:
            vec = fasttext_model[word]
            pos_sim = np.dot(vec, fast_pos_vec) / (np.linalg.norm(vec) * np.linalg.norm(fast_pos_vec))
            neg_sim = np.dot(vec, fast_neg_vec) / (np.linalg.norm(vec) * np.linalg.norm(fast_neg_vec))
            lexicon[word] = pos_sim - neg_sim
        else:
            lexicon[word] = 0.0
    return lexicon

# 6. Normalize and Combine Lexicons

def normalize_dict(d):
    vals = np.array(list(d.values()))
    if np.max(np.abs(vals)) == 0:
        return d
    factor = 1 / np.max(np.abs(vals))
    return {k: float(v * factor) for k, v in d.items()}

def combine_lexicons(pmi_lex, emb_lex, ft_lex, weights=(0.4,0.4,0.2)):
    pmi_norm = normalize_dict(pmi_lex)
    emb_norm = normalize_dict(emb_lex)
    ft_norm = normalize_dict(ft_lex)
    combined = {}
    for word in set(list(pmi_norm.keys()) + list(emb_norm.keys())):
        combined[word] = (
            weights[0]*pmi_norm.get(word,0) +
            weights[1]*emb_norm.get(word,0) +
            weights[2]*ft_norm.get(word,0)
        )
    return combined

# 7. VADER++ Style Sentiment Score

def vaderpp_score(sentence, lexicon):
    words = preprocess_text(sentence).split()
    scores = [lexicon.get(w,0) for w in words if w in lexicon]

    if not scores:
        return {'neg':0.0,'neu':1.0,'pos':0.0,'compound':0.0}

    compound = np.mean(scores)
    pos_count = sum(1 for s in scores if s>0.05)
    neg_count = sum(1 for s in scores if s<-0.05)
    neu_count = len(scores)-pos_count-neg_count
    total = len(scores)

    return {
        'neg': round(neg_count/total,3),
        'neu': round(neu_count/total,3),
        'pos': round(pos_count/total,3),
        'compound': round(compound,4)
    }

# 8. Main Function to Build Lexicon

def build_vaderpp_lexicon():
    positive_seed = ["good", "great", "excellent", "amazing", "fantastic", "love"]
    negative_seed = ["bad", "terrible", "awful", "poor", "hate", "worst"]

    texts = load_corpus()
    tokenized = preprocess_corpus(texts)
    word_counts = Counter(itertools.chain(*tokenized))

    pmi_lex = build_pmi_lexicon(tokenized, positive_seed, negative_seed)
    emb_lex = build_embedding_lexicon(tokenized, positive_seed, negative_seed)
    ft_lex  = build_fasttext_lexicon(word_counts, positive_seed, negative_seed)

    combined = combine_lexicons(pmi_lex, emb_lex, ft_lex)
    return combined
texts = load_corpus()
tokenized = preprocess_corpus(texts)
word_counts = Counter(itertools.chain(*tokenized))
positive_seed = ["good", "great", "excellent", "amazing", "fantastic", "love"]
negative_seed = ["bad", "terrible", "awful", "poor", "hate", "worst", "not","never","no","none","stopped"]
pmi_lex = build_pmi_lexicon(tokenized, positive_seed, negative_seed)
emb_lex = build_embedding_lexicon(tokenized, positive_seed, negative_seed)
ft_lex  = build_fasttext_lexicon(word_counts, positive_seed, negative_seed)
combined = combine_lexicons(pmi_lex, emb_lex, ft_lex)
lexicon = combined # build_vaderpp_lexicon()
examples = [
    "The laptop performance is excellent and the battery lasts long.",
    "The vacuum cleaner stopped working after a week, terrible product.",
    "This smartwatch is good but the strap quality feels poor.",
    "The sound quality of these headphones is amazing for the price!"
]

def domainExpansion(s):
    texts = load_corpus()
    tokenized = preprocess_corpus(texts)
    word_counts = Counter(itertools.chain(*tokenized))
    positive_seed = ["good", "great", "excellent", "amazing", "fantastic", "love"]
    negative_seed = ["bad", "terrible", "awful", "poor", "hate", "worst", "not","never","no","none","stopped"]
    pmi_lex = build_pmi_lexicon(tokenized, positive_seed, negative_seed)
    emb_lex = build_embedding_lexicon(tokenized, positive_seed, negative_seed)
    ft_lex  = build_fasttext_lexicon(word_counts, positive_seed, negative_seed)
    combined = combine_lexicons(pmi_lex, emb_lex, ft_lex)
    lexicon = combined # build_vaderpp_lexicon()
    examples = [
        "The laptop performance is excellent and the battery lasts long.",
        "The vacuum cleaner stopped working after a week, terrible product.",
        "This smartwatch is good but the strap quality feels poor.",
        "The sound quality of these headphones is amazing for the price!"
    ]
    # for s in examples:
    score = vaderpp_score(s, lexicon)
    return score

# for s in examples:
#     score = vaderpp_score(s, lexicon)
#     print(f"Sentence: {s}\nSentiment Scores: {score}\n")