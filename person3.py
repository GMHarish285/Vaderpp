import re

# -----------------------------
# Sentiment Lexicon
# -----------------------------
LEXICON = {
    "good": 2.0,
    "bad": -2.0,
    "happy": 2.5,
    "sad": -2.5,
    "amazing": 3.0,
    "terrible": -3.0,
    "great": 3.2,
    "awful": -3.2,
    "fantastic": 3.5,
    "horrible": -3.5,
    "love": 3.0,
    "hate": -3.0
}

# Intensifiers: scaling factors for sentiment
INTENSIFIERS = {
    "very": 1.5,
    "extremely": 2.0,
    "quite": 1.2,
    "slightly": 0.5,
    "barely": 0.3,
    "really": 1.4,
    "too": 1.6,
    "absolutely": 2.0,
    "incredibly": 2.2,
}

# Negation markers
NEGATIONS = {"not", "never", "no", "nothing", "nowhere", 
             "hardly", "scarcely", "barely", "isn't", "wasn't", 
             "don't", "doesn't", "didn't", "won't", "can't"}

# Sentence boundary markers (end negation scope)
BOUNDARIES = {".", ",", ";", "!", "?"}


# -----------------------------
# Utility Functions
# -----------------------------

def is_negation_in_scope(tokens, idx, window=3):
    """
    Checks if there is a negation word within the scope before the sentiment word.
    Scope is cut off if punctuation appears.
    """
    start = max(0, idx - window)
    for j in range(start, idx):
        if tokens[j] in NEGATIONS:
            return True
        if tokens[j] in BOUNDARIES:  # negation does not cross punctuation
            break
    return False


def apply_intensifiers(tokens, idx, base_score):
    """
    Looks backward from sentiment word and applies multiplier if intensifiers found.
    Can handle multiple intensifiers.
    """
    score = base_score
    j = idx - 1
    while j >= 0 and tokens[j] not in BOUNDARIES:
        if tokens[j] in INTENSIFIERS:
            score *= INTENSIFIERS[tokens[j]]
        else:
            break
        j -= 1
    return score


# -----------------------------
# Core Scoring Function
# -----------------------------

def person3_negation_intensifier(sentence: str) -> float:
    """
    Handles negation and intensifier logic for sentiment analysis.
    Returns a raw sentiment score (positive or negative).
    """
    tokens = re.findall(r"\w+|[.,!?;]", sentence.lower())  # tokenize words + punctuation
    total_score = 0.0

    for i, word in enumerate(tokens):
        if word in LEXICON:
            score = LEXICON[word]

            # Apply intensifiers before this sentiment word
            score = apply_intensifiers(tokens, i, score)

            # Flip sentiment if negation is in scope
            if is_negation_in_scope(tokens, i):
                score *= -1

            total_score += score

    return total_score


# -----------------------------
# VADER++ Style Output
# -----------------------------

def person3_analyze(sentence: str):
    """
    Returns sentiment in VADER-like dictionary format.
    """
    raw_score = person3_negation_intensifier(sentence)

    # Normalize raw_score to [-1, 1] like VADER’s compound
    compound = raw_score / (abs(raw_score) + 4) if raw_score != 0 else 0.0
    compound = max(-1.0, min(1.0, compound))    # clamp between -1 and 1

    # Proportions
    if raw_score > 0:
        pos = min(1.0, compound if compound > 0 else 0.0)
        neg = 0.0
    elif raw_score < 0:
        neg = min(1.0, abs(compound))
        pos = 0.0
    else:
        pos = neg = 0.0

    neu = max(0.0, 1.0 - (pos + neg))

    return {
        "neg": round(neg, 3),
        "neu": round(neu, 3),
        "pos": round(pos, 3),
        "compound": round(compound, 3)
    }


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    examples = [
        "The food was absolutely fantastic!",
        "This movie is not good.",
        "She is very happy today!",
        "That was extremely bad service.",
        "I am barely happy with this.",
        "The food was not terrible.",
        "He is really very amazing.",
        "I don't think this is good, but not awful either."
    ]

    for s in examples:
        print(f"{s}\n -> {person3_analyze(s)}\n")

def analyze(sentence):
    return person3_analyze(sentence)
