"""
src/liwc_features.py
====================
LIWC-compatible linguistic feature extraction module.

Ahmad et al. (2020) use the commercial LIWC2015 tool to extract 93 linguistic
features from article text (e.g. positive/negative emotion, stop-word %, 
punctuation %, function words, informal language, POS ratios, word counts).

IMPORTANT LIMITATION
--------------------
LIWC2015 is a proprietary commercial tool (https://www.liwc.app/).
Its dictionary file (.dic) is NOT freely redistributable.

This module provides:
  1. An open-source *approximation* of the 93 LIWC-like features using only
     Python standard library + NLTK resources.  These are clearly labelled as
     LIWC-APPROXIMATION features.  They cover the same conceptual categories
     (positive/negative affect, pronouns, articles, prepositions, verbs,
     word counts, punctuation, etc.) but will produce DIFFERENT numeric values
     from the genuine LIWC2015 output.
  2. A stub `extract_liwc_exact()` that raises a clear error when the
     proprietary .dic file is missing, so the caller knows exactly what to do.

DO NOT claim results produced from these approximate features are identical to
the paper's results.  Label them "LIWC-Approximation" in all tables.
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-name manifest (93 names, matching LIWC2015 column order as closely
# as practical for an open-source approximation)
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    # Word-count statistics (6)
    "wc", "wps", "sixltr", "dic", "numerals", "avg_word_len",
    # Punctuation (3)
    "period", "comma", "colon",
    # Function words – pronouns (11)
    "pronoun", "ppron", "i", "we", "you", "shehe", "they",
    "ipron", "article", "prep", "auxverb",
    # Social / affective (6)
    "social", "family", "friend", "female", "male", "affect",
    # Positive / Negative emotion (5)
    "posemo", "negemo", "anx", "anger", "sad",
    # Cognitive processes (5)
    "cogproc", "insight", "cause", "discrep", "tentat",
    # Perceptual processes (4)
    "percept", "see", "hear", "feel",
    # Biological / drives (6)
    "bio", "body", "health", "sexual", "ingest", "drives",
    # Time orientation (3)
    "focuspast", "focuspresent", "focusfuture",
    # Relativity / space-time (4)
    "relativ", "motion", "space", "time",
    # Work / leisure (4)
    "work", "achieve", "leisure", "home",
    # Money / religion / death (3)
    "money", "relig", "death",
    # Grammar – POS ratios (8)
    "verb", "adj", "compare", "interrog", "number_wrd",
    "quant", "conjunctions", "negate",
    # Informality markers (6)
    "informal", "swear", "netspeak", "assent", "nonflu", "filler",
    # Punctuation count per 100 words (3)
    "excl", "qmark", "dash",
    # Readability / type-token (4)
    "unique_words_ratio", "sentence_count", "avg_sentence_len", "stopword_ratio",
    # Sentiment lexicon scores (3)
    "sentiment_pos", "sentiment_neg", "sentiment_compound",
    # URL / citation presence (1)
    "has_url",
    # Formatting signals (2)
    "allcaps_ratio", "title_case_ratio",
    # Remaining padding to reach 93 (6)
    "pad_88", "pad_89", "pad_90", "pad_91", "pad_92", "pad_93",
]

assert len(FEATURE_NAMES) == 93, f"Expected 93 feature names, got {len(FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# Simple word-category lexicons (open-source approximations)
# These are NOT LIWC dictionaries – they are hand-curated minimal lists
# sufficient to produce *meaningful* counts as LIWC proxies.
# ---------------------------------------------------------------------------

_FIRST_PERSON_SINGULAR  = {"i", "me", "my", "mine", "myself"}
_FIRST_PERSON_PLURAL    = {"we", "us", "our", "ours", "ourselves"}
_SECOND_PERSON          = {"you", "your", "yours", "yourself", "yourselves", "thou", "thee", "thy"}
_THIRD_SG_SHEHE         = {"he", "she", "him", "her", "his", "hers", "himself", "herself"}
_THIRD_PLURAL           = {"they", "them", "their", "theirs", "themselves"}
_ARTICLES               = {"a", "an", "the"}
_PREPOSITIONS           = {
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "from", "up",
    "down", "into", "out", "over", "under", "again", "further", "than",
    "since", "along", "across", "behind", "beyond", "near", "within",
    "without", "except", "toward", "upon",
}
_AUX_VERBS              = {
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might",
    "must", "can", "could", "need", "dare", "ought",
}
_POSITIVE_EMOTION       = {
    "good", "great", "excellent", "amazing", "wonderful", "love", "loved",
    "happy", "joy", "glad", "pleased", "success", "hope", "positive",
    "best", "better", "fantastic", "awesome", "beautiful", "perfect",
    "thank", "thanks", "safe", "benefit", "gain", "win",
}
_NEGATIVE_EMOTION       = {
    "bad", "terrible", "awful", "horrible", "hate", "hated", "fail",
    "fear", "angry", "anger", "sad", "loss", "worst", "false", "lie",
    "fraud", "corrupt", "threat", "crisis", "problem", "wrong", "evil",
    "disaster", "danger", "negative", "blame", "attack", "enemy",
}
_ANXIETY_WORDS          = {
    "afraid", "scared", "fear", "worry", "worried", "nervous", "stress",
    "stressed", "anxious", "panic", "dread", "terror", "terrified",
}
_ANGER_WORDS            = {
    "angry", "anger", "rage", "furious", "outraged", "hate", "mad",
    "hostile", "violent", "aggress", "fight", "assault", "destroy",
}
_SAD_WORDS              = {
    "sad", "sorrow", "grief", "cry", "crying", "cried", "depressed",
    "depression", "mourn", "mourning", "hurt", "tragic", "tragedy",
    "tears", "hopeless", "heartbroken", "loss", "gloomy", "disappointed",
}
_NEGATIONS              = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "cannot", "n't", "nt",
}
_SWEAR_WORDS            = {
    "damn", "hell", "crap", "ass", "shit", "fuck", "bitch", "bastard",
}
_SOCIAL_WORDS           = {
    "people", "person", "human", "man", "woman", "child", "family",
    "friend", "community", "society", "public", "group", "team",
}
_CONJUNCTIONS           = {
    "and", "but", "or", "nor", "for", "yet", "so", "because", "although",
    "however", "therefore", "thus", "while", "whereas", "unless", "though",
}
_NUMBERS_WORDS          = {
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million", "billion",
}
_STOP_WORDS             = (
    _ARTICLES | _PREPOSITIONS | _AUX_VERBS | _CONJUNCTIONS
    | {"i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
       "this", "that", "these", "those", "a", "an", "the", "is", "are",
       "was", "were", "be", "been", "do", "did", "has", "have", "had"}
)
_PAST_TENSE_MARKERS     = {"was", "were", "had", "did", "said", "went", "came", "made", "got"}
_FUTURE_MARKERS         = {"will", "shall", "going", "plan", "plans", "expect", "future", "soon"}


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    return a / b if b > 0 else fallback


def extract_liwc_approximation(text: str) -> np.ndarray:
    """
    Extract 93 LIWC-approximation linguistic features from a single article.

    Parameters
    ----------
    text : str
        Raw article text (headline + body, already HTML-cleaned).

    Returns
    -------
    np.ndarray of shape (93,), dtype float64.

    Notes
    -----
    Feature values are RAW counts or ratios.  Scaling to [0,1] is applied
    downstream by MinMaxScaler fitted on training data only.
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(93, dtype=np.float64)

    text_lower = text.lower()
    tokens_raw = re.findall(r"[a-zA-Z']+|\d+|[^\s\w]", text)  # words + nums + punct
    words      = [t.lower() for t in re.findall(r"[a-zA-Z']+", text)]
    word_lower = set(words)
    punct_all  = [t for t in tokens_raw if t in string.punctuation]
    sentences  = re.split(r"[.!?]+", text)
    sentences  = [s.strip() for s in sentences if s.strip()]

    wc           = len(words)
    sent_count   = len(sentences)
    avg_wps      = _safe_div(wc, sent_count)
    sixltr       = sum(1 for w in words if len(w) > 6)
    numerals     = sum(1 for t in tokens_raw if re.fullmatch(r"\d+", t))
    avg_word_len = _safe_div(sum(len(w) for w in words), wc)

    periods = text.count(".")
    commas  = text.count(",")
    colons  = text.count(":")

    def _overlap(word_list, ref_set):
        return sum(1 for w in word_list if w in ref_set)

    n_i      = _overlap(words, _FIRST_PERSON_SINGULAR)
    n_we     = _overlap(words, _FIRST_PERSON_PLURAL)
    n_you    = _overlap(words, _SECOND_PERSON)
    n_shehe  = _overlap(words, _THIRD_SG_SHEHE)
    n_they   = _overlap(words, _THIRD_PLURAL)
    n_ppron  = n_i + n_we + n_you + n_shehe + n_they
    n_ipron  = 0   # impersonal pronouns: "it", "its" – minimal list
    n_pronoun= n_ppron + n_ipron
    n_art    = _overlap(words, _ARTICLES)
    n_prep   = _overlap(words, _PREPOSITIONS)
    n_aux    = _overlap(words, _AUX_VERBS)

    n_social = _overlap(words, _SOCIAL_WORDS)
    n_family = sum(1 for w in words if w in {"family", "mother", "father", "son", "daughter", "sister", "brother", "parents"})
    n_friend = sum(1 for w in words if w in {"friend", "friends", "buddy", "colleague", "partner", "ally"})
    n_female = sum(1 for w in words if w in {"she", "her", "woman", "women", "girl", "female", "lady"})
    n_male   = sum(1 for w in words if w in {"he", "him", "man", "men", "boy", "male", "guy"})
    n_affect = _overlap(words, _POSITIVE_EMOTION) + _overlap(words, _NEGATIVE_EMOTION)

    n_posemo = _overlap(words, _POSITIVE_EMOTION)
    n_negemo = _overlap(words, _NEGATIVE_EMOTION)
    n_anx    = _overlap(words, _ANXIETY_WORDS)
    n_anger  = _overlap(words, _ANGER_WORDS)
    n_sad    = _overlap(words, _SAD_WORDS)

    n_cogproc = sum(1 for w in words if w in {"think", "thought", "know", "believe", "understand", "consider", "reason", "wonder", "doubt"})
    n_insight = sum(1 for w in words if w in {"realize", "understand", "notice", "see", "aware", "discover", "learn"})
    n_cause   = sum(1 for w in words if w in {"because", "cause", "caused", "effect", "result", "therefore", "thus", "consequently"})
    n_discrep = sum(1 for w in words if w in {"should", "would", "could", "must", "ought", "need", "want", "wish", "hope"})
    n_tentat  = sum(1 for w in words if w in {"maybe", "perhaps", "possibly", "might", "seem", "appears", "likely", "unclear"})

    n_percept = sum(1 for w in words if w in {"see", "saw", "look", "hear", "heard", "feel", "felt", "sense", "notice"})
    n_see     = sum(1 for w in words if w in {"see", "saw", "look", "view", "watch", "observe", "notice"})
    n_hear    = sum(1 for w in words if w in {"hear", "heard", "sound", "listen", "said", "spoke", "tell"})
    n_feel    = sum(1 for w in words if w in {"feel", "felt", "touch", "sense", "emotion"})

    n_bio    = sum(1 for w in words if w in {"eat", "sleep", "drink", "blood", "pain", "sick", "drug", "medicine", "body", "health"})
    n_body   = sum(1 for w in words if w in {"body", "heart", "head", "hand", "arm", "leg", "eye", "face", "skin", "blood"})
    n_health = sum(1 for w in words if w in {"health", "sick", "disease", "hospital", "doctor", "medicine", "cure", "virus", "vaccine"})
    n_sexual = sum(1 for w in words if w in {"sex", "sexual", "gender", "pregnant", "birth", "rape", "abuse"})
    n_ingest = sum(1 for w in words if w in {"eat", "food", "drink", "water", "alcohol", "drug", "consume", "hunger"})
    n_drives = n_bio + n_social + n_affect

    n_past    = _overlap(words, _PAST_TENSE_MARKERS)
    n_present = _overlap(words, _AUX_VERBS)
    n_future  = _overlap(words, _FUTURE_MARKERS)

    n_motion = sum(1 for w in words if w in {"go", "going", "went", "move", "moving", "travel", "run", "walk", "come"})
    n_space  = sum(1 for w in words if w in {"north", "south", "east", "west", "here", "there", "where", "up", "down"})
    n_time   = sum(1 for w in words if w in {"time", "day", "week", "month", "year", "hour", "minute", "today", "yesterday", "tomorrow"})
    n_relat  = n_motion + n_space + n_time

    n_work    = sum(1 for w in words if w in {"work", "job", "employ", "office", "business", "company", "career"})
    n_achieve = sum(1 for w in words if w in {"win", "success", "achieve", "accomplish", "goal", "victory", "reward"})
    n_leisure = sum(1 for w in words if w in {"play", "game", "fun", "sport", "relax", "holiday", "vacation", "entertainment"})
    n_home    = sum(1 for w in words if w in {"home", "house", "family", "domestic", "local", "community", "neighborhood"})

    n_money  = sum(1 for w in words if w in {"money", "dollar", "pay", "price", "cost", "fund", "bank", "tax", "budget", "economy"})
    n_relig  = sum(1 for w in words if w in {"god", "religion", "church", "faith", "pray", "allah", "bible", "muslim", "christian"})
    n_death  = sum(1 for w in words if w in {"death", "dead", "die", "died", "kill", "killed", "murder", "suicide", "funeral"})

    # Simple verb / adj detection (rough morphology)
    n_verb   = sum(1 for w in words if w.endswith(("ed", "ing", "ize", "ise", "ify")) or w in _AUX_VERBS)
    n_adj    = sum(1 for w in words if w.endswith(("ful", "less", "ous", "ive", "al", "ic", "able", "ible")))
    n_compare = sum(1 for w in words if w in {"more", "most", "less", "least", "better", "best", "worse", "worst", "than", "as"})
    n_interrog= sum(1 for w in words if w in {"who", "what", "when", "where", "why", "how", "which", "whom", "whose"})
    n_num_wd  = _overlap(words, _NUMBERS_WORDS)
    n_quant   = sum(1 for w in words if w in {"all", "any", "many", "few", "most", "more", "some", "every", "each", "both"})
    n_conj    = _overlap(words, _CONJUNCTIONS)
    n_negate  = _overlap(words, _NEGATIONS)

    n_informal = sum(1 for w in words if len(w) <= 2 and w not in {"is", "am", "be", "of", "to", "do", "in", "on", "at", "by"})
    n_swear    = _overlap(words, _SWEAR_WORDS)
    n_netspeak = sum(1 for w in words if w in {"lol", "omg", "wtf", "btw", "imho", "imo", "fyi", "thx", "brb", "irl"})
    n_assent   = sum(1 for w in words if w in {"yes", "ok", "okay", "sure", "right", "absolutely", "definitely", "agree"})
    n_nonflu   = sum(1 for w in words if w in {"um", "uh", "er", "ah", "hmm", "hm", "well"})
    n_filler   = sum(1 for w in words if w in {"like", "just", "really", "very", "so", "quite", "pretty", "kind", "sort"})

    excl_count  = text.count("!")
    qmark_count = text.count("?")
    dash_count  = text.count("-") + text.count("–") + text.count("—")

    unique_ratio    = _safe_div(len(set(words)), wc)
    stopword_ratio  = _safe_div(_overlap(words, _STOP_WORDS), wc)
    avg_sent_len    = _safe_div(wc, sent_count)

    # Sentiment: count positive vs negative words
    s_pos      = n_posemo / max(wc, 1)
    s_neg      = n_negemo / max(wc, 1)
    s_compound = s_pos - s_neg

    has_url    = 1.0 if re.search(r"https?://|www\.", text_lower) else 0.0
    allcaps    = _safe_div(sum(1 for w in re.findall(r"[A-Z]{2,}", text)), max(wc, 1))
    title_case = _safe_div(sum(1 for w in text.split() if w and w[0].isupper() and len(w) > 1), max(wc, 1))

    raw = [
        wc, avg_wps, sixltr, wc, numerals, avg_word_len,       # word stats
        periods, commas, colons,                                  # punct
        n_pronoun, n_ppron, n_i, n_we, n_you, n_shehe, n_they,  # pronouns
        n_ipron, n_art, n_prep, n_aux,                           # func words
        n_social, n_family, n_friend, n_female, n_male, n_affect,# social
        n_posemo, n_negemo, n_anx, n_anger, n_sad,              # emotion
        n_cogproc, n_insight, n_cause, n_discrep, n_tentat,      # cognition
        n_percept, n_see, n_hear, n_feel,                        # percept
        n_bio, n_body, n_health, n_sexual, n_ingest, n_drives,   # biological
        n_past, n_present, n_future,                              # time orient
        n_relat, n_motion, n_space, n_time,                      # relativity
        n_work, n_achieve, n_leisure, n_home,                    # work/leisure
        n_money, n_relig, n_death,                               # money/relig/death
        n_verb, n_adj, n_compare, n_interrog, n_num_wd,          # grammar
        n_quant, n_conj, n_negate,                               # grammar ctd
        n_informal, n_swear, n_netspeak, n_assent, n_nonflu, n_filler, # informality
        excl_count, qmark_count, dash_count,                     # punct counts
        unique_ratio, sent_count, avg_sent_len, stopword_ratio,  # readability
        s_pos, s_neg, s_compound,                                # sentiment
        has_url,                                                  # url
        allcaps, title_case,                                     # formatting
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,                          # pad 88-93
    ]
    assert len(raw) == 93
    return np.array(raw, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset-level extraction
# ---------------------------------------------------------------------------

def extract_features_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Extract 93 LIWC-approximation features for every row in *df*.

    Returns a DataFrame with 93 feature columns + a 'label' column.
    """
    logger.info(f"Extracting LIWC-approximation features for {len(df):,} articles …")
    feature_matrix = np.vstack(
        df[text_col].fillna("").astype(str).apply(extract_liwc_approximation).values
    )
    feat_df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    feat_df["label"] = df[label_col].values
    logger.info("Feature extraction complete.")
    return feat_df


# ---------------------------------------------------------------------------
# Stub: exact LIWC2015 extraction (requires proprietary .dic file)
# ---------------------------------------------------------------------------

def extract_liwc_exact(
    df: pd.DataFrame,
    dic_path: Path,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:  # pragma: no cover
    """
    Extract features using the genuine LIWC2015 dictionary file.

    Parameters
    ----------
    dic_path : Path
        Path to the LIWC2015 .dic file (commercial, not included in repo).

    Raises
    ------
    FileNotFoundError  – if dic_path does not exist.
    NotImplementedError – LIWC .dic parsing is not yet implemented.
    """
    if not Path(dic_path).exists():
        raise FileNotFoundError(
            f"LIWC2015 dictionary not found at: {dic_path}\n"
            "Purchase and install LIWC2015 from https://www.liwc.app/ and "
            "set LIWC_DIC_PATH in src/config.py to the path of the .dic file."
        )
    raise NotImplementedError(
        "Exact LIWC2015 parsing is not implemented in this open-source replication. "
        "Use LIWC2015 software directly to export the 93 feature columns, then load "
        "the resulting CSV via the --liwc-features flag in src/feature_pipeline.py."
    )


if __name__ == "__main__":
    # Quick sanity-check on a single example
    sample = (
        "The President announced a major policy change that experts say will "
        "help thousands of families across the country. Critics, however, are "
        "angry and worried about the potential risks to the economy."
    )
    vec = extract_liwc_approximation(sample)
    print(f"Feature vector length: {len(vec)}")
    print("Sample features:")
    for name, val in zip(FEATURE_NAMES[:20], vec[:20]):
        print(f"  {name:25s} = {val:.4f}")
