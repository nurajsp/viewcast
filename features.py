"""Feature builder shared by the dashboard's 'make a prediction' form.
Uses exactly the same rules as the cleaning step in the notebook."""
import re
import numpy as np
import pandas as pd

SINHALA = re.compile(r"[඀-෿]")
LETTERS = re.compile(r"[A-Za-z඀-෿]")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
KEYWORDS = {
    "kw_uom": r"uom|moratuwa|mora\b|මොරටුව",
    "kw_al_ol": r"a/l|o/l|al results|3a|9a",
    "kw_university": r"universit|campus|undergrad|uni\b|batch|faculty",
    "kw_vlog": r"vlog|trip|hike|day out|family",
    "kw_interview": r"interview|asking|questions|what do|how many",
    "kw_festival": r"avurudu|අවුරුදු|valentine|christmas|carol|dansal|දන්සල",
    "kw_tech": r"laptop|camera|iphone|ipad|macbook|robot|project|tech|unbox",
}


def make_features(title: str, duration_sec: float, publish_date, age_days: int) -> dict:
    title = re.sub(r"\s+", " ", str(title)).strip()
    d = pd.Timestamp(publish_date)
    letters = LETTERS.findall(title)
    low = title.lower()
    f = {
        "duration_min": duration_sec / 60,
        "is_short": int(duration_sec <= 181),
        "publish_year": d.year,
        "publish_month": d.month,
        "publish_weekday": d.dayofweek,
        "is_weekend": int(d.dayofweek >= 5),
        "video_age_days": age_days,
        "avurudu_season": int(d.month in (4, 5)),
        "al_results_season": int(d.month in (4, 5, 8, 9)),
        "sinhala_ratio": len(SINHALA.findall(title)) / len(letters) if letters else 0.0,
        "title_len": len(title),
        "title_words": len(title.split()),
        "emoji_count": len(EMOJI.findall(title)),
        "hashtag_count": len(re.findall(r"#\w+", title)),
        "has_mention": int("@" in title),
        "has_question": int("?" in title),
        "has_pipe": int("|" in title),
    }
    for col, pattern in KEYWORDS.items():
        f[col] = int(bool(re.search(pattern, low)))
    return f
