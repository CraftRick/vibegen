"""
VibeGen v2
- Mobile-first layout (form-based search, big tap targets, no wrapping columns)
- Parallel + cached network calls (fast search)
- A real neural network (numpy MLP, Adam, online learning) that learns from your
  feedback, grows new neurons as it gains experience and is saved to disk.

requirements: streamlit, requests, numpy
"""
import os
import re
import json
import math
import time
import html
import zlib
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import requests
import streamlit as st

st.set_page_config(
    page_title="VibeGen",
    page_icon="🎧",
    layout="centered",               # "wide" is awful on phones
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, .stApp { background:#07090F; color:#E2E8F0;
  font-family: system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.block-container { padding:1.1rem .9rem 4rem !important; max-width:720px !important; }
header[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { letter-spacing:-.02em; }

/* cards */
.card { display:flex; align-items:center; gap:12px; background:#0E1320; border:1px solid #1C2333;
  border-radius:14px; padding:12px 14px; margin:12px 0 4px; }
.card img { width:52px; height:52px; border-radius:8px; object-fit:cover; flex-shrink:0; background:#1C2333; }
.card .meta { flex:1; min-width:0; }
.card .t { font-weight:700; font-size:1rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card .a { color:#64748B; font-size:.85rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pill { background:#1C1F3A; color:#818CF8; border:1px solid #3730A3; padding:1px 8px;
  border-radius:20px; font-size:.68rem; font-weight:600; margin-left:6px; vertical-align:middle; }
.vibe-box { background:#0B0F1C; border:1px solid #1E2A45; border-left:3px solid #4F46E5;
  border-radius:10px; padding:10px 14px; margin:14px 0 6px; font-size:.85rem; color:#94A3B8; line-height:1.7; }

/* inputs + buttons: 16px font stops iOS from zooming, 46px = comfortable tap target */
input, textarea { font-size:16px !important; }
.stTextInput input { background:#0E1320 !important; color:#E2E8F0 !important;
  border:1px solid #1C2333 !important; border-radius:12px !important; min-height:46px; }
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {
  background:linear-gradient(135deg,#4F46E5,#7C3AED) !important; color:#fff !important; border:none !important;
  border-radius:12px !important; font-weight:700 !important; min-height:46px; width:100%; }
[data-testid="stForm"] { border:none !important; padding:0 !important; }

/* keep button rows side by side on phones instead of stacking */
div[data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; gap:.5rem !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { min-width:0 !important; flex:1 1 0 !important; }
audio { width:100%; }
@media (max-width:480px) { .card .t { font-size:.95rem; } }
</style>
""", unsafe_allow_html=True)


def _get_key():
    k = os.getenv("LABUBU", "")
    if not k:
        try:
            k = st.secrets.get("LABUBU", "")
        except Exception:
            k = ""
    # fallback = the key that was hard-coded in the original file
    return (k or " ").strip()


LASTFM_API_KEY = _get_key()
LASTFM_ROOT = "https://ws.audioscrobbler.com/2.0/"
ITUNES_ROOT = "https://itunes.apple.com/search"
DATAMUSE_ROOT = "https://api.datamuse.com/words"
DATA_DIR = Path(os.getenv("VIBEGEN_DATA", "."))

ss = st.session_state
ss.setdefault("banned_genres", set())
ss.setdefault("results", [])
ss.setdefault("reserve", [])
ss.setdefault("vibe_meta", {})
ss.setdefault("last_scenario", "")
ss.setdefault("profile", "default")
ss.setdefault("notice", "")

LANGUAGE_CONFIG = {
    "hindi":      {"lastfm_tags": ["bollywood", "hindi pop", "indian pop", "desi"], "itunes_country": "in", "search_suffix": "hindi"},
    "urdu":       {"lastfm_tags": ["urdu", "ghazal", "qawwali", "pakistan"], "itunes_country": "pk", "search_suffix": "urdu"},
    "punjabi":    {"lastfm_tags": ["punjabi", "bhangra", "punjabi pop"], "itunes_country": "in", "search_suffix": "punjabi"},
    "spanish":    {"lastfm_tags": ["latin", "reggaeton", "spanish pop", "flamenco"], "itunes_country": "mx", "search_suffix": "spanish"},
    "korean":     {"lastfm_tags": ["k-pop", "korean", "kpop", "k-indie"], "itunes_country": "kr", "search_suffix": "korean"},
    "japanese":   {"lastfm_tags": ["j-pop", "jpop", "japanese", "city pop", "anime"], "itunes_country": "jp", "search_suffix": "japanese"},
    "chinese":    {"lastfm_tags": ["c-pop", "mandopop", "chinese", "cantopop"], "itunes_country": "cn", "search_suffix": "chinese"},
    "french":     {"lastfm_tags": ["french pop", "chanson", "french indie", "variete"], "itunes_country": "fr", "search_suffix": "french"},
    "portuguese": {"lastfm_tags": ["mpb", "bossa nova", "sertanejo", "fado"], "itunes_country": "br", "search_suffix": "portuguese"},
    "arabic":     {"lastfm_tags": ["arabic pop", "khaleeji", "shaabi", "rai"], "itunes_country": "sa", "search_suffix": "arabic"},
    "turkish":    {"lastfm_tags": ["turkish pop", "arabesque", "turk muzik"], "itunes_country": "tr", "search_suffix": "turkish"},
    "italian":    {"lastfm_tags": ["italian pop", "cantautore", "italo disco"], "itunes_country": "it", "search_suffix": "italian"},
    "german":     {"lastfm_tags": ["german pop", "schlager", "deutschpop"], "itunes_country": "de", "search_suffix": "german"},
    "russian":    {"lastfm_tags": ["russian pop", "russian rock", "russkiy rok"], "itunes_country": "ru", "search_suffix": "russian"},
    "tamil":      {"lastfm_tags": ["kollywood", "tamil pop", "carnatic"], "itunes_country": "in", "search_suffix": "tamil"},
    "telugu":     {"lastfm_tags": ["tollywood", "telugu songs"], "itunes_country": "in", "search_suffix": "telugu"},
    "bengali":    {"lastfm_tags": ["bengali songs", "rabindra sangeet", "bangla"], "itunes_country": "in", "search_suffix": "bengali"},
    "marathi":    {"lastfm_tags": ["marathi songs", "lavani"], "itunes_country": "in", "search_suffix": "marathi"},
    "gujarati":   {"lastfm_tags": ["garba", "gujarati folk"], "itunes_country": "in", "search_suffix": "gujarati"},
    "nepali":     {"lastfm_tags": ["nepali pop", "nepali folk"], "itunes_country": "np", "search_suffix": "nepali"},
    "thai":       {"lastfm_tags": ["thai pop", "luk thung", "t-pop"], "itunes_country": "th", "search_suffix": "thai"},
    "vietnamese": {"lastfm_tags": ["v-pop", "vietnamese", "nhac tre"], "itunes_country": "vn", "search_suffix": "vietnamese"},
    "indonesian": {"lastfm_tags": ["dangdut", "indonesian pop", "indo pop"], "itunes_country": "id", "search_suffix": "indonesian"},
    "malay":      {"lastfm_tags": ["malay pop", "pop melayu"], "itunes_country": "my", "search_suffix": "malay"},
    "swahili":    {"lastfm_tags": ["afrobeats", "bongo flava", "east african"], "itunes_country": "tz", "search_suffix": "swahili"},
    "afrikaans":  {"lastfm_tags": ["afrikaans", "boeremusiek", "south african"], "itunes_country": "za", "search_suffix": "afrikaans"},
    "greek":      {"lastfm_tags": ["greek pop", "laika", "rebetiko"], "itunes_country": "gr", "search_suffix": "greek"},
    "polish":     {"lastfm_tags": ["polish pop", "polska muzyka"], "itunes_country": "pl", "search_suffix": "polish"},
    "bollywood":  {"lastfm_tags": ["bollywood", "hindi film", "filmi"], "itunes_country": "in", "search_suffix": "bollywood"},
    "k-pop":      {"lastfm_tags": ["k-pop", "kpop", "korean"], "itunes_country": "kr", "search_suffix": "kpop"},
    "kpop":       {"lastfm_tags": ["k-pop", "kpop", "korean"], "itunes_country": "kr", "search_suffix": "kpop"},
    "latin":      {"lastfm_tags": ["latin", "reggaeton", "salsa", "bachata"], "itunes_country": "mx", "search_suffix": "latin"},
    "reggaeton":  {"lastfm_tags": ["reggaeton", "latin urban", "trap latino"], "itunes_country": "pr", "search_suffix": "reggaeton"},
}

LANGUAGE_ALIASES = {
    "hindi songs": "hindi", "in hindi": "hindi", "indian": "hindi",
    "in korean": "korean", "korean songs": "korean",
    "in japanese": "japanese", "japanese songs": "japanese",
    "in spanish": "spanish", "spanish songs": "spanish",
    "in french": "french", "french songs": "french",
    "in arabic": "arabic", "arabic songs": "arabic",
    "in turkish": "turkish", "turkish songs": "turkish",
    "in chinese": "chinese", "chinese songs": "chinese",
    "in punjabi": "punjabi", "punjabi songs": "punjabi",
    "in urdu": "urdu", "urdu songs": "urdu",
    "in tamil": "tamil", "tamil songs": "tamil",
    "in telugu": "telugu", "telugu songs": "telugu",
    "bengali songs": "bengali", "in bengali": "bengali",
    "in portuguese": "portuguese", "brazilian": "portuguese",
    "in russian": "russian", "russian songs": "russian",
    "in german": "german", "german songs": "german",
    "in italian": "italian", "italian songs": "italian",
}

WORD_VIBES = {
    "happy": [0.68, 0.92, 0.25, 0.45, 0.62],
    "sad": [0.18, 0.08, 0.35, 0.72, 0.22],
    "angry": [0.88, 0.12, 0.42, 0.18, 0.86],
    "calm": [0.08, 0.65, 0.20, 0.72, 0.12],
    "excited": [0.86, 0.82, 0.30, 0.20, 0.88],
    "anxious": [0.60, 0.18, 0.66, 0.30, 0.70],
    "peaceful": [0.05, 0.75, 0.18, 0.78, 0.08],
    "nostalgic": [0.28, 0.55, 0.36, 0.66, 0.28],
    "lonely": [0.12, 0.12, 0.40, 0.72, 0.18],
    "hopeful": [0.42, 0.78, 0.28, 0.62, 0.40],
    "melancholic": [0.18, 0.12, 0.48, 0.66, 0.18],
    "euphoric": [0.92, 0.95, 0.42, 0.18, 0.90],
    "depressed": [0.06, 0.03, 0.30, 0.68, 0.06],
    "frustrated": [0.72, 0.18, 0.40, 0.28, 0.70],
    "content": [0.24, 0.74, 0.18, 0.68, 0.28],
    "scared": [0.62, 0.08, 0.58, 0.38, 0.68],
    "confused": [0.40, 0.40, 0.82, 0.42, 0.42],
    "romantic": [0.30, 0.82, 0.24, 0.74, 0.28],
    "energetic": [0.92, 0.72, 0.28, 0.16, 0.92],
    "tired": [0.06, 0.40, 0.14, 0.68, 0.06],
    "motivated": [0.82, 0.70, 0.28, 0.22, 0.84],
    "love": [0.34, 0.90, 0.24, 0.74, 0.30],
    "hate": [0.82, 0.05, 0.40, 0.14, 0.74],
    "grief": [0.08, 0.04, 0.42, 0.72, 0.10],
    "joy": [0.72, 0.94, 0.22, 0.50, 0.72],
    "fear": [0.66, 0.06, 0.58, 0.30, 0.68],
    "wonder": [0.48, 0.72, 0.68, 0.50, 0.42],
    "awe": [0.48, 0.74, 0.66, 0.50, 0.32],
    "rage": [0.95, 0.04, 0.44, 0.10, 0.92],
    "bliss": [0.56, 0.96, 0.32, 0.60, 0.52],
    "dread": [0.44, 0.04, 0.64, 0.36, 0.42],
    "longing": [0.22, 0.28, 0.44, 0.68, 0.22],
    "pride": [0.66, 0.78, 0.36, 0.40, 0.62],
    "shame": [0.18, 0.10, 0.40, 0.56, 0.18],
    "guilt": [0.16, 0.08, 0.42, 0.58, 0.14],
    "relief": [0.34, 0.70, 0.20, 0.66, 0.30],
    "jealousy": [0.60, 0.20, 0.48, 0.30, 0.58],
    "envy": [0.56, 0.18, 0.46, 0.34, 0.54],
    "courage": [0.72, 0.72, 0.34, 0.36, 0.70],
    "despair": [0.10, 0.02, 0.52, 0.62, 0.10],
    "ecstasy": [0.92, 0.96, 0.48, 0.20, 0.92],
    "agony": [0.56, 0.02, 0.56, 0.38, 0.50],
    "serenity": [0.04, 0.76, 0.16, 0.80, 0.06],
    "tension": [0.66, 0.24, 0.58, 0.34, 0.66],
    "urgency": [0.84, 0.34, 0.46, 0.20, 0.86],
    "running": [0.90, 0.62, 0.20, 0.10, 0.90],
    "sleeping": [0.02, 0.50, 0.10, 0.74, 0.02],
    "studying": [0.30, 0.52, 0.42, 0.52, 0.26],
    "coding": [0.40, 0.52, 0.48, 0.20, 0.40],
    "driving": [0.60, 0.62, 0.30, 0.28, 0.62],
    "cooking": [0.42, 0.72, 0.26, 0.62, 0.48],
    "dancing": [0.84, 0.84, 0.26, 0.16, 0.92],
    "meditating": [0.02, 0.64, 0.26, 0.76, 0.02],
    "gaming": [0.72, 0.62, 0.48, 0.18, 0.72],
    "partying": [0.92, 0.82, 0.20, 0.08, 0.92],
    "fighting": [0.92, 0.08, 0.40, 0.10, 0.92],
    "crying": [0.18, 0.06, 0.40, 0.72, 0.16],
    "laughing": [0.62, 0.92, 0.28, 0.52, 0.62],
    "dreaming": [0.20, 0.62, 0.64, 0.52, 0.20],
    "falling": [0.52, 0.28, 0.44, 0.40, 0.60],
    "flying": [0.62, 0.82, 0.42, 0.40, 0.70],
    "gym": [0.92, 0.62, 0.18, 0.08, 0.92],
    "workout": [0.90, 0.62, 0.18, 0.08, 0.90],
    "exercise": [0.82, 0.62, 0.18, 0.18, 0.82],
    "healing": [0.20, 0.72, 0.30, 0.72, 0.18],
    "escaping": [0.72, 0.30, 0.48, 0.32, 0.74],
    "surviving": [0.66, 0.34, 0.48, 0.38, 0.66],
    "rain": [0.20, 0.32, 0.30, 0.72, 0.20],
    "snow": [0.10, 0.42, 0.22, 0.72, 0.10],
    "storm": [0.82, 0.20, 0.50, 0.40, 0.72],
    "thunder": [0.80, 0.18, 0.52, 0.40, 0.70],
    "lightning": [0.82, 0.38, 0.48, 0.32, 0.80],
    "sunshine": [0.52, 0.92, 0.18, 0.62, 0.52],
    "fog": [0.18, 0.40, 0.52, 0.52, 0.18],
    "ocean": [0.28, 0.62, 0.42, 0.62, 0.28],
    "beach": [0.40, 0.82, 0.18, 0.62, 0.42],
    "forest": [0.18, 0.62, 0.30, 0.82, 0.18],
    "city": [0.62, 0.52, 0.52, 0.18, 0.62],
    "night": [0.28, 0.38, 0.42, 0.42, 0.28],
    "dark": [0.38, 0.08, 0.52, 0.40, 0.38],
    "bright": [0.60, 0.82, 0.28, 0.52, 0.58],
    "soft": [0.18, 0.72, 0.20, 0.72, 0.18],
    "hard": [0.72, 0.38, 0.40, 0.28, 0.65],
    "deep": [0.30, 0.40, 0.72, 0.52, 0.22],
    "weird": [0.50, 0.50, 0.92, 0.42, 0.52],
    "strange": [0.42, 0.42, 0.85, 0.42, 0.42],
    "crazy": [0.82, 0.62, 0.72, 0.28, 0.82],
    "epic": [0.82, 0.72, 0.50, 0.40, 0.72],
    "chaos": [0.82, 0.28, 0.92, 0.20, 0.82],
    "silence": [0.02, 0.52, 0.28, 0.72, 0.02],
    "noise": [0.82, 0.40, 0.72, 0.20, 0.72],
    "dream": [0.18, 0.62, 0.65, 0.52, 0.18],
    "nightmare": [0.50, 0.02, 0.72, 0.32, 0.50],
    "peace": [0.08, 0.82, 0.28, 0.72, 0.08],
    "war": [0.82, 0.08, 0.50, 0.28, 0.72],
    "poop": [0.52, 0.40, 0.72, 0.28, 0.52],
    "trash": [0.40, 0.28, 0.60, 0.30, 0.40],
    "fart": [0.50, 0.45, 0.65, 0.32, 0.52],
    "blood": [0.72, 0.05, 0.52, 0.30, 0.62],
    "robot": [0.62, 0.52, 0.60, 0.08, 0.62],
    "alien": [0.52, 0.42, 0.82, 0.20, 0.52],
    "ghost": [0.28, 0.18, 0.62, 0.42, 0.22],
    "monster": [0.72, 0.08, 0.52, 0.28, 0.62],
    "dragon": [0.82, 0.42, 0.52, 0.30, 0.72],
    "cat": [0.30, 0.72, 0.40, 0.52, 0.30],
    "dog": [0.52, 0.82, 0.20, 0.52, 0.52],
    "rocket": [0.88, 0.65, 0.45, 0.15, 0.88],
    "book": [0.20, 0.60, 0.50, 0.65, 0.18],
    "phone": [0.50, 0.52, 0.42, 0.28, 0.50],
    "computer": [0.48, 0.52, 0.52, 0.12, 0.48],
    "mirror": [0.28, 0.45, 0.55, 0.48, 0.25],
    "window": [0.28, 0.58, 0.38, 0.55, 0.25],
    "door": [0.38, 0.52, 0.38, 0.50, 0.35],
    "crown": [0.65, 0.72, 0.45, 0.40, 0.60],
    "mask": [0.48, 0.30, 0.58, 0.38, 0.45],
    "flower": [0.30, 0.80, 0.28, 0.72, 0.30],
    "pizza": [0.50, 0.82, 0.18, 0.42, 0.50],
    "sugar": [0.48, 0.85, 0.18, 0.50, 0.48],
    "fire": [0.88, 0.42, 0.50, 0.12, 0.88],
    "ice": [0.18, 0.45, 0.30, 0.55, 0.18],
    "water": [0.22, 0.60, 0.28, 0.68, 0.20],
    "earth": [0.25, 0.58, 0.35, 0.75, 0.22],
    "wind": [0.45, 0.62, 0.38, 0.48, 0.48],
    "sun": [0.55, 0.90, 0.20, 0.58, 0.52],
    "moon": [0.18, 0.52, 0.40, 0.60, 0.15],
    "star": [0.42, 0.75, 0.45, 0.35, 0.38],
    "death": [0.28, 0.04, 0.55, 0.40, 0.25],
    "life": [0.55, 0.75, 0.38, 0.55, 0.52],
    "birth": [0.42, 0.78, 0.30, 0.65, 0.40],
    "time": [0.30, 0.45, 0.55, 0.42, 0.28],
    "space": [0.22, 0.50, 0.72, 0.28, 0.18],
    "home": [0.28, 0.75, 0.22, 0.72, 0.25],
    "road": [0.55, 0.58, 0.28, 0.35, 0.60],
    "fight": [0.90, 0.10, 0.42, 0.12, 0.90],
    "run": [0.88, 0.60, 0.20, 0.12, 0.88],
    "walk": [0.35, 0.60, 0.22, 0.55, 0.32],
    "swim": [0.38, 0.68, 0.25, 0.62, 0.35],
    "jump": [0.82, 0.72, 0.28, 0.18, 0.82],
    "break": [0.65, 0.15, 0.48, 0.28, 0.62],
    "build": [0.55, 0.68, 0.42, 0.38, 0.52],
    "destroy": [0.85, 0.08, 0.52, 0.12, 0.82],
    "create": [0.52, 0.75, 0.55, 0.42, 0.48],
    "lost": [0.28, 0.15, 0.55, 0.52, 0.25],
    "found": [0.45, 0.72, 0.32, 0.58, 0.42],
    "broken": [0.25, 0.08, 0.52, 0.58, 0.22],
    "healed": [0.25, 0.72, 0.28, 0.72, 0.22],
    "missing": [0.20, 0.15, 0.45, 0.65, 0.18],
    "searching": [0.48, 0.38, 0.52, 0.38, 0.45],
    "waiting": [0.18, 0.35, 0.42, 0.58, 0.15],
    "leaving": [0.38, 0.22, 0.45, 0.48, 0.35],
    "arriving": [0.52, 0.68, 0.32, 0.48, 0.50],
    "beginning": [0.48, 0.70, 0.38, 0.52, 0.45],
    "ending": [0.28, 0.25, 0.50, 0.55, 0.25],
    "forever": [0.32, 0.52, 0.42, 0.55, 0.28],
    "never": [0.35, 0.20, 0.48, 0.45, 0.32],
    "always": [0.38, 0.58, 0.35, 0.52, 0.35],
    "beautiful": [0.38, 0.85, 0.28, 0.68, 0.35],
    "ugly": [0.52, 0.12, 0.58, 0.30, 0.48],
    "powerful": [0.82, 0.60, 0.42, 0.25, 0.78],
    "weak": [0.15, 0.28, 0.35, 0.55, 0.12],
    "fast": [0.90, 0.58, 0.22, 0.12, 0.92],
    "slow": [0.12, 0.48, 0.28, 0.65, 0.10],
    "loud": [0.85, 0.42, 0.55, 0.08, 0.82],
    "quiet": [0.08, 0.55, 0.22, 0.78, 0.06],
    "cold": [0.18, 0.28, 0.38, 0.55, 0.15],
    "hot": [0.72, 0.58, 0.35, 0.22, 0.68],
    "empty": [0.08, 0.12, 0.45, 0.52, 0.06],
    "full": [0.52, 0.65, 0.28, 0.48, 0.50],
    "alone": [0.12, 0.15, 0.42, 0.68, 0.10],
    "together": [0.52, 0.78, 0.25, 0.55, 0.50],
    "free": [0.62, 0.80, 0.35, 0.45, 0.60],
    "trapped": [0.45, 0.08, 0.58, 0.38, 0.42],
    "drunk": [0.65, 0.55, 0.60, 0.25, 0.62],
    "sober": [0.30, 0.52, 0.30, 0.55, 0.28],
    "high": [0.60, 0.70, 0.62, 0.30, 0.58],
    "stoned": [0.25, 0.55, 0.58, 0.52, 0.22],
    "party": [0.90, 0.82, 0.22, 0.08, 0.92],
    "festival": [0.85, 0.82, 0.28, 0.18, 0.88],
    "funeral": [0.12, 0.04, 0.40, 0.68, 0.10],
    "wedding": [0.65, 0.88, 0.22, 0.52, 0.62],
    "birthday": [0.72, 0.88, 0.20, 0.42, 0.70],
    "heartbreak": [0.22, 0.06, 0.48, 0.68, 0.18],
    "breakup": [0.25, 0.08, 0.50, 0.65, 0.22],
    "betrayal": [0.42, 0.06, 0.55, 0.48, 0.38],
    "revenge": [0.75, 0.15, 0.52, 0.22, 0.72],
    "victory": [0.85, 0.80, 0.35, 0.30, 0.82],
    "defeat": [0.22, 0.08, 0.48, 0.55, 0.18],
    "chase": [0.88, 0.38, 0.35, 0.12, 0.90],
    "escape": [0.75, 0.32, 0.48, 0.25, 0.78],
    "hide": [0.28, 0.22, 0.48, 0.52, 0.25],
    "seek": [0.48, 0.52, 0.45, 0.38, 0.45],
    "pray": [0.18, 0.68, 0.35, 0.72, 0.15],
    "worship": [0.35, 0.72, 0.45, 0.62, 0.30],
    "rebel": [0.82, 0.32, 0.52, 0.15, 0.80],
    "surrender": [0.22, 0.30, 0.42, 0.60, 0.18],
    "rise": [0.72, 0.72, 0.38, 0.35, 0.70],
    "fall": [0.42, 0.22, 0.45, 0.48, 0.38],
    "crash": [0.82, 0.12, 0.55, 0.18, 0.80],
    "glow": [0.42, 0.78, 0.38, 0.52, 0.40],
    "fade": [0.18, 0.28, 0.42, 0.58, 0.15],
    "burn": [0.82, 0.35, 0.48, 0.15, 0.80],
    "float": [0.22, 0.65, 0.38, 0.55, 0.20],
    "sink": [0.22, 0.12, 0.48, 0.52, 0.18],
    "drift": [0.18, 0.42, 0.45, 0.62, 0.15],
    "wander": [0.35, 0.45, 0.48, 0.55, 0.32],
    "shatter": [0.72, 0.08, 0.58, 0.22, 0.68],
    "melt": [0.22, 0.52, 0.35, 0.60, 0.18],
    "explode": [0.95, 0.30, 0.58, 0.08, 0.95],
    "implode": [0.65, 0.08, 0.62, 0.25, 0.60],
    "whisper": [0.08, 0.52, 0.35, 0.72, 0.06],
    "scream": [0.92, 0.25, 0.52, 0.08, 0.90],
    "sing": [0.55, 0.75, 0.30, 0.62, 0.52],
    "play": [0.65, 0.80, 0.30, 0.42, 0.62],
    "suffer": [0.35, 0.05, 0.52, 0.55, 0.30],
    "thrive": [0.65, 0.80, 0.30, 0.42, 0.62],
    "collapse": [0.38, 0.08, 0.55, 0.50, 0.35],
    "soar": [0.72, 0.82, 0.38, 0.38, 0.70],
    "spiral": [0.58, 0.20, 0.65, 0.30, 0.55],
    "flow": [0.35, 0.65, 0.38, 0.60, 0.32],
    "freeze": [0.15, 0.25, 0.42, 0.60, 0.12],
    "spark": [0.72, 0.72, 0.45, 0.28, 0.70],
    "crush": [0.68, 0.12, 0.52, 0.25, 0.65],
    "touch": [0.30, 0.68, 0.28, 0.58, 0.28],
    "hold": [0.25, 0.65, 0.25, 0.65, 0.22],
    "release": [0.42, 0.62, 0.35, 0.55, 0.38],
    "remember": [0.28, 0.50, 0.42, 0.60, 0.25],
    "forget": [0.22, 0.25, 0.45, 0.52, 0.20],
    "imagine": [0.38, 0.68, 0.62, 0.52, 0.35],
    "believe": [0.45, 0.72, 0.38, 0.55, 0.42],
    "doubt": [0.35, 0.22, 0.55, 0.48, 0.32],
    "trust": [0.35, 0.72, 0.28, 0.62, 0.32],
    "lie": [0.48, 0.18, 0.55, 0.38, 0.45],
    "truth": [0.45, 0.68, 0.38, 0.55, 0.42],
    "lost": [0.28, 0.15, 0.55, 0.52, 0.25],
    "hollow": [0.12, 0.12, 0.52, 0.55, 0.10],
    "void": [0.08, 0.06, 0.55, 0.42, 0.06],
    "numb": [0.08, 0.10, 0.42, 0.60, 0.06],
    "shiver": [0.45, 0.22, 0.50, 0.48, 0.42],
    "tremble": [0.55, 0.15, 0.55, 0.42, 0.52],
    "sweat": [0.72, 0.38, 0.35, 0.22, 0.70],
    "bleed": [0.65, 0.06, 0.52, 0.30, 0.60],
    "heal": [0.25, 0.70, 0.30, 0.70, 0.22],
    "scar": [0.30, 0.12, 0.50, 0.52, 0.28],
    "memory": [0.22, 0.45, 0.48, 0.62, 0.20],
    "future": [0.52, 0.62, 0.45, 0.40, 0.50],
    "past": [0.22, 0.38, 0.48, 0.62, 0.20],
    "present": [0.42, 0.60, 0.38, 0.50, 0.40],
    "distance": [0.20, 0.28, 0.45, 0.55, 0.18],
    "close": [0.38, 0.68, 0.28, 0.58, 0.35],
    "far": [0.25, 0.32, 0.45, 0.52, 0.22],
    "heaven": [0.28, 0.85, 0.35, 0.65, 0.25],
    "hell": [0.75, 0.05, 0.55, 0.22, 0.72],
    "angel": [0.32, 0.82, 0.30, 0.65, 0.28],
    "demon": [0.72, 0.08, 0.55, 0.22, 0.68],
    "chaos": [0.82, 0.28, 0.92, 0.20, 0.82],
    "order": [0.38, 0.60, 0.28, 0.55, 0.35],
    "magic": [0.52, 0.75, 0.62, 0.48, 0.48],
    "trap": [0.82, 0.42, 0.38, 0.08, 0.85],
    "bounce": [0.80, 0.78, 0.25, 0.15, 0.82],
    "groove": [0.72, 0.78, 0.28, 0.22, 0.75],
    "wave": [0.45, 0.65, 0.40, 0.58, 0.42],
    "pulse": [0.65, 0.55, 0.38, 0.22, 0.68],
    "bass": [0.75, 0.55, 0.35, 0.12, 0.72],
    "beat": [0.78, 0.60, 0.30, 0.15, 0.80],
    "rhythm": [0.68, 0.65, 0.28, 0.25, 0.70],
    "melody": [0.40, 0.70, 0.38, 0.55, 0.38],
    "harmony": [0.32, 0.75, 0.30, 0.60, 0.30],
    "silence": [0.02, 0.52, 0.28, 0.72, 0.02],
    "echo": [0.25, 0.42, 0.55, 0.52, 0.22],
    "reverb": [0.28, 0.42, 0.58, 0.48, 0.25],
    "distortion": [0.78, 0.22, 0.68, 0.12, 0.75],
    "static": [0.55, 0.25, 0.65, 0.20, 0.52],
    "glitch": [0.65, 0.38, 0.75, 0.15, 0.62],
    "neon": [0.72, 0.65, 0.52, 0.12, 0.70],
    "retro": [0.48, 0.62, 0.45, 0.48, 0.45],
    "vintage": [0.32, 0.58, 0.42, 0.62, 0.28],
    "modern": [0.58, 0.58, 0.48, 0.22, 0.55],
    "ancient": [0.25, 0.45, 0.55, 0.68, 0.20],
    "sky": [0.35, 0.72, 0.38, 0.55, 0.32],
    "ground": [0.28, 0.52, 0.30, 0.72, 0.25],
    "underground": [0.52, 0.28, 0.60, 0.42, 0.48],
    "mountain": [0.38, 0.60, 0.40, 0.72, 0.35],
    "valley": [0.25, 0.62, 0.32, 0.75, 0.22],
    "desert": [0.22, 0.28, 0.45, 0.55, 0.18],
    "jungle": [0.58, 0.52, 0.55, 0.65, 0.55],
    "arctic": [0.15, 0.35, 0.38, 0.65, 0.12],
    "tropical": [0.55, 0.78, 0.30, 0.58, 0.52],
    "swamp": [0.28, 0.25, 0.58, 0.65, 0.25],
    "cave": [0.22, 0.22, 0.55, 0.58, 0.18],
    "throne": [0.65, 0.62, 0.45, 0.40, 0.62],
    "kingdom": [0.62, 0.65, 0.45, 0.42, 0.58],
    "empire": [0.72, 0.52, 0.52, 0.32, 0.68],
    "rebellion": [0.80, 0.35, 0.55, 0.18, 0.78],
    "revolution": [0.82, 0.42, 0.55, 0.20, 0.80],
    "apocalypse": [0.78, 0.08, 0.62, 0.22, 0.72],
    "utopia": [0.45, 0.82, 0.42, 0.55, 0.42],
    "dystopia": [0.55, 0.12, 0.65, 0.25, 0.52],
    "galaxy": [0.35, 0.55, 0.72, 0.30, 0.30],
    "cosmos": [0.28, 0.55, 0.72, 0.32, 0.25],
    "dimension": [0.35, 0.48, 0.75, 0.35, 0.32],
    "portal": [0.52, 0.55, 0.72, 0.32, 0.50],
    "mirror": [0.28, 0.45, 0.55, 0.48, 0.25],
    "illusion": [0.42, 0.42, 0.68, 0.42, 0.38],
    "reality": [0.45, 0.52, 0.42, 0.50, 0.42],
    "gravity": [0.42, 0.38, 0.52, 0.45, 0.38],
    "pressure": [0.65, 0.22, 0.58, 0.30, 0.62],
    "weight": [0.35, 0.22, 0.48, 0.55, 0.32],
    "light": [0.45, 0.82, 0.32, 0.52, 0.42],
    "shadow": [0.25, 0.22, 0.52, 0.48, 0.22],
    "darkness": [0.32, 0.08, 0.55, 0.42, 0.28],
    "dawn": [0.40, 0.72, 0.28, 0.62, 0.38],
    "dusk": [0.25, 0.38, 0.42, 0.58, 0.22],
    "midnight": [0.20, 0.25, 0.48, 0.45, 0.18],
    "sunrise": [0.45, 0.80, 0.25, 0.62, 0.42],
    "sunset": [0.30, 0.62, 0.32, 0.60, 0.28],
    "heartbeat": [0.52, 0.60, 0.35, 0.48, 0.52],
    "breathe": [0.22, 0.62, 0.28, 0.68, 0.20],
    "chasing": [0.85, 0.45, 0.38, 0.15, 0.88],
    "crumbling": [0.35, 0.10, 0.55, 0.48, 0.32],
    "glowing": [0.45, 0.75, 0.40, 0.50, 0.42],
    "fading": [0.18, 0.28, 0.42, 0.58, 0.15],
    "burning": [0.82, 0.38, 0.50, 0.15, 0.80],
    "frozen": [0.12, 0.28, 0.38, 0.62, 0.10],
    "shattering": [0.75, 0.08, 0.60, 0.20, 0.72],
    "rising": [0.65, 0.72, 0.38, 0.35, 0.62],
    "drowning": [0.35, 0.08, 0.55, 0.50, 0.32],
    "soaring": [0.72, 0.80, 0.38, 0.38, 0.70],
    "hunting": [0.80, 0.28, 0.45, 0.22, 0.82],
    "hiding": [0.22, 0.20, 0.48, 0.52, 0.20],
    "wandering": [0.32, 0.42, 0.48, 0.55, 0.30],
    "spinning": [0.68, 0.48, 0.55, 0.25, 0.65],
    "breaking": [0.68, 0.12, 0.52, 0.25, 0.65],
    "morphing": [0.48, 0.42, 0.70, 0.32, 0.45],
    "evolving": [0.52, 0.62, 0.58, 0.35, 0.50],
    "collapsing": [0.42, 0.08, 0.58, 0.38, 0.38],
    "ascending": [0.62, 0.72, 0.40, 0.38, 0.58],
    "descending": [0.28, 0.22, 0.45, 0.52, 0.25],
    "transcending": [0.52, 0.72, 0.58, 0.45, 0.48],
    "awakening": [0.58, 0.72, 0.45, 0.48, 0.55],
    "slumber": [0.05, 0.48, 0.18, 0.72, 0.04],
    "chaos": [0.82, 0.28, 0.92, 0.20, 0.82],
    "anarchy": [0.85, 0.25, 0.88, 0.15, 0.82],
    "harmony": [0.32, 0.78, 0.28, 0.62, 0.30],
    "infinity": [0.35, 0.58, 0.68, 0.45, 0.32],
    "zero": [0.08, 0.28, 0.35, 0.55, 0.06],
    "hero": [0.75, 0.72, 0.38, 0.35, 0.72],
    "villain": [0.68, 0.15, 0.55, 0.25, 0.65],
    "ghost": [0.28, 0.18, 0.62, 0.42, 0.22],
    "zombie": [0.55, 0.08, 0.58, 0.30, 0.52],
    "vampire": [0.52, 0.15, 0.60, 0.28, 0.48],
    "witch": [0.48, 0.28, 0.65, 0.38, 0.45],
    "wizard": [0.52, 0.55, 0.68, 0.42, 0.48],
    "warrior": [0.82, 0.52, 0.42, 0.22, 0.82],
    "soldier": [0.72, 0.38, 0.42, 0.25, 0.70],
    "lover": [0.35, 0.88, 0.25, 0.68, 0.32],
    "stranger": [0.38, 0.32, 0.52, 0.48, 0.35],
    "rebel": [0.80, 0.35, 0.52, 0.18, 0.78],
    "nomad": [0.45, 0.45, 0.48, 0.52, 0.42],
    "pilgrim": [0.35, 0.58, 0.42, 0.62, 0.32],
    "exile": [0.25, 0.18, 0.52, 0.55, 0.22],
    "king": [0.65, 0.65, 0.42, 0.38, 0.62],
    "queen": [0.62, 0.70, 0.40, 0.42, 0.58],
    "god": [0.55, 0.68, 0.55, 0.50, 0.50],
    "devil": [0.72, 0.10, 0.58, 0.20, 0.68],
    "child": [0.45, 0.80, 0.30, 0.65, 0.42],
    "adult": [0.40, 0.52, 0.38, 0.50, 0.38],
    "old": [0.20, 0.45, 0.40, 0.68, 0.18],
    "young": [0.62, 0.72, 0.30, 0.48, 0.60],
    "ancient": [0.25, 0.45, 0.55, 0.68, 0.20],
    "new": [0.58, 0.65, 0.40, 0.38, 0.55],
    "broken": [0.25, 0.08, 0.52, 0.58, 0.22],
    "whole": [0.42, 0.68, 0.30, 0.58, 0.38],
    "sick": [0.28, 0.10, 0.48, 0.52, 0.25],
    "healthy": [0.55, 0.75, 0.25, 0.58, 0.52],
    "dying": [0.22, 0.05, 0.52, 0.55, 0.18],
    "living": [0.55, 0.72, 0.35, 0.52, 0.52],
    "reborn": [0.62, 0.72, 0.42, 0.48, 0.58],
    "cursed": [0.42, 0.08, 0.60, 0.40, 0.38],
    "blessed": [0.42, 0.80, 0.30, 0.62, 0.38],
    "hollow": [0.12, 0.12, 0.52, 0.55, 0.10],
    "solid": [0.52, 0.55, 0.30, 0.58, 0.48],
    "ethereal": [0.22, 0.65, 0.62, 0.55, 0.18],
    "physical": [0.55, 0.52, 0.30, 0.45, 0.52],
    "digital": [0.55, 0.52, 0.60, 0.10, 0.52],
    "analog": [0.38, 0.55, 0.40, 0.65, 0.35],
    "matrix": [0.58, 0.42, 0.68, 0.12, 0.55],
    "simulation": [0.48, 0.42, 0.68, 0.18, 0.45],
    "apocalyptic": [0.72, 0.08, 0.65, 0.22, 0.68],
    "celestial": [0.30, 0.72, 0.60, 0.50, 0.28],
    "infernal": [0.72, 0.08, 0.60, 0.18, 0.68],
    "surreal": [0.42, 0.48, 0.80, 0.38, 0.38],
    "mundane": [0.30, 0.50, 0.25, 0.60, 0.28],
    "abstract": [0.38, 0.48, 0.80, 0.35, 0.35],
    "concrete": [0.42, 0.52, 0.28, 0.58, 0.38],
}

PHRASE_VIBES = {
    "falling apart": [0.25, 0.05, 0.55, 0.55, 0.22],
    "falling in love": [0.38, 0.88, 0.30, 0.68, 0.35],
    "end of the world": [0.40, 0.05, 0.65, 0.38, 0.38],
    "late at night": [0.18, 0.30, 0.48, 0.42, 0.15],
    "middle of the night": [0.15, 0.25, 0.52, 0.42, 0.12],
    "road trip": [0.62, 0.72, 0.28, 0.38, 0.65],
    "broken heart": [0.18, 0.05, 0.50, 0.68, 0.15],
    "deep space": [0.20, 0.48, 0.70, 0.25, 0.18],
    "ghost town": [0.18, 0.18, 0.58, 0.48, 0.15],
    "poop falling": [0.52, 0.45, 0.72, 0.25, 0.55],
    "falling from the sky": [0.55, 0.35, 0.62, 0.38, 0.58],
    "end of days": [0.40, 0.05, 0.65, 0.35, 0.38],
    "dark night of the soul": [0.18, 0.05, 0.65, 0.45, 0.15],
    "long drive": [0.40, 0.55, 0.30, 0.45, 0.38],
    "late night drive": [0.38, 0.38, 0.42, 0.38, 0.35],
    "crying in the rain": [0.18, 0.05, 0.45, 0.72, 0.15],
    "crying in the car": [0.20, 0.06, 0.42, 0.68, 0.18],
    "crying in the gym": [0.72, 0.06, 0.55, 0.22, 0.70],
    "running from something": [0.88, 0.18, 0.48, 0.12, 0.90],
    "running toward something": [0.88, 0.65, 0.35, 0.12, 0.90],
    "lost in the woods": [0.25, 0.20, 0.52, 0.65, 0.22],
    "lost in the city": [0.42, 0.25, 0.55, 0.28, 0.40],
    "staring at the ceiling": [0.08, 0.22, 0.45, 0.60, 0.06],
    "end of summer": [0.30, 0.45, 0.38, 0.55, 0.28],
    "walking alone": [0.22, 0.25, 0.42, 0.62, 0.20],
    "last dance": [0.55, 0.38, 0.40, 0.45, 0.52],
    "first love": [0.45, 0.88, 0.30, 0.65, 0.42],
    "deep underwater": [0.15, 0.32, 0.58, 0.65, 0.12],
    "outer space": [0.20, 0.50, 0.72, 0.25, 0.18],
    "middle of nowhere": [0.18, 0.28, 0.52, 0.60, 0.15],
    "new beginnings": [0.52, 0.72, 0.35, 0.52, 0.50],
    "letting go": [0.30, 0.45, 0.38, 0.58, 0.28],
    "moving on": [0.42, 0.55, 0.35, 0.52, 0.40],
    "starting over": [0.48, 0.62, 0.38, 0.50, 0.45],
    "holding on": [0.38, 0.40, 0.42, 0.55, 0.35],
    "giving up": [0.12, 0.10, 0.45, 0.60, 0.10],
    "never giving up": [0.75, 0.72, 0.38, 0.32, 0.72],
    "fighting back": [0.85, 0.42, 0.45, 0.15, 0.85],
    "standing tall": [0.68, 0.70, 0.35, 0.38, 0.65],
    "hitting rock bottom": [0.18, 0.04, 0.55, 0.55, 0.15],
    "coming back stronger": [0.72, 0.70, 0.38, 0.30, 0.70],
    "poop falling from the sky": [0.55, 0.42, 0.75, 0.22, 0.55],
    "walking in the rain": [0.22, 0.32, 0.35, 0.70, 0.20],
    "dancing in the rain": [0.65, 0.72, 0.28, 0.55, 0.62],
    "stuck in traffic": [0.35, 0.22, 0.48, 0.30, 0.32],
    "3am": [0.12, 0.15, 0.52, 0.42, 0.10],
    "2am": [0.12, 0.18, 0.50, 0.42, 0.10],
    "golden hour": [0.35, 0.80, 0.28, 0.62, 0.32],
    "blue hour": [0.20, 0.42, 0.42, 0.58, 0.18],
    "after the storm": [0.32, 0.55, 0.35, 0.62, 0.28],
    "before the storm": [0.42, 0.38, 0.52, 0.48, 0.40],
}

INTENSIFIERS = {"very", "extremely", "super", "ultra", "mega", "really", "so", "too", "quite", "deeply", "truly", "absolutely", "incredibly", "insanely", "mad", "hella", "totally", "completely", "utterly", "profoundly", "intensely"}
NEGATORS = {"not", "no", "never", "without", "barely", "hardly", "nothing", "neither", "nor", "anti"}


# ===================================================================
#  RUNTIME: shared HTTP session, thread pool and cache
#  (st.cache_resource keeps these alive between reruns. The original
#   module-level dict caches were wiped on EVERY rerun = slow.)
# ===================================================================
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "from", "and", "or", "is", "are", "was", "were",
    "with", "for", "my", "me", "i", "it", "its", "this", "that", "when", "while", "as", "by", "into",
    "out", "up", "down", "over", "under", "about", "like", "just", "some", "songs", "song", "music",
    "feel", "feeling", "vibe", "vibes", "listening", "i'm", "im", "be", "been", "am", "you", "your",
    "we", "they", "he", "she", "have", "has", "had", "of", "off", "get", "got", "then", "than",
}


@st.cache_resource
def get_runtime():
    s = requests.Session()
    ad = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32)
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": "VibeGen/2.0"})
    return {"session": s, "pool": ThreadPoolExecutor(max_workers=16), "cache": {}, "lock": threading.Lock()}


RT = get_runtime()


def cached_get(url, params, ttl=6 * 3600, timeout=4.0):
    """GET -> json with a process-wide TTL cache. Failures are not cached."""
    key = (url, tuple(sorted(params.items())))
    now = time.time()
    hit = RT["cache"].get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        r = RT["session"].get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    with RT["lock"]:
        if len(RT["cache"]) > 4000:
            RT["cache"].clear()
        RT["cache"][key] = (now, data)
    return data


def pmap(fn, items, timeout=8.0):
    """Run fn(item) for all items in parallel. Failed / late items -> None."""
    items = list(items)
    futs = [RT["pool"].submit(fn, it) for it in items]
    done, _ = wait(futs, timeout=timeout)
    out = []
    for f in futs:
        if f in done:
            try:
                out.append(f.result())
            except Exception:
                out.append(None)
        else:
            out.append(None)
    return out


# ===================================================================
#  WORD -> VIBE VECTOR
# ===================================================================
def datamuse_related(word):
    data = cached_get(DATAMUSE_ROOT, {"ml": word, "max": 12}, ttl=7 * 86400, timeout=2.5)
    return [d.get("word", "") for d in data] if isinstance(data, list) else []


def _local_resolve(word):
    vec = WORD_VIBES.get(word)
    if vec:
        return list(vec), 1.0
    for suffix in ["s", "ing", "ed", "ly", "er", "est", "tion", "ness", "ful", "less", "ment"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            vec = WORD_VIBES.get(word[:-len(suffix)])
            if vec:
                return list(vec), 0.9
    if len(word) > 4:
        for known in WORD_VIBES:
            if known.startswith(word[:4]) or word.startswith(known[:4]):
                if abs(len(word) - len(known)) <= 3:
                    return list(WORD_VIBES[known]), 0.75
    return None, 0.0


def _related_resolve(related):
    best_vec, best_w = None, 0.0
    for i, rw in enumerate(related[:12]):
        cand = WORD_VIBES.get(rw)
        if cand is None:
            for suffix in ["s", "ing", "ed", "ly"]:
                if rw.endswith(suffix):
                    cand = WORD_VIBES.get(rw[:-len(suffix)])
                    if cand:
                        break
        if cand:
            decay = 0.80 - i * 0.04
            if decay > best_w:
                best_vec, best_w = list(cand), decay
    return best_vec, best_w


def resolve_words(tokens):
    """Resolve all tokens locally first; look unknown ones up on Datamuse IN PARALLEL."""
    out, unknown = {}, []
    for w in dict.fromkeys(tokens):
        v, c = _local_resolve(w)
        if v is not None:
            out[w] = (v, c)
        elif len(w) >= 4 and not w.isdigit():
            unknown.append(w)
    unknown = unknown[:5]
    if unknown:
        for w, rel in zip(unknown, pmap(datamuse_related, unknown, timeout=3.5)):
            v, c = _related_resolve(rel or [])
            if v is not None:
                out[w] = (v, c)
    return out

def detect_language(text):
    tl = text.lower()
    for phrase, lang_key in LANGUAGE_ALIASES.items():
        if phrase in tl:
            return LANGUAGE_CONFIG.get(lang_key)
    words = re.findall(r"[a-z]+", tl)
    for word in words:
        if word in LANGUAGE_CONFIG:
            return LANGUAGE_CONFIG[word]
    return None

def clean_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())

def dedupe_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def normalize_vector(v):
    return [max(0.0, min(1.0, x)) for x in v]


def extract_vector(text):
    text_lower = text.lower().strip()
    tokens = clean_words(text_lower)
    skip = INTENSIFIERS | NEGATORS | STOPWORDS | set(LANGUAGE_CONFIG)
    resolved = resolve_words([t for t in tokens if t not in skip])

    vectors, weights = [], []
    for phrase, vec in PHRASE_VIBES.items():
        if phrase in text_lower:
            vectors.append(list(vec))
            weights.append(2.5)

    intensity, negate = 1.0, False
    for w in tokens:
        if w in INTENSIFIERS:
            intensity = min(1.8, intensity + 0.4)
            continue
        if w in NEGATORS:
            negate = True
            continue
        if w in STOPWORDS or w in LANGUAGE_CONFIG:
            continue
        hit = resolved.get(w)
        if hit:
            v = list(hit[0])
            if negate:
                v[0], v[1] = 1.0 - v[0], 1.0 - v[1]
                negate = False
            vectors.append([x * intensity for x in v])
            weights.append(hit[1])
            intensity = 1.0
        else:
            negate, intensity = False, 1.0

    if not vectors:
        return [0.5] * 5
    total = sum(weights)
    result = [0.0] * 5
    for v, wt in zip(vectors, weights):
        for j in range(5):
            result[j] += v[j] * wt / total
    return normalize_vector(result)

def vector_to_tags(vec, text):
    energy, valence, complexity, acoustic, tempo = vec
    tags = []

    if energy >= 0.80:
        tags.extend(["energetic", "upbeat", "anthem"])
    elif energy >= 0.60:
        tags.extend(["dynamic", "driving", "groovy"])
    elif energy >= 0.35:
        tags.extend(["chill", "mellow", "moderate"])
    elif energy >= 0.15:
        tags.extend(["soft", "gentle", "quiet"])
    else:
        tags.extend(["ambient", "minimal", "drone"])

    if valence >= 0.75:
        tags.extend(["happy", "feel good", "bright", "warm"])
    elif valence >= 0.50:
        tags.extend(["neutral", "cinematic", "indie"])
    elif valence >= 0.25:
        tags.extend(["melancholic", "moody", "bittersweet"])
    else:
        tags.extend(["sad", "dark", "depressing"])

    if complexity >= 0.70:
        tags.extend(["experimental", "progressive", "art rock"])
    elif complexity >= 0.50:
        tags.extend(["alternative", "layered", "interesting"])
    else:
        tags.extend(["simple", "clean", "direct"])

    if acoustic >= 0.70:
        tags.extend(["acoustic", "piano", "folk", "organic"])
    elif acoustic >= 0.45:
        tags.extend(["semi-acoustic", "indie folk", "singer-songwriter"])
    elif acoustic <= 0.30:
        tags.extend(["electronic", "synth", "digital", "beats"])

    if tempo >= 0.80:
        tags.extend(["fast", "high bpm", "frantic"])
    elif tempo >= 0.60:
        tags.extend(["upbeat", "driving rhythm"])
    elif tempo <= 0.20:
        tags.extend(["slow", "glacial", "meditative"])

    words = clean_words(text)
    special_map = {
        "poop": ["novelty", "quirky", "funny"],
        "fart": ["novelty", "quirky", "comedy"],
        "trash": ["lo-fi", "bedroom pop", "raw"],
        "weird": ["experimental", "avant-garde", "quirky"],
        "chaos": ["noise", "industrial", "chaotic"],
        "rain": ["rainy day", "cozy", "atmospheric"],
        "storm": ["dramatic", "intense", "cinematic"],
        "night": ["night drive", "late night", "dark"],
        "space": ["space ambient", "cosmic", "sci-fi"],
        "gym": ["workout", "hype", "aggressive"],
        "workout": ["workout", "high energy", "pump up"],
        "coding": ["focus", "study", "instrumental"],
        "study": ["lofi", "study beats", "focus"],
        "reading": ["instrumental", "ambient", "quiet"],
        "gaming": ["electronic", "epic", "intense"],
        "sad": ["sad songs", "heartbreak", "emotional"],
        "love": ["love songs", "romantic", "r&b"],
        "angry": ["metal", "punk", "aggressive"],
        "sleep": ["sleep music", "ambient", "lullaby"],
        "dream": ["dreamy", "ethereal", "ambient pop"],
        "drunk": ["party", "feel good", "loose"],
        "heartbreak": ["sad", "breakup songs", "emotional"],
        "breakup": ["sad", "heartbreak", "indie"],
        "running": ["running", "workout", "high bpm"],
        "fire": ["epic", "intense", "powerful"],
        "ocean": ["oceanic", "surf", "calm"],
        "forest": ["nature", "folk", "peaceful"],
        "apocalypse": ["post-apocalyptic", "dark ambient", "industrial"],
        "heaven": ["celestial", "ethereal", "choir"],
        "hell": ["dark", "metal", "sinister"],
        "robot": ["electronic", "synth", "futuristic"],
        "alien": ["experimental", "cosmic", "strange"],
        "war": ["epic", "orchestral", "intense"],
        "peace": ["peaceful", "ambient", "healing"],
        "magic": ["fantasy", "orchestral", "ethereal"],
        "party": ["party", "dance", "upbeat"],
        "rebel": ["punk", "alternative", "rock"],
        "hollow": ["empty", "sad", "minimal"],
        "glitch": ["glitch", "electronic", "IDM"],
        "neon": ["synthwave", "retrowave", "80s"],
        "retro": ["retro", "vintage", "classic"],
        "trap": ["trap", "hip hop", "bass heavy"],
        "bass": ["bass heavy", "electronic", "dubstep"],
        "bounce": ["bounce", "upbeat", "dance"],
        "groove": ["groove", "funk", "soul"],
    }
    for word in words:
        if word in special_map:
            tags.extend(special_map[word])

    return dedupe_keep_order(tags)

def vector_to_queries(vec, tags, text, lang_config=None):
    energy, valence, complexity, acoustic, tempo = vec
    queries = []

    suffix = f" {lang_config['search_suffix']}" if lang_config else ""

    queries.extend([t + suffix for t in tags[:4]])
    if len(tags) >= 2:
        queries.append(tags[0] + " " + tags[1] + suffix)

    if energy >= 0.8 and valence >= 0.7:
        queries.extend([f"upbeat dance music{suffix}", f"feel good pop{suffix}", f"energetic electronic{suffix}"])
    elif energy >= 0.8 and valence <= 0.35:
        queries.extend([f"aggressive electronic{suffix}", f"dark industrial{suffix}", f"heavy{suffix}"])
    elif energy <= 0.2 and valence <= 0.35:
        queries.extend([f"sad ambient{suffix}", f"melancholic piano{suffix}", f"slow sad{suffix}"])
    elif energy <= 0.2 and valence >= 0.7:
        queries.extend([f"peaceful piano{suffix}", f"healing ambient{suffix}", f"soft acoustic{suffix}"])
    elif complexity >= 0.65:
        queries.extend([f"experimental{suffix}", f"art rock{suffix}", f"progressive{suffix}"])
    elif acoustic >= 0.7:
        queries.extend([f"acoustic folk{suffix}", f"piano solo{suffix}", f"singer songwriter{suffix}"])
    elif acoustic <= 0.3:
        queries.extend([f"synthwave{suffix}", f"electronic beats{suffix}", f"digital{suffix}"])

    words = clean_words(text)
    keyword_queries = {
        "poop": ["novelty song", "comedy music", "quirky"],
        "fart": ["comedy music", "novelty", "funny songs"],
        "rain": [f"rainy day music{suffix}", f"rain ambient{suffix}"],
        "storm": [f"storm ambient{suffix}", f"dramatic music{suffix}"],
        "night": [f"late night vibe{suffix}", f"dark chill{suffix}"],
        "space": [f"space ambient{suffix}", f"cosmic music{suffix}"],
        "gym": [f"workout music{suffix}", f"gym hype{suffix}"],
        "workout": [f"workout music{suffix}", f"high energy songs{suffix}"],
        "coding": [f"focus music{suffix}", f"study beats{suffix}"],
        "study": [f"lofi study{suffix}", f"focus music{suffix}"],
        "sleep": [f"sleep music{suffix}", f"ambient lullaby{suffix}"],
        "gaming": [f"gaming music{suffix}", f"epic electronic{suffix}"],
        "love": [f"love songs{suffix}", f"romantic music{suffix}"],
        "heartbreak": [f"heartbreak songs{suffix}", f"sad love songs{suffix}"],
        "angry": [f"angry music{suffix}", f"aggressive songs{suffix}"],
        "party": [f"party music{suffix}", f"dance songs{suffix}"],
        "drunk": [f"party anthems{suffix}", f"feel good music{suffix}"],
        "driving": [f"driving music{suffix}", f"road trip songs{suffix}"],
        "meditation": [f"meditation music{suffix}", f"mindfulness{suffix}"],
        "yoga": [f"yoga music{suffix}", f"calm instrumental{suffix}"],
        "running": [f"running music{suffix}", f"workout playlist{suffix}"],
        "apocalypse": [f"post apocalyptic music{suffix}", f"dark ambient{suffix}"],
        "war": [f"war music{suffix}", f"epic battle{suffix}"],
        "rebellion": [f"punk{suffix}", f"alternative rock{suffix}"],
        "glitch": [f"glitch electronic{suffix}", f"IDM music{suffix}"],
        "neon": [f"synthwave{suffix}", f"retrowave{suffix}"],
        "trap": [f"trap music{suffix}", f"hip hop beats{suffix}"],
    }
    for word in words:
        if word in keyword_queries:
            queries.extend(keyword_queries[word])

    if lang_config:
        queries.extend(lang_config["lastfm_tags"][:3])

    queries.append(f"instrumental music{suffix}")

    return dedupe_keep_order([q for q in queries if q.strip()])[:12]

def vibe_label(vec):
    e, v, c, a, t = vec
    energy_word = ["ethereal", "serene", "mellow", "dynamic", "intense", "explosive"][min(5, int(e * 6))]
    valence_word = ["bleak", "somber", "neutral", "warm", "uplifting", "radiant"][min(5, int(v * 6))]
    tempo_word = ["glacial", "slow", "steady", "brisk", "fast", "frantic"][min(5, int(t * 6))]
    return energy_word, valence_word, tempo_word, int(e * 100)


# ===================================================================
#  DATA SOURCES (all cached + called in parallel)
# ===================================================================
def lastfm_top_tracks(tag, limit=14):
    if not LASTFM_API_KEY:
        return []
    data = cached_get(LASTFM_ROOT, {"method": "tag.getTopTracks", "tag": tag, "limit": limit,
                                    "api_key": LASTFM_API_KEY, "format": "json"})
    raw = data.get("tracks", {}).get("track", []) if isinstance(data, dict) else []
    if isinstance(raw, dict):
        raw = [raw]
    out = []
    for item in raw:
        name = item.get("name", "")
        a = item.get("artist")
        artist = a.get("name", "") if isinstance(a, dict) else ""
        if name and artist:
            out.append({
                "track_id": f"{artist.lower()}::{name.lower()}", "track_name": name, "artist_name": artist,
                "genre": tag, "preview_url": "", "url": item.get("url", ""), "image": "",
                "source": f"lastfm:{tag}",
            })
    return out


def lastfm_similar_tags(tag, limit=5):
    if not LASTFM_API_KEY:
        return []
    data = cached_get(LASTFM_ROOT, {"method": "tag.getSimilar", "tag": tag, "limit": limit,
                                    "api_key": LASTFM_API_KEY, "format": "json"})
    raw = data.get("similartags", data.get("toptags", {})).get("tag", []) if isinstance(data, dict) else []
    if isinstance(raw, dict):
        raw = [raw]
    return [(i.get("name") or "").strip() for i in raw if (i.get("name") or "").strip()]


def normalize_itunes(t, source):
    return {
        "track_id": str(t.get("trackId", "")), "track_name": t.get("trackName", ""),
        "artist_name": t.get("artistName", ""), "genre": t.get("primaryGenreName", "Unknown"),
        "preview_url": t.get("previewUrl", ""), "url": t.get("trackViewUrl", ""),
        "image": t.get("artworkUrl100", ""), "source": source,
    }


def search_itunes(query, country=None, limit=25):
    params = {"term": query, "entity": "song", "limit": limit, "media": "music"}
    if country:
        params["country"] = country
    data = cached_get(ITUNES_ROOT, params, timeout=4.5)
    out = []
    for t in (data.get("results", []) if isinstance(data, dict) else []):
        if t.get("wrapperType") != "track" or t.get("kind") != "song":
            continue
        if t.get("trackName") and t.get("artistName"):
            out.append(normalize_itunes(t, f"itunes:{query}"))
    return out


def _key(t):
    return f"{t['artist_name'].lower()}::{t['track_name'].lower()}"


def merge_track(old, new):
    """Fold `new` into `old`, preferring playable preview + artwork."""
    old["_srcs"] = (old.get("_srcs") or old.get("source", "")) + " " + new.get("source", "")
    if new.get("preview_url") and not old.get("preview_url"):
        for f in ("preview_url", "image", "url", "genre", "track_id"):
            if new.get(f):
                old[f] = new[f]


def collect_candidates(tags, queries, lang_config):
    country = lang_config["itunes_country"] if lang_config else None
    primary = tags[:8]
    if lang_config:
        primary = lang_config["lastfm_tags"][:4] + tags[:4]

    jobs = [("lf", t) for t in primary] if LASTFM_API_KEY else []
    jobs += [("it", q) for q in queries[:8]]

    def run(job):
        kind, arg = job
        return lastfm_top_tracks(arg) if kind == "lf" else search_itunes(arg, country)

    collected = {}

    def add_all(batches):
        for batch in batches:
            for t in (batch or []):
                k = _key(t)
                if k in collected:
                    collected[k]["_seen"] += 1
                    merge_track(collected[k], t)
                else:
                    t["_seen"] = 1
                    t["_srcs"] = t.get("source", "")
                    collected[k] = t

    add_all(pmap(run, jobs, timeout=8))

    # thin results -> branch out through similar Last.fm tags (2 parallel waves)
    if len(collected) < 20 and LASTFM_API_KEY:
        sims = [s for lst in pmap(lastfm_similar_tags, primary[:3], timeout=4) for s in (lst or [])][:8]
        add_all(pmap(lastfm_top_tracks, sims, timeout=6))

    if not collected:
        fb = lang_config["lastfm_tags"][0] if lang_config and lang_config.get("lastfm_tags") else "lofi chill instrumental"
        add_all([search_itunes(fb, country)])
    return list(collected.values())


def enrich_preview(track, country):
    """Last.fm tracks have no audio -> find the same song on iTunes for a playable preview."""
    an, tn = track["artist_name"].lower(), track["track_name"].lower()
    for r in search_itunes(f"{track['artist_name']} {track['track_name']}", country, limit=5):
        ra, rt = r["artist_name"].lower(), r["track_name"].lower()
        if r.get("preview_url") and (an in ra or ra in an) and (tn[:14] in rt or rt[:14] in tn):
            return r
    return None


# ===================================================================
#  THE BRAIN  -  a real neural network (numpy MLP, Adam, online learning)
#
#  input  : scenario vibe (5) + hand features (9) + hashed track tokens (48)
#           + hashed vibe x genre interaction tokens (64)  = 126 numbers
#  hidden : two leaky-ReLU layers that GROW with experience (16/8 -> 64/32)
#  output : P(you will like this track in this situation)
# ===================================================================
N_HAND, H_A, H_B = 9, 48, 64
IN_DIM = 5 + N_HAND + H_A + H_B
MAX_H1, MAX_H2 = 64, 32
BUFFER_MAX = 300


def _hash(token, dim):
    c = zlib.crc32(token.encode("utf-8"))
    return c % dim, (1.0 if (c >> 16) & 1 else -1.0)


def _hash_into(x, offset, dim, tokens):
    if not tokens:
        return
    scale = 1.0 / math.sqrt(len(tokens))
    for t in tokens:
        i, s = _hash(t, dim)
        x[offset + i] += s * scale


def _sparse(x):
    return [[int(i), round(float(x[i]), 4)] for i in np.nonzero(x)[0]]


def _dense(pairs):
    x = np.zeros(IN_DIM)
    for i, v in pairs:
        if 0 <= i < IN_DIM:
            x[i] = v
    return x


class VibeNet:
    def __init__(self, in_dim=IN_DIM, h1=16, h2=8, seed=7):
        rng = np.random.default_rng(seed)
        self.in_dim, self.h1, self.h2 = in_dim, h1, h2
        self.W1 = rng.normal(0, math.sqrt(2 / in_dim), (in_dim, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, math.sqrt(2 / h1), (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, math.sqrt(1 / h2), (h2, 1))
        self.b3 = np.zeros(1)
        self.n_seen = 0
        self.ema_loss = 0.69
        self.buffer = []          # replay memory: (sparse x, label, weight)
        self.growth_log = []
        self.t = 0
        self._init_adam()

    # ---------- core maths ----------
    def _init_adam(self):
        names = ("W1", "b1", "W2", "b2", "W3", "b3")
        self.m = {k: np.zeros_like(getattr(self, k)) for k in names}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in names}

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.where(z1 > 0, z1, 0.05 * z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.where(z2 > 0, z2, 0.05 * z2)
        z3 = a2 @ self.W3 + self.b3
        p = 1.0 / (1.0 + np.exp(-np.clip(z3, -30, 30)))
        return z1, a1, z2, a2, p

    def predict(self, X):
        return self._forward(np.atleast_2d(X))[4].ravel()

    def _step(self, X, y, w, lr=0.01, l2=1e-4):
        z1, a1, z2, a2, p = self._forward(X)
        y, w = y.reshape(-1, 1), w.reshape(-1, 1)
        sw = float(w.sum())
        eps = 1e-7
        loss = float(np.sum(-w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / sw)
        d3 = w * (p - y) / sw
        g = {"W3": a2.T @ d3 + l2 * self.W3, "b3": d3.sum(0)}
        d2 = (d3 @ self.W3.T) * np.where(z2 > 0, 1.0, 0.05)
        g["W2"], g["b2"] = a1.T @ d2 + l2 * self.W2, d2.sum(0)
        d1 = (d2 @ self.W2.T) * np.where(z1 > 0, 1.0, 0.05)
        g["W1"], g["b1"] = X.T @ d1 + l2 * self.W1, d1.sum(0)
        self.t += 1
        b1_, b2_ = 0.9, 0.999
        for k, grad in g.items():
            self.m[k] = b1_ * self.m[k] + (1 - b1_) * grad
            self.v[k] = b2_ * self.v[k] + (1 - b2_) * grad * grad
            mh = self.m[k] / (1 - b1_ ** self.t)
            vh = self.v[k] / (1 - b2_ ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + 1e-8))
        return loss

    # ---------- learning ----------
    def learn(self, x, y, weight=1.0, steps=6):
        self.buffer.append((_sparse(x), float(y), float(weight)))
        self.buffer = self.buffer[-BUFFER_MAX:]
        self.n_seen += 1
        loss = self.ema_loss
        for _ in range(steps):
            k = min(31, len(self.buffer))
            sample = random.sample(self.buffer, k) if len(self.buffer) > 1 else []
            batch = [self.buffer[-1]] + sample          # always include the newest lesson
            X = np.stack([_dense(b[0]) for b in batch])
            Y = np.array([b[1] for b in batch])
            W = np.array([b[2] for b in batch])
            loss = self._step(X, Y, W)
        self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss
        self._maybe_grow()

    # ---------- growth (function-preserving: new neurons start with zero output weights) ----------
    def _maybe_grow(self):
        s = math.sqrt(self.n_seen)
        a1 = min(MAX_H1, 16 + int(3 * s)) - self.h1
        a2 = min(MAX_H2, 8 + int(1.5 * s)) - self.h2
        if a1 >= 4 or a2 >= 2:
            self._grow(a1 if a1 >= 4 else 0, a2 if a2 >= 2 else 0)

    def _grow(self, add1, add2):
        rng = np.random.default_rng(self.n_seen + 101)
        if add1:
            self.W1 = np.hstack([self.W1, rng.normal(0, 0.3 * math.sqrt(2 / self.in_dim), (self.in_dim, add1))])
            self.b1 = np.concatenate([self.b1, np.zeros(add1)])
            self.W2 = np.vstack([self.W2, np.zeros((add1, self.h2))])
            self.h1 += add1
        if add2:
            self.W2 = np.hstack([self.W2, rng.normal(0, 0.3 * math.sqrt(2 / self.h1), (self.h1, add2))])
            self.b2 = np.concatenate([self.b2, np.zeros(add2)])
            self.W3 = np.vstack([self.W3, np.zeros((add2, 1))])
            self.h2 += add2
        self._init_adam()
        self.growth_log.append([self.n_seen, self.h1, self.h2])
        self.growth_log = self.growth_log[-30:]

    # ---------- (de)serialisation ----------
    def to_dict(self):
        r = lambda a: np.round(a, 5).tolist()
        return {"in_dim": self.in_dim, "h1": self.h1, "h2": self.h2,
                "W1": r(self.W1), "b1": r(self.b1), "W2": r(self.W2), "b2": r(self.b2),
                "W3": r(self.W3), "b3": r(self.b3), "n_seen": self.n_seen,
                "ema_loss": self.ema_loss, "buffer": self.buffer, "growth_log": self.growth_log}

    @classmethod
    def from_dict(cls, d):
        if d.get("in_dim") != IN_DIM:
            raise ValueError("incompatible brain")
        n = cls(IN_DIM, int(d["h1"]), int(d["h2"]))
        n.W1, n.b1 = np.array(d["W1"]), np.array(d["b1"])
        n.W2, n.b2 = np.array(d["W2"]), np.array(d["b2"])
        n.W3, n.b3 = np.array(d["W3"]), np.array(d["b3"])
        if n.W1.shape != (IN_DIM, n.h1) or n.W2.shape != (n.h1, n.h2) or n.W3.shape != (n.h2, 1):
            raise ValueError("bad shapes")
        n.n_seen = int(d.get("n_seen", 0))
        n.ema_loss = float(d.get("ema_loss", 0.69))
        n.buffer = [tuple(b) for b in d.get("buffer", [])][-BUFFER_MAX:]
        n.growth_log = d.get("growth_log", [])
        n._init_adam()
        return n


class Brain:
    """Network + the small 'memories' that feed it (genre taste, artist affinity). Thread-safe, persisted."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self._blank()

    def _blank(self):
        self.net = VibeNet()
        self.genre_profile = {}     # genre -> {"sum": [5], "n": int}   (which vibes you like it in)
        self.artist_aff = {}        # artist -> -3..3
        self.likes = 0
        self.dislikes = 0

    # --- features the network sees ---
    def genre_fit(self, genre, vec):
        gp = self.genre_profile.get(genre)
        if not gp or gp["n"] == 0:
            return 0.5
        mean = np.array(gp["sum"]) / gp["n"]
        fit = 1.0 - float(np.linalg.norm(np.array(vec) - mean)) / math.sqrt(5)
        conf = min(1.0, gp["n"] / 3)
        return 0.5 + (max(0.0, min(1.0, fit)) - 0.5) * conf

    def artist_affinity(self, artist):
        return math.tanh(self.artist_aff.get(artist, 0.0) / 2)

    def predict(self, X):
        with self.lock:
            return self.net.predict(X)

    def alpha(self):
        """How much we trust the network vs. the built-in heuristic (grows with experience)."""
        return min(0.7, self.net.n_seen / 60 * 0.7)

    # --- learning ---
    def feedback(self, x, label, weight, genre, artist, vec):
        with self.lock:
            if x is not None and len(x) == IN_DIM:
                self.net.learn(np.asarray(x, dtype=float), label, weight)
            a = (artist or "").lower()
            if a:
                self.artist_aff[a] = max(-3.0, min(3.0, self.artist_aff.get(a, 0.0) + (0.6 if label > 0.5 else -0.6)))
                if len(self.artist_aff) > 2000:
                    for k in list(self.artist_aff)[:500]:
                        self.artist_aff.pop(k, None)
            g = (genre or "").lower()
            if label > 0.5:
                self.likes += 1
                if g:
                    gp = self.genre_profile.setdefault(g, {"sum": [0.0] * 5, "n": 0})
                    gp["sum"] = [s + v for s, v in zip(gp["sum"], vec)]
                    gp["n"] += 1
            else:
                self.dislikes += 1
            self.save()

    def stats(self):
        n = self.net
        return {"h1": n.h1, "h2": n.h2, "seen": n.n_seen, "likes": self.likes,
                "dislikes": self.dislikes, "alpha": self.alpha(), "genres": len(self.genre_profile)}

    # --- persistence ---
    def to_json(self):
        with self.lock:
            return json.dumps({"net": self.net.to_dict(), "genre_profile": self.genre_profile,
                               "artist_aff": self.artist_aff, "likes": self.likes, "dislikes": self.dislikes})

    def load_json(self, s):
        d = json.loads(s)
        net = VibeNet.from_dict(d["net"])
        with self.lock:
            self.net = net
            self.genre_profile = d.get("genre_profile", {})
            self.artist_aff = d.get("artist_aff", {})
            self.likes = int(d.get("likes", 0))
            self.dislikes = int(d.get("dislikes", 0))

    def save(self):
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(self.to_json())
            os.replace(tmp, self.path)
        except Exception:
            pass

    def reset(self):
        with self.lock:
            self._blank()
            self.save()


@st.cache_resource
def _brains():
    return {}


def slug(name):
    return re.sub(r"[^a-z0-9_-]", "", (name or "").lower())[:24] or "default"


def get_brain(profile):
    store = _brains()
    if profile not in store:
        b = Brain(DATA_DIR / f"vibegen_brain_{profile}.json")
        try:
            if b.path.exists():
                b.load_json(b.path.read_text())
        except Exception:
            b._blank()
        store[profile] = b
    return store[profile]


# ===================================================================
#  SCORING: heuristic (cold start) blended with the network
# ===================================================================
def heuristic_score(track, ctx):
    vec = ctx["vec"]
    energy, valence = vec[0], vec[1]
    text_bits = " ".join([track.get("track_name", ""), track.get("artist_name", ""),
                          track.get("genre", ""), track.get("_srcs", track.get("source", ""))]).lower()
    score = 0.0
    for w in ctx["words"]:
        if len(w) < 3:
            continue
        if w in text_bits:
            score += 2.3
        if w in track.get("track_name", "").lower():
            score += 1.6
        if w in track.get("artist_name", "").lower():
            score += 0.8
    for tl in ctx["tags_l"]:
        if tl in text_bits:
            score += 1.7
    genre = track.get("genre", "").lower()
    if genre and any(t in genre for t in ctx["tags_l"][:8]):
        score += 1.4
    if track.get("preview_url"):
        score += 0.8
    if track.get("source", "").startswith("lastfm:"):
        score += 1.0
    if energy > 0.7 and ("live" in text_bits or "acoustic version" in text_bits):
        score -= 0.4
    if valence < 0.3 and ("happy" in text_bits or "very bright" in text_bits):
        score -= 0.3
    return score


def build_features(track, ctx, h_norm, brain):
    vec = ctx["vec"]
    genre = (track.get("genre") or "").lower()
    artist = (track.get("artist_name") or "").lower()
    srcs = (track.get("_srcs") or track.get("source", "")).lower()
    text_bits = f"{track.get('track_name', '')} {artist} {genre} {srcs}".lower()

    words_hit = sum(1 for w in ctx["words"] if len(w) >= 3 and w in text_bits)
    tags_hit = sum(1 for t in ctx["tags_l"] if t in text_bits)
    genre_tag = 1.0 if genre and any(t in genre for t in ctx["tags_l"][:8]) else 0.0

    hand = [
        1.0 if track.get("preview_url") else 0.0,
        1.0 if "lastfm:" in srcs else 0.0,
        min(1.0, words_hit / 3),
        min(1.0, tags_hit / 4),
        genre_tag,
        brain.genre_fit(genre, vec),
        brain.artist_affinity(artist),
        float(h_norm),
        min(1.0, track.get("_seen", 1) / 4),
    ]
    x = np.zeros(IN_DIM)
    x[:5] = vec
    x[5:5 + N_HAND] = hand

    e, v, c, a, t = vec
    ev, tb, cb = f"{min(2, int(e * 3))}{min(2, int(v * 3))}", min(2, int(t * 3)), min(2, int(c * 3))
    src_toks = [s for s in re.findall(r"[a-z0-9]+", srcs) if s not in ("lastfm", "itunes", "enrich")][:6]
    g_toks = re.findall(r"[a-z0-9]+", genre)
    _hash_into(x, 5 + N_HAND, H_A, [f"g:{g}" for g in g_toks] + [f"a:{artist}"] + [f"s:{s}" for s in src_toks])
    _hash_into(x, 5 + N_HAND + H_A, H_B,
               [f"g:{genre}|{ev}", f"g:{genre}|t{tb}", f"g:{genre}|c{cb}"] + [f"s:{s}|{ev}" for s in src_toks[:4]])
    return x


def rank_candidates(cands, ctx, brain):
    if not cands:
        return []
    raws = np.array([heuristic_score(t, ctx) for t in cands])
    span = raws.max() - raws.min()
    hn = (raws - raws.min()) / span if span > 1e-9 else np.full(len(raws), 0.5)
    X = np.stack([build_features(t, ctx, h, brain) for t, h in zip(cands, hn)])
    p = brain.predict(X)
    a = brain.alpha()
    final = (1 - a) * hn + a * p + np.random.uniform(0, 0.04, len(cands))
    for t, xv, pv, fv in zip(cands, X, p, final):
        t["_x"] = np.round(xv, 4).tolist()
        t["_p"] = float(pv)
        t["_score"] = float(fv)
    return sorted(cands, key=lambda t: t["_score"], reverse=True)


def diversify(tracks, limit=8):
    out, seen_a, seen_g = [], set(), {}
    for t in tracks:
        a, g = t.get("artist_name", "").lower(), t.get("genre", "").lower()
        if (a and a in seen_a) or seen_g.get(g, 0) >= 2:
            continue
        out.append(t)
        seen_a.add(a)
        seen_g[g] = seen_g.get(g, 0) + 1
        if len(out) >= limit:
            return out
    for t in tracks:                                   # top-up pass if the rules were too strict
        if len(out) >= limit:
            break
        if t not in out and t.get("artist_name", "").lower() not in seen_a:
            out.append(t)
            seen_a.add(t.get("artist_name", "").lower())
    return out


def generate_recommendations(text, brain, banned):
    lang_config = detect_language(text)
    vec = extract_vector(text)
    tags = vector_to_tags(vec, text)
    queries = vector_to_queries(vec, tags, text, lang_config)
    ctx = {"words": clean_words(text), "tags": tags, "tags_l": [t.lower() for t in tags], "vec": vec}
    country = lang_config["itunes_country"] if lang_config else None

    cands = [t for t in collect_candidates(tags, queries, lang_config) if t.get("genre") not in banned]
    ranked = rank_candidates(cands, ctx, brain)

    pool, rest = ranked[:24], ranked[24:]
    need = [t for t in pool if not t.get("preview_url")]
    if need:
        for t, hit in zip(need, pmap(lambda t: enrich_preview(t, country), need, timeout=6)):
            if hit:
                merge_track(t, hit)
        pool = rank_candidates([t for t in pool if t.get("genre") not in banned], ctx, brain)

    final = diversify(pool, 8)
    shown = {id(t) for t in final}
    reserve = [t for t in pool + rest if id(t) not in shown and t.get("genre") not in banned][:14]
    return final, reserve, vec, tags, queries, lang_config


# ===================================================================
#  FEEDBACK (button callbacks -> no manual st.rerun needed)
# ===================================================================
def _replace_card(idx, exclude_artists):
    reserve = ss.reserve
    while reserve:
        t = reserve.pop(0)
        if t.get("genre") in ss.banned_genres or t.get("artist_name", "").lower() in exclude_artists:
            continue
        ss.results[idx] = t
        return
    ss.results.pop(idx)


def handle_feedback(idx, kind):
    if idx >= len(ss.results):
        return
    t = ss.results[idx]
    brain = get_brain(ss.profile)
    vec = ss.vibe_meta.get("vec", [0.5] * 5)
    genre, artist = t.get("genre", ""), t.get("artist_name", "")
    if kind == "like":
        brain.feedback(t.get("_x"), 1.0, 1.0, genre, artist, vec)
        t["_liked"] = True
        return
    brain.feedback(t.get("_x"), 0.0, 1.3 if kind == "style" else 1.0, genre, artist, vec)
    taken = {r.get("artist_name", "").lower() for r in ss.results}
    if kind == "style" and genre:
        ss.banned_genres.add(genre)
        for j in range(len(ss.results) - 1, -1, -1):   # sweep other cards of that genre too
            if ss.results[j].get("genre") == genre:
                _replace_card(j, taken)
    else:
        _replace_card(idx, taken)


def reset_brain():
    get_brain(ss.profile).reset()


# ===================================================================
#  UI
# ===================================================================
st.markdown("## 🎧 VibeGen")
st.caption("Describe any moment. The brain learns from every 👍 / 👎 and grows over time.")

with st.expander("🧠 Brain & settings"):
    name = st.text_input("Profile (each profile has its own brain)", value=ss.profile)
    ss.profile = slug(name)
    brain = get_brain(ss.profile)
    s = brain.stats()
    st.markdown(
        f"**Neurons:** {s['h1']} + {s['h2']} &nbsp;·&nbsp; **Lessons learned:** {s['seen']} "
        f"&nbsp;·&nbsp; 👍 {s['likes']} &nbsp; 👎 {s['dislikes']} &nbsp;·&nbsp; "
        f"**Brain influence:** {int(s['alpha'] * 100)}%"
    )
    if brain.net.growth_log:
        n_seen, h1, h2 = brain.net.growth_log[-1]
        st.caption(f"Last growth spurt at lesson {n_seen}: now {h1}+{h2} neurons.")
    if ss.banned_genres:
        keep = st.multiselect("Blocked genres (remove to unblock)", sorted(ss.banned_genres),
                              default=sorted(ss.banned_genres))
        ss.banned_genres = set(keep)
    st.download_button("⬇ Download brain", data=brain.to_json(), file_name=f"vibegen_brain_{ss.profile}.json",
                       mime="application/json")
    up = st.file_uploader("Import a brain (.json)", type=["json"])
    if up is not None and st.button("Load imported brain"):
        try:
            brain.load_json(up.getvalue().decode("utf-8"))
            brain.save()
            st.success("Brain loaded.")
        except Exception:
            st.error("That file isn't a compatible VibeGen brain.")
    st.button("Reset brain", on_click=reset_brain)

# A form is what makes search work on phones: the text and the button press are
# submitted together, so the mobile keyboard's "Go" key and the button both work.
with st.form("search_form", clear_on_submit=False):
    scenario = st.text_input("Scenario", value=ss.last_scenario, label_visibility="collapsed",
                             placeholder='"crying in the gym" or "late night drive in the rain"')
    submitted = st.form_submit_button("✦  Recommend")

if submitted:
    text = scenario.strip()
    if not text:
        st.warning("Type something first.")
    else:
        ss.last_scenario = text
        with st.spinner("Decoding your vibe…"):
            brain = get_brain(ss.profile)
            results, reserve, vec, tags, queries, lang_config = generate_recommendations(
                text, brain, ss.banned_genres)
        ss.results, ss.reserve = results, reserve
        ev, vv, tv, epct = vibe_label(vec)
        ss.vibe_meta = {"vec": vec, "tags": tags, "queries": queries, "energy_word": ev, "valence_word": vv,
                        "tempo_word": tv, "energy_pct": epct,
                        "lang": lang_config["search_suffix"].upper() if lang_config else None}
        if not results:
            st.warning("Couldn't reach the music services. Check your connection and try again.")

meta = ss.vibe_meta
if meta and ss.results:
    col = "#22C55E" if meta["energy_pct"] < 33 else "#F59E0B" if meta["energy_pct"] < 66 else "#EF4444"
    lang_badge = f' · <span style="color:#A78BFA">🌐 {html.escape(meta["lang"])}</span>' if meta.get("lang") else ""
    st.markdown(
        f'<div class="vibe-box"><strong style="color:#818CF8">{meta["energy_word"].upper()}</strong> · '
        f'<strong style="color:#34D399">{meta["valence_word"].upper()}</strong> · '
        f'<strong style="color:#FB923C">{meta["tempo_word"].upper()} TEMPO</strong> · '
        f'<span style="color:{col}">ENERGY {meta["energy_pct"]}%</span>{lang_badge}</div>',
        unsafe_allow_html=True)
    st.caption("Tags: " + ", ".join(meta["tags"][:10]))

for i, track in enumerate(ss.results):
    tid = track.get("track_id") or f"{track.get('artist_name', '')}_{track.get('track_name', '')}"
    esc = lambda v: html.escape(str(v or ""))
    img = esc(track.get("image")) or ""
    st.markdown(
        f'<div class="card"><img src="{img}" loading="lazy" alt="">'
        f'<div class="meta"><div class="t">{esc(track.get("track_name"))}</div>'
        f'<div class="a">{esc(track.get("artist_name"))}<span class="pill">{esc(track.get("genre", "Unknown"))}</span>'
        f'</div></div></div>', unsafe_allow_html=True)

    if track.get("preview_url"):
        st.audio(track["preview_url"], format="audio/mp4")
    else:
        st.caption("Preview unavailable")

    c1, c2, c3 = st.columns(3)
    c1.button("💜" if track.get("_liked") else "👍", key=f"l_{i}_{tid}", on_click=handle_feedback, args=(i, "like"))
    c2.button("👎", key=f"d_{i}_{tid}", on_click=handle_feedback, args=(i, "vibe"))
    c3.button("🚫", key=f"b_{i}_{tid}", on_click=handle_feedback, args=(i, "style"))

if ss.vibe_meta and ss.last_scenario and len(ss.results) < 3:
    st.info("Running low on fresh picks, hit Recommend again for a new batch.")

if ss.results:
    st.caption("👍 love it  ·  👎 doesn't fit the vibe  ·  🚫 not my style (blocks that genre)")
