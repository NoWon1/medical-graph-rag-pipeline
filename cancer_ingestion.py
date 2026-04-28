# =============================================================================
# cancer_ingestion.py — v5 Graph RAG Pipeline
# 
# CHANGES FROM v4:
#   - Qdrant replaced by Neo4jVector (single database)
#   - config.py imported for all constants
#   - output/graph/ directory added to ensure_dirs()
#   - build_vector_store() now writes :Chunk nodes to Neo4j
#     with HNSW vector index — same chunk JSON files written
#     to disk so graph builder and BM25 can read them
#
# PIPELINE:
#   Phase 1  — Extract text  (Docling → PyMuPDF fallback)
#   Phase 2  — Caption + image extraction
#   Phase 3  — Clean markdown + inject [IMAGE:] tags
#   Phase 4  — Chunk text (header-aware two-stage)
#   Phase 5  — Save chunk JSONs + build Neo4j vector index
#
# RUN ORDER:
#   1. python cancer_ingestion.py   ← this file
#   2. python cancer_graph_builder.py
#   3. streamlit run cancer_app.py
# =============================================================================

from __future__ import annotations

import re
import io
import json
import math
import uuid
import hashlib
import unicodedata
from pathlib import Path
from typing import Optional

import fitz
from PIL import Image
from tqdm import tqdm

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document as LCDoc
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

from config import (
    BASE_DIR, PDF_DIR, MD_DIR, CHUNK_DIR, CAP_DIR,
    IMAGE_DIR, EMBED_DIR, LOG_DIR, GRAPH_DIR,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE,
    NEO4J_CHUNK_INDEX, NEO4J_CHUNK_LABEL,
    NEO4J_CHUNK_TEXT_PROP, NEO4J_CHUNK_EMBEDDING_PROP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    MIN_TEXT_CHARS, MIN_IMAGE_PX, MAX_ASPECT_RATIO,
    DPI_VECTOR, DPI_OCR, CHUNK_SIZE, CHUNK_OVERLAP,
    MIN_CHUNK_CHARS, HEADERS_TO_SPLIT, DOCLING_IMG_SCALE,
    EMBED_PREVIEW_DIMS, UPSERT_BATCH,
    get_source_url, detect_cancer_type, ensure_dirs,
)

# ── Optional dependencies ──────────────────────────────────
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import PictureItem, TableItem
    _DOCLING = True
except Exception:
    _DOCLING = False
    print("ℹ️  Docling not installed — PyMuPDF fallback active")

try:
    import easyocr
    _EASYOCR    = True
    _ocr_reader = None
except Exception:
    _EASYOCR = False

try:
    import numpy as np
    _NUMPY = True
except Exception:
    _NUMPY = False

# =============================================================================
# PAGE CLASSIFIER
# =============================================================================

class PageProfile:
    def __init__(self, page: fitz.Page) -> None:
        raw                = page.get_text().strip()
        self.text_chars    = len(raw)
        self.image_count   = len(page.get_images(full=True))
        self.drawing_count = len(page.get_drawings())
        self.is_text_rich    = self.text_chars >= MIN_TEXT_CHARS
        self.has_images      = self.image_count > 0
        self.is_vector_heavy = self.drawing_count > 25
        self.is_scanned      = not self.is_text_rich and not self.has_images
        self.is_mixed        = self.is_text_rich and (
            self.has_images or self.is_vector_heavy
        )

    def strategy(self) -> str:
        if self.text_chars < 10:  return "skip"
        if self.is_scanned:       return "scanned"
        if self.is_mixed:         return "mixed"
        if self.is_vector_heavy:  return "vector"
        return "text"

# =============================================================================
# TEXT CLEANING — 12-step medical PDF cleaner
# =============================================================================

def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    # 1. Mathematical bold digit repair
    math_digits = {
        '\U0001D7CE': '0', '\U0001D7CF': '1', '\U0001D7D0': '2',
        '\U0001D7D1': '3', '\U0001D7D2': '4', '\U0001D7D3': '5',
        '\U0001D7D4': '6', '\U0001D7D5': '7', '\U0001D7D6': '8',
        '\U0001D7D7': '9', '\U0001D7EC': '0', '\U0001D7ED': '1',
        '\U0001D7EE': '2', '\U0001D7EF': '3', '\U0001D7F0': '4',
        '\U0001D7F1': '5', '\U0001D7F2': '6', '\U0001D7F3': '7',
        '\U0001D7F4': '8', '\U0001D7F5': '9',
    }
    for weird, normal in math_digits.items():
        text = text.replace(weird, normal)

    # 2. Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 3. Raw glyph codes
    glyph_codes = {
        r'\s*/uniFB01\s*': 'fi', r'\s*/uniFB02\s*': 'fl',
        r'\s*/uniFB00\s*': 'ff', r'\s*/uniFB03\s*': 'ffi',
        r'\s*/uniFB04\s*': 'ffl', r'\s*/uniF642\s*': '%',
        r'\s*/C15\s*': '-', r'\s*/C19\s*': 'e',
        r'\s*/C20\s*': 'c', r'\s*/C211\s*': ' ',
    }
    for pattern, fixed in glyph_codes.items():
        text = re.sub(pattern, fixed, text)

    # 4. Unicode ligatures
    for bad, good in {
        '\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff',
        '\ufb03': 'ffi', '\ufb04': 'ffl', '\ufb05': 'st',
    }.items():
        text = text.replace(bad, good)

    # 5. Broken word stitching
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)', r'\1\2\3', text)
    text = re.sub(r'([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)(?=[a-zA-Z])',    r'\1\2',   text)
    text = re.sub(r'(?<=[a-zA-Z])(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)',   r'\1\2',   text)

    # 6. Docling placeholder tokens
    text = re.sub(r'', '', text, flags=re.IGNORECASE)

    # 7. Running journal headers/footers
    text = re.sub(
        r'(?m)^.{0,80}(?:guidelines?\s+version|©\s*\d{4}|page\s+\d+\s+of\s+\d+).{0,80}$\n?',
        '', text, flags=re.IGNORECASE
    )

    # 8. "Author et al." running headers
    text = re.sub(r'(?m)^[A-Z][a-z]+\s+et al\..{0,120}$\n?', '', text)

    # 9. DOI / URL lines
    text = re.sub(r'(?m)^(?:https?://|doi:\s*|www\.)\S+\s*$\n?', '', text)

    # 10. References section cutoff
    ref_match = re.compile(
        r'^#{0,3}\s*\*{0,2}'
        r'(references?|bibliography|works cited|literature cited)'
        r'\*{0,2}\s*$',
        re.IGNORECASE | re.MULTILINE,
    ).search(text)
    if ref_match:
        text = text[: ref_match.start()]

    # 11. Excess whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 12. Isolated page numbers
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$\n?', '', text)

    return text.strip()

# =============================================================================
# CONTENT TYPE DETECTION
# =============================================================================

def detect_content_type(text: str) -> str:
    t = text.lower()
    if re.search(r'\bfig(?:ure)?\.?\s*\d+', t):  return "figure_caption"
    if re.search(r'\btable\s*\d+', t):            return "table_caption"
    if any(w in t for w in [
        "p-value", "p < 0.", "confidence interval", "mann-whitney",
        "chi-square", "statistical analysis", "multivariate", "hazard ratio",
    ]):                                            return "statistical_methods"
    if any(w in t for w in [
        "recommend", "guideline", "should be", "must be", "standard of care",
        "first-line", "indicated", "contraindicated", "treatment plan",
        "protocol", "regimen", "adjuvant", "neo-adjuvant",
    ]):                                            return "clinical_recommendation"
    if any(w in t for w in [
        "survival", "prognosis", "outcome", "recurrence", "5-year",
        "overall survival", "disease-free", "mortality", "remission", "relapse",
    ]):                                            return "prognosis_data"
    return "clinical_text"

# =============================================================================
# IMAGE DEDUPLICATION + NOISE FILTER
# =============================================================================

_img_hashes: set = set()

def _is_duplicate_image(b: bytes) -> bool:
    h = hashlib.md5(b).hexdigest()
    if h in _img_hashes:
        return True
    _img_hashes.add(h)
    return False

NOISE_MIN_PX         = 100
NOISE_MAX_ASPECT     = 6.0
NOISE_SMALL_BADGE_PX = 200
NOISE_BW_RATIO       = 0.70
NOISE_LOGO_BW_RATIO  = 0.55
NOISE_GREEN_RATIO    = 0.45
NOISE_TEAL_RATIO     = 0.35
NOISE_ORANGE_RATIO   = 0.30
NOISE_SEPIA_RATIO    = 0.35
NOISE_QR_EDGE_RATIO  = 0.30
NOISE_FLAT_EDGE_RATIO = 0.05
PHASH_HAMMING_THRESH = 10

_phash_blocklist: list = []
_rejected_log:    list = []

NOISE_PUBLISHER_PATTERNS = [
    r'\belsevier\b', r'\bmdpi\b', r'\bspringer\b', r'\bwiley\b',
    r'\btaylor\s*&\s*francis\b', r'\blippincott\b', r'\bwolters\b',
    r'\bcheck\s+for\s+updates\b', r'\bcrossmark\b', r'\bpublished\s+by\b',
    r'\ball\s+rights\s+reserved\b', r'\bopen\s+access\b',
    r'\bcc\s+by\b', r'\bcreative\s+commons\b',
    r'\borcid\b', r'\bissn\b', r'\beissn\b',
    r'©\s*\d{4}', r'\bdoi:\s*10\.',
    r'\bvolume\s+\d+\b', r'\bissue\s+\d+\b',
    r'\bjournal\s+of\b', r'\bcopyright\b',
    r'\bavailable\s+online\b', r'\bwww\.\w+\.\w+\b',
    r'\bofficial\s+journal\b',
]

NOISE_FIGURE_OVERRIDE_PATTERNS = [
    r'\bfig(?:ure)?\.?\s*\d+', r'\btable\s*\d+',
    r'\bpanel\s*[a-d]\b', r'\bsupplementary\b',
    r'\bp\s*[<>=]\s*0\.\d+', r'\bhazard\s+ratio\b',
    r'\bkaplan.meier\b', r'\boverall\s+survival\b',
    r'\bflow\s+cytometry\b', r'\bimmunohistochem\b',
    r'\bscale\s+bar\b', r'\bmagnification\b',
    r'\bstaining\b', r'\bmicroscopy\b',
]

def _compute_phash(img: Image.Image, hash_size: int = 8) -> int:
    grey   = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(grey.getdata())
    bits   = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left  = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits  = (bits << 1) | (1 if left > right else 0)
    return bits

def _phash_hamming(h1: int, h2: int) -> int:
    x = h1 ^ h2; count = 0
    while x: count += x & 1; x >>= 1
    return count

def _is_in_phash_blocklist(ph: int) -> bool:
    return any(_phash_hamming(ph, k) <= PHASH_HAMMING_THRESH for k in _phash_blocklist)

def _log_rejection(filename: str, reason: str, detail: str) -> None:
    _rejected_log.append({"filename": filename, "reason": reason, "detail": detail})

def _color_analysis(img: Image.Image) -> dict:
    rgb   = img.convert("RGB")
    w, h  = rgb.size
    total = w * h
    if total == 0:
        return {k: 0 for k in ["bw_ratio","green_ratio","teal_ratio",
                                "orange_ratio","sepia_ratio",
                                "dominant_hue_frac","edge_ratio"]}
    pixels = list(rgb.getdata())
    BW_THRESH   = 30
    bw_count = green_count = 0
    hue_buckets = [0] * 36

    for r, g, b in pixels:
        lo, hi = min(r, g, b), max(r, g, b)
        if (hi - lo) < BW_THRESH and (hi < 50 or lo > 205):
            bw_count += 1
        if r < 120 and 160 <= g <= 230 and b < 120:
            green_count += 1
        delta = max(r, g, b) - min(r, g, b)
        if delta > 40 and max(r, g, b) > 0:
            mc = max(r, g, b)
            if mc == r:   hue = (60 * ((g - b) / delta)) % 360
            elif mc == g: hue = 60 * ((b - r) / delta) + 120
            else:         hue = 60 * ((r - g) / delta) + 240
            hue_buckets[int(hue / 10) % 36] += 1

    sat_total = sum(hue_buckets)
    if sat_total > total * 0.10:
        tb  = max(range(36), key=lambda i: hue_buckets[i])
        tc  = sum(hue_buckets[(tb + d) % 36] for d in [-1, 0, 1])
        dhf = tc / sat_total
    else:
        dhf = 0.0

    grey  = img.convert("L").resize((64, 64), Image.LANCZOS)
    gpix  = list(grey.getdata())
    gw = gh = 64; ec = 0; ET = 30
    for row in range(gh - 1):
        for col in range(gw - 1):
            idx = row * gw + col
            if (abs(int(gpix[idx]) - int(gpix[idx + 1])) > ET or
                    abs(int(gpix[idx]) - int(gpix[idx + gw])) > ET):
                ec += 1
    edge_ratio = ec / (gw * gh)

    teal_count   = sum(1 for r,g,b in pixels if r<100 and g>150 and b>150 and abs(int(g)-int(b))<40)
    orange_count = sum(1 for r,g,b in pixels if r>180 and 80<=g<=160 and b<80)
    sepia_count  = sum(1 for r,g,b in pixels if 100<=r<=210 and 60<=g<=150 and 20<=b<=110 and r>g>b and (r-b)>40)

    return {
        "bw_ratio":          bw_count     / total,
        "green_ratio":       green_count  / total,
        "teal_ratio":        teal_count   / total,
        "orange_ratio":      orange_count / total,
        "sepia_ratio":       sepia_count  / total,
        "dominant_hue_frac": dhf,
        "edge_ratio":        edge_ratio,
    }

def _classify_by_context(context_text: str) -> str:
    if not context_text:
        return "unknown"
    t = context_text.lower()
    if any(re.search(p, t) for p in NOISE_FIGURE_OVERRIDE_PATTERNS):
        return "figure"
    if any(re.search(p, t) for p in NOISE_PUBLISHER_PATTERNS):
        return "noise"
    return "unknown"

def _is_noise_image(img: Image.Image, filename: str = "",
                    context_text: str = "") -> bool:
    w, h = img.size

    if context_text:
        ctx = _classify_by_context(context_text)
        if ctx == "figure":
            return False
        if ctx == "noise":
            _log_rejection(filename, "layer5_context", context_text[:80])
            try: _phash_blocklist.append(_compute_phash(img))
            except Exception: pass
            return True

    if w < NOISE_MIN_PX or h < NOISE_MIN_PX:
        _log_rejection(filename, "layer1_too_small", f"{w}×{h}"); return True
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > NOISE_MAX_ASPECT:
        _log_rejection(filename, "layer1_banner", f"aspect={aspect:.1f}"); return True
    if w <= NOISE_SMALL_BADGE_PX and h <= NOISE_SMALL_BADGE_PX and aspect < 1.6:
        _log_rejection(filename, "layer1_small_badge", f"{w}×{h}"); return True

    try:
        ca = _color_analysis(img)
    except Exception:
        ca = {k: 0 for k in ["bw_ratio","green_ratio","teal_ratio",
                              "orange_ratio","sepia_ratio","dominant_hue_frac","edge_ratio"]}

    if ca["bw_ratio"] > NOISE_BW_RATIO and ca["edge_ratio"] > NOISE_QR_EDGE_RATIO:
        _log_rejection(filename, "layer2_qr_code", f"bw={ca['bw_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["green_ratio"] > NOISE_GREEN_RATIO:
        _log_rejection(filename, "layer2_orcid_green", f"green={ca['green_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["teal_ratio"] > NOISE_TEAL_RATIO:
        _log_rejection(filename, "layer2_teal_icon", f"teal={ca['teal_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["orange_ratio"] > NOISE_ORANGE_RATIO and w < 350 and h < 350:
        _log_rejection(filename, "layer2_orange_badge", f"orange={ca['orange_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["sepia_ratio"] > NOISE_SEPIA_RATIO and w < 500 and h < 400:
        _log_rejection(filename, "layer2_sepia_logo", f"sepia={ca['sepia_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["bw_ratio"] > NOISE_LOGO_BW_RATIO and w < 500 and h < 350:
        _log_rejection(filename, "layer2_bw_logo", f"bw={ca['bw_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if ca["edge_ratio"] < NOISE_FLAT_EDGE_RATIO and w < 300 and h < 300:
        _log_rejection(filename, "layer3_flat_icon", f"edge={ca['edge_ratio']:.2f}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True
    if (w > 400 and h > 400 and 0.55 < (w / h) < 0.82 and ca["dominant_hue_frac"] > 0.28):
        _log_rejection(filename, "layer6_journal_cover",
                       f"portrait_cover size={w}×{h}")
        try: _phash_blocklist.append(_compute_phash(img))
        except Exception: pass
        return True

    try:
        ph = _compute_phash(img)
        if _is_in_phash_blocklist(ph):
            _log_rejection(filename, "layer4_phash_match", "near-duplicate")
            return True
    except Exception:
        pass

    return False

def save_rejection_log() -> None:
    log_path = LOG_DIR / "rejected_images.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(_rejected_log, f, indent=2, ensure_ascii=False)

# =============================================================================
# EASYOCR
# =============================================================================

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _ocr_reader

def _ocr_page(page: fitz.Page) -> str:
    if not _EASYOCR:
        return ""
    try:
        pil     = Image.open(io.BytesIO(page.get_pixmap(dpi=DPI_OCR).tobytes("png")))
        results = _get_ocr_reader().readtext(pil, detail=1, paragraph=True)
        return "\n".join(
            t.strip()
            for _, t, conf in sorted(results, key=lambda r: r[0][0][1])
            if conf >= 0.4 and t.strip()
        )
    except Exception:
        return ""

# =============================================================================
# CAPTION CHUNK BUILDER
# =============================================================================

class CaptionChunkBuilder:
    _CAPTION_RE = re.compile(
        r'((?:Supplementary\s+)?(?:Fig(?:ure)?\.?|Table)\s*\d+[a-zA-Z]?'
        r'[\.\-\u2013\s].{20,500}?)'
        r'(?=\n\n|\Z)',
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, source_name: str, cancer_type: str) -> None:
        self.source_name   = source_name
        self.cancer_type   = cancer_type
        self._seen_caps:   set        = set()
        self._master_meta: list[dict] = []

    def build_from_docling(self, conv_result, local_image_dir: Path) -> tuple[list[LCDoc], list[dict]]:
        items        = list(conv_result.document.iterate_items())
        caption_docs = []
        pic_counter  = 0
        tbl_counter  = 0

        for i, (element, _level) in enumerate(items):
            is_picture = isinstance(element, PictureItem)
            is_table   = isinstance(element, TableItem)
            if not (is_picture or is_table):
                continue

            img = element.get_image(conv_result.document)
            if img is None:
                continue

            context_parts = []
            for offset in [-3, -2, -1, 1, 2, 3]:
                ni = i + offset
                if 0 <= ni < len(items):
                    el = items[ni][0]
                    if hasattr(el, "text") and el.text:
                        context_parts.append(el.text.strip()[:120])
            context_text = " ".join(context_parts)

            try:
                pil_check = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img))
                if _is_noise_image(pil_check.convert("RGB"), filename=f"{self.source_name}_element_{i}", context_text=context_text):
                    continue
            except Exception:
                pass

            if is_picture:
                pic_counter += 1
                img_filename = f"{self.source_name}_picture_{pic_counter}.png"
            else:
                tbl_counter += 1
                img_filename = f"{self.source_name}_table_{tbl_counter}.png"

            img_path = local_image_dir / img_filename
            with open(img_path, "wb") as fp:
                img.save(fp, "PNG")

            caption_text, tier = self._extract_caption_docling(element, items, i)
            doc_chunk, meta    = self._make_caption_chunk(caption_text, img_filename, img_path, "Picture" if is_picture else "Table", tier)
            if doc_chunk is not None:
                caption_docs.append(doc_chunk)
                self._master_meta.append(meta)

        print(f"    📸 Docling: {tbl_counter} tables + {pic_counter} figures → {len(caption_docs)} caption chunks")
        return caption_docs, self._master_meta

    def build_from_pymupdf(self, doc: fitz.Document, local_image_dir: Path) -> tuple[list[LCDoc], list[dict]]:
        caption_docs = []
        img_counter  = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            p    = PageProfile(page)
            if p.has_images:
                for ref in page.get_images(full=True):
                    try:
                        b = doc.extract_image(ref[0])["image"]
                        if _is_duplicate_image(b):
                            continue
                        pil = Image.open(io.BytesIO(b)).convert("RGB")
                        if _is_noise_image(pil):
                            continue
                        img_counter += 1
                        fp = local_image_dir / f"{self.source_name}_p{page_idx+1}_img{img_counter}.png"
                        pil.save(fp)
                    except Exception:
                        pass
            elif p.is_vector_heavy:
                try:
                    b = page.get_pixmap(dpi=DPI_VECTOR).tobytes("png")
                    if not _is_duplicate_image(b):
                        pil = Image.open(io.BytesIO(b))
                        if not _is_noise_image(pil):
                            img_counter += 1
                            fp = local_image_dir / f"{self.source_name}_p{page_idx+1}_vector.png"
                            pil.save(fp)
                except Exception:
                    pass

        seen: set = set()
        for page_idx in range(len(doc)):
            for match in self._CAPTION_RE.findall(doc[page_idx].get_text()):
                cleaned = re.sub(r'\s+', ' ', match.strip())
                if len(cleaned) < 30 or cleaned in seen:
                    continue
                seen.add(cleaned)
                asset_type = "Table" if re.match(r'table', cleaned, re.IGNORECASE) else "Picture"
                img_fname  = f"{self.source_name}_regex_cap_{len(caption_docs)+1:04d}.png"
                doc_chunk, meta = self._make_caption_chunk(cleaned, img_fname, local_image_dir / img_fname, asset_type, "pymupdf_regex")
                if doc_chunk is not None:
                    caption_docs.append(doc_chunk)
                    self._master_meta.append(meta)

        print(f"    📸 PyMuPDF fallback: {img_counter} images | {len(caption_docs)} caption chunks")
        return caption_docs, self._master_meta

    def _extract_caption_docling(self, element, items, idx) -> tuple[str, str]:
        captions = getattr(element, "captions", [])
        parts    = [c.text for c in captions if hasattr(c, "text") and c.text]
        if parts:
            return " ".join(parts).strip(), "docling_native"
        if idx + 1 < len(items):
            nxt = items[idx + 1][0]
            if hasattr(nxt, "text") and nxt.text:
                t = nxt.text.strip()
                if re.match(r'(?i)(fig|table)', t):
                    return t, "sliding_window_next"
        if idx - 1 >= 0:
            prv = items[idx - 1][0]
            if hasattr(prv, "text") and prv.text:
                t = prv.text.strip()
                if re.match(r'(?i)(fig|table)', t):
                    return t, "sliding_window_prev"
        if idx + 1 < len(items):
            nxt = items[idx + 1][0]
            if hasattr(nxt, "text") and nxt.text:
                return f"Context: {nxt.text.strip()[:300]}...", "context_fallback"
        return "Visual asset extracted from document.", "no_caption"

    def _make_caption_chunk(self, caption_text: str, img_filename: str, img_path: Path, asset_type: str, tier: str) -> tuple[Optional[LCDoc], dict]:
        norm = re.sub(r'\s+', ' ', caption_text.lower().strip())
        if norm in self._seen_caps:
            return None, {}
        self._seen_caps.add(norm)

        try:
            rel_path = str(img_path.relative_to(BASE_DIR))
        except ValueError:
            rel_path = str(img_path)

        cap_idx      = len(self._seen_caps)
        chunk_id     = f"{self.source_name}_cap_{cap_idx:04d}"
        content_type = "table_caption" if asset_type == "Table" else "figure_caption"

        metadata = {
            "chunk_id":        chunk_id,
            "source_file":     self.source_name,
            "source_url":      get_source_url(self.source_name),
            "cancer_type":     self.cancer_type,
            "content_type":    content_type,
            "asset_type":      asset_type,
            "image_filename":  img_filename,
            "image_path":      rel_path,
            "extraction_tier": tier,
            "char_length":     len(caption_text),
        }
        return LCDoc(page_content=caption_text, metadata=metadata), {**metadata, "caption": caption_text}

# =============================================================================
# TABLE DATA EXTRACTOR
# =============================================================================

def extract_table_data(doc: fitz.Document) -> str:
    seen: set = set(); table_blocks: list = []
    for page_num in range(len(doc)):
        for block in doc[page_num].get_text("blocks", sort=True):
            if len(block) < 5: continue
            raw = block[4].strip()
            if len(re.findall(r'\d+\.?\d*', raw)) >= 3 and len(raw) > 40:
                cleaned = re.sub(r'\s+', ' ', raw)
                if cleaned not in seen:
                    seen.add(cleaned); table_blocks.append(cleaned)
    return (
        "\n\n## Key Numerical Data\n\n" + "\n\n".join(table_blocks[:25])
    ) if table_blocks else ""

# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def _pymupdf_extract(doc: fitz.Document) -> str:
    parts: list = []; scanned = 0; ocr_done = 0
    for i in range(len(doc)):
        page  = doc[i]; strat = PageProfile(page).strategy()
        if strat == "skip": continue
        elif strat == "scanned":
            scanned += 1
            if _EASYOCR:
                t = _ocr_page(page)
                parts.append(f"\n[OCR p{i+1}]\n{t}" if t else f"\n[p{i+1}: OCR empty]\n")
                if t: ocr_done += 1
            else:
                parts.append(f"\n[p{i+1}: scanned — pip install easyocr]\n")
        else:
            t = page.get_text()
            if t.strip(): parts.append(t)
    if scanned: print(f"    🔍 Scanned: {scanned} pages, {ocr_done} OCR'd")
    return "\n\n".join(parts)

# =============================================================================
# IMAGE TAG SECTION BUILDER
# =============================================================================

def _build_image_tag_section(paper_images: list[dict]) -> str:
    if not paper_images:
        return ""
    section  = "\n\n## Extracted Visual Assets Database\n\n"
    section += (
        "The following figures, tables and flowcharts were extracted from "
        "this paper. When answering questions about these visuals, reference "
        "them using their exact tag.\n\n"
    )
    for meta in paper_images:
        section += f"- [IMAGE: {meta['image_filename']}]\n  Caption: {meta['caption']}\n\n"
    return section

# =============================================================================
# SINGLE PDF PROCESSOR
# =============================================================================

def process_single_pdf(pdf_path: Path) -> dict:
    name        = pdf_path.stem
    cancer_type = detect_cancer_type(name)

    print(f"\n{'─'*62}")
    print(f"📄  {name}")
    print(f"    Cancer: {cancer_type}")
    print(f"    Size  : {pdf_path.stat().st_size // 1024} KB")

    try:
        doc        = fitz.open(pdf_path)
        page_count = len(doc)
        text_rich  = sum(1 for i in range(page_count) if len(doc[i].get_text().strip()) >= MIN_TEXT_CHARS)
        strats     = [PageProfile(doc[i]).strategy() for i in range(page_count)]
        counts     = {s: strats.count(s) for s in set(strats)}
        print(f"    Pages : {page_count}  Text-rich: {text_rich}")
        print("    📊 " + " | ".join(f"{k}={v}" for k, v in sorted(counts.items())))

        caption_builder = CaptionChunkBuilder(name, cancer_type)
        caption_docs: list[LCDoc] = []
        method = "pymupdf"

        if _DOCLING:
            try:
                po = PdfPipelineOptions()
                po.images_scale            = DOCLING_IMG_SCALE
                po.generate_page_images    = True
                po.generate_picture_images = True
                conv_result = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=po)}
                ).convert(pdf_path, max_num_pages=200)
                raw_md = conv_result.document.export_to_markdown()

                if raw_md and len(raw_md.strip()) > 300:
                    print("    ✅ Docling text extraction succeeded")
                    method = "docling"
                    caption_docs, _ = caption_builder.build_from_docling(conv_result, IMAGE_DIR)
                    raw = raw_md
                else:
                    print("    ⚠️  Docling sparse — PyMuPDF fallback")
                    raw = _pymupdf_extract(doc)
                    caption_docs, _ = caption_builder.build_from_pymupdf(doc, IMAGE_DIR)
            except Exception as e:
                print(f"    ⚠️  Docling error ({e}) — fallback")
                raw = _pymupdf_extract(doc)
                caption_docs, _ = caption_builder.build_from_pymupdf(doc, IMAGE_DIR)
        else:
            raw = _pymupdf_extract(doc)
            caption_docs, _ = caption_builder.build_from_pymupdf(doc, IMAGE_DIR)

        tabs = extract_table_data(doc)
        if tabs:
            raw += tabs
        doc.close()

        cleaned = clean_text(raw)
        if len(cleaned) < 200:
            print(f"    ⚠️  Low text yield ({len(cleaned)} chars)")

        tag_section = _build_image_tag_section(caption_builder._master_meta)
        if tag_section:
            cleaned += tag_section
            print(f"    🖼️  Injected {len(caption_builder._master_meta)} [IMAGE:] tags")

        (MD_DIR / f"{name}.md").write_text(cleaned, encoding="utf-8")

        text_chunks = _chunk_text(cleaned, name, cancer_type)

        with open(CHUNK_DIR / f"{name}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(text_chunks, f, indent=2, ensure_ascii=False)

        cap_payload = [{**d.metadata, "content": d.page_content} for d in caption_docs]
        with open(CAP_DIR / f"{name}_caption_chunks.json", "w", encoding="utf-8") as f:
            json.dump(cap_payload, f, indent=2, ensure_ascii=False)

        _update_master_image_metadata(caption_builder._master_meta)

        img_chunks = sum(1 for c in text_chunks if c.get("has_image_tags"))
        print(f"    ✅ {method} | Text: {len(text_chunks)} chunks ({img_chunks} with image tags) | Captions: {len(caption_docs)}")

        return {
            "file": name, "cancer_type": cancer_type,
            "pages": page_count, "text_pages": text_rich,
            "text_chunks": len(text_chunks), "image_tag_chunks": img_chunks,
            "caption_chunks": len(caption_docs),
            "images_extracted": len(caption_builder._master_meta),
            "method": method, "status": "success",
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"file": name, "status": "failed", "error": str(e)}


def _update_master_image_metadata(new_entries: list[dict]) -> None:
    meta_path = IMAGE_DIR / "image_metadata.json"
    existing: list = []
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f: existing = json.load(f)
        except Exception: existing = []
    seen = {e["image_filename"] for e in existing}
    for entry in new_entries:
        if entry.get("image_filename") not in seen:
            existing.append(entry); seen.add(entry["image_filename"])
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

# =============================================================================
# CHUNKING
# =============================================================================

def _chunk_text(cleaned: str, source_name: str, cancer_type: str) -> list:
    h_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    t_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    try:
        chunks = t_splitter.split_documents(h_splitter.split_text(cleaned))
    except Exception:
        chunks = t_splitter.split_documents([LCDoc(page_content=cleaned)])

    payload: list = []
    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        hierarchy = " > ".join(
            v for k, v in chunk.metadata.items() if k.startswith("H") and v
        ) or "Body"
        payload.append({
            "chunk_id":          f"{source_name}_{i:04d}",
            "source_file":       source_name,
            "source_url":        get_source_url(source_name),
            "cancer_type":       cancer_type,
            "content_type":      detect_content_type(content),
            "section_hierarchy": hierarchy,
            "content":           content,
            "char_length":       len(content),
            "has_image_tags":    "[IMAGE:" in content,
        })
    return payload

# =============================================================================
# NEO4J VECTOR STORE BUILD
# — replaces the old Qdrant build_vector_store()
# — stores :Chunk nodes with embedding vectors in Neo4j Aura
# =============================================================================

def build_neo4j_vector_store(force_rebuild: bool = False) -> None:
    """
    Embed all text + caption chunks and store in Neo4j as :Chunk nodes.

    Neo4j replaces Qdrant entirely:
      - :Chunk nodes hold text, metadata, and 768-dim embedding
      - HNSW vector index created on the embedding property
      - chunk_id property links to graph entities built later
        by cancer_graph_builder.py

    The same chunk JSON files written to output/chunks/ are
    read here for embedding AND by cancer_graph_builder.py for
    graph construction AND by BM25Retriever at query time.
    Three consumers, one source of truth.
    """
    print(f"\n{'─'*62}")
    print(f"🗄️   NEO4J VECTOR STORE BUILD")
    print(f"    URI: {NEO4J_URI}")

    # 1 — Load text docs
    text_docs: list[LCDoc] = []
    for jp in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(jp, encoding="utf-8") as f:
            for c in json.load(f):
                text_docs.append(LCDoc(
                    page_content=c["content"],
                    metadata={k: v for k, v in c.items() if k != "content"}
                ))

    # 2 — Load caption docs
    caption_docs: list[LCDoc] = []
    for jp in sorted(CAP_DIR.glob("*_caption_chunks.json")):
        with open(jp, encoding="utf-8") as f:
            for c in json.load(f):
                content = c.pop("content", "")
                caption_docs.append(LCDoc(page_content=content, metadata=c))

    all_docs = text_docs + caption_docs
    img_tag_count = sum(1 for d in text_docs if d.metadata.get("has_image_tags"))
    print(f"\n    📦 {len(text_docs)} text ({img_tag_count} with image tags) "
          f"+ {len(caption_docs)} caption = {len(all_docs)} total")

    if not all_docs:
        print("    ❌ No documents — run PDF processing first"); return

    # 3 — Load embedding model
    print(f"    🔢 Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # 4 — Store in Neo4j
    # Neo4jVector.from_documents() handles:
    #   a) Creating :Chunk nodes with text + metadata properties
    #   b) Computing and storing embeddings
    #   c) Creating HNSW vector index named NEO4J_CHUNK_INDEX
    print(f"\n    💾 Writing {len(all_docs)} chunks to Neo4j...")
    print(f"    ⏳ This takes ~2–5 minutes for 900 chunks on Aura free tier...")

    try:
        # Process in batches to avoid timeout on Aura free tier
        batch_size = UPSERT_BATCH
        for start in range(0, len(all_docs), batch_size):
            batch = all_docs[start:start + batch_size]

            # First batch creates the index; subsequent batches add to it
            if start == 0:
                neo4j_vs = Neo4jVector.from_documents(
                    documents=batch,
                    embedding=embed_model,
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database=NEO4J_DATABASE,
                    index_name=NEO4J_CHUNK_INDEX,
                    node_label=NEO4J_CHUNK_LABEL,
                    text_node_property=NEO4J_CHUNK_TEXT_PROP,
                    embedding_node_property=NEO4J_CHUNK_EMBEDDING_PROP,
                    pre_delete_collection=force_rebuild,
                )
            else:
                neo4j_vs.add_documents(batch)

            print(f"    ↳ {min(start + batch_size, len(all_docs)):4d} / {len(all_docs)} stored")

        print(f"    ✅ Neo4j vector store ready — index: '{NEO4J_CHUNK_INDEX}'")
        print(f"    ✅ {len(all_docs)} :Chunk nodes created in Neo4j Aura")

        # Save local summary for debugging
        summary = {
            "total_chunks":     len(all_docs),
            "text_chunks":      len(text_docs),
            "caption_chunks":   len(caption_docs),
            "image_tag_chunks": img_tag_count,
            "neo4j_index":      NEO4J_CHUNK_INDEX,
            "embedding_model":  EMBEDDING_MODEL,
            "embedding_dim":    EMBEDDING_DIM,
        }
        with open(EMBED_DIR / "neo4j_store_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    except Exception as e:
        print(f"    ❌ Neo4j store build failed: {e}")
        import traceback; traceback.print_exc()
        raise

# =============================================================================
# MAIN
# =============================================================================

def main(force_rebuild: bool = False) -> None:
    print("=" * 62)
    print("  cancer_ingestion.py — v5 Graph RAG (Neo4j backend)")
    print("=" * 62)
    print(f"  Docling : {'✅' if _DOCLING else '⬜ fallback active'}")
    print(f"  EasyOCR : {'✅' if _EASYOCR  else '⬜ optional'}")
    print(f"  Neo4j   : {NEO4J_URI[:40]}...")

    ensure_dirs()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌  No PDFs in: {PDF_DIR}"); return

    print(f"\n📂  {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"    • {p.name:<45} [{detect_cancer_type(p.stem)}]")

    results: list = []
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        results.append(process_single_pdf(pdf))

    with open(LOG_DIR / "pipeline_log.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_rejection_log()
    build_neo4j_vector_store(force_rebuild=force_rebuild)

    success        = [r for r in results if r.get("status") == "success"]
    total_text     = sum(r.get("text_chunks", 0)      for r in success)
    total_captions = sum(r.get("caption_chunks", 0)   for r in success)
    total_images   = sum(r.get("images_extracted", 0) for r in success)

    print(f"\n{'='*62}")
    print(f"  INGESTION COMPLETE — v5")
    print(f"{'='*62}")
    print(f"  ✅  PDFs processed   : {len(success)} / {len(results)}")
    print(f"  📦  Text chunks      : {total_text}")
    print(f"  📸  Caption chunks   : {total_captions}")
    print(f"  🖼️   Images saved     : {total_images}")
    print(f"  🗄️   Neo4j index      : '{NEO4J_CHUNK_INDEX}'")
    print(f"\n  ➡️   Next: python cancer_graph_builder.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    # force_rebuild=True clears existing Neo4j :Chunk nodes before rebuilding
    # force_rebuild=False skips rebuild if index already exists
    main(force_rebuild=True)
