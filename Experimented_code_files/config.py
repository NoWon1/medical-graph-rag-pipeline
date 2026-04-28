
# Attempt 1:

# =============================================================================
# config.py — Single source of truth for all constants
# All other files import from here. No hardcoded strings anywhere else.
#
# CHANGES FROM v1:
#   - Removed: GRAPH_BUILD_BATCH, GRAPH_CONTENT_TYPES, ALLOWED_NODES,
#               ALLOWED_RELATIONSHIPS, USE_GRAPH_RAG, GRAPH_HOP_DEPTH,
#               GRAPH_TOP_K_ENTITIES  (all were for LLMGraphTransformer)
#   - Added:   QUERY MODE constants  (three-mode toggle)
#   - Added:   GRAPH NODE LABELS     (hand-crafted graph node types)
#   - Added:   GRAPH RELATIONSHIP TYPES (from .cypher files)
#   - Added:   GRAPH ENTITY LISTS    (for intent detection)
#   - Added:   GRAPH RETRIEVAL settings
#   - Added:   CYPHER QUERY KEYS     (maps intent to query name)
#   - Added:   IMAGE TAG constant    (preserved from v3 retrieval)
# =============================================================================

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# DIRECTORIES
# =============================================================================

BASE_DIR  = Path(__file__).parent
PDF_DIR   = BASE_DIR / "data"

MD_DIR    = BASE_DIR / "output" / "markdown"
CHUNK_DIR = BASE_DIR / "output" / "chunks"
CAP_DIR   = BASE_DIR / "output" / "caption_chunks"
IMAGE_DIR = BASE_DIR / "output" / "images"
EMBED_DIR = BASE_DIR / "output" / "embedding_export"
LOG_DIR   = BASE_DIR / "output" / "logs"
GRAPH_DIR = BASE_DIR / "output" / "graph"


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in [MD_DIR, CHUNK_DIR, CAP_DIR, IMAGE_DIR,
              EMBED_DIR, LOG_DIR, GRAPH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# NEO4J — single database for vectors + graph
# =============================================================================

NEO4J_URI      = os.getenv("NEO4J_URI",      "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME",  "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",  "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE",  "neo4j")

# ── Chunk vector index (created by cancer_ingestion.py) ───────
NEO4J_CHUNK_INDEX          = "chunk_vector_index"
NEO4J_CHUNK_LABEL          = "Chunk"
NEO4J_CHUNK_TEXT_PROP      = "text"
NEO4J_CHUNK_EMBEDDING_PROP = "embedding"

# ── Graph node vector indexes (from 01_schema_constraints.cypher) ─
# Used for semantic search directly on graph nodes
NEO4J_CHEMO_VECTOR_INDEX         = "chemo_vector_index"
NEO4J_CANCER_VECTOR_INDEX        = "cancer_vector_index"
NEO4J_EATING_EFFECT_VECTOR_INDEX = "eating_effect_vector_index"
NEO4J_NON_CHEMO_VECTOR_INDEX     = "non_chemo_vector_index"

# ── Graph fulltext indexes (from 01_schema_constraints.cypher) ────
# Used for keyword fallback when exact entity name not found
NEO4J_CHEMO_FULLTEXT_INDEX       = "chemo_text_index"
NEO4J_CANCER_FULLTEXT_INDEX      = "cancer_text_index"
NEO4J_EATING_FULLTEXT_INDEX      = "eating_effect_text_index"
NEO4J_FOOD_FULLTEXT_INDEX        = "food_item_text_index"
NEO4J_INTERACTION_FULLTEXT_INDEX = "interaction_text_index"

# =============================================================================
# MODELS
# =============================================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_QUERY = "llama-3.3-70b-versatile"   # mixtral for better value extraction in answers
GROQ_TEMP_QUERY  = 0.1

# =============================================================================
# INGESTION  (cancer_ingestion.py — do not change these)
# =============================================================================

MIN_TEXT_CHARS    = 80
MIN_IMAGE_PX      = 100
MAX_ASPECT_RATIO  = 6.0
DPI_VECTOR        = 150
DPI_OCR           = 200

CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
MIN_CHUNK_CHARS   = 120

HEADERS_TO_SPLIT   = [("#", "H1"), ("##", "H2"), ("###", "H3")]
DOCLING_IMG_SCALE  = 0.8
EMBED_PREVIEW_DIMS = 8
UPSERT_BATCH       = 64

SOURCE_URLS: dict[str, str] = {
    "osteosarcoma-review":         "https://pubmed.ncbi.nlm.nih.gov/33795081/",
    "acute-leukemia-review":       "https://pubmed.ncbi.nlm.nih.gov/32093433/",
    "breast-cancer-review":        "https://pubmed.ncbi.nlm.nih.gov/31735550/",
    "lung-cancer-review":          "https://pubmed.ncbi.nlm.nih.gov/33207404/",
    "melanoma-skin-cancer-review": "https://pubmed.ncbi.nlm.nih.gov/32887954/",
    "skin-cancer-types-review":    "https://pubmed.ncbi.nlm.nih.gov/30609218/",
}

def get_source_url(source_name: str) -> str:
    return SOURCE_URLS.get(source_name, "")

# =============================================================================
# QUERY MODE — three-mode toggle
# Used by cancer_app.py (UI) and cancer_retrieval.py (routing logic)
# =============================================================================

# Mode string constants — passed from app.py into generate_answer()
QUERY_MODE_RESEARCH = "research"   # PDF chunks only — clinical literature
QUERY_MODE_GRAPH    = "graph"      # Hand-crafted graph — drug/food/interaction
QUERY_MODE_AUTO     = "auto"       # Both sources merged — default for new users

QUERY_MODE_DEFAULT  = QUERY_MODE_AUTO

# Display labels shown in the Streamlit sidebar radio selector
QUERY_MODE_LABELS = {
    QUERY_MODE_AUTO:     "Auto (recommended)",
    QUERY_MODE_RESEARCH: "Research & Literature",
    QUERY_MODE_GRAPH:    "Treatment & Nutrition",
}

# Tooltip help text shown below each mode in the sidebar
QUERY_MODE_DESCRIPTIONS = {
    QUERY_MODE_AUTO: (
        "Searches both clinical literature and the treatment knowledge graph. "
        "Best for complex or mixed questions."
    ),
    QUERY_MODE_RESEARCH: (
        "Ask about clinical trials, survival rates, staging, prognosis, "
        "and treatment rationale from peer-reviewed papers."
    ),
    QUERY_MODE_GRAPH: (
        "Ask about chemotherapy eating side effects, food to avoid or eat, "
        "drug interactions, nutrition guidelines, and treatment protocols."
    ),
}

# =============================================================================
# GRAPH NODE LABELS
# Exactly match the labels defined in 01_schema_constraints.cypher
# =============================================================================

GRAPH_LABEL_CANCER           = "Cancer"
GRAPH_LABEL_CHEMO_DRUG       = "ChemoDrug"
GRAPH_LABEL_EATING_EFFECT    = "EatingAdverseEffect"
GRAPH_LABEL_FOOD_ITEM        = "FoodItem"
GRAPH_LABEL_NUTRITION_GUIDE  = "NutritionGuideline"
GRAPH_LABEL_BIOMARKER        = "Biomarker"
GRAPH_LABEL_TREATMENT_PROTO  = "TreatmentProtocol"
GRAPH_LABEL_NON_CHEMO_DRUG   = "NonChemoDrug"
GRAPH_LABEL_AILMENT          = "Ailment"
GRAPH_LABEL_DRUG_INTERACTION = "DrugInteraction"
GRAPH_LABEL_SIDE_EFFECT      = "SideEffect"

# =============================================================================
# GRAPH RELATIONSHIP TYPES
# Exactly match relationship types in 02 and 03 .cypher files
# =============================================================================

REL_TREATED_WITH         = "TREATED_WITH"
REL_CAUSES_EATING_EFFECT = "CAUSES_EATING_EFFECT"
REL_WORSENED_BY          = "WORSENED_BY"
REL_RELIEVED_BY          = "RELIEVED_BY"
REL_HAS_BIOMARKER        = "HAS_BIOMARKER"
REL_TREATED_BY_PROTOCOL  = "TREATED_BY_PROTOCOL"
REL_INCLUDES_DRUG        = "INCLUDES_DRUG"
REL_HAS_INTERACTION_WITH = "HAS_INTERACTION_WITH"
REL_DESCRIBED_BY         = "DESCRIBED_BY"
REL_TREATS               = "TREATS"
REL_MAY_CAUSE            = "MAY_CAUSE"
REL_COMPOUNDS_EATING     = "COMPOUNDS_EATING_EFFECT"
REL_LEADS_TO_EATING      = "LEADS_TO_EATING_EFFECT"
REL_MANAGES              = "MANAGES"
REL_REQUIRED_FOR         = "REQUIRED_FOR"
REL_WARNS_ABOUT          = "WARNS_ABOUT"

# =============================================================================
# GRAPH ENTITY LISTS
# Used by intent detection in cancer_retrieval.py to identify what the
# user is asking about and route to the correct Cypher query.
# All values lowercase — queries are lowercased before matching.
# =============================================================================

# Exact Cancer.name values from 02_cancer_chemo_eating.cypher (lowercase)
KNOWN_CANCERS: set[str] = {
    "lung cancer",
    "breast cancer",
    "acute leukemia",
    "leukemia",
    "osteosarcoma",
    "skin cancer",
    "melanoma",
    "nsclc",
    "sclc",
    "aml",
    "all",
    "tnbc",
    "her2",
}

# Exact ChemoDrug.name values (lowercase)
KNOWN_CHEMO_DRUGS: set[str] = {
    "cisplatin", "carboplatin", "paclitaxel", "docetaxel",
    "gemcitabine", "pemetrexed", "vinorelbine", "doxorubicin",
    "cyclophosphamide", "capecitabine", "methotrexate", "vincristine",
    "cytarabine", "daunorubicin", "idarubicin", "etoposide",
    "ifosfamide", "mercaptopurine", "thioguanine", "l-asparaginase",
    "asparaginase", "ipilimumab", "nivolumab", "pembrolizumab",
    "atezolizumab", "vemurafenib", "dabrafenib", "trametinib",
    "trastuzumab", "pertuzumab",
}

# Exact NonChemoDrug.name values (lowercase)
KNOWN_NON_CHEMO_DRUGS: set[str] = {
    "warfarin", "aspirin", "ibuprofen", "metformin", "omeprazole",
    "phenytoin", "carbamazepine", "fluconazole", "voriconazole",
    "ciprofloxacin", "co-trimoxazole", "tmp-smx", "atorvastatin",
    "sertraline", "amitriptyline", "amlodipine", "furosemide",
    "allopurinol", "morphine", "levothyroxine", "dexamethasone",
}

# Exact TreatmentProtocol.name values (lowercase)
KNOWN_PROTOCOLS: set[str] = {
    "ac-t", "ac-t (breast)",
    "map", "map (osteosarcoma)",
    "7+3", "7+3 aml induction",
    "hyper-cvad", "hyper-cvad (all)",
    "carboplatin/paclitaxel", "carboplatin/paclitaxel (nsclc)",
    "ipi+nivo", "ipi+nivo (melanoma)",
}

# EatingAdverseEffect.name values (lowercase)
KNOWN_EATING_EFFECTS: set[str] = {
    "nausea", "vomiting", "diarrhoea", "diarrhea",
    "constipation", "mucositis", "sore mouth",
    "appetite loss", "weight loss", "weight gain",
    "taste changes", "smell changes", "taste and smell changes",
    "dry mouth", "fatigue", "sore mouth (mucositis)",
}

# Keywords that strongly signal a graph query in AUTO mode
FOOD_KEYWORDS: set[str] = {
    "eat", "eating", "food", "foods", "diet", "nutrition", "drink",
    "avoid", "meal", "appetite", "nausea", "vomiting", "taste",
    "swallow", "mouth", "stomach", "digest", "supplement", "vitamin",
    "protein", "calorie", "hydration", "water", "grapefruit", "milk",
    "alcohol", "what to eat", "what to avoid", "side effect", "sore mouth",
    "constipation", "diarrhoea", "diarrhea", "weight",
}

# Keywords that signal a drug interaction query
INTERACTION_KEYWORDS: set[str] = {
    "interaction", "interact", "combined", "together", "mixing",
    "blood thinner", "anticoagulant", "antibiotic", "antifungal",
    "diabetes medication", "blood pressure", "epilepsy", "seizure",
    "painkiller", "pain medication", "antidepressant", "statin",
}

# =============================================================================
# GRAPH RETRIEVAL SETTINGS
# =============================================================================

# Max graph result rows returned per Cypher query
GRAPH_TOP_K_RESULTS = 15

# Number of vector chunks added as narrative context in GRAPH mode
# Graph structured results appear first, these provide clinical backing
GRAPH_MODE_VECTOR_ENRICHMENT = 3

# Number of vector chunks in RESEARCH mode (pure literature)
RESEARCH_MODE_TOP_K = 8

# =============================================================================
# CYPHER QUERY INTENT KEYS
# Maps detected query intent → query name in CYPHER_QUERIES dict
# in cancer_retrieval.py
# =============================================================================

INTENT_CANCER_DRUGS_EFFECTS  = "cancer_drugs_effects"
INTENT_DRUG_INTERACTIONS     = "chemo_drug_interactions"
INTENT_FOOD_GUIDANCE         = "food_guidance_for_drug"
INTENT_PROTOCOL_DETAIL       = "protocol_full_detail"
INTENT_NON_CHEMO_INTERACTION = "non_chemo_interaction"
INTENT_GENERAL_GRAPH         = "general_graph_search"

# =============================================================================
# RETRIEVAL — vector pipeline settings (unchanged from v3)
# =============================================================================

K_DENSE     = 20
K_SPARSE    = 20
K_RRF_FINAL = 20
K_MMR_FINAL = 8
MMR_LAMBDA  = 0.6
RRF_K       = 60

# =============================================================================
# IMAGE TAGS — preserved from original v3 retrieval pipeline
# Injected into markdown during ingestion, retrieved in chunks,
# reproduced by LLM in answers, rendered by render_message_with_images()
# in cancer_app.py using st.image()
# =============================================================================

IMAGE_TAG_PREFIX  = "[IMAGE:"
IMAGE_TAG_SUFFIX  = "]"
IMAGE_TAG_PATTERN = r'\[IMAGE:\s*([^\]]+)\]'

# =============================================================================
# WEB FALLBACK — DuckDuckGo search trigger phrases
# =============================================================================

NO_ANSWER_PHRASES = [
    "do not have enough information",
    "not mentioned in",
    "not provided in",
    "not found in the context",
    "cannot find",
    "no information",
    "not in the context",
    "not recognized",
    "not described",
    "no mention of",
]

# =============================================================================
# CANCER TYPE DETECTION (used by cancer_ingestion.py only)
# =============================================================================

CANCER_KEYWORDS = [
    ("osteosarcoma", "osteosarcoma"),
    ("leukemia",     "leukemia"),
    ("melanoma",     "melanoma"),
    ("breast",       "breast"),
    ("lung",         "lung"),
    ("skin",         "skin"),
]

def detect_cancer_type(filename: str) -> str:
    fn = filename.lower()
    for keyword, label in CANCER_KEYWORDS:
        if keyword in fn:
            return label
    return "general"