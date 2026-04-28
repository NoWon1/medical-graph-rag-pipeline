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

#  (added Google API key to .env and config.py)

# =============================================================================
# config.py — Single source of truth for all constants
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
    for d in [MD_DIR, CHUNK_DIR, CAP_DIR, IMAGE_DIR, EMBED_DIR, LOG_DIR, GRAPH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# NEO4J
# =============================================================================

NEO4J_URI      = os.getenv("NEO4J_URI",      "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

NEO4J_CHUNK_INDEX          = "chunk_vector_index"
NEO4J_CHUNK_LABEL          = "Chunk"
NEO4J_CHUNK_TEXT_PROP      = "text"
NEO4J_CHUNK_EMBEDDING_PROP = "embedding"

NEO4J_CHEMO_VECTOR_INDEX         = "chemo_vector_index"
NEO4J_CANCER_VECTOR_INDEX        = "cancer_vector_index"
NEO4J_EATING_EFFECT_VECTOR_INDEX = "eating_effect_vector_index"
NEO4J_NON_CHEMO_VECTOR_INDEX     = "non_chemo_vector_index"

NEO4J_CHEMO_FULLTEXT_INDEX       = "chemo_text_index"
NEO4J_CANCER_FULLTEXT_INDEX      = "cancer_text_index"
NEO4J_EATING_FULLTEXT_INDEX      = "eating_effect_text_index"
NEO4J_FOOD_FULLTEXT_INDEX        = "food_item_text_index"
NEO4J_INTERACTION_FULLTEXT_INDEX = "interaction_text_index"

# =============================================================================
# LLM CONFIGURATION (PLUGGABLE ARCHITECTURE)
# =============================================================================

# 1. API Keys
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 2. Pipeline "Student" Config (The model answering the questions)
PIPELINE_LLM_PROVIDER = os.getenv("PIPELINE_LLM_PROVIDER", "groq")
GROQ_MODEL_QUERY      = os.getenv("PIPELINE_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_TEMP_QUERY       = 0.1

# 3. RAGAS "Judge" Config (The model grading the answers)
JUDGE_LLM_PROVIDER    = os.getenv("JUDGE_LLM_PROVIDER", "google")
RAGAS_JUDGE_MODEL     = os.getenv("JUDGE_LLM_MODEL", "gemini-2.5-pro")

# 4. Embeddings (Used for both Retrieval and RAGAS)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM   = 768

# =============================================================================
# INGESTION & DOCUMENTS
# =============================================================================

MIN_TEXT_CHARS   = 80
MIN_IMAGE_PX     = 100
MAX_ASPECT_RATIO = 6.0
DPI_VECTOR       = 300
DPI_OCR          = 300

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 150
MIN_CHUNK_CHARS  = 120

HEADERS_TO_SPLIT   = [("#", "H1"), ("##", "H2"), ("###", "H3")]
DOCLING_IMG_SCALE  = 3.0  # Increase image scale for better OCR and visual clarity in markdown
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
# QUERY MODES & GRAPH ENUMS
# =============================================================================

QUERY_MODE_RESEARCH = "research"
QUERY_MODE_GRAPH    = "graph"
QUERY_MODE_AUTO     = "auto"
QUERY_MODE_DEFAULT  = QUERY_MODE_AUTO

QUERY_MODE_LABELS = {
    QUERY_MODE_AUTO:     "Auto (recommended)",
    QUERY_MODE_RESEARCH: "Research & Literature",
    QUERY_MODE_GRAPH:    "Treatment & Nutrition",
}


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

KNOWN_CANCERS: set[str] = {"lung cancer", "breast cancer", "acute leukemia", "leukemia", "osteosarcoma", "skin cancer", "melanoma", "nsclc", "sclc", "aml", "all", "tnbc", "her2"}
KNOWN_CHEMO_DRUGS: set[str] = {"cisplatin", "carboplatin", "paclitaxel", "docetaxel", "gemcitabine", "pemetrexed", "vinorelbine", "doxorubicin", "cyclophosphamide", "capecitabine", "methotrexate", "vincristine", "cytarabine", "daunorubicin", "idarubicin", "etoposide", "ifosfamide", "mercaptopurine", "thioguanine", "l-asparaginase", "asparaginase", "ipilimumab", "nivolumab", "pembrolizumab", "atezolizumab", "vemurafenib", "dabrafenib", "trametinib", "trastuzumab", "pertuzumab"}
KNOWN_NON_CHEMO_DRUGS: set[str] = {"warfarin", "aspirin", "ibuprofen", "metformin", "omeprazole", "phenytoin", "carbamazepine", "fluconazole", "voriconazole", "ciprofloxacin", "co-trimoxazole", "tmp-smx", "atorvastatin", "sertraline", "amitriptyline", "amlodipine", "furosemide", "allopurinol", "morphine", "levothyroxine", "dexamethasone"}
# Add these right below KNOWN_NON_CHEMO_DRUGS in config.py

KNOWN_PROTOCOLS: set[str] = {
    "ac-t", "ac-t (breast)", "map", "map (osteosarcoma)",
    "7+3", "7+3 aml induction", "hyper-cvad", "hyper-cvad (all)",
    "carboplatin/paclitaxel", "carboplatin/paclitaxel (nsclc)",
    "ipi+nivo", "ipi+nivo (melanoma)",
}

KNOWN_EATING_EFFECTS: set[str] = {
    "nausea", "vomiting", "diarrhoea", "diarrhea",
    "constipation", "mucositis", "sore mouth",
    "appetite loss", "weight loss", "weight gain",
    "taste changes", "smell changes", "taste and smell changes",
    "dry mouth", "fatigue", "sore mouth (mucositis)",
}

FOOD_KEYWORDS: set[str] = {
    "eat", "eating", "food", "foods", "diet", "nutrition", "drink",
    "avoid", "meal", "appetite", "nausea", "vomiting", "taste",
    "swallow", "mouth", "stomach", "digest", "supplement", "vitamin",
    "protein", "calorie", "hydration", "water", "grapefruit", "milk",
    "alcohol", "what to eat", "what to avoid", "side effect", "sore mouth",
    "constipation", "diarrhoea", "diarrhea", "weight",
}

INTERACTION_KEYWORDS: set[str] = {
    "interaction", "interact", "combined", "together", "mixing",
    "blood thinner", "anticoagulant", "antibiotic", "antifungal",
    "diabetes medication", "blood pressure", "epilepsy", "seizure",
    "painkiller", "pain medication", "antidepressant", "statin",
}

INTENT_CANCER_DRUGS_EFFECTS  = "cancer_drugs_effects"
INTENT_DRUG_INTERACTIONS     = "chemo_drug_interactions"
INTENT_FOOD_GUIDANCE         = "food_guidance_for_drug"
INTENT_PROTOCOL_DETAIL       = "protocol_full_detail"
INTENT_NON_CHEMO_INTERACTION = "non_chemo_interaction"
INTENT_GENERAL_GRAPH         = "general_graph_search"

GRAPH_TOP_K_RESULTS          = 15
GRAPH_MODE_VECTOR_ENRICHMENT = 3
RESEARCH_MODE_TOP_K          = 8

K_DENSE     = 20
K_SPARSE    = 20
K_RRF_FINAL = 20
K_MMR_FINAL = 8
MMR_LAMBDA  = 0.6
RRF_K       = 60

IMAGE_TAG_PREFIX  = "[IMAGE:"
IMAGE_TAG_SUFFIX  = "]"
IMAGE_TAG_PATTERN = r'\[IMAGE:\s*([^\]]+)\]'

NO_ANSWER_PHRASES = [
    "do not have enough information", "not mentioned in", "not provided in", 
    "not found in the context", "cannot find", "no information", 
    "not in the context", "not recognized", "not described", "no mention of",
    "no specific", "does not mention"  
]

CANCER_KEYWORDS = [("osteosarcoma", "osteosarcoma"), ("leukemia", "leukemia"), ("melanoma", "melanoma"), ("breast", "breast"), ("lung", "lung"), ("skin", "skin")]

def detect_cancer_type(filename: str) -> str:
    fn = filename.lower()
    for keyword, label in CANCER_KEYWORDS:
        if keyword in fn:
            return label
    return "general"