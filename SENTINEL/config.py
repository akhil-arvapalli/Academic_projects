"""
SENTINEL — Configuration & Theme Constants
Satellite ENvironmental Terrain INtelligence EvaLuation
"""
import os

# ── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(env_key: str, default_rel_path: str) -> str:
	value = os.getenv(env_key, default_rel_path)
	if os.path.isabs(value):
		return value
	return os.path.join(BASE_DIR, value)


MODEL_DIR = _resolve_path("MODEL_DIR", "model")
SAMPLE_DIR = _resolve_path("SAMPLE_DIR", "sampleImages")
DATASET_DIR = _resolve_path("DATASET_DIR", "Dataset")

FEATURES_X_FILE = os.getenv("FEATURES_X_FILE", "X.txt.npy")
FEATURES_Y_FILE = os.getenv("FEATURES_Y_FILE", "Y.txt.npy")
WEIGHTS_FILE = os.getenv("WEIGHTS_FILE", "model_weights.h5")
WEIGHTS_V2_FILE = os.getenv("WEIGHTS_V2_FILE", "model_weights_v2.h5")
HISTORY_FILE = os.getenv("HISTORY_FILE", "history.pckl")
HISTORY_V2_FILE = os.getenv("HISTORY_V2_FILE", "history_v2.pckl")
METRICS_V2_FILE = os.getenv("METRICS_V2_FILE", "metrics_v2.pckl")

WEIGHTS_PATH = os.path.join(MODEL_DIR, WEIGHTS_FILE)
WEIGHTS_V2_PATH = os.path.join(MODEL_DIR, WEIGHTS_V2_FILE)
FEATURES_X_PATH = os.path.join(MODEL_DIR, FEATURES_X_FILE)
FEATURES_Y_PATH = os.path.join(MODEL_DIR, FEATURES_Y_FILE)
HISTORY_PATH = os.path.join(MODEL_DIR, HISTORY_FILE)
HISTORY_V2_PATH = os.path.join(MODEL_DIR, HISTORY_V2_FILE)
METRICS_V2_PATH = os.path.join(MODEL_DIR, METRICS_V2_FILE)

# ── Theme Colors ─────────────────────────────────────────
BG_PRIMARY = "#0a0e17"          # Deep dark navy
BG_SECONDARY = "#111827"        # Card/panel backgrounds
BG_TERTIARY = "#1a2332"         # Input fields, hover
BG_SIDEBAR = "#080c14"          # Sidebar background
ACCENT_GREEN = "#00ff88"        # Primary accent — neon green
ACCENT_GREEN_DIM = "#0a3d2a"    # Dimmed green for borders/bg
ACCENT_AMBER = "#ffaa00"        # Warning / secondary accent
ACCENT_RED = "#ff3366"          # Error / danger
TEXT_PRIMARY = "#e0e6ed"        # Main text
TEXT_SECONDARY = "#8899aa"      # Subdued text
TEXT_DIM = "#4a5568"            # Very dim (labels, hints)
BORDER_COLOR = "#1e3a2f"        # Green-tinted borders
BORDER_GLOW = "#00ff8833"       # Glow effect (with alpha)

# ── Fonts ────────────────────────────────────────────────
FONT_MONO = ("Consolas", 11)
FONT_MONO_SMALL = ("Consolas", 9)
FONT_MONO_LARGE = ("Consolas", 14, "bold")
FONT_HEADING = ("Segoe UI", 20, "bold")
FONT_SUBHEADING = ("Segoe UI", 14, "bold")
FONT_LABEL = ("Segoe UI", 11)
FONT_LABEL_BOLD = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_BUTTON = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Consolas", 24, "bold")

# ── Model Parameters ────────────────────────────────────
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 4
LABELS = ["Urban Land", "Agricultural Land", "Range Land", "Forest Land"]
LABEL_ICONS = ["🏙️", "🌾", "🏜️", "🌲"]
BATCH_SIZE = 32
EPOCHS = 20

# ── App Settings ─────────────────────────────────────────
APP_NAME = "SENTINEL"
APP_SUBTITLE = "Satellite Environmental Terrain Intelligence Evaluation"
APP_VERSION = "2.0.0"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750
SIDEBAR_WIDTH = 220
