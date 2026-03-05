# 🛰️ SENTINEL
### Satellite Environmental Terrain Intelligence Evaluation

A deep learning-powered satellite image classification system with a military tactical HUD interface. Classifies satellite imagery into 4 terrain types using an enhanced CNN architecture.

![Dashboard](docs/dashboard.png)

---

## 🎯 Classification Categories

| Category | Icon | Description |
|----------|------|-------------|
| Urban Land | 🏙️ | Built-up areas, buildings, roads |
| Agricultural Land | 🌾 | Farming, cultivated areas |
| Range Land | 🏜️ | Open grassland, shrubland |
| Forest Land | 🌲 | Dense vegetation, tree cover |

## 🏗️ Architecture

```
SENTINEL/
├── app.py                  # Flask API server (entry point)
├── config.py               # Configuration & theme constants
├── model_engine.py         # CNN v2 model (build, train, predict)
├── utils.py                # Image processing helpers & logging
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── frontend/               # React (Vite) web app
│   ├── src/
│   │   ├── App.jsx         # Router + state management
│   │   ├── index.css       # Military HUD theme (500+ lines)
│   │   ├── pages/          # Dashboard, Classify, Training, Analytics
│   │   └── components/     # Sidebar, StatusBar, GlowCard, ConfidenceBar
│   ├── package.json
│   └── vite.config.js      # Proxy /api → Flask
├── model/                  # Pre-trained model files (not in git)
├── sampleImages/           # Test satellite images
└── Dataset/                # Training dataset (not in git)
```

## 🧠 CNN Model v2

| Feature | Detail |
|---------|--------|
| Architecture | 3 Conv blocks (32→64→128) + BatchNorm + Dropout |
| Input | 64×64×3 RGB images |
| Augmentation | Rotation, flip, zoom, brightness |
| Validation | 80/20 stratified split |
| Early Stopping | Patience=5 on val_loss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5) |
| Metrics | Confusion matrix, precision, recall, F1-score |

## 🚀 Setup & Installation

### Prerequisites
- Python 3.12+
- Node.js 18+
- npm 9+

### Backend Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Data Setup
Place your dataset and model files:
```
model/
├── X.txt.npy          # Feature array (images)
├── Y.txt.npy          # Label array
└── model_weights.h5   # Pre-trained weights (optional)

Dataset/               # Raw satellite images by class
├── urban/
├── agricultural/
├── rangeland/
└── forest/
```

## ▶️ Running the App

### Terminal 1 — Backend
```bash
.\venv\Scripts\python app.py
# Server starts at http://localhost:5000
```

### Terminal 2 — Frontend
```bash
cd frontend
npm run dev
# App opens at http://localhost:5173
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status, model info |
| `/api/features/load` | POST | Load training features |
| `/api/model/load` | POST | Load pre-trained weights |
| `/api/model/train` | POST | Train CNN (with augmentation) |
| `/api/model/info` | GET | Model architecture details |
| `/api/predict` | POST | Classify single image |
| `/api/predict/batch` | POST | Classify multiple images |
| `/api/history` | GET | Training accuracy/loss history |
| `/api/metrics` | GET | Confusion matrix & per-class metrics |
| `/api/sample-images` | GET | List sample images |

## 🛠️ Tech Stack

- **Backend:** Python 3.12, TensorFlow 2.20, Keras 3, Flask, OpenCV
- **Frontend:** React 19, Vite 7, Recharts, Lucide Icons
- **ML:** Scikit-learn (metrics), NumPy, Matplotlib
- **Theme:** Custom military HUD CSS with 12+ animations

## 📄 License

Academic project — CMR University, Semester 6 (IOMP)
