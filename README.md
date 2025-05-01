# <img src="data/NextGen.png" alt="NextGen Analytics Logo" width="60"/> Next Generation Analytics Dashboard

Welcome to the **Next Generation Analytics** dashboard — an interactive, user-friendly platform for biomedical data visualization, machine learning, and medical image analysis. This tool is built using **Python**, **Streamlit**, **scikit-learn**, **Plotly**, and **SimpleITK**, and is designed for both clinical researchers and data scientists.

---

## 🚀 Features

- 📁 Upload and preview biomedical datasets (.csv, .xlsx, etc.)
- 📊 Visualize data distributions with histograms, boxplots, scatterplots, etc.
- 🧼 Clean data: handle missing values, duplicates, outliers, and normalization
- 🔬 Dimensionality reduction: PCA, t-SNE, UMAP
- 🤖 Train machine learning models with cross-validation and performance plots
- 🖼️ Analyze medical images (.nii, .nii.gz) and apply segmentation (LungMask, TotalSegmentator)
- 🧪 Natural image processing: grayscale, edge detection, face detection, histogram equalization, etc.
- 💬 Built-in AI assistant for dataset-specific Q&A (optional)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/suleimantaofik6/Next-Generation-Visual-Analytics-Dashboard.git
cd Next-Generation-Visual-Analytics-Dashboard
```

#### 📁 File Structure
```bash
.
├── app.py                     # Main Streamlit dashboard
├── requirements.txt           # Python package dependencies
├── ai_assistant.py            # Optional AI assistant module
├── README.md                  # This file
├── data/
│   └── NextGen.png            # Dashboard logo
├── demo/                      # (Optional) Video tutorials or recordings
└── data/                      # Sample input files
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Running the App

```bash
streamlit run main.py
```

This will launch the app in your browser. You can upload datasets, explore and clean them, train ML models, and analyse medical or natural images interactively.
