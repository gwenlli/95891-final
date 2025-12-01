# Multi-Modal Attribute Extraction & Retrieval for Secondhand E-Commerce

**Version:** 0.1 (MVP)  
**Platform:** Streamlit (Python)  
**Target Audience:** AI/ML Course Instructors & Stakeholders

## 🎯 Project Overview

This Streamlit application serves as the interactive evaluation interface for a multi-modal deep learning model designed to clean noisy secondhand clothing listings. It demonstrates the model's ability to:

- **Ingest** user-generated images and unstructured text
- **Output** structured verified attributes with confidence scores
- **Retrieve** visually/semantically similar items from a catalog

### Core Value Proposition

Solving the "discovery problem" in sustainable fashion by automating accurate item labeling and enabling intelligent similarity search.

## 🚀 Quick Start

### Installation

1. Clone or download this repository

2. Create a virtual environment (recommended, especially on macOS):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Create assets directory structure:

```bash
mkdir -p assets/images
```

### Running the Application

**Option 1: Using the quick start script (easiest)**

```bash
./run.sh
```

The script will automatically create a virtual environment if needed and install dependencies.

**Option 2: Manual start**

If you already have the virtual environment activated:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

**Note**: If you encounter `ModuleNotFoundError`, make sure you've activated the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 📋 Features

### 1. Input Module - The "Noisy" Listing

Two input methods available:

**Method 1: Upload Image & Text**
- **Image Upload**: Upload a single clothing item image (JPG/PNG)
- **Text Input**: Enter or use sample "noisy" seller descriptions
- **Process Button**: Trigger the inference pipeline

**Method 2: Poshmark Link Scraping** 🆕
- **URL Input**: Paste a Poshmark listing URL
- **Automatic Scraping**: Extracts item name, size, description, and images
- **Security Features**: 
  - Rotating user agents to avoid detection
  - Random delays between requests (2-5 seconds)
  - Proper error handling and timeout management
- **Process Button**: Process the scraped listing data

### 2. Output Module A: Structured Attribute Verification

- **Clean vs Noisy Description**: Side-by-side comparison
- **Attribute Extraction Table**: 
  - Category, Color, Material, Style, Condition
  - Confidence scores (0-100%) with color-coded badges
  - Green (High: ≥80%), Yellow (Medium: ≥60%), Red (Low: <60%)

### 3. Output Module B: Similarity Search & Retrieval

- **Embedding Visualization**: Optional 2D t-SNE scatter plot showing latent space
- **Top-5 Ranked Results**: 
  - Retrieved items with similarity scores
  - Cosine similarity displayed for each result

## 🏗️ Architecture

### File Structure

### File Structure

```text
.
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── assets/
│   └── images/              # Asset directory for images
├── models/                  # Local model artifacts and configurations
│   ├── fashion_cleaner_model/
│   └── final_vision_model/
├── scrapers/                # Poshmark scraping scripts
│   ├── poshmark_scraper.py
│   └── poshmark_scraper_selenium.py
├── src/                     # Core application logic and utilities
│   ├── cleaned_captions.json
│   ├── data_utils.py        # Catalog loading and data utilities
│   └── model_inference.py   # Model wrapper and inference logic
├── venv/                    # Virtual environment
├── .gitattributes
├── .gitignore
├── app.py                   # Main Streamlit application entry point
├── config.json              # General configuration
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── run.sh                   # Quick start script
```

### Key Components

1. **MultiModalModel** (`model_inference.py`):
   - Handles image/text embedding extraction
   - Attribute classification
   - Similarity search
   - Embedding visualization

2. **Data Utilities** (`data_utils.py`):
   - Catalog loading (Pandas DataFrame)
   - Sample data generation
   - Pre-computed embedding support

3. **Poshmark Scraper** (`poshmark_scraper.py`):
   - Web scraping functionality for Poshmark listings
   - Security measures (user agent rotation, delays, timeouts)
   - Extracts: title, size, description, images, price, brand

4. **Streamlit App** (`app.py`):
   - User interface with dual input methods
   - Input/output handling
   - Metrics dashboard
   - Validation mode
   - Poshmark link integration

## 🔧 Models

1. **Text-to-text**
	- Look at the fashion_cleaner_model for the training data and weights 

2. **Vision-encoder**
	- Look at the final_image_processor, final_vision_model and final_tokenizer for model weights and python file. 


## 📊 User Flow

### Upload Method:
1. **Start**: User opens app → Sidebar shows Model Metrics
2. **Input**: Choose "Upload Image & Text" → Upload image + enter description
3. **Process**: Click "Process Listing" → Loading spinner appears
4. **Output A (Classification)**: Displays extracted attributes with confidence scores
5. **Output B (Retrieval)**: Shows Top-5 similar items with similarity scores

### Poshmark Link Method:
1. **Start**: User opens app → Sidebar shows Model Metrics
2. **Input**: Choose "Poshmark Link" → Paste Poshmark listing URL
3. **Scrape**: Click "🔍 Scrape Listing" → Extracts item information automatically
4. **Review**: Review scraped data (title, size, description, images)
5. **Process**: Click "🚀 Process Scraped Listing" → Loading spinner appears
6. **Output**: Same as Upload Method (attributes + similar items)

## 📝 Technical Stack

- **Frontend**: Streamlit
- **Model Inference**: PyTorch (mock/production models)
- **Data Storage**: Pandas DataFrame (local) or FAISS index
- **Visualization**: Plotly
- **Image Processing**: PIL/Pillow

## 📚 References

- Fashion-IQ dataset style for validation data
- Depop/Poshmark style for noisy input simulation
- Multi-modal embedding architectures for similarity search

## 📄 License

This project is created for educational purposes (AI/ML Course).

---


