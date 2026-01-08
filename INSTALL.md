# Installation Guide

## Quick Installation

The dependencies are being installed. If you need to install manually, follow these steps:

### Method 1: Using requirements.txt (Recommended)

```bash
# Make sure you're in the project directory
cd "d:\AI System Monitoring"

# Activate virtual environment
.\venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Method 2: Install Packages Individually

If you encounter issues with requirements.txt, install packages one by one:

```bash
# Activate virtual environment first
.\venv\Scripts\activate

# Install core packages
pip install streamlit
pip install psutil
pip install pandas
pip install numpy

# Install visualization
pip install plotly
pip install matplotlib

# Install ML
pip install scikit-learn

# Install RAG components
pip install chromadb
pip install sentence-transformers

# Install utilities
pip install python-dotenv
```

### Method 3: Minimal Installation (Start Small)

If you have limited bandwidth or want to test quickly:

```bash
# Essential packages only
pip install streamlit psutil pandas plotly scikit-learn
```

Then add more later:
```bash
pip install chromadb sentence-transformers
```

## Troubleshooting Installation

### Issue: "ERROR: Could not install packages"

**Solution**: Try installing without version constraints
```bash
pip install --upgrade pip setuptools wheel
pip install streamlit psutil pandas numpy plotly scikit-learn
```

### Issue: "No module named 'sklearn'"

**Solution**: Install scikit-learn
```bash
pip install scikit-learn
```

### Issue: ChromaDB installation fails

**Solution**: Install dependencies first
```bash
pip install pydantic numpy
pip install chromadb
```

### Issue: sentence-transformers download fails

**Solution**: Install PyTorch first (CPU version)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

## Verify Installation

After installation, test if everything works:

```bash
python test_modules.py
```

This will test all components and show which are working.

## Running Without All Dependencies

The app can run with minimal dependencies. Features will be disabled gracefully:

**Minimum to run**:
- streamlit
- psutil
- pandas
- plotly

**Without ChromaDB**: RAG features won't work
**Without sentence-transformers**: RAG features won't work  
**Without scikit-learn**: ML anomaly detection won't work

The app will still show:
- ✅ Live monitoring
- ✅ Rule-based alerts
- ✅ Basic explanations
- ✅ Visualizations

## Package Sizes (Approximate)

| Package | Size | Purpose |
|---------|------|---------|
| streamlit | ~30 MB | Dashboard |
| psutil | ~500 KB | System metrics |
| pandas | ~50 MB | Data processing |
| numpy | ~20 MB | Numerical ops |
| plotly | ~30 MB | Charts |
| scikit-learn | ~30 MB | ML |
| chromadb | ~50 MB | Vector DB |
| sentence-transformers | ~400 MB | Embeddings |

**Total**: ~600-700 MB

## Alternative: Use Conda

If you prefer conda:

```bash
conda create -n sysmon python=3.9
conda activate sysmon
conda install -c conda-forge streamlit psutil pandas numpy plotly scikit-learn
pip install chromadb sentence-transformers
```

## Need Help?

1. Check Python version: `python --version` (need 3.8+)
2. Check pip version: `pip --version`
3. Try: `pip install --upgrade pip`
4. Use: `pip list` to see installed packages

## Once Installed

Run the dashboard:
```bash
streamlit run app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open the Local URL in your browser!
