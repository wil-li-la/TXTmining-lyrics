FROM python:3.11-slim

# Light system deps (git for any pip git+ install, curl for the streamlit healthcheck script)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: non-root user "user" with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install Python deps first (cache layer)
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# Pre-download NLTK + TextBlob data so first request doesn't pay for it
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('cmudict', quiet=True); \
    nltk.download('punkt_tab', quiet=True)" \
    && python -m textblob.download_corpora

# Pre-download SBERT model (~90 MB) into the user cache so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the app
COPY --chown=user . .

# HF Spaces routes to port 7860 by default
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 7860
CMD ["streamlit", "run", "app.py"]
