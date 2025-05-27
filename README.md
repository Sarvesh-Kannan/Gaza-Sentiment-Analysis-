# SentiNa - Gaza Reddit Posts Sentiment Analysis

This project performs sentiment analysis on Reddit posts related to the Gaza issue, including various NLP tasks such as topic modeling, named entity recognition, and explainable AI. The analysis is accelerated using GPU with CUDA for faster processing and model training.

## Features

- GPU-accelerated sentiment analysis using BERT-based models
- Topic modeling with Latent Dirichlet Allocation (LDA)
- Named Entity Recognition (NER) using spaCy
- Explainable AI with LIME to understand sentiment decisions
- Comprehensive visualizations including word clouds and sentiment distributions
- HTML report generation with detailed findings
- CUDA-enabled PyTorch for high-performance deep learning

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- CuDNN compatible with your CUDA version

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: For GPU support, you may need to install PyTorch with CUDA manually if the requirements installation doesn't configure it correctly:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace 'cu118' with your CUDA version, e.g., 'cu117' for CUDA 11.7)

4. Additional setup:
   - Download spaCy model:
     ```
     python -m spacy download en_core_web_sm
     ```
   - NLTK resources will be downloaded automatically on first run

## Usage

1. Ensure your JSON data file is named `reddit_posts.json` and is in the root directory of the project.

2. Run the sentiment analysis script:
   ```
   python sentiment_analysis.py
   ```
   The script will automatically detect and use your GPU if available.

3. View the results:
   - Open `gaza_reddit_sentiment_analysis_report.html` in a web browser
   - Check generated visualization files:
     - `sentiment_distribution.png`
     - `wordcloud.png`
     - `sentiment_vs_wordcount.png`
     - `entity_types.png`

## Expected JSON Format

The script is designed to handle various JSON formats, but ideally, your data should be either:
- A list of post objects
- A dictionary with a 'posts' key containing a list of post objects
- A nested dictionary that can be flattened to post objects

Each post object should include text content in fields like 'title', 'text', 'body', 'content', 'selftext', or 'comments'.

## GPU Acceleration Benefits

The implementation uses GPU acceleration for:
- BERT model fine-tuning and inference
- Sentiment analysis with Hugging Face transformers
- Large-scale data processing

This provides several advantages:
- Significantly faster training and inference times
- Ability to process larger datasets
- More complex models with better accuracy
- Enhanced explainability through deeper analysis

## Explainable AI

The project uses LIME (Local Interpretable Model-agnostic Explanations) with GPU-accelerated BERT models to explain individual sentiment predictions. This helps identify which words or phrases in a post most strongly influence its sentiment classification.

## Customization

You can modify `sentiment_analysis.py` to adjust:
- Number of topics in topic modeling (default: 5)
- Sample size for detailed analysis (default: up to 1000 posts)
- Number of features used for explanations
- Visualization styles and formats
- BERT model hyperparameters (batch size, learning rate, etc.)
- GPU memory usage via batch size 