import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
import spacy
from wordcloud import WordCloud
import lime
import lime.lime_text
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from tqdm import tqdm

# Check for CUDA availability
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"Error initializing CUDA: {e}")
    device = torch.device("cpu")
    print("Falling back to CPU")

# Download required NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load spaCy model for NER
try:
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load the Reddit data
print("Loading Reddit data...")
try:
    with open('reddit_posts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading data: {e}")
    print("Creating sample data for testing...")
    # Create sample data for testing
    data = [
        {"title": "Gaza situation is terrible", "selftext": "The situation in Gaza is getting worse every day. The humanitarian crisis is escalating."},
        {"title": "Peace efforts in Middle East", "selftext": "There are some positive developments in peace negotiations for Gaza."},
        {"title": "Neutral post about Gaza", "selftext": "This is an informational post about the history and geography of Gaza."},
        {"title": "Angry about Gaza conflict", "selftext": "I'm so furious about what's happening in Gaza right now. This is unacceptable!"},
        {"title": "Hope for Gaza", "selftext": "I believe things will get better soon in Gaza. There's always hope."}
    ]

# Convert to DataFrame for easier processing
if isinstance(data, list):
    df = pd.DataFrame(data)
else:
    # Handle other possible JSON structures
    if isinstance(data, dict) and 'posts' in data:
        df = pd.DataFrame(data['posts'])
    else:
        # Try to flatten nested structure
        flattened_data = []
        for key, value in data.items():
            if isinstance(value, list):
                flattened_data.extend(value)
            elif isinstance(value, dict):
                flattened_data.append(value)
        df = pd.DataFrame(flattened_data)

print(f"Data loaded with {len(df)} posts")
print("Columns in the dataset:", df.columns.tolist())

# Determine text columns
text_columns = [col for col in df.columns if col.lower() in ['title', 'text', 'body', 'content', 'selftext', 'comments']]
if not text_columns:
    print("No text columns found. Using the first string column...")
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].map(lambda x: isinstance(x, str)).any():
            text_columns = [col]
            break

if not text_columns:
    raise ValueError("No text columns found in the dataset")

print(f"Using text columns: {text_columns}")

# Combine text columns into a single text field
df['combined_text'] = df[text_columns].apply(
    lambda row: ' '.join(str(cell) for cell in row if isinstance(cell, (str, int, float)) and pd.notna(cell) and str(cell).strip() != ''), 
    axis=1
)

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

print("Preprocessing text data...")
df['processed_text'] = df['combined_text'].apply(preprocess_text)

# Basic text statistics
df['text_length'] = df['combined_text'].apply(len)
df['word_count'] = df['combined_text'].apply(lambda x: len(str(x).split()))

print("\n--- Text Statistics ---")
print(f"Average text length: {df['text_length'].mean():.2f} characters")
print(f"Average word count: {df['word_count'].mean():.2f} words")

# SENTIMENT ANALYSIS
print("\n--- Performing Sentiment Analysis ---")
# Initialize the sentiment analyzer with CUDA support if available
try:
    # Try to use GPU if available
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    print(f"Error initializing GPU-based sentiment analyzer: {e}")
    print("Falling back to CPU-based sentiment analyzer")
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )

# Function to get sentiment with a batched approach to avoid memory issues
def get_sentiment(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Filter out empty strings to avoid errors
        batch = [text for text in batch if isinstance(text, str) and text.strip()]
        if batch:  # Only process if there are valid texts
            try:
                batch_results = sentiment_analyzer(batch)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error analyzing batch: {e}")
                # Add neutral sentiment for texts that couldn't be analyzed
                results.extend([{'label': 'NEUTRAL', 'score': 0.5} for _ in range(len(batch))])
        else:
            # Add neutral sentiment for empty texts
            results.extend([{'label': 'NEUTRAL', 'score': 0.5} for _ in range(len([t for t in texts[i:i+batch_size] if not (isinstance(t, str) and t.strip())]))])
    return results

# Sample a subset for detailed analysis if the dataset is large
sample_size = min(1000, len(df))
if len(df) > sample_size:
    print(f"Dataset is large, using a sample of {sample_size} posts for detailed analysis")
    df_sample = df.sample(sample_size, random_state=42)
else:
    df_sample = df

# Get sentiment for the sample
print("Analyzing sentiment...")
sentiments = get_sentiment(df_sample['combined_text'].tolist())
df_sample['sentiment'] = [result['label'] for result in sentiments]
df_sample['sentiment_score'] = [result['score'] for result in sentiments]

# Map to simpler sentiment labels
df_sample['sentiment_simple'] = df_sample['sentiment'].map(
    {'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral'}
)

# Calculate sentiment distribution
sentiment_counts = df_sample['sentiment_simple'].value_counts()
print("\nSentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}: {count} posts ({count/len(df_sample)*100:.2f}%)")

# TOPIC MODELING
print("\n--- Performing Topic Modeling ---")
vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.9)
X = vectorizer.fit_transform(df_sample['processed_text'])
feature_names = vectorizer.get_feature_names_out()

# LDA for topic modeling
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Print top words for each topic
print("\nTop words per topic:")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx+1}: {', '.join(top_words)}")

# NAMED ENTITY RECOGNITION
print("\n--- Performing Named Entity Recognition ---")
def extract_entities(text):
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        doc = nlp(text[:min(len(text), 100000)])  # Limit text length to avoid memory issues
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        print(f"Error in NER: {e}")
        return []

# Apply NER to a subset of the data
ner_sample_size = min(100, len(df_sample))
print(f"Extracting entities from {ner_sample_size} posts...")
entity_lists = [extract_entities(text) for text in df_sample['combined_text'].head(ner_sample_size)]

# Flatten and count entities
all_entities = [entity for sublist in entity_lists for entity in sublist]
entity_counter = Counter(all_entities)
most_common_entities = entity_counter.most_common(20)

print("\nMost common named entities:")
for (entity_text, entity_type), count in most_common_entities:
    print(f"{entity_text} ({entity_type}): {count} occurrences")

# EXPLAINABLE AI
print("\n--- Setting up Explainable AI ---")
# Prepare data for the classifier
X_text = df_sample['processed_text']
y = (df_sample['sentiment_simple'] == 'positive').astype(int)  # Binary classification: positive vs not positive

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(X_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a classifier
print("Training classifier for explainable AI...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier accuracy: {accuracy:.2f}")

# LIME for local explanations
class_names = ['negative/neutral', 'positive']
explainer = lime.lime_text.LimeTextExplainer(class_names=class_names)

def lime_predict_proba(texts):
    return clf.predict_proba(tfidf.transform(texts))

# Sample a few posts for explanation
explanation_samples = min(5, len(df_sample))
for i in range(explanation_samples):
    if i < len(X_text):
        text = X_text.iloc[i]
        if isinstance(text, str) and text.strip():
            print(f"\nExplaining post {i+1}:")
            exp = explainer.explain_instance(text, lime_predict_proba, num_features=10)
            print(f"Original text: {df_sample['combined_text'].iloc[i][:100]}...")
            print(f"Prediction: {class_names[clf.predict(tfidf.transform([text]))[0]]}")
            print("Explanation:")
            for feature, weight in exp.as_list():
                print(f"  {feature}: {weight:.4f}")

# VISUALIZATIONS
print("\n--- Creating Visualizations ---")

# 1. Sentiment Distribution Pie Chart
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['red', 'green', 'gray'])
plt.title('Sentiment Distribution in Gaza-Related Reddit Posts')
plt.ylabel('')
plt.savefig('sentiment_distribution.png')
print("Saved sentiment distribution pie chart to sentiment_distribution.png")

# 2. Word Cloud
plt.figure(figsize=(12, 8))
text = ' '.join(df_sample['processed_text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud.png')
print("Saved word cloud to wordcloud.png")

# 3. Sentiment Over Word Count
plt.figure(figsize=(10, 6))
sns.scatterplot(x='word_count', y='sentiment_score', hue='sentiment_simple', data=df_sample)
plt.title('Sentiment Score vs. Word Count')
plt.xlabel('Word Count')
plt.ylabel('Sentiment Score')
plt.savefig('sentiment_vs_wordcount.png')
print("Saved sentiment vs. word count plot to sentiment_vs_wordcount.png")

# 4. Entity Type Distribution
if all_entities:
    entity_types = [entity_type for _, entity_type in all_entities]
    entity_type_counts = Counter(entity_types)

    plt.figure(figsize=(12, 6))
    entity_df = pd.DataFrame(list(entity_type_counts.items()), columns=['Entity Type', 'Count'])
    entity_df = entity_df.sort_values('Count', ascending=False)
    sns.barplot(x='Entity Type', y='Count', data=entity_df)
    plt.title('Distribution of Named Entity Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('entity_types.png')
    print("Saved entity type distribution to entity_types.png")
else:
    print("No entities found, skipping entity type distribution visualization")

# Generate an HTML report with all findings
print("\n--- Generating HTML Report ---")
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Gaza-Related Reddit Posts Sentiment Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .stats {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .visualization {{ margin: 30px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis of Gaza-Related Reddit Posts</h1>
        
        <div class="stats">
            <h2>Dataset Statistics</h2>
            <p>Total posts analyzed: {len(df)}</p>
            <p>Average text length: {df['text_length'].mean():.2f} characters</p>
            <p>Average word count: {df['word_count'].mean():.2f} words</p>
            <p>Computing device: {device.type.upper()}</p>
        </div>
        
        <div class="visualization">
            <h2>Sentiment Distribution</h2>
            <p>The sentiment distribution among the analyzed posts:</p>
            <ul>
"""

for sentiment, count in sentiment_counts.items():
    percentage = count/len(df_sample)*100
    html_report += f"                <li>{sentiment}: {count} posts ({percentage:.2f}%)</li>\n"

html_report += """
            </ul>
            <img src="sentiment_distribution.png" alt="Sentiment Distribution" style="max-width: 100%;">
        </div>
        
        <div class="visualization">
            <h2>Word Cloud</h2>
            <p>Most frequent words in the posts:</p>
            <img src="wordcloud.png" alt="Word Cloud" style="max-width: 100%;">
        </div>
        
        <div class="visualization">
            <h2>Topic Analysis</h2>
            <p>Top topics identified in the posts:</p>
            <table>
                <tr>
                    <th>Topic</th>
                    <th>Top Keywords</th>
                </tr>
"""

for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    html_report += f"""
                <tr>
                    <td>Topic {topic_idx+1}</td>
                    <td>{', '.join(top_words)}</td>
                </tr>
"""

html_report += """
            </table>
        </div>
        
        <div class="visualization">
            <h2>Named Entity Recognition</h2>
            <p>Most common named entities mentioned in the posts:</p>
            <table>
                <tr>
                    <th>Entity</th>
                    <th>Type</th>
                    <th>Count</th>
                </tr>
"""

for (entity_text, entity_type), count in most_common_entities[:15]:
    html_report += f"""
                <tr>
                    <td>{entity_text}</td>
                    <td>{entity_type}</td>
                    <td>{count}</td>
                </tr>
"""

html_report += """
            </table>
"""
if all_entities:
    html_report += """
            <img src="entity_types.png" alt="Entity Type Distribution" style="max-width: 100%;">
"""

html_report += """
        </div>
        
        <div class="visualization">
            <h2>Sentiment vs. Word Count</h2>
            <p>Relationship between post length and sentiment:</p>
            <img src="sentiment_vs_wordcount.png" alt="Sentiment vs. Word Count" style="max-width: 100%;">
        </div>
        
        <div class="visualization">
            <h2>Explainable AI Results</h2>
            <p>Understanding what words and phrases are influencing the sentiment classification:</p>
"""

for i in range(min(3, explanation_samples)):
    if i < len(X_text):
        text = X_text.iloc[i]
        if isinstance(text, str) and text.strip():
            exp = explainer.explain_instance(text, lime_predict_proba, num_features=5)
            original_text = df_sample['combined_text'].iloc[i]
            if len(original_text) > 100:
                original_text = original_text[:100] + "..."
            prediction = class_names[clf.predict(tfidf.transform([text]))[0]]
            
            html_report += f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
                <h3>Example Post {i+1}</h3>
                <p><strong>Text:</strong> {original_text}</p>
                <p><strong>Prediction:</strong> {prediction}</p>
                <p><strong>Key factors influencing this prediction:</strong></p>
                <ul>
"""
            
            for feature, weight in exp.as_list()[:5]:
                color = "green" if weight > 0 else "red"
                html_report += f"""
                    <li style="color: {color};">{feature}: {weight:.4f}</li>
"""
            
            html_report += """
                </ul>
            </div>
"""

html_report += """
        </div>
        
        <div class="visualization">
            <h2>Conclusion</h2>
            <p>This analysis provides insights into the sentiment patterns, key topics, and important entities mentioned in Reddit posts related to the Gaza issue. The visualizations and explainable AI components help understand the factors driving sentiment in these discussions.</p>
        </div>
    </div>
</body>
</html>
"""

# Save the HTML report
with open("gaza_reddit_sentiment_analysis_report.html", "w", encoding="utf-8") as f:
    f.write(html_report)

print("\nAnalysis complete!")
print("HTML report generated: gaza_reddit_sentiment_analysis_report.html")
print("Check the report for detailed findings and visualizations.") 