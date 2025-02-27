import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import networkx as nx
from collections import defaultdict, Counter

# Set page config
st.set_page_config(
    page_title="Financial Statements Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_nlp_resources():
    # Load RoBERTa model for NER
    model_name = "Jean-Baptiste/roberta-large-ner-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    
    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Initialize VADER sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    
    return ner_pipeline, nlp, vader_analyzer

# Regex patterns for financial metrics
currency_pattern = re.compile(r'(EUR|\$)[\s]?[0-9,.]+[\s]?(mn|m|million|billion)?')
percentage_pattern = re.compile(r'[0-9,.]+[\s]?%')
ticker_pattern = re.compile(r'\b[A-Z]{2,4}\b(?![a-z])')

# Financial extraction functions
def extract_financial_metrics(text):
    """Extract financial metrics using regex patterns"""
    currencies = [''.join(match).strip() for match in currency_pattern.findall(text)]
    percentages = percentage_pattern.findall(text)
    tickers = ticker_pattern.findall(text)
    return {"currencies": currencies, "percentages": percentages, "tickers": tickers}

def extract_financial_keywords(text):
    """Extract positive and negative financial keywords"""
    positive_terms = ["growth", "increase", "rising", "profit", "gain", "surge", "up"]
    negative_terms = ["decline", "drop", "decrease", "loss", "down", "fall", "negative"]
    positive_matches = [word for word in positive_terms if word in text.lower()]
    negative_matches = [word for word in negative_terms if word in text.lower()]
    return {"positive": positive_matches, "negative": negative_matches}

def check_negation_and_modifiers(doc, entity_text):
    """Check if an entity is connected to a negation or modifier"""
    for token in doc:
        # Check for negations (e.g., "no", "not")
        if token.dep_ == "neg" and token.head.text in entity_text:
            return "negative"

        # Check for positive/negative modifiers (e.g., "growth", "decline")
        if token.text.lower() in ["growth", "increase", "rising"] and token.head.text in entity_text:
            return "positive"
        if token.text.lower() in ["decline", "drop", "decrease", "no", "not"] and token.head.text in entity_text:
            return "negative"

    return "neutral"

def extract_dependency_relations(text, nlp):
    """Extract dependency relations from text"""
    doc = nlp(text)
    relations = []

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr"):  # Extract subject-object relations
            relations.append((token.head.text, token.dep_, token.text))

    return relations

def map_financial_performance(sentiment, entities, metrics, text):
    """Determine financial performance category based on analysis"""
    financial_keywords = extract_financial_keywords(text)
    
    if sentiment == 1 and (financial_keywords["positive"] or "growth" in entities):
        return "High Growth Potential"
    elif sentiment == 0 and (financial_keywords["negative"] or "decline" in entities):
        return "High Risk"
    else:
        return "Stable"

# IMPROVED: Semantic Graph functions - updated to match your code
def extract_entities(text, nlp):
    """Extract named entities using spaCy."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "MONEY", "GPE", "PRODUCT", "EVENT"}]
    return entities

def get_top_entities(df, nlp, num_top=15, num_rows=100):
    """Find top occurring financial entities in dataset."""
    entity_counter = Counter()
    for text in df["Sentence"][:num_rows]:
        entities = extract_entities(text, nlp)
        entity_counter.update(entities)
    top_entities = {entity for entity, _ in entity_counter.most_common(num_top)}
    return top_entities

def extract_entity_pairs(text, nlp, top_entities):
    """Extract entity pairs using dependency parsing."""
    doc = nlp(text)
    pairs = []
    for token in doc:
        if token.text in top_entities and token.head.text in top_entities:
            relation = token.dep_
            pairs.append((token.text, token.head.text, relation))
    return pairs

def build_semantic_graph(df, nlp, num_rows=100):
    """Builds a directed graph using named entities and sentiment-based edges."""
    G = nx.DiGraph()
    edge_weights = defaultdict(int)
    
    # Identify top financial entities
    top_entities = get_top_entities(df, nlp, num_rows=num_rows)
    
    df_sample = df.head(num_rows)
    
    for _, row in df_sample.iterrows():
        text = row["Sentence"]
        sentiment = row.get("Sentiment", 0)  # Default sentiment to 0 if missing
        
        entity_pairs = extract_entity_pairs(text, nlp, top_entities)
        
        for entity1, entity2, relation in entity_pairs:
            if entity1 not in G:
                G.add_node(entity1, sentiment=0)
            if entity2 not in G:
                G.add_node(entity2, sentiment=0)
            
            # Assign sentiment weight
            sentiment_weight = 1 if sentiment == 1 else -1
            edge_weights[(entity1, entity2)] += sentiment_weight
            G.add_edge(entity1, entity2, relation=relation, weight=edge_weights[(entity1, entity2)])
            
            # Update node sentiment
            G.nodes[entity1]["sentiment"] += sentiment_weight
            G.nodes[entity2]["sentiment"] += sentiment_weight
    
    return G

# Function to analyze a manually entered tweet
def analyze_tweet(text, ner_pipeline, nlp, vader_analyzer):
    """Analyze a manually entered tweet"""
    results = {}
    
    # VADER sentiment analysis
    vader_scores = vader_analyzer.polarity_scores(text)
    results["vader_sentiment"] = vader_scores
    results["sentiment_label"] = "Positive" if vader_scores["compound"] > 0 else "Negative"
    
    # Extract entities
    ner_results = ner_pipeline(text)
    results["entities"] = [
        {"Entity": entity['word'], 
         "Type": entity['entity_group'], 
         "Confidence": f"{entity['score']:.4f}"} 
        for entity in ner_results if entity['score'] > 0.7
    ]
    
    # Extract financial keywords
    financial_keywords = extract_financial_keywords(text)
    results["financial_keywords"] = financial_keywords
    
    # Extract financial metrics
    financial_metrics = extract_financial_metrics(text)
    results["financial_metrics"] = financial_metrics
    
    # Map financial performance
    entities_list = [entity['word'].lower() for entity in ner_results if entity['score'] > 0.8]
    sentiment_val = 1 if vader_scores["compound"] > 0 else 0
    performance = map_financial_performance(sentiment_val, entities_list, financial_metrics, text)
    results["performance_category"] = performance
    
    return results

# Main application
def main():
    st.title("Financial Statements Analysis Dashboard")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Load data
    try:
        df = pd.read_csv("data.csv")
        # Limit to 200 rows as requested
        df = df.head(200)
    except FileNotFoundError:
        st.sidebar.error("data.csv not found. Using example data instead.")
        # Use example data if file is not found
        example_data = {
            "Sentiment": [0, 1, 1, 1, 1, 1, 1, 1, 1],
            "Sentence": [
                "According to Gran, the company has no plans to...",
                "For the last quarter of 2010, Componenta's net profit...",
                "In the third quarter of 2010, net sales increased...",
                "Operating profit rose to EUR 13.1 mn from EUR 8.7...",
                "Operating profit totalled EUR 21.1 mn, up from...",
                "Finnish Talentum reports its operating profit in...",
                "Clothing retail chain Sepp+fl+ñ's sales increased...",
                "Consolidated net sales increased 16 % to reach...",
                "Foundries division reports its sales increased by..."
            ]
        }
        df = pd.DataFrame(example_data)
    
    # Make sure the dataframe has the required columns
    if "Sentiment" not in df.columns or "Sentence" not in df.columns:
        st.error("The dataset must contain 'Sentiment' and 'Sentence' columns.")
        return
    
    # Clean the data
    df = df.dropna(subset=['Sentence'])
    
    # Load NLP resources
    with st.spinner("Loading NLP models..."):
        ner_pipeline, nlp, vader_analyzer = load_nlp_resources()
    
    # Navigation tabs
    tabs = st.tabs(["Overview", "Manual Tweet Analysis", "Entity Analysis", "Sentiment Analysis", "Dependency Graph"])
    
    with tabs[0]:  # Overview tab
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Statements", len(df))
            st.metric("Positive Statements", df[df["Sentiment"] == 1].shape[0])
            st.metric("Negative Statements", df[df["Sentiment"] == 0].shape[0])
        
        with col2:
            # Sentiment distribution pie chart
            fig, ax = plt.subplots(figsize=(6, 6))
            sentiment_counts = df["Sentiment"].value_counts()
            ax.pie(
                sentiment_counts, 
                labels=["Positive" if idx == 1 else "Negative" for idx in sentiment_counts.index],
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336']
            )
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)
        
        st.subheader("Sample Statements")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tabs[1]:  # Manual Tweet Analysis tab
        st.header("Manual Tweet Analysis")
        
        # Text input for manual tweet
        tweet_text = st.text_area("Enter a financial statement or tweet to analyze:", 
                                 height=150,
                                 placeholder="Example: The company reported a 15% increase in quarterly profits, exceeding market expectations.")
        
        if st.button("Analyze Tweet"):
            if tweet_text:
                with st.spinner("Analyzing tweet..."):
                    analysis_results = analyze_tweet(tweet_text, ner_pipeline, nlp, vader_analyzer)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Sentiment Analysis")
                        compound_score = analysis_results["vader_sentiment"]["compound"]
                        sentiment_color = "green" if compound_score > 0 else "red"
                        
                        st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color}'>{analysis_results['sentiment_label']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Compound Score:** {compound_score:.3f}")
                        st.markdown(f"**Positive Score:** {analysis_results['vader_sentiment']['pos']:.3f}")
                        st.markdown(f"**Negative Score:** {analysis_results['vader_sentiment']['neg']:.3f}")
                        st.markdown(f"**Neutral Score:** {analysis_results['vader_sentiment']['neu']:.3f}")
                        
                        # Financial keywords
                        st.subheader("Financial Keywords")
                        positive_terms = analysis_results["financial_keywords"]["positive"]
                        negative_terms = analysis_results["financial_keywords"]["negative"]
                        
                        if positive_terms:
                            st.markdown("**Positive Terms:** " + ", ".join(positive_terms))
                        else:
                            st.markdown("**Positive Terms:** None detected")
                            
                        if negative_terms:
                            st.markdown("**Negative Terms:** " + ", ".join(negative_terms))
                        else:
                            st.markdown("**Negative Terms:** None detected")
                        
                        # Performance category
                        st.subheader("Financial Performance Category")
                        perf_category = analysis_results["performance_category"]
                        perf_color = "green" if perf_category == "High Growth Potential" else "red" if perf_category == "High Risk" else "blue"
                        st.markdown(f"**Category:** <span style='color:{perf_color}'>{perf_category}</span>", unsafe_allow_html=True)
                    
                    with col2:
                        # Named entities
                        st.subheader("Named Entities")
                        if analysis_results["entities"]:
                            entities_df = pd.DataFrame(analysis_results["entities"])
                            st.dataframe(entities_df, use_container_width=True)
                        else:
                            st.info("No entities detected.")
                        
                        # Financial metrics
                        st.subheader("Financial Metrics")
                        financial_metrics = analysis_results["financial_metrics"]
                        metrics_data = []
                        
                        if financial_metrics["currencies"]:
                            metrics_data.append({"Type": "Currency", "Values": ", ".join(financial_metrics["currencies"])})
                        if financial_metrics["percentages"]:
                            metrics_data.append({"Type": "Percentage", "Values": ", ".join(financial_metrics["percentages"])})
                        if financial_metrics["tickers"]:
                            metrics_data.append({"Type": "Ticker", "Values": ", ".join(financial_metrics["tickers"])})
                        
                        if metrics_data:
                            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                        else:
                            st.info("No financial metrics detected.")
            else:
                st.warning("Please enter a statement to analyze.")
    
    with tabs[2]:  # Entity Analysis tab
        st.header("Named Entity Recognition")
        
        # Sample selection
        sample_size = st.slider("Number of statements to analyze", min_value=1, max_value=min(200, len(df)), value=5)
        sample_df = df.head(sample_size)
        
        for i, (text, sentiment) in enumerate(zip(sample_df["Sentence"], sample_df["Sentiment"])):
            with st.expander(f"Statement {i+1} - {'Positive' if sentiment == 1 else 'Negative'}"):
                st.write(text)
                
                # Extract named entities
                ner_results = ner_pipeline(text)
                
                if ner_results:
                    entities_df = pd.DataFrame([
                        {"Entity": entity['word'], 
                         "Type": entity['entity_group'], 
                         "Confidence": f"{entity['score']:.4f}"} 
                        for entity in ner_results if entity['score'] > 0.7
                    ])
                    
                    if not entities_df.empty:
                        st.dataframe(entities_df, use_container_width=True)
                    else:
                        st.info("No high-confidence entities found.")
                else:
                    st.info("No entities detected.")
                
                # Extract financial metrics
                financial_metrics = extract_financial_metrics(text)
                metrics_data = []
                
                if financial_metrics["currencies"]:
                    metrics_data.append({"Type": "Currency", "Values": ", ".join(financial_metrics["currencies"])})
                if financial_metrics["percentages"]:
                    metrics_data.append({"Type": "Percentage", "Values": ", ".join(financial_metrics["percentages"])})
                if financial_metrics["tickers"]:
                    metrics_data.append({"Type": "Ticker", "Values": ", ".join(financial_metrics["tickers"])})
                
                if metrics_data:
                    st.subheader("Financial Metrics")
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        
        # Generate word cloud of entities
        if st.button("Generate Entity Word Cloud"):
            all_entities = []
            for text in df["Sentence"].head(50):  # Use first 50 statements
                try:
                    ner_results = ner_pipeline(text)
                    all_entities.extend([entity['word'] for entity in ner_results if entity['score'] > 0.7])
                except Exception as e:
                    st.warning(f"Error processing text: {e}")
            
            if all_entities:
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues')
                wordcloud.generate(" ".join(all_entities))
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("Not enough entities found for word cloud.")
    
    with tabs[3]:  # Sentiment Analysis tab
        st.header("Sentiment Analysis")
        
        # Financial performance classification
        st.subheader("Financial Performance Classification")
        
        # Process sample statements for detailed analysis
        sample_size = st.slider("Statements to analyze", min_value=1, max_value=min(200, len(df)), value=5, key="sentiment_slider")
        sample_df = df.head(sample_size)
        
        performance_results = []
        
        for i, (text, sentiment) in enumerate(zip(sample_df["Sentence"], sample_df["Sentiment"])):
            # Extract entities
            ner_results = ner_pipeline(text)
            entities = [entity['word'].lower() for entity in ner_results if entity['score'] > 0.8]
            
            # Extract financial metrics
            financial_metrics = extract_financial_metrics(text)
            
            # Map financial performance
            performance = map_financial_performance(sentiment, entities, financial_metrics, text)
            
            # VADER sentiment
            vader_score = vader_analyzer.polarity_scores(text)["compound"]
            
            performance_results.append({
                "Statement": text,
                "Original Sentiment": "Positive" if sentiment == 1 else "Negative",
                "VADER Score": vader_score,
                "Performance Category": performance
            })
        
        # Convert to DataFrame and display
        performance_df = pd.DataFrame(performance_results)
        st.dataframe(performance_df, use_container_width=True)
        
        # Summary chart
        performance_counts = Counter(performance_df["Performance Category"])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(performance_counts.keys(), performance_counts.values())
        ax.set_ylabel("Count")
        ax.set_title("Financial Performance Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Sentiment keywords analysis
        st.subheader("Financial Sentiment Keywords")
        
        positive_keywords = []
        negative_keywords = []
        
        analysis_limit = min(200, len(df))  # Process 200 rows as requested
        
        for text in df["Sentence"].head(analysis_limit):
            keywords = extract_financial_keywords(text)
            positive_keywords.extend(keywords["positive"])
            negative_keywords.extend(keywords["negative"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Positive Financial Terms")
            if positive_keywords:
                pos_counts = Counter(positive_keywords)
                pos_df = pd.DataFrame({
                    "Term": list(pos_counts.keys()),
                    "Count": list(pos_counts.values())
                }).sort_values("Count", ascending=False)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(pos_df["Term"], pos_df["Count"], color="green")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No positive financial terms found.")
        
        with col2:
            st.write("Negative Financial Terms")
            if negative_keywords:
                neg_counts = Counter(negative_keywords)
                neg_df = pd.DataFrame({
                    "Term": list(neg_counts.keys()),
                    "Count": list(neg_counts.values())
                }).sort_values("Count", ascending=False)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(neg_df["Term"], neg_df["Count"], color="red")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("No negative financial terms found.")
    
    with tabs[4]:  # Dependency Graph tab
        st.header("Improved Semantic Relationship Graph")
        
        # UPDATED: Graph visualization controls
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.slider("Number of statements to analyze", min_value=10, max_value=min(200, len(df)), value=50)
        
        with col2:
            num_entities = st.slider("Number of top entities to include", min_value=5, max_value=30, value=15)
        
        if st.button("Generate Improved Semantic Graph"):
            with st.spinner("Building semantic graph... This may take a moment."):
                # UPDATED: Use the improved graph building function
                G = build_semantic_graph(df, nlp, num_rows=num_rows)
                
                if G.number_of_nodes() == 0:
                    st.warning("Not enough entities found to create a graph. Try using more data.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Node sizes based on sentiment influence - increased scaling for visibility
                    node_sizes = [abs(G.nodes[node]["sentiment"]) * 500 + 500 for node in G.nodes]
                    
                    # Node colors (green = positive, red = negative, gray = neutral)
                    node_colors = [
                        "green" if G.nodes[node]["sentiment"] > 0 else 
                        "red" if G.nodes[node]["sentiment"] < 0 else "gray"
                        for node in G.nodes
                    ]
                    
                    # Edge widths based on sentiment strength
                    edge_widths = [abs(G.edges[u, v]["weight"]) for u, v in G.edges]
                    
                    # Edge labels (relationship types)
                    edge_labels = {(u, v): G.edges[u, v]["relation"] for u, v in G.edges}
                    
                    # Layout adjustment for better visibility - increased k parameter
                    pos = nx.spring_layout(G, seed=42, k=0.6)
                    
                    # Draw the graph
                    nx.draw(
                        G, pos, with_labels=True, 
                        node_color=node_colors, 
                        edge_color="gray",
                        node_size=node_sizes, 
                        font_size=10,
                        width=edge_widths,
                        alpha=0.7
                    )
                    
                    # Draw edge labels for relationship types
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="blue")
                    
                    plt.title("Improved Semantic Graph of Financial Entities")
                    st.pyplot(fig)
                    
                    # Entity table
                    st.subheader("Entity Sentiment Summary")
                    entity_data = [
                        {"Entity": node, 
                         "Sentiment Score": G.nodes[node]["sentiment"],
                         "Sentiment": "Positive" if G.nodes[node]["sentiment"] > 0 else 
                                     "Negative" if G.nodes[node]["sentiment"] < 0 else "Neutral",
                         "Connections": G.degree(node)}
                        for node in G.nodes
                    ]
                    entity_df = pd.DataFrame(entity_data).sort_values("Sentiment Score", ascending=False)
                    st.dataframe(entity_df)
                    
                    # Edge relationship summary
                    st.subheader("Entity Relationship Summary")
                    edge_data = [
                        {"Source": u, 
                         "Target": v, 
                         "Relationship": G.edges[u, v]["relation"],
                         "Sentiment Weight": G.edges[u, v]["weight"]}
                        for u, v in G.edges
                    ]
                    edge_df = pd.DataFrame(edge_data).sort_values("Sentiment Weight", ascending=False)
                    st.dataframe(edge_df)

if __name__ == "__main__":
    main()