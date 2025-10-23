"""
Streamlit frontend demo for CBM NLP API service.
"""

import streamlit as st
import requests
import pandas as pd
import json
from typing import Dict, Any, Optional


# Page configuration
st.set_page_config(
    page_title="CBM NLP Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available models and modes
AVAILABLE_MODELS = ["bert-base-uncased", "gpt2", "roberta-base", "lstm"]
AVAILABLE_MODES = ["standard", "joint"]

# Concept name mapping
CONCEPT_FULL_NAMES = {
    # QA Dataset concepts
    "FC": "Focus/Clarity",
    "CC": "Coherence/Cohesion", 
    "TU": "Task Understanding",
    "CP": "Critical Thinking",
    "R": "Relevance",
    "DU": "Depth/Understanding",
    "EE": "Evidence/Examples",
    "FR": "Flow/Readability",
    
    # Essay Dataset concepts
    "TC": "Task Completion",
    "UE": "Understanding/Explanation",
    "OC": "Organization/Clarity",
    "GM": "Grammar/Mechanics",
    "VA": "Vocabulary/Accuracy",
    "SV": "Support/Validation",
    "CTD": "Critical Thinking/Depth",
    "FR": "Flow/Readability",
    
    # CEBaB Dataset concepts
    "Food": "Food Quality",
    "Ambiance": "Ambiance/Atmosphere",
    "Service": "Service Quality",
    "Noise": "Noise Level",
    "cleanliness": "Cleanliness",
    "price": "Price/Value",
    "location": "Location",
    "menu_variety": "Menu Variety",
    "waiting_time": "Waiting Time",
    "waiting_area": "Waiting Area",
    
    # IMDB Dataset concepts
    "acting": "Acting Performance",
    "storyline": "Storyline/Plot",
    "emotional": "Emotional Impact",
    "cinematography": "Cinematography",
    "soundtrack": "Soundtrack/Music",
    "directing": "Directing",
    "background": "Background Setting",
    "editing": "Editing"
}


def check_backend_connection(base_url: str) -> Dict[str, Any]:
    """Check if backend is accessible and get status."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "connected", "data": response.json()}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": str(e)}


def get_available_models(base_url: str) -> Optional[Dict[str, Any]]:
    """Get available models from backend."""
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def predict_single_text(base_url: str, text: str, model_name: str, mode: str) -> Optional[Dict[str, Any]]:
    """Send prediction request to backend."""
    try:
        payload = {
            "text": text,
            "model_name": model_name,
            "mode": mode
        }
        response = requests.post(f"{base_url}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None




def format_prediction_icon(prediction: int, num_classes: int) -> str:
    """Format prediction icon based on number of classes."""
    if num_classes == 2:
        return "✅" if prediction == 1 else "❌"
    elif num_classes == 6:
        return "📝" + "⭐" * (prediction + 1)  # Essay scoring with stars
    else:
        return "⭐" * (prediction + 1)


def display_rating_highlight(rating: int, num_classes: int, confidence: float):
    """Display rating as large prominent text with confidence."""
    max_rating = num_classes
    confidence_pct = confidence * 100
    
    # Create prominent display
    st.markdown("### 🎯 Prediction Result")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"## Rating: **{rating}/{max_rating}**")
        st.markdown(f"**{format_prediction_icon(rating-1, num_classes)}**")
    
    with col2:
        st.metric("Confidence", f"{confidence_pct:.1f}%")
    
    with col3:
        st.metric("Max Score", max_rating)


def display_probability_chart(probabilities: list, rating: int, num_classes: int, confidence: float):
    """Create 2:8 column layout with rating info on left and horizontal bar chart on right."""
    st.markdown("### 📊 Probability Distribution")
    
    # Create 2:8 column layout
    col1, col2 = st.columns([2, 8])
    
    with col1:
        st.markdown("**Rating Summary**")
        st.metric("Predicted Rating", f"{rating}/{num_classes}")
        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.metric("Max Score", num_classes)
        
        # Show highest probability info
        max_prob_idx = probabilities.index(max(probabilities))
        st.markdown(f"**Highest:** Score {max_prob_idx + 1}")
        st.markdown(f"**Probability:** {max(probabilities)*100:.1f}%")
    
    with col2:
        # Create horizontal bar chart
        data = {
            'Score': [f"{i+1}" for i in range(len(probabilities))],
            'Probability': probabilities
        }
        
        df = pd.DataFrame(data)
        df = df.set_index('Score')
        
        # Display horizontal bar chart
        st.bar_chart(df, height=300)


def display_concept_cards(concept_predictions: list):
    """Display 8 concept cards in 4 rows x 2 columns with prominent borders and color coding."""
    if not concept_predictions:
        return
        
    st.markdown("### 🎯 Concept Analysis")
    
    # Color mapping for icons - handle different sentiment labels dynamically
    def get_icon_for_sentiment(sentiment: str) -> str:
        """Get appropriate icon based on sentiment label."""
        sentiment_lower = sentiment.lower()
        
        # Handle numeric labels (1-5 scale)
        if sentiment.isdigit():
            score = int(sentiment)
            if score <= 2:
                return "🔴"  # Low scores (1-2)
            elif score == 3:
                return "🟡"  # Medium score (3)
            else:
                return "🟢"  # High scores (4-5)
        
        # Handle text labels
        if any(word in sentiment_lower for word in ['negative', 'low', 'very low']):
            return "🔴"
        elif any(word in sentiment_lower for word in ['neutral', 'medium']):
            return "🟡"
        elif any(word in sentiment_lower for word in ['positive', 'high', 'very high']):
            return "🟢"
        else:
            return "⚪"  # Default for unknown labels
    
    # Display cards in 2 rows, 4 cards per row
    for row in range(2):
        cols = st.columns(4)
        
        for col_idx in range(4):
            concept_idx = row * 4 + col_idx
            if concept_idx < len(concept_predictions):
                concept = concept_predictions[concept_idx]
                
                with cols[col_idx]:
                    # Get concept info
                    concept_name = concept['concept_name']
                    full_name = CONCEPT_FULL_NAMES.get(concept_name, concept_name)
                    prediction = concept['prediction']
                    probs = concept['probabilities']
                    
                    # Get styling
                    icon = get_icon_for_sentiment(prediction)
                    
                    # Get top probability info
                    top_prob = max(probs.values())
                    top_sentiment = max(probs, key=probs.get)
                    
                    # Create simple card without borders
                    card_html = f"""
                    <div style="
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                    ">
                        <h4 style="margin: 0 0 5px 0; color: #333;">
                            {icon} {concept_name}
                        </h4>
                        <p style="margin: 0 0 10px 0; color: #666; font-size: 12px;">
                            {full_name}
                        </p>
                        <p style="margin: 5px 0; color: #333;">
                            <strong>Prediction:</strong> {prediction}
                        </p>
                        <p style="margin: 5px 0; color: #666; font-size: 12px;">
                            <strong>Top:</strong> {top_sentiment} ({top_prob*100:.1f}%)
                        </p>
                    </div>
                    """
                    
                    st.markdown(card_html, unsafe_allow_html=True)
                    
                    # Add bar chart inside the card
                    concept_data = {
                        'Sentiment': list(probs.keys()),
                        'Probability': list(probs.values())
                    }
                    
                    concept_df = pd.DataFrame(concept_data)
                    concept_df = concept_df.set_index('Sentiment')
                    
                    # Display horizontal bar chart with smaller height
                    st.bar_chart(concept_df, height=150)
                    
                    # Add probability details below chart
                    prob_text = " | ".join([f"{label}: {probs[label]*100:.1f}%" for label in probs.keys()])
                    st.markdown(f"""
                    <div style="margin-top: 5px;">
                        <small style="color: #888;">
                            {prob_text}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)


def main():
    # Title and description
    st.title("🤖 CBM NLP Demo")
    st.markdown("**Concept Bottleneck Model for Natural Language Processing**")
    st.info("📝 This demo uses the Essay dataset for programming answer quality assessment")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Backend URL input
    backend_url = st.sidebar.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="URL of the FastAPI backend service"
    )
    
    # Connection status
    st.sidebar.markdown("### 🔗 Connection Status")
    connection_status = check_backend_connection(backend_url)
    
    if connection_status["status"] == "connected":
        st.sidebar.success("✅ Connected")
        health_data = connection_status["data"]
        st.sidebar.json(health_data)
    else:
        st.sidebar.error(f"❌ Connection Failed")
        st.sidebar.error(connection_status["message"])
        st.sidebar.warning("Make sure the backend service is running!")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📈 Backend Status"])
    
    with tab1:
        st.header("Single Text Prediction")
        st.markdown("Predict sentiment/rating for a single text input.")
        
        if connection_status["status"] != "connected":
            st.warning("⚠️ Please ensure the backend service is running and accessible.")
            return
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.selectbox(
                    "Model",
                    AVAILABLE_MODELS,
                    help="Select the model to use for prediction"
                )
            
            with col2:
                mode = st.radio(
                    "Mode",
                    AVAILABLE_MODES,
                    help="Standard: basic sentiment analysis, Joint: with concept analysis"
                )
            
            text_input = st.text_area(
                "Text to analyze",
                value="Q: What is a pointer in C++?\nA: A pointer is a variable that stores the memory address of another variable.",
                height=100,
                help="Enter programming Q&A text to analyze"
            )
            
            predict_button = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        # Process prediction
        if predict_button and text_input.strip():
            with st.spinner("Analyzing text..."):
                result = predict_single_text(backend_url, text_input, model_name, mode)
            
            if result:
                st.success("✅ Prediction completed!")
                
                # Get number of classes and confidence
                num_classes = len(result['probabilities'])
                max_probability = max(result['probabilities'])
                
                # 1. Prediction Summary (prominent)
                display_rating_highlight(result['rating'], num_classes, max_probability)
                
                # 2. Probability Distribution
                display_probability_chart(result['probabilities'], result['rating'], num_classes, max_probability)
                
                # 3. Concept Analysis (only for joint mode)
                if result.get("concept_predictions"):
                    display_concept_cards(result["concept_predictions"])
                
                # 4. Raw Model Results (collapsible at bottom)
                with st.expander("📋 Raw Model Results", expanded=False):
                    st.json(result)
    
    with tab2:
        st.header("Backend Status")
        st.markdown("Monitor backend service status and available models.")
        
        # Refresh button
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Health status
        st.markdown("### 🏥 Health Status")
        if connection_status["status"] == "connected":
            st.success("✅ Backend service is healthy")
            health_data = connection_status["data"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(health_data)
        else:
            st.error("❌ Backend service is not accessible")
            st.error(connection_status["message"])
        
        # Available models
        st.markdown("### 🤖 Available Models")
        models_data = get_available_models(backend_url)
        
        if models_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Available Models:**")
                for model in models_data["available_models"]:
                    st.write(f"• {model}")
                
                st.markdown("**Available Modes:**")
                for mode in models_data["available_modes"]:
                    st.write(f"• {mode}")
            
            with col2:
                st.markdown("**Currently Loaded:**")
                if models_data["loaded_models"]:
                    for model, modes in models_data["loaded_models"].items():
                        st.write(f"• **{model}**: {', '.join(modes)}")
                else:
                    st.write("No models currently loaded")
        else:
            st.warning("⚠️ Could not retrieve model information")


if __name__ == "__main__":
    main()
