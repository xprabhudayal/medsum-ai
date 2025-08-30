# MedSumAI Pro - Complete Implementation
# Install required packages first: pip install transformers torch streamlit pandas accelerate datasets

import json
import pandas as pd
import streamlit as st
import torch
from transformers import pipeline
import logging
import warnings
import os
import re
from typing import Dict, List, Any

# --- Configuration ---
FINETUNED_MODEL_PATH = "./models/medsum-bart-finetuned"
BASE_MODEL = "facebook/bart-large-cnn"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

class LocalSummarizer:
    def __init__(self, model_path: str):
        """Initialize local summarization model."""
        self.model_path = model_path
        self.summarizer = None
        self.model_name = os.path.basename(model_path) if os.path.exists(model_path) and os.listdir(model_path) else BASE_MODEL

    def load_model(self):
        """Load the summarization model with memory optimization."""
        model_to_load = self.model_path
        
        try:
            # Prioritize loading the fine-tuned model if it exists
            if os.path.exists(self.model_path) and os.listdir(self.model_path):
                logger.info(f"Found fine-tuned model. Loading from: {self.model_path}")
                # Use a session state to show toast only once
                if 'toast_shown' not in st.session_state:
                    st.toast(f"🚀 Using fine-tuned model: {self.model_name}")
                    st.session_state.toast_shown = True
            else:
                logger.warning(f"Fine-tuned model not found at '{self.model_path}'. Falling back to base model.")
                if 'warning_shown' not in st.session_state:
                    st.warning(f"**Fine-tuned model not found.** Falling back to the pre-trained `{BASE_MODEL}` model. Run `train.py` to create a fine-tuned version.")
                    st.session_state.warning_shown = True
                model_to_load = BASE_MODEL
                self.model_name = BASE_MODEL

            self.summarizer = pipeline(
                "summarization",
                model=model_to_load,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            logger.info(f"✅ Successfully loaded model: {self.model_name}")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            st.error(f"Fatal Error: Could not load any model. Please check logs. Error: {e}")
            self.summarizer = None

    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary for given text."""
        if not self.summarizer:
            return "Model not loaded. Please check the logs for errors."
        
        try:
            text = self.clean_text(text)
            if len(text.split()) < min_length:
                return "Input text is too short for a meaningful summary."

            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Summarization failed: {str(e)}"
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?\-]', '', text)
        return text.strip()

class PerspectiveProcessor:
    def __init__(self):
        """Initialize perspective detection and processing."""
        self.perspective_patterns = {
            'INFORMATION': [r'\b(definition|meaning|what is|explained|describes|causes?|symptoms?|condition|disease)\b'],
            'SUGGESTION': [r'\b(should|recommend|suggest|try|consider|treatment|therapy|medication)\b'],
            'EXPERIENCE': [r'\b(I have|I feel|my experience|happened to me|I noticed|I tried)\b'],
            'CAUSE': [r'\b(because|due to|caused by|results from|leads to|triggers)\b']
        }
    
    def extract_perspectives(self, entry: Dict) -> Dict[str, List[str]]:
        """Extract different perspective categories from medical Q&A entry."""
        perspectives = {k: [] for k in self.perspective_patterns}
        
        all_text = f"{entry.get('question', '')} {entry.get('context', '')} {" ".join(entry.get('answers', []))}"
        sentences = re.split(r'[.!?]+', all_text.lower())
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                for perspective, patterns in self.perspective_patterns.items():
                    if any(re.search(p, sentence, re.IGNORECASE) for p in patterns):
                        perspectives[perspective].append(sentence.strip())
                        break
        return perspectives

class DualAudienceGenerator:
    def __init__(self, summarizer: LocalSummarizer, perspective_processor: PerspectiveProcessor):
        """Initialize dual-audience summary generator."""
        self.summarizer = summarizer
        self.perspective_processor = perspective_processor
        self.patient_templates = {
            'INFORMATION': "Here's what you should know: {}",
            'SUGGESTION': "Your healthcare team suggests: {}",
            'EXPERIENCE': "Other patients have shared: {}",
            'CAUSE': "This happens because: {}"
        }
        self.clinician_templates = {
            'INFORMATION': "Clinical Information: {}",
            'SUGGESTION': "Treatment Protocol: {}",
            'EXPERIENCE': "Patient-Reported Outcomes: {}",
            'CAUSE': "Etiology & Pathophysiology: {}"
        }
    
    def _generate_summary_for_audience(self, entry: Dict, audience: str) -> str:
        """Generic summary generation logic for either audience."""
        templates = self.patient_templates if audience == 'patient' else self.clinician_templates
        max_len, min_len = (80, 20) if audience == 'patient' else (120, 30)
        
        try:
            perspectives = self.perspective_processor.extract_perspectives(entry)
            sections = []

            for perspective, segments in perspectives.items():
                if segments:
                    combined_text = ". ".join(segments[:3] if audience == 'patient' else segments)
                    summary = self.summarizer.summarize_text(combined_text, max_length=max_len, min_length=min_len)
                    sections.append(templates.get(perspective, "{}").format(summary))
            
            if not sections:
                return self._generate_fallback_summary(entry, audience)

            full_summary = "\n\n".join(sections)
            disclaimer = self._get_disclaimer(audience, entry)
            return full_summary + disclaimer

        except Exception as e:
            logger.error(f"{audience.capitalize()} summary generation failed: {e}")
            return self._generate_fallback_summary(entry, audience)

    def generate_patient_summary(self, entry: Dict) -> str:
        return self._generate_summary_for_audience(entry, 'patient')

    def generate_clinician_summary(self, entry: Dict) -> str:
        return self._generate_summary_for_audience(entry, 'clinician')

    def _get_disclaimer(self, audience: str, entry: Dict) -> str:
        if audience == 'patient':
            return (
                    "\n\n⚠️ **Important**: This summary is for informational purposes only. "
                    "Always consult with your healthcare provider.")
        else: # Clinician
            metadata = [f"Source: {entry.get('uri', 'N/A')}", f"Response Count: {len(entry.get('answers', []))}"]
            return (
                    f"\n\n📋 **Metadata**: { ' | '.join(metadata)}\n\n" 
                    "🔬 **Note**: AI-generated summary for clinical reference. Verify information independently.")

    def _generate_fallback_summary(self, entry: Dict, audience: str) -> str:
        """Fallback summary when perspective processing fails."""
        logger.warning(f"Generating fallback summary for {audience}.")
        text_to_summarize = f"Question: {entry.get('question', '')} Answers: {" ".join(entry.get('answers', []))}"
        max_len = 100 if audience == 'patient' else 120
        summary = self.summarizer.summarize_text(text_to_summarize, max_length=max_len)
        return summary + self._get_disclaimer(audience, entry)

@st.cache_resource
def get_summarizer() -> LocalSummarizer:
    """Initializes and loads the summarizer model, caching it."""
    logger.info("Initializing MedSumAI Pro...")
    summarizer = LocalSummarizer(model_path=FINETUNED_MODEL_PATH)
    summarizer.load_model()
    logger.info("MedSumAI Pro initialization complete!")
    return summarizer

def create_streamlit_app():
    """Create the Streamlit web interface."""
    st.set_page_config(page_title="MedSumAI Pro", page_icon="🏥", layout="wide")

    # Load model and processors
    summarizer = get_summarizer()
    if not summarizer or not summarizer.summarizer:
        st.error("Model could not be loaded. The application cannot proceed.")
        return

    perspective_processor = PerspectiveProcessor()
    audience_generator = DualAudienceGenerator(summarizer, perspective_processor)

    st.title("🏥 MedSumAI Pro - Medical Q&A Summarizer")
    st.markdown("*Perspective-aware medical summarization for patients and clinicians*")

    st.sidebar.header("⚙️ Options")
    processing_mode = st.sidebar.selectbox("Processing Mode", ["Quick Summary", "Detailed Analysis"], help="Quick mode is faster, Detailed is more comprehensive.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📝 Input")
        input_text = st.text_area("Paste Medical Q&A Content:", height=200, placeholder="Enter medical question and answers here...")
        st.markdown("**Or upload a JSON file:**")
        uploaded_file = st.file_uploader("Choose JSON file", type=['json'])
        process_button = st.button("🔄 Generate Summaries", type="primary")

    with col2:
        st.header("📊 Output")
        if process_button:
            if not input_text and not uploaded_file:
                st.error("Please provide text input or upload a JSON file.")
                return

            with st.spinner("Processing medical content..."):
                entry = {}
                if uploaded_file:
                    try:
                        file_content = json.load(uploaded_file)
                        entry = file_content[0] if isinstance(file_content, list) else file_content
                    except json.JSONDecodeError:
                        st.error("Invalid JSON file. Please check the file format.")
                        return
                else:
                    # Simple parsing for text input
                    lines = input_text.splitlines()
                    entry = {
                        'question': lines[0] if lines else "",
                        'answers': lines[1:] if len(lines) > 1 else [],
                        'uri': 'user-input'
                    }

                patient_summary = audience_generator.generate_patient_summary(entry)
                clinician_summary = audience_generator.generate_clinician_summary(entry)

            tab1, tab2, tab3 = st.tabs(["👤 Patient View", "🩺 Clinician View", "📋 Details"])
            with tab1:
                st.markdown(patient_summary)
                st.download_button("📥 Download Patient Summary", patient_summary, "patient_summary.txt")
            with tab2:
                st.markdown(clinician_summary)
                st.download_button("📥 Download Clinical Summary", clinician_summary, "clinical_summary.txt")
            with tab3:
                st.metric("Model Used", summarizer.model_name)
                with st.expander("Raw Input Data"):
                    st.json(entry)

    st.markdown("---")
    st.markdown("*MedSumAI Pro v1.0 - AI-powered medical summarization for educational purposes.*")

if __name__ == "__main__":
    create_streamlit_app()