<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# MedSumAI Pro - Product Requirements Document

## Project Overview

**Build Time Constraint:** 1.5 hours
**Goal:** Create a locally-deployed, perspective-aware medical summarization system that processes healthcare Q\&A data and generates tailored summaries for patients and clinicians.

**Core Philosophy:** "Just Works" - Minimal viable product with essential functionality.

***

## Technical Stack

- **Development Environment:** Jupyter Notebook
- **LLM:** Local model (Phi-2 or similar lightweight model)
- **Frontend:** Streamlit
- **Key Libraries:** transformers, torch, streamlit, pandas, json

***

## Jupyter Notebook Structure

### Cell 1: Environment Setup \& Imports

```python
# Install and import required libraries
# Set up local model loading
# Configure basic logging
```

**Time Allocation:** 10 minutes

### Cell 2: Data Loading \& Preprocessing

```python
# Load train/test/valid JSON files
# Parse data structure (uri, question, context, answers, labelled_summaries)
# Create simple data validation
# Sample data exploration
```

**Key Functions:**

- `load_medical_data(file_path)`
- `parse_json_structure(data)`
- `validate_data_format(entry)`

**Time Allocation:** 15 minutes

### Cell 3: Local Model Setup

```python
# Initialize lightweight summarization model (Phi-2 or DistilBART)
# Configure model for local inference
# Test basic model functionality
```

**Model Requirements:**

- Must run on standard laptop (8GB RAM constraint)
- No API calls - fully local
- Fast inference time

**Time Allocation:** 20 minutes

### Cell 4: Perspective-Aware Processing Engine

```python
# Extract different perspective categories (INFORMATION, CAUSE, SUGGESTION, EXPERIENCE)
# Implement basic span detection
# Create perspective-specific prompt templates
```

**Core Functions:**

- `extract_perspectives(entry)`
- `generate_perspective_summary(text, perspective_type)`
- `combine_perspectives(summaries)`

**Time Allocation:** 25 minutes

### Cell 5: Dual-Audience Output Generator

```python
# Patient-friendly summary generator (simple language, actionable)
# Clinician summary generator (detailed, evidence-based)
# Basic safety disclaimers
```

**Output Formats:**

- **Patient View:** Simple, conversational, with health disclaimers
- **Clinician View:** Detailed, structured, with source references

**Time Allocation:** 15 minutes

### Cell 6: Streamlit Frontend Integration

```python
# Create Streamlit app structure
# Input handling (paste Q&A text or upload JSON)
# Summary generation interface
# Results display with perspective tabs
```

**Frontend Features:**

- Text input area for medical Q\&A
- Processing button
- Tabbed output (Patient/Clinician views)
- Basic error handling

**Time Allocation:** 20 minutes

### Cell 7: Testing \& Validation

```python
# Test with sample data entries
# Validate output quality
# Performance benchmarking
# Error handling verification
```

**Time Allocation:** 5 minutes

***

## Core Features (MVP)

### Essential Features

1. **Data Ingestion:** Load and parse the provided JSON medical Q\&A data
2. **Local Summarization:** Generate summaries using local LLM
3. **Perspective Awareness:** Basic categorization of content types
4. **Dual Output:** Patient vs. Clinician summary formats
5. **Web Interface:** Simple Streamlit frontend

### Simplified Architecture

```
Input (Medical Q&A) → Data Processing → Local LLM → Perspective Classification → Dual Summary Generation → Streamlit Display
```


### Safety \& Disclaimers

- **Patient Mode:** Always include "Consult healthcare professional" disclaimer
- **Clinician Mode:** Mark as "AI-generated content for reference only"

***

## Data Processing Pipeline

### Input Processing

1. Parse JSON structure from provided datasets
2. Extract key fields: question, context, answers, existing summaries
3. Clean and validate text data

### Perspective Detection (Simplified)

- **INFORMATION:** Factual medical knowledge
- **SUGGESTION:** Treatment/management recommendations
- **EXPERIENCE:** Patient experience sharing
- **CAUSE:** Causal relationships and explanations


### Output Generation

- Use existing `labelled_summaries` as training examples
- Generate new summaries for unseen data
- Apply perspective-specific formatting

***

## Streamlit Frontend Specifications

### Page Layout

```
Title: MedSumAI Pro - Medical Q&A Summarizer

Input Section:
├── Text Area: "Paste medical Q&A content"
├── File Upload: "Upload JSON file (optional)"
└── Process Button

Output Section:
├── Tab 1: Patient Summary (simplified language)
├── Tab 2: Clinician Summary (detailed, technical)
└── Processing Status/Errors
```


### User Experience

- **Input:** Paste text or upload JSON file
- **Processing:** Show progress indicator
- **Output:** Tabbed interface for different audiences
- **Error Handling:** Clear error messages for invalid input

***

## Success Criteria

### Functional Requirements

- [ ] Successfully loads and processes provided JSON data
- [ ] Generates coherent summaries using local model
- [ ] Distinguishes between patient and clinician outputs
- [ ] Runs entirely offline (no API dependencies)
- [ ] Completes processing within reasonable time (< 30 seconds per query)


### Technical Requirements

- [ ] Fits within 8GB RAM constraint
- [ ] Jupyter notebook runs end-to-end without errors
- [ ] Streamlit app launches and responds to user input
- [ ] Handles edge cases (empty inputs, malformed data)

***

## Implementation Priority

**High Priority (Must Have):**

- Data loading and basic processing
- Local model integration
- Simple summarization pipeline
- Basic Streamlit frontend

**Medium Priority (Nice to Have):**

- Perspective categorization
- Dual-audience formatting
- Enhanced error handling

**Low Priority (Future Enhancement):**

- Advanced factuality checking
- Citation linking
- Model fine-tuning capabilities

***

## Risk Mitigation

### Technical Risks

- **Model Size:** Use DistilBART if Phi-2 too large
- **Processing Speed:** Implement basic caching
- **Memory Issues:** Process data in smaller batches


### Time Constraints

- **Fallback Plan:** Use pre-existing summaries from dataset if model generation fails
- **Minimum Viable:** Text input → basic summary → display (30-minute version)

This PRD provides a realistic roadmap for building a functional medical summarization tool within the 1.5-hour constraint while maintaining the core vision of perspective-aware, locally-deployed AI assistance.

