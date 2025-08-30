# MedSumAI Pro - Installation and Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install transformers torch streamlit pandas accelerate psutil
```

### 2. Run the Complete Application
```bash
python medsumai_complete.py
```

### 3. Run Streamlit Web Interface
```bash
streamlit run medsumai_complete.py
```

## Project Structure

```
Med-Sum-AI/
├── data/
│   ├── train.json
│   ├── test.json
│   └── valid.json
├── docs/
│   └── MedSumAI Pro - Product Requirements Document.md
├── medsumai_complete.py      # Complete implementation
├── test_validation.py        # Testing suite
└── README.md                # This file
```

## Usage Examples

### As a Python Module
```python
from medsumai_complete import *

# Initialize components
summarizer = LocalSummarizer()
summarizer.load_model()
perspective_processor = PerspectiveProcessor()
audience_generator = DualAudienceGenerator(summarizer, perspective_processor)

# Process medical data
sample_entry = {
    'question': 'What causes diabetes?',
    'context': 'Medical information about diabetes',
    'answers': ['Diabetes is caused by...'],
    'uri': 'example',
    'labelled_summaries': {}
}

patient_summary = audience_generator.generate_patient_summary(sample_entry)
clinician_summary = audience_generator.generate_clinician_summary(sample_entry)
```

### Web Interface Features
- **Text Input**: Paste medical Q&A content directly
- **File Upload**: Upload JSON files with medical data
- **Dual Summaries**: Get both patient and clinician views
- **Download**: Save summaries as text files
- **Processing Details**: View model info and performance metrics

## Model Information

**Default Model**: facebook/bart-large-cnn
**Fallback Models**: 
- sshleifer/distilbart-cnn-12-6
- facebook/bart-base
- t5-small (emergency fallback)

**Memory Requirements**: Optimized for 8GB RAM systems
**Device Support**: CUDA (if available) or CPU

## Key Features

1. **Perspective-Aware Processing**: Categorizes content into:
   - 📚 INFORMATION: Medical facts and definitions
   - 💊 SUGGESTION: Treatment recommendations  
   - 👤 EXPERIENCE: Patient experiences
   - 🔍 CAUSE: Causal relationships

2. **Dual Audience Output**:
   - **Patient View**: Simple language with safety disclaimers
   - **Clinician View**: Technical terminology with metadata

3. **Local Deployment**: No API dependencies, runs entirely offline

4. **Error Handling**: Graceful fallbacks for edge cases

## Testing

Run the validation suite:
```bash
python test_validation.py
```

## Data Format

Expected JSON structure:
```json
{
    "uri": "unique-identifier",
    "question": "Medical question",
    "context": "Additional context",
    "answers": ["Answer 1", "Answer 2"],
    "labelled_summaries": {
        "perspective_type": "existing_summary"
    }
}
```

## Safety & Disclaimers

- **Patient summaries** include medical disclaimers
- **Clinical summaries** marked as AI-generated reference material
- System designed for educational and reference purposes only
- Always consult healthcare professionals for medical decisions

## Troubleshooting

### Memory Issues
- Model will automatically fallback to lighter versions
- Close other applications to free RAM
- Use CPU mode if GPU memory insufficient

### Model Loading Errors
- Check internet connection for first-time model download
- Ensure sufficient disk space (2-3GB for models)
- Try running with `--no-cache-dir` pip flag

### Performance Optimization
- Use CUDA-compatible GPU for faster processing
- Reduce input text length for quicker summaries
- Process data in smaller batches

## System Requirements

- **Python**: 3.8+
- **RAM**: 8GB recommended
- **Storage**: 5GB for models and dependencies
- **GPU**: Optional but recommended for faster processing

## Version History

- **v1.0**: Initial release with core functionality
  - Local model integration
  - Perspective-aware processing
  - Dual-audience summaries
  - Streamlit web interface

---

*MedSumAI Pro - Local AI-powered medical summarization system*