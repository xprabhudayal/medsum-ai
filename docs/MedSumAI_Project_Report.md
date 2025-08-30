# MedSumAI Pro: Project Report and Technical Walkthrough

## 1. Executive Summary

This report details the architecture and development journey of the MedSumAI Pro project. The project's primary goal is to provide a web-based tool for generating perspective-aware summaries of medical question-and-answer content, tailored for both patients and clinicians.

The final iteration of the project implements a robust, two-part system that separates the machine learning model fine-tuning from the user-facing application. This architecture consists of:

1.  **A dedicated training script (`train.py`):** This script leverages the Hugging Face `transformers` library to fine-tune a pre-trained language model (Facebook's BART) on the project's custom medical dataset.
2.  **A Streamlit web application (`medsumai_complete.py`):** This script serves as the interactive user interface, loading the fine-tuned model to perform on-demand summarization.

This modular design is efficient, scalable, and aligns with modern machine learning best practices.

## 2. Project Journey: From Monolith to a Modular Pipeline

The project underwent a significant architectural evolution based on performance bottlenecks and the need for true model customization.

### Initial State

The project began as a single, monolithic Python script (`medsumai_complete.py`). This script was responsible for all tasks:
- Loading the pre-trained model from the internet.
- Loading and parsing the entire `train`, `valid`, and `test` datasets.
- Defining the Streamlit user interface.
- Handling user input and performing summarization.

### Identified Problems

Running the initial script revealed several critical issues:

1.  **Performance Bottlenecks:** The application was extremely sluggish. The root cause was that the entire script, including the slow processes of model and data loading, was being re-executed on every user interaction. This is inherent to Streamlit's execution model but must be managed correctly.
2.  **No Fine-Tuning:** The script loaded a generic, pre-trained model but never actually fine-tuned it on the provided `train.json` data. The model's summaries were based only on its general knowledge, not on the specific medical data provided in the project.
3.  **Lack of Modularity:** Mixing training, inference, and UI logic in one file makes the codebase difficult to read, maintain, and scale. Changes to one component could inadvertently break another.

### Architectural Evolution

To address these problems, a strategic decision was made to refactor the project into a modular pipeline, separating the concerns of training and inference. This change directly addresses the core requirement of creating a model that is specialized for the custom dataset.

## 3. Current Architecture Deep Dive

The final architecture consists of two primary components that work in tandem.

### 3.1. Model Fine-Tuning (`train.py`)

This script is the heart of the machine learning pipeline. Its sole purpose is to create a specialized summarization model.

**Process:**

1.  **Data Loading:** The script begins by loading the `train.json` and `valid.json` files from the `/data` directory.
2.  **Preprocessing:** It processes each entry, creating a structured input for the model by combining the `question` and `answers` fields. The target output is the `labelled_summaries`.
3.  **Tokenization:** Using the tokenizer from the base model, it converts the text-based inputs and targets into numerical IDs that the transformer model can process.
4.  **Training:** It leverages the high-level Hugging Face `Trainer` API, which handles the complexities of the training loop, including gradient descent, evaluation, and checkpointing. The script fine-tunes the `facebook/bart-large-cnn` model on the prepared dataset.
5.  **Output:** Upon completion, the script saves the resulting fine-tuned model—a highly specialized asset—to the `./models/medsum-bart-finetuned` directory.

### 3.2. Inference and User Interface (`medsumai_complete.py`)

This script is the user-facing component, designed to be a lightweight and responsive web application.

**Process:**

1.  **Optimized Model Loading:** On startup, the application's first action is to check for the existence of the locally saved fine-tuned model in `./models/medsum-bart-finetuned`.
2.  **Dynamic Fallback:** If a fine-tuned model is not found, the application gracefully falls back to loading the generic `facebook/bart-large-cnn` model from the internet. It displays a clear warning to the user, ensuring the app is always functional even before the training script has been run.
3.  **Resource Caching:** The application uses Streamlit's `@st.cache_resource` decorator. This powerful feature ensures that the model, whether local or remote, is loaded into memory only **once**. This completely resolves the initial performance issues, leading to a fast and interactive user experience.
4.  **Inference:** It takes user input (either from a text area or a JSON file upload) and passes it to the loaded model to generate the patient- and clinician-focused summaries.

## 4. Rationale for Architectural Changes

The shift to a two-script system provides numerous advantages:

-   **Efficiency:** The Streamlit app is now lean and fast. It doesn't waste time or memory loading training data, and it caches the inference model intelligently.
-   **Modularity & Maintainability:** The codebase is clean and organized. A developer can work on the user interface in `medsumai_complete.py` without any risk of interfering with the complex training logic in `train.py`.
-   **Scalability:** This architecture allows for a professional deployment strategy. The `train.py` script can be run on a powerful, GPU-equipped machine for hours, while the resulting `medsumai_complete.py` app can be deployed on a cost-effective, standard server for user access.
-   **Reproducibility:** Having a saved model file allows for versioning and guarantees that the exact same model is used for all inference tasks, leading to consistent and reproducible results.

## 5. How to Use the System: A Two-Step Guide

The new workflow is simple and powerful.

### Step 1: Train the Model

To create your custom-tuned model, run the training script from your terminal. This process will take time and is best run on a machine with a GPU.

```bash
# This command will read your data and create the fine-tuned model
python train.py
```

### Step 2: Run the Application

Once the training is complete and the model is saved, you can launch the web application.

```bash
# This command starts the web server for the user interface
streamlit run medsumai_complete.py
```

The application will now automatically detect and use your custom-trained model, providing superior, domain-specific summaries.
