
import json
import logging
import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BASE_MODEL = "facebook/bart-large-cnn"
OUTPUT_DIR = "./models/medsum-bart-finetuned"
TRAIN_FILE = "data/train.json"
VALID_FILE = "data/valid.json"

# Training hyperparameters
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4 
PER_DEVICE_EVAL_BATCH_SIZE = 4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LEARNING_RATE = 2e-5
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256

def load_data_from_json(file_path: str) -> Dataset:
    """Loads data from a JSON file and converts it to a Hugging Face Dataset."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # We need to structure the data into a format that the model can use.
        # We'll create a prompt from the question and answers, and the summary is our target.
        processed_records = []
        for record in data:
            question = record.get("question", "")
            # Combine all answers into a single text block
            answers = " ".join([ans if isinstance(ans, str) else ans.get('txt', '') for ans in record.get("answers", [])])
            
            # The input to the model will be the question and the answers
            input_text = f"Question: {question} Answers: {answers}"
            
            # The target is the summary. We'll use the first summary found.
            summary = ""
            if "labelled_summaries" in record and record["labelled_summaries"]:
                summary = next(iter(record["labelled_summaries"].values()), "")

            if input_text and summary:
                processed_records.append({"input_text": input_text, "target_text": summary})

        df = pd.DataFrame(processed_records)
        return Dataset.from_pandas(df)

    except Exception as e:
        logger.error(f"Error loading or processing data from {file_path}: {e}")
        return None

def preprocess_function(examples, tokenizer):
    """Tokenizes the input and target texts."""
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    """Main function to run the fine-tuning process."""
    logger.info("Starting fine-tuning process...")

    # --- 1. Load Model and Tokenizer ---
    logger.info(f"Loading base model and tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    
    # --- 2. Load and Preprocess Data ---
    logger.info("Loading and preprocessing datasets...")
    train_dataset = load_data_from_json(TRAIN_FILE)
    valid_dataset = load_data_from_json(VALID_FILE)

    if train_dataset is None or valid_dataset is None:
        logger.error("Could not load datasets. Aborting training.")
        return

    tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_valid_dataset = valid_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # --- 3. Define Training Arguments ---
    logger.info("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir='./logs',
        logging_steps=LOGGING_STEPS,
        evaluation_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        learning_rate=LEARNING_RATE,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(), # Enable mixed precision training if CUDA is available
        report_to="tensorboard",
    )

    # --- 4. Create Trainer ---
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 5. Start Training ---
    logger.info("🚀 Starting model fine-tuning!")
    trainer.train()
    logger.info("✅ Fine-tuning complete.")

    # --- 6. Save the Model ---
    logger.info(f"Saving fine-tuned model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("✨ Model saved successfully! You can now use it for inference.")

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
