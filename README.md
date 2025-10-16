# 🧬 Fine-tuning-Pretrained-Transformers
A comprehensive Streamlit application for fine-tuning BioBERT models on biomedical question-answering tasks using both full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA techniques.

## 🎯 Overview

This project explores and compares two prominent fine-tuning strategies—full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA—on the BioBERT model. The goal is to analyze the trade-offs between model performance, computational efficiency, and deployment practicalities for a question-answering task on the complex BioASQ dataset.

The project includes an interactive Streamlit application to demonstrate and compare the models' predictions in real-time.

## ✨ Features

### 🚀 **Two Fine-Tuning Pipelines**
Implements both traditional full fine-tuning and modern PEFT with LoRA.

### 📊 **Biomedical QA**
Leverages the domain-specific BioBERT model to answer questions from the BioASQ dataset.

### 🔬 **Performance Evaluation**
Automatically calculates metrics like Exact Match, F1 Score, Precision, and Recall.

### 🎨 **Interactive Demo**
A Streamlit app (`main.py`) allows you to input questions and see live predictions from both models side-by-side.

## 📂 Project Structure
```
├── data/
│   ├── __init__.py
│   ├── data_loader.py       # Loads the BioASQ dataset
│   └── data_processor.py    # Prepares data for the model
├── models/
│   ├── __init__.py
│   ├── full_finetuning.py   # Logic for the full fine-tuning model
│   └── peft_lora.py         # Logic for the PEFT with LoRA model
├── utils/
│   ├── __init__.py
│   └── metrics.py           # Functions for performance evaluation
├── .gitignore
├── BioASQ-train-factoid-6b.json # The training dataset
├── config.py                # Central configuration for models and training
├── main.py                  # The main Streamlit application
├── README.md
└── requirements.txt         # Project dependencies
```
### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- GPU support (optional but recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fine-tuning-Pretrained-Transformers
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download BioASQ dataset**
   - Place `BioASQ-train-factoid-6b-full-annotated.json` in the project root
   - The dataset will be automatically loaded when you run the application
