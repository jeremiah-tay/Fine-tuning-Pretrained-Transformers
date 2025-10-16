# 🧬 Fine-tuning-Pretrained-Transformers
A comprehensive Streamlit application for fine-tuning BioBERT models on biomedical question-answering tasks using both full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA techniques.

## 🎯 Overview

This project explores and compares two prominent fine-tuning strategies—full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA—on the BioBERT model. The goal is to analyze the trade-offs between model performance, computational efficiency, and deployment practicalities for a question-answering task on the complex BioASQ dataset.

The project includes an interactive Streamlit application to demonstrate and compare the models' predictions in real-time.

## ✨ Features

### 🚀 **Two Fine-Tuning Pipelines**
- Implements both traditional full fine-tuning and modern PEFT with LoRA.

### 📊 **Biomedical QA**
- Leverages the domain-specific BioBERT model to answer questions from the BioASQ dataset.

### 🔬 **Performance Evaluation**
- Automatically calculates metrics like Exact Match, F1 Score, Precision, and Recall.

### 🎨 **Interactive Demo**
- A Streamlit app (`main.py`) allows you to input questions and see live predictions from both models side-by-side.

## 📂 Project Structure
```
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # Loads the BioASQ dataset
│   └── data_processor.py          # Prepares data for the model
├── models/
│   ├── __init__.py
│   ├── full_finetuning.py         # Logic for the full fine-tuning model
│   └── peft_lora.py               # Logic for the PEFT with LoRA model
├── utils/
│   ├── __init__.py
│   └── metrics.py                 # Functions for performance evaluation
├── .gitignore
├── config.py                      # Central configuration for models and training
├── main.py                        # The main Streamlit application
├── README.md
└── requirements.txt               # Project dependencies
```
## 🌟 Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- GPU support (optional but recommended)

## ⚙️ **Setup and Installation**

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
   - Download the BioASQ dataset from Kaggle (https://www.kaggle.com/datasets/maverickss26/bioasq-dataset/data)
   - Place `BioASQ-train-factoid-6b-full-annotated.json` in the project root
   - The dataset will be automatically loaded when you run the application
  
## **▶️ How to Run**
Once the setup is complete, you can launch the interactive Streamlit application.

1. Make sure you are in the project's root directory and your virtual environment is activated, then run:
   ```bash
   streamlit run main.py
   ```
   Your web browser should automatically open with the application running.

2. **Using the Application:**
   - In the sidebar on the left, use the **"Select Model Type"** dropdown to choose which fine-tuning strategy you want to execute: `Full Fine-tuning`, `PEFT LoRA`, or `Compare Both`.
   - Once you've made your selection, click the main **"Start Training"** button on the page.
   - The application will then load the data, train the selected model(s), and display a detailed comparison of the results, including performance metrics and sample predictions.
