# Fine-tuning-Pretrained-Transformers
A comprehensive Streamlit application for fine-tuning BioBERT models on biomedical question-answering tasks using both full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) with LoRA techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This application provides an interactive interface for training and comparing BioBERT models on the BioASQ dataset. It supports two fine-tuning approaches:

1. **Full Fine-tuning**: Traditional end-to-end fine-tuning of the entire model
2. **PEFT with LoRA**: Parameter-efficient fine-tuning using Low-Rank Adaptation

The application automatically detects and optimizes for the best available compute resources (CUDA GPU, Apple Silicon MPS, or CPU).

## ✨ Features

### 🚀 **Automatic Device Detection**
- **CUDA GPU**: Enables fp16 mixed precision for maximum performance
- **Apple Silicon MPS**: Optimized for M1/M2/M3 Macs with 3-7x speedup
- **CPU Fallback**: Works on any system without GPU acceleration

### 📊 **Interactive Dashboard**
- Real-time training progress with spinners and progress bars
- Comprehensive metrics visualization using Matplotlib
- Side-by-side model comparison
- Sample predictions display

### 🔬 **Advanced Metrics**
- **SQuAD Metrics**: Exact Match and F1 Score
- **BERTScore**: Precision, Recall, and F1 Score
- **Performance Comparison**: Detailed model comparison tables

### 🎨 **Rich Visualizations**
- Bar charts for individual model performance
- Side-by-side comparison plots
- Interactive Streamlit components

## 🛠️ Installation

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
