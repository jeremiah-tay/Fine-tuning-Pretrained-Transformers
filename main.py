# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import io
import base64

# Import your modules
from data.data_loader import load_bioasq_data, extract_data, create_train_val_test_split
from data.data_processor import preprocess_qa_examples, postprocess_qa_predictions
from models.full_finetuning import FullFineTuningModel
from models.peft_lora import PEFTLoRAModel
from utils.metrics import calculate_squad_metrics, calculate_bertscore, compare_models
from config import MODEL_CONFIG, LORA_CONFIG

# Page configuration
st.set_page_config(
    page_title="BioBERT Fine-tuning Results",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 BioBERT Fine-tuning Results Dashboard")
st.markdown("Compare Full Fine-tuning vs PEFT with LoRA on BioASQ Dataset")

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("### Model Settings")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Full Fine-tuning", "PEFT LoRA", "Compare Both"]
)

# Data loading section
st.header("📊 Dataset Information")

@st.cache_data
def load_and_process_data():
    """Load and process the BioASQ data."""
    data_path = "./BioASQ-train-factoid-6b-full-annotated.json"
    
    if not Path(data_path).exists():
        st.error(f"Data file not found: {data_path}")
        return None, None, None
    
    # Load data
    with st.spinner("📂 Loading BioASQ dataset..."):
        raw_data = load_bioasq_data(data_path)
        df = extract_data(raw_data)
        datasets = create_train_val_test_split(df)
    
    return raw_data, df, datasets

def create_matplotlib_plot(metrics_data, title, colors=['#4c72b0', '#55a868', '#c44e52', '#DBE070']):
    """Create matplotlib bar plot for metrics."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(list(metrics_data.items()), columns=['Metric', 'Score'])
    
    # Create bar plot
    bars = ax.bar(metrics_df['Metric'], metrics_df['Score'], color=colors)
    
    # Add title and labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Metric Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.0)
    
    # Add score values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, 
                f'{yval:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize appearance
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_comparison_plot(fft_metrics, lora_metrics):
    """Create comparison plot between FFT and LoRA."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Prepare data
    metrics = ['Exact Match', 'F1 Score', 'Precision', 'Recall']
    fft_scores = [fft_metrics['exact_match'], fft_metrics['f1'], 
                  fft_metrics['precision'], fft_metrics['recall']]
    lora_scores = [lora_metrics['exact_match'], lora_metrics['f1'], 
                   lora_metrics['precision'], lora_metrics['recall']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, fft_scores, width, label='Full Fine-tuning', 
                   color='#4c72b0', alpha=0.8)
    bars2 = ax.bar(x + width/2, lora_scores, width, label='PEFT LoRA', 
                   color='#55a868', alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def matplotlib_to_streamlit(fig):
    """Convert matplotlib figure to streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

def display_single_model_results(model_name, squad_results, bertscore_results, predictions, test_examples):
    """Display results for a single model."""
    
    try:
        # Metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {model_name} - SQuAD Metrics")
            if squad_results and 'exact_match' in squad_results and 'f1' in squad_results:
                st.metric("Exact Match", f"{squad_results['exact_match']:.4f}")
                st.metric("F1 Score", f"{squad_results['f1']:.4f}")
            else:
                st.error("❌ SQuAD metrics not available")
        
        with col2:
            st.subheader(f"📊 {model_name} - BERTScore")
            if bertscore_results and all(key in bertscore_results for key in ['precision', 'recall', 'f1']):
                st.metric("Precision", f"{bertscore_results['precision']:.4f}")
                st.metric("Recall", f"{bertscore_results['recall']:.4f}")
                st.metric("F1 Score", f"{bertscore_results['f1']:.4f}")
            else:
                st.error("❌ BERTScore metrics not available")
    except Exception as e:
        st.error(f"❌ Error displaying metrics: {str(e)}")
        return
    
    # Matplotlib visualization
    st.subheader(f"📊 {model_name} - Performance Visualization")
    
    try:
        with st.spinner("📊 Creating visualization..."):
            # Prepare metrics data
            if squad_results and bertscore_results:
                metrics_data = {
                    'Exact Match': squad_results.get('exact_match', 0),
                    'F1 Score': squad_results.get('f1', 0),
                    'Precision': bertscore_results.get('precision', 0),
                    'Recall': bertscore_results.get('recall', 0)
                }
                
                # Create and display plot
                fig = create_matplotlib_plot(metrics_data, f'Evaluation Metrics for {model_name}')
                st.pyplot(fig)
            else:
                st.error("❌ Cannot create visualization - metrics data not available")
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")
    
    # Sample predictions
    st.subheader("Sample Predictions")
    try:
        if predictions and test_examples:
            sample_df = pd.DataFrame([
                {
                    "Question": test_examples[i]["question"],
                    "Answer": test_examples[i]["answer"],
                    "Prediction": predictions.get(test_examples[i]["id"], "No prediction")
                }
                for i in range(min(5, len(test_examples)))
            ])
            st.dataframe(sample_df)
        else:
            st.error("❌ Sample predictions not available")
    except Exception as e:
        st.error(f"❌ Error displaying sample predictions: {str(e)}")

def display_comparison_results(fft_squad, fft_bertscore, fft_predictions,
                              lora_squad, lora_bertscore, lora_predictions,
                              test_examples):
    """Display comparison results."""
    
    # Metrics comparison
    st.subheader("📊 Metrics Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Exact Match", 
                 f"{lora_squad['exact_match']:.4f}",
                 f"{lora_squad['exact_match'] - fft_squad['exact_match']:.4f}")
    
    with col2:
        st.metric("F1 Score (SQuAD)", 
                 f"{lora_squad['f1']:.4f}",
                 f"{lora_squad['f1'] - fft_squad['f1']:.4f}")
    
    with col3:
        st.metric("F1 Score (BERTScore)", 
                 f"{lora_bertscore['f1']:.4f}",
                 f"{lora_bertscore['f1'] - fft_bertscore['f1']:.4f}")
    
    # Individual model visualizations
    st.subheader("📈 Individual Model Performance")
    
    with st.spinner("📊 Creating individual model visualizations..."):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Full Fine-tuning")
            fft_metrics_data = {
                'Exact Match': fft_squad['exact_match'],
                'F1 Score': fft_squad['f1'],
                'Precision': fft_bertscore['precision'],
                'Recall': fft_bertscore['recall']
            }
            fft_fig = create_matplotlib_plot(fft_metrics_data, 'Evaluation Metrics for Full Fine-tuning')
            st.pyplot(fft_fig)
        
        with col2:
            st.markdown("### PEFT with LoRA")
            lora_metrics_data = {
                'Exact Match': lora_squad['exact_match'],
                'F1 Score': lora_squad['f1'],
                'Precision': lora_bertscore['precision'],
                'Recall': lora_bertscore['recall']
            }
            lora_fig = create_matplotlib_plot(lora_metrics_data, 'Evaluation Metrics for PEFT with LoRA')
            st.pyplot(lora_fig)
    
    # Side-by-side comparison
    st.subheader("📊 Side-by-Side Comparison")
    with st.spinner("📊 Creating comparison visualization..."):
        comparison_fig = create_comparison_plot(
            {
                'exact_match': fft_squad['exact_match'],
                'f1': fft_squad['f1'],
                'precision': fft_bertscore['precision'],
                'recall': fft_bertscore['recall']
            },
            {
                'exact_match': lora_squad['exact_match'],
                'f1': lora_squad['f1'],
                'precision': lora_bertscore['precision'],
                'recall': lora_bertscore['recall']
            }
        )
        st.pyplot(comparison_fig)
    
    # Performance summary table
    st.subheader("📋 Performance Summary")
    
    summary_data = {
        'Model': ['Full Fine-tuning', 'PEFT LoRA'],
        'Exact Match': [fft_squad['exact_match'], lora_squad['exact_match']],
        'F1 Score (SQuAD)': [fft_squad['f1'], lora_squad['f1']],
        'Precision': [fft_bertscore['precision'], lora_bertscore['precision']],
        'Recall': [fft_bertscore['recall'], lora_bertscore['recall']],
        'F1 Score (BERTScore)': [fft_bertscore['f1'], lora_bertscore['f1']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Sample predictions comparison
    st.subheader("Sample Predictions Comparison")
    
    comparison_df = pd.DataFrame([
        {
            "Question": test_examples[i]["question"],
            "Answer": test_examples[i]["answer"],
            "Full Fine-tuning": fft_predictions[test_examples[i]["id"]],
            "PEFT LoRA": lora_predictions[test_examples[i]["id"]]
        }
        for i in range(min(5, len(test_examples)))
    ])
    st.dataframe(comparison_df, use_container_width=True)

raw_data, df, datasets = load_and_process_data()

if df is not None:
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Examples", len(df))
    with col2:
        st.metric("Training Examples", len(datasets['train']))
    with col3:
        st.metric("Validation Examples", len(datasets['validation']))
    with col4:
        st.metric("Test Examples", len(datasets['test']))
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Training section
    st.header("🚀 Model Training")
    
    if st.button("Start Training", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_text = st.empty()
        
        # Initialize models
        with st.spinner("🔄 Initializing models..."):
            fft_model = FullFineTuningModel()
            lora_model = PEFTLoRAModel()
        
        # Load tokenizer for preprocessing
        with st.spinner("🔄 Loading tokenizer..."):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        
        # Preprocess data
        with st.spinner("🔄 Preprocessing data..."):
            status_text.text("Preprocessing data...")
            progress_bar.progress(20)
            progress_text.text("20% - Preprocessing data")
            
            tokenized_datasets = datasets.map(
                lambda examples: preprocess_qa_examples(
                    examples, tokenizer, 
                    MODEL_CONFIG["max_length"], 
                    MODEL_CONFIG["stride"]
                ),
                batched=True,
                remove_columns=datasets["train"].column_names
            )
        
        # Prepare test data (needed for both model types)
        with st.spinner("🔄 Preparing test data..."):
            test_features = tokenized_datasets["test"]
            test_examples = datasets["test"]
            sample_mapping = test_features["overflow_to_sample_mapping"]
            example_ids = [test_examples[i]["id"] for i in sample_mapping]
            test_features = test_features.add_column("example_id", example_ids)
        
        # Initialize result variables
        fft_squad_results = None
        fft_bertscore_results = None
        fft_final_predictions = None
        lora_squad_results = None
        lora_bertscore_results = None
        lora_final_predictions = None
        
        # Training
        if model_type in ["Full Fine-tuning", "Compare Both"]:
            with st.spinner("🚀 Training Full Fine-tuning model... This may take several minutes."):
                status_text.text("Training Full Fine-tuning model...")
                if model_type == "Full Fine-tuning":
                    progress_bar.progress(40)
                    progress_text.text("40% - Training Full Fine-tuning model")
                else:  # Compare Both
                    progress_bar.progress(20)
                    progress_text.text("20% - Training Full Fine-tuning model")
                
                fft_trainer = fft_model.train(
                    tokenized_datasets["train"],
                    tokenized_datasets["validation"]
                )
            
            # Make predictions
            with st.spinner("🔍 Making predictions with Full Fine-tuning..."):
                status_text.text("Making predictions with Full Fine-tuning...")
                if model_type == "Full Fine-tuning":
                    progress_bar.progress(60)
                    progress_text.text("60% - Making predictions")
                else:  # Compare Both
                    progress_bar.progress(30)
                    progress_text.text("30% - Making predictions")
                
                fft_raw_predictions = fft_trainer.predict(test_features)
                fft_final_predictions = postprocess_qa_predictions(
                    test_examples, test_features, fft_raw_predictions.predictions, tokenizer
                )
            
            # Calculate metrics
            with st.spinner("📊 Calculating metrics..."):
                status_text.text("Calculating metrics...")
                if model_type == "Full Fine-tuning":
                    progress_bar.progress(80)
                    progress_text.text("80% - Calculating metrics")
                else:  # Compare Both
                    progress_bar.progress(40)
                    progress_text.text("40% - Calculating metrics")
                
                try:
                    fft_squad_results = calculate_squad_metrics(fft_final_predictions, test_examples)
                    fft_bertscore_results = calculate_bertscore(fft_final_predictions, test_examples)
                    st.write("✅ Full Fine-tuning metrics calculated successfully")
                except Exception as e:
                    st.error(f"❌ Error calculating Full Fine-tuning metrics: {str(e)}")
                    fft_squad_results = None
                    fft_bertscore_results = None
        
        if model_type in ["PEFT LoRA", "Compare Both"]:
            with st.spinner("🚀 Training PEFT LoRA model... This may take several minutes."):
                status_text.text("Training PEFT LoRA model...")
                if model_type == "PEFT LoRA":
                    progress_bar.progress(40)
                    progress_text.text("40% - Training PEFT LoRA model")
                else:  # Compare Both
                    progress_bar.progress(50)
                    progress_text.text("50% - Training PEFT LoRA model")
                
                lora_trainer = lora_model.train(
                    tokenized_datasets["train"],
                    tokenized_datasets["validation"]
                )
            
            # Make predictions
            with st.spinner("🔍 Making predictions with PEFT LoRA..."):
                status_text.text("Making predictions with PEFT LoRA...")
                if model_type == "PEFT LoRA":
                    progress_bar.progress(60)
                    progress_text.text("60% - Making predictions")
                else:  # Compare Both
                    progress_bar.progress(60)
                    progress_text.text("60% - Making predictions")
                
                lora_raw_predictions = lora_trainer.predict(test_features)
                lora_final_predictions = postprocess_qa_predictions(
                    test_examples, test_features, lora_raw_predictions.predictions, tokenizer
                )
            
            # Calculate metrics
            with st.spinner("📊 Calculating metrics..."):
                status_text.text("Calculating metrics...")
                if model_type == "PEFT LoRA":
                    progress_bar.progress(80)
                    progress_text.text("80% - Calculating metrics")
                else:  # Compare Both
                    progress_bar.progress(70)
                    progress_text.text("70% - Calculating metrics")
                
                try:
                    lora_squad_results = calculate_squad_metrics(lora_final_predictions, test_examples)
                    lora_bertscore_results = calculate_bertscore(lora_final_predictions, test_examples)
                    st.write("✅ PEFT LoRA metrics calculated successfully")
                except Exception as e:
                    st.error(f"❌ Error calculating PEFT LoRA metrics: {str(e)}")
                    lora_squad_results = None
                    lora_bertscore_results = None
        
        # Completion
        with st.spinner("✅ Finalizing results..."):
            progress_bar.progress(100)
            progress_text.text("100% - Training completed!")
            status_text.text("Training completed!")
        
        # Clear progress indicators
        progress_bar.empty()
        progress_text.empty()
        status_text.empty()
        
        # Show success message
        st.success("🎉 Training completed successfully! View results below.")
        
        # Debug information
        with st.expander("🔍 Debug Information"):
            st.write(f"Model type: {model_type}")
            st.write(f"FFT results available: {fft_squad_results is not None}")
            st.write(f"LoRA results available: {lora_squad_results is not None}")
            if fft_squad_results is not None:
                st.write(f"FFT Exact Match: {fft_squad_results.get('exact_match', 'N/A')}")
            if lora_squad_results is not None:
                st.write(f"LoRA Exact Match: {lora_squad_results.get('exact_match', 'N/A')}")
        
        # Results section
        st.header("📈 Results")
        
        # Check what results are available
        fft_available = fft_squad_results is not None and fft_bertscore_results is not None and fft_final_predictions is not None
        lora_available = lora_squad_results is not None and lora_bertscore_results is not None and lora_final_predictions is not None
        
        if model_type == "Full Fine-tuning":
            if fft_available:
                st.success("✅ Full Fine-tuning results ready!")
                display_single_model_results("Full Fine-tuning", fft_squad_results, fft_bertscore_results, fft_final_predictions, test_examples)
            else:
                st.error("❌ Full Fine-tuning results not available. Please check the training logs for errors.")
                
        elif model_type == "PEFT LoRA":
            if lora_available:
                st.success("✅ PEFT LoRA results ready!")
                display_single_model_results("PEFT LoRA", lora_squad_results, lora_bertscore_results, lora_final_predictions, test_examples)
            else:
                st.error("❌ PEFT LoRA results not available. Please check the training logs for errors.")
                
        elif model_type == "Compare Both":
            if fft_available and lora_available:
                st.success("✅ Both model results ready for comparison!")
                display_comparison_results(fft_squad_results, fft_bertscore_results, fft_final_predictions,
                                         lora_squad_results, lora_bertscore_results, lora_final_predictions,
                                         test_examples)
            elif fft_available and not lora_available:
                st.warning("⚠️ Only Full Fine-tuning results available. LoRA training may have failed.")
                st.info("Displaying Full Fine-tuning results only:")
                display_single_model_results("Full Fine-tuning", fft_squad_results, fft_bertscore_results, fft_final_predictions, test_examples)
            elif not fft_available and lora_available:
                st.warning("⚠️ Only PEFT LoRA results available. Full Fine-tuning may have failed.")
                st.info("Displaying PEFT LoRA results only:")
                display_single_model_results("PEFT LoRA", lora_squad_results, lora_bertscore_results, lora_final_predictions, test_examples)
            else:
                st.error("❌ No results available for comparison. Please check the training logs for errors.")
        else:
            st.error("❌ Unknown model type selected.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | BioBERT Fine-tuning Project")