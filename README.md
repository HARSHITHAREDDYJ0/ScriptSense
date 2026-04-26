ScriptSense: AI-Powered Language Identification & Script Analysis System

ScriptSense is an advanced NLP pipeline for language identification, script recognition, and linguistic analysis across 12 Indic languages plus English. Built on the AI4Bharat Pralekha dataset, it combines modern machine learning with deep Unicode-level analysis to deliver high-precision multilingual understanding.

Live Platforms

Website: https://scripttsensee.netlify.app/
 Hugging Face Demo: https://huggingface.co/spaces/HarshithaaaaReddddyyy/Scriptt
 
 Supported Languages
Language	Script	Code
Bengali	Bengali	ben
English	Latin	eng
Gujarati	Gujarati	guj
Hindi	Devanagari	hin
Kannada	Kannada	kan
Malayalam	Malayalam	mal
Marathi	Devanagari	mar
Odia	Odia	ori
Punjabi	Gurmukhi	pan
Tamil	Tamil	tam
Telugu	Telugu	tel
Urdu	Perso-Arabic	urd

 System Architecture

scriptsense/
├── src/
│   ├── data/        # Dataset ingestion, preprocessing, streaming
│   ├── models/      # ML models: char-ngram, transformer, ensemble
│   ├── analysis/    # Script detection, Unicode inspection, statistics
│   ├── utils/       # Logging, metrics, helper utilities
│   └── api/         # FastAPI-based REST service
├── notebooks/       # EDA & experimentation
├── configs/         # YAML configuration files
├── tests/           # Unit & integration tests
└── scripts/         # CLI tools for training, evaluation, inference

Quick Start
# Install dependencies
pip install -r requirements.txt

# Train all models
python scripts/train_pipeline.py --config configs/default.yaml

# Start API server
uvicorn src.api.server:app --reload --port 8000

# Analyze text
python scripts/analyze.py --text "नमस्ते दुनिया"

# Run evaluation
python scripts/evaluate.py --split test

Core Features

Multi-Model Intelligence
Character n-gram models for lightweight efficiency
Transformer-based deep learning models for contextual understanding
Ensemble framework for improved robustness and accuracy

 Script-Level Analysis
Unicode block detection
Script mixing identification
Directionality handling (LTR / RTL)
Character distribution insights

 Scalable Data Pipeline
Optimized streaming for datasets exceeding 1.5M+ samples
Memory-efficient preprocessing
Modular and extensible pipeline design

 Production-Ready API
FastAPI-based REST service
Async support for high throughput
Easy deployment and integration

Evaluation & Metrics
Per-language precision, recall, and F1-score
Confusion matrix visualization
Confidence calibration and error analysis

 Interactive UI
Gradio-powered dashboard
Real-time predictions and script analysis
User-friendly interface for experimentation
 Vision

ScriptSense aims to bridge the gap in Indic language processing by delivering a unified system capable of understanding diverse scripts, multilingual inputs, and code-mixed text—making it highly relevant for real-world applications such as:

Social media text analysis
Multilingual search engines
OCR post-processing
Content moderation
Language-aware AI systems
