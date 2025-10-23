"""
FastAPI application for CBM NLP inference service.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    PredictRequest, PredictResponse, EvaluateResponse, 
    HealthResponse, ModelsResponse, ConceptPrediction
)
from model_manager import model_manager
from inference import predict_single, evaluate_batch
from pathlib import Path

# =========================
# Manual configuration
# =========================
# Select which dataset's models to use (matches folder under `saved_models/`)
CBM_DATASET = "essay"  # e.g., "essay", "imdb", "cebab"

# Make dataset selection visible to the model loader
os.environ["CBM_DATASET"] = CBM_DATASET

# Project root and model discovery
PROJECT_ROOT = Path(__file__).parent.parent

def discover_available_models(dataset: str) -> List[str]:
    base_dir = PROJECT_ROOT / "saved_models" / dataset
    if not base_dir.exists():
        return []
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])


# Create FastAPI app
app = FastAPI(
    title="CBM NLP Inference API",
    description="API for Concept Bottleneck Model inference on text data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models and modes
AVAILABLE_MODELS = discover_available_models(CBM_DATASET)
AVAILABLE_MODES = ["standard", "joint"]


@app.on_event("startup")
async def startup_event():
    """Load default models at startup."""
    print("Starting CBM NLP API service...")
    print(f"Using device: {model_manager.device}")
    model_manager.load_default_models()
    print("API service ready!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    loaded_models = model_manager.get_loaded_models()
    return HealthResponse(
        status="healthy",
        loaded_models=loaded_models
    )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """Get available models and currently loaded models."""
    loaded_models = model_manager.get_loaded_models()
    return ModelsResponse(
        available_models=AVAILABLE_MODELS,
        available_modes=AVAILABLE_MODES,
        loaded_models=loaded_models
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment/rating for a single text."""
    # Validate model name
    if request.model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_name. Available models: {AVAILABLE_MODELS}"
        )
    
    # Validate mode
    if request.mode not in AVAILABLE_MODES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode. Available modes: {AVAILABLE_MODES}"
        )
    
    try:
        # Perform prediction
        result = predict_single(request.text, request.model_name, request.mode)
        
        # Format concept predictions if present
        concept_predictions = None
        if result['concept_predictions']:
            concept_predictions = [
                ConceptPrediction(**cp) for cp in result['concept_predictions']
            ]
        
        return PredictResponse(
            prediction=result['prediction'],
            rating=result['rating'],
            probabilities=result['probabilities'],
            concept_predictions=concept_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    mode: str = Form(...),
    show_details: bool = Form(False)
):
    """Evaluate batch of texts with labels from CSV file."""
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model_name. Available models: {AVAILABLE_MODELS}"
        )
    
    # Validate mode
    if mode not in AVAILABLE_MODES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode. Available modes: {AVAILABLE_MODES}"
        )
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a CSV file"
        )
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
        
        # Validate CSV structure
        if 'text' not in df.columns or 'label' not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain 'text' and 'label' columns"
            )
        
        # Extract texts and labels
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # Validate labels (support 2, 5, and 6 class models)
        max_label = max(labels) if labels else 0
        if max_label > 5:
            raise HTTPException(
                status_code=400, 
                detail="Labels must be integers between 0 and 5 (for 6-class models)"
            )
        elif max_label > 4:
            # 6-class model (0-5)
            pass
        elif max_label > 1:
            # 5-class model (0-4)
            pass
        else:
            # 2-class model (0-1)
            pass
        
        # Perform evaluation
        result = evaluate_batch(texts, labels, model_name, mode, show_details)
        
        return EvaluateResponse(
            accuracy=result['accuracy'],
            macro_f1=result['macro_f1'],
            weighted_f1=result['weighted_f1'],
            num_samples=result['num_samples'],
            predictions=result['predictions']
        )
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CBM NLP Inference API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Single text prediction",
            "evaluate": "POST /evaluate - Batch evaluation with CSV upload",
            "health": "GET /health - Health check",
            "models": "GET /models - Available models",
            "docs": "GET /docs - API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
