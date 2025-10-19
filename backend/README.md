# CBM NLP Backend API

FastAPI backend service for Concept Bottleneck Model inference on text data.

## Features

- **Single Text Prediction**: `/predict` endpoint for individual text sentiment analysis
- **Batch Evaluation**: `/evaluate` endpoint for CSV file upload with metrics calculation
- **Model Management**: Automatic model loading at startup with caching
- **Multiple Models**: Support for BERT, GPT-2, RoBERTa, and LSTM
- **Two Modes**: Standard and Joint (concept-aware) prediction modes

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Running the Service

```bash
# Navigate to backend directory
cd backend

# Development mode
uvicorn main:app --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```
Returns service status and loaded models.

### 2. Available Models
```bash
GET /models
```
Lists available models, modes, and currently loaded models.

### 3. Single Text Prediction
```bash
POST /predict
Content-Type: application/json

{
  "text": "This restaurant has amazing food!",
  "model_name": "bert-base-uncased",
  "mode": "standard"
}
```

**Response:**
```json
{
  "prediction": 4,
  "rating": 5,
  "probabilities": [0.1, 0.1, 0.1, 0.1, 0.6],
  "concept_predictions": null
}
```

### 4. Batch Evaluation
```bash
POST /evaluate
Content-Type: multipart/form-data

file: [CSV file with 'text' and 'label' columns]
model_name: bert-base-uncased
mode: standard
show_details: false
```

**CSV Format:**
```csv
text,label
"Great restaurant!",4
"Terrible food.",0
"Average experience.",2
```

**Response:**
```json
{
  "accuracy": 0.8,
  "macro_f1": 0.75,
  "weighted_f1": 0.78,
  "num_samples": 3,
  "predictions": null
}
```

## Available Models

- **bert-base-uncased**: BERT model
- **gpt2**: GPT-2 model  
- **roberta-base**: RoBERTa model
- **lstm**: BiLSTM with attention

## Available Modes

- **standard**: Standard sentiment classification (5 classes: 0-4)
- **joint**: Joint prediction with concept analysis (Food, Ambiance, Service, Noise)

## API Documentation

Once the service is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Example Usage

```bash
# Test single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great food!", "model_name": "bert-base-uncased", "mode": "standard"}'

# Test batch evaluation
curl -X POST "http://localhost:8000/evaluate" \
  -F "file=@test_data.csv" \
  -F "model_name=bert-base-uncased" \
  -F "mode=standard"
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid model names or modes
- Malformed CSV files
- Missing required columns
- Model loading failures
- Prediction errors

## Performance

- Models are loaded at startup for fast inference
- Automatic device detection (CPU/CUDA)
- Batch processing for evaluation endpoint
- CORS enabled for web client integration
