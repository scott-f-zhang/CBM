# CBM NLP Frontend Demo

A Streamlit web application that provides an intuitive interface for interacting with the CBM NLP FastAPI backend service. This demo is configured for the Essay dataset, which focuses on programming answer quality assessment.

## Features

- **🔮 Single Text Prediction**: Interactive text analysis with real-time results
- **📊 Batch Evaluation**: CSV file upload for batch processing with metrics
- **📈 Backend Status**: Real-time monitoring of backend service health
- **🎯 Concept Analysis**: Joint mode predictions with concept breakdown
- **📊 Visualizations**: Interactive charts and probability distributions
- **📥 Export Results**: Download evaluation results as CSV

## Installation

```bash
cd frontend
pip install -r requirements.txt
```

## Running the Application

### Prerequisites
Make sure the backend service is running first:

```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload
```

### Start Frontend
```bash
# Terminal 2: Start frontend
cd frontend
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Single Text Prediction

1. Navigate to the "🔮 Single Prediction" tab
2. Select your preferred model (BERT, GPT-2, RoBERTa, or LSTM)
3. Choose the mode:
   - **Standard**: Basic answer quality assessment (binary classification)
   - **Joint**: Advanced analysis with concept predictions (8 programming concepts)
4. Enter your programming Q&A text in the text area
5. Click "🔮 Predict" to get results

**Results include:**
- Predicted answer quality (Correct/Incorrect with emoji display)
- Probability distribution chart
- Confidence score
- Concept predictions (for joint mode)

### 2. Batch Evaluation

1. Navigate to the "📊 Batch Evaluation" tab
2. Select model and mode
3. Upload a CSV file with the following format:
   ```csv
   text,label
   "Q: What is a pointer in C++?\nA: A pointer is a variable that stores the memory address of another variable.",1
   "Q: What is a class in C++?\nA: A banana is a yellow fruit that grows on trees.",0
   "Q: How do you declare an array?\nA: You declare an array by specifying the data type, followed by the array name and size in square brackets.",1
   ```
4. Optionally check "Show detailed predictions"
5. Click "📊 Evaluate" to process

**Results include:**
- Overall metrics (Accuracy, Macro F1, Weighted F1)
- Number of samples processed
- Detailed predictions table (if requested)
- Download button for results

### 3. Backend Status

1. Navigate to the "📈 Backend Status" tab
2. View real-time connection status
3. See available models and modes
4. Monitor currently loaded models
5. Use "🔄 Refresh Status" to update information

## Configuration

### Backend URL
You can configure the backend URL in the sidebar:
- Default: `http://localhost:8000`
- Change this if your backend is running on a different host/port

### Connection Status
The sidebar shows real-time connection status:
- ✅ **Connected**: Backend is accessible and healthy
- ❌ **Connection Failed**: Backend is not reachable

## File Format Requirements

### CSV Upload Format
For batch evaluation, your CSV file must contain:

**Required columns:**
- `text`: The programming Q&A text to analyze (string)
- `label`: True labels (integer, 0=Incorrect, 1=Correct)

**Example:**
```csv
text,label
"Q: What is a pointer in C++?\nA: A pointer is a variable that stores the memory address of another variable.",1
"Q: What is a class in C++?\nA: A banana is a yellow fruit that grows on trees.",0
"Q: How do you declare an array?\nA: You declare an array by specifying the data type, followed by the array name and size in square brackets.",1
"Q: What is inheritance?\nA: Inheritance allows a class to inherit properties and methods from another class.",1
"Q: What is polymorphism?\nA: I don't know what that means.",0
```

## Models and Modes

### Available Models
- **bert-base-uncased**: BERT transformer model
- **gpt2**: GPT-2 transformer model
- **roberta-base**: RoBERTa transformer model
- **lstm**: BiLSTM with attention mechanism

### Available Modes
- **standard**: Standard answer quality classification
  - Output: 2 classes (0=Incorrect, 1=Correct)
  - Use case: Binary answer quality assessment

- **joint**: Joint prediction with concept analysis
  - Output: Answer quality + concept predictions
  - Concepts: FC (Focus), CC (Coherence), TU (Thesis/Unity), CP (Content/Precision), R (Reasoning), DU (Development/Unity), EE (Evidence/Examples), FR (Flow/Readability)
  - Each concept: Negative, Neutral, Positive
  - Use case: Detailed analysis with programming concept breakdown

## Troubleshooting

### Common Issues

1. **"Connection Failed" Error**
   - Ensure backend service is running (`uvicorn main:app --reload`)
   - Check if backend URL is correct in sidebar
   - Verify backend is accessible at the specified URL

2. **"Prediction Failed" Error**
   - Check if the selected model is loaded in backend
   - Verify backend service logs for detailed error messages
   - Try a different model or mode

3. **CSV Upload Issues**
   - Ensure CSV has 'text' and 'label' columns
   - Verify labels are integers (0=Incorrect, 1=Correct)
   - Check file size (large files may timeout)

4. **Slow Performance**
   - Large text inputs or batch files may take time
   - Consider using smaller batch sizes
   - Check backend service performance

### Getting Help

1. Check the Backend Status tab for service health
2. Review backend service logs for detailed error messages
3. Ensure all dependencies are installed correctly
4. Verify file formats match requirements

## Development

### Project Structure
```
frontend/
├── __init__.py          # Package initialization
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### Key Components
- **Connection Management**: Real-time backend health monitoring
- **API Integration**: HTTP requests to FastAPI backend
- **Data Visualization**: Charts and metrics display
- **File Handling**: CSV upload and download functionality
- **Error Handling**: User-friendly error messages and validation

## Screenshots

The application features:
- Clean, modern interface with tabbed navigation
- Real-time connection status indicators
- Interactive charts and visualizations
- Responsive design for different screen sizes
- Intuitive form controls and file uploads
