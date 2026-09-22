# AI Analytics (Conversational Analytics Platform)

A powerful Streamlit web application that combines the simplicity of natural language with advanced AI capabilities to perform comprehensive data analytics on CSV files. Built with PandasAI and Google's Generative AI, this platform makes data analysis accessible to everyone.

## Features
- **Natural Language Queries:** Ask questions about your data in plain English.
- **Interactive Data Visualization:** Generate charts and graphs automatically.
- **AI-Powered Insights:** Leverage Google's PaLM AI for intelligent data analysis.
- **CSV File Support:** Easy drag-and-drop file upload.
- **Statistical Analysis:** Perform complex statistical operations with simple queries.

## Project Structure
- `streamlit_app.py`: The main Streamlit web application.
- `requirements.txt`: Python dependencies.
- `diabetes.csv`: Sample dataset.
- `pandasai.ipynb`: Jupyter notebook for experimentation.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed. It is recommended to use a virtual environment. You will also need a Google PaLM API key.

```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
git clone https://github.com/Apurva16032005/Conversational-Analytics.git
cd Conversational-Analytics
pip install -r requirements.txt
```

### 3. Configuration
Set up your Google PaLM API key. You can get your API key from Google AI Studio. Replace the hardcoded API key in `streamlit_app.py` with your own:

```python
llm = GooglePalm(api_key="YOUR_API_KEY_HERE")
```

For production, you can use environment variables:
```python
import os
llm = GooglePalm(api_key=os.getenv("GOOGLE_PALM_API_KEY"))
```

### 4. Running the Application
```bash
streamlit run streamlit_app.py
```
The dashboard will open in your browser at `http://localhost:8501`.

### 5. Usage Guide
1. **Upload Data:** Click "Upload a CSV file" and select your dataset.
2. **View Data:** The uploaded data will be displayed in a table format.
3. **Ask Questions:** Type natural language questions about your data (e.g., "What is the average age of patients with diabetes?").
4. **Get Results:** Click "Analyze" to get AI-powered insights and visualizations.
