import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import GoogleGemini, GooglePalm
import matplotlib.pyplot as plt
from PIL import Image

# Load environment variables from .env file (override=True ensures real-time updates when .env is saved)
load_dotenv(override=True)

# Set page config
st.set_page_config(
    page_title="🧠 AI Analytics - Intelligent Data Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium UI styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #718096;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    .stDownloadButton>button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        border: none;
        padding: 0.5rem 1.8rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(17, 153, 142, 0.3);
    }
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(17, 153, 142, 0.45);
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Read API Key and Model from environment variables (.env file)
raw_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_PALM_API_KEY", os.getenv("PALM_API_KEY", "")))
api_key = raw_key.strip().strip('\'"')
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip().strip('\'"')

# Initialize LLM
llm = None
if api_key and api_key != "YOUR_API_KEY_HERE":
    try:
        model_name = gemini_model if gemini_model.startswith("models/") else f"models/{gemini_model}"
        GoogleGemini.model = model_name
        llm = GoogleGemini(api_key=api_key)
    except Exception as e:
        st.error(f"⚠️ Error initializing AI Model: {str(e)}")

# Header
st.markdown("<h1 class='main-title'>🧠 AI Analytics Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze your datasets using natural language queries powered by Google AI.</p>", unsafe_allow_html=True)

# Data source selection
data_source = st.radio(
    "Choose Dataset Source",
    ["Upload a CSV file", "Use Sample Diabetes Dataset"],
    horizontal=True
)

df = None
if data_source == "Upload a CSV file":
    uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
else:
    sample_path = "diabetes.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        st.info("Loaded pre-bundled Diabetes Dataset.")
    else:
        st.error(f"Sample dataset '{sample_path}' not found.")

if df is not None:
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.subheader("📊 Data Preview")
    with col2:
        st.write("") # Vertical alignment spacing
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv_data,
            file_name="full_dataset.csv",
            mime="text/csv",
            help="Download the complete dataset containing all rows and columns."
        )
    
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Showing preview of top 10 rows from the dataset ({len(df)} total rows, {len(df.columns)} columns). Use the button above to download the full dataset.")
    
    st.subheader("🤖 Query Your Data")
    query = st.text_input(
        'Ask anything about the dataset (e.g. "What is the average age of patients?" or "Plot a histogram of glucose")',
        placeholder="Enter your question here..."
    )
    
    analyze_btn = st.button(label='Analyze')
    
    if analyze_btn:
        if not llm:
            st.error("⚠️ Please provide a valid Google Gemini API Key in your `.env` file (e.g., `GEMINI_API_KEY=your_actual_key`).")
        elif not query:
            st.warning("Please enter a query first.")
        else:
            with st.spinner("Analyzing data and generating insights..."):
                try:
                    sdf = SmartDataframe(df, config={"llm": llm})
                    result = sdf.chat(query)
                    
                    st.success("Analysis Complete!")
                    
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                        st.download_button(
                            label="📥 Download Result Dataset (CSV)",
                            data=result.to_csv(index=False).encode('utf-8'),
                            file_name="query_result.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write(result)
                    
                    # Display any generated charts/plots if pandasai generated them
                    # PandasAI usually saves the chart to a file and returns the path or displays it.
                    # We can check if the result is a string representing a path to an image.
                    if isinstance(result, str) and (result.endswith('.png') or result.endswith('.jpg')):
                        if os.path.exists(result):
                            st.image(result, caption="Generated Visualisation")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        models = [m.name for m in genai.list_models()]
                        st.info("ℹ️ Available models under this API Key:")
                        st.write(models)
                    except Exception as diag_err:
                        st.warning(f"Could not retrieve model list: {str(diag_err)}")
else:
    st.info("Please upload a CSV file or choose the sample dataset to get started.")