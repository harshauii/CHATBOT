from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure templates
current_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(current_dir))

# API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")
OPENFDA_URL = "https://api.fda.gov/drug/label.json"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def get_openfda_medications(condition: str):
    """Fetch FDA-approved medications from OpenFDA API"""
    try:
        response = requests.get(
            OPENFDA_URL,
            params={
                "search": f'indications_and_usage:"{condition}"',
                "limit": 5,
                "api_key": OPENFDA_API_KEY
            },
            timeout=10
        )
        
        if response.status_code != 200:
            return []

        medications = []
        for drug in response.json().get("results", []):
            try:
                brand_name = drug["openfda"]["brand_name"][0]
                dosage = next(iter(drug.get("dosage_and_administration", ["Not specified"])), "Not specified")
                purpose = next(iter(drug.get("indications_and_usage", ["Unknown"])), "Unknown")
                
                medications.append({
                    "name": brand_name,
                    "dosage": dosage.split('.')[0][:100],  # Truncate long text
                    "purpose": purpose.split('.')[0][:100]
                })
            except (KeyError, IndexError):
                continue

        return medications

    except Exception as e:
        logger.error(f"OpenFDA API error: {str(e)}")
        return []

def generate_treatment_recommendations(analysis_text: str):
    """Generate other recommendations using Groq"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json={
                "model": "llama-3-70b-8192",
                "messages": [{
                    "role": "system",
                    "content": "Generate treatment plan in JSON format with treatments, precautions, and follow_up arrays"
                }, {
                    "role": "user", 
                    "content": analysis_text
                }],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 500
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            content = json.loads(result["choices"][0]["message"]["content"])
            return {
                "treatments": content.get("treatments", []),
                "precautions": content.get("precautions", []),
                "follow_up": content.get("follow_up", [])
            }
        return {}

    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return {}

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Image processing and analysis (same as before)
        # ... [keep existing image analysis code]
        
        # Get medications from OpenFDA
        medications = await get_openfda_medications(analysis_text)
        
        # Get other recommendations from Groq
        other_recommendations = generate_treatment_recommendations(analysis_text)

        return JSONResponse({
            "analysis": analysis_text,
            "recommendations": {
                "medications": medications,
                "treatments": other_recommendations.get("treatments", []),
                "precautions": other_recommendations.get("precautions", []),
                "follow_up": other_recommendations.get("follow_up", [])
            }
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, detail="Processing error")
