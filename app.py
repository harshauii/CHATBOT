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

# Set templates directory to current folder
current_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(current_dir))

# Ensure index.html exists
template_path = current_dir / "index.html"
if not template_path.exists():
    raise RuntimeError(f"Template not found at {template_path}")

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_recommendations(analysis_text: str) -> dict:
    """Generate real-time medical recommendations using Groq API"""
    recommendation_prompt = f"""
    Analyze this radiology report and generate structured medical recommendations:
    {analysis_text}

    Provide recommendations in this exact JSON format:
    {{
        "medications": [
            {{
                "name": "Medication Name",
                "dosage": "Dosage information",
                "purpose": "Therapeutic purpose"
            }}
        ],
        "treatments": ["Treatment option 1", "Treatment option 2"],
        "precautions": ["Important precaution 1", "Important precaution 2"],
        "follow_up": ["Follow-up action 1", "Follow-up action 2"]
    }}

    Follow these guidelines:
    1. Use only medications appropriate for the condition
    2. Consider patient safety and contraindications
    3. Include both immediate and long-term actions
    4. Base recommendations on latest medical guidelines
    5. Use realistic dosages and administration frequencies
    """

    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama-3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a medical recommendation system. Generate structured treatment recommendations based on radiology findings."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 1000
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Recommendation API error: {response.text}")
            return {}

        result = response.json()
        recommendations = json.loads(result["choices"][0]["message"]["content"])
        return recommendations

    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return {}

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Process image
        image_content = await image.read()
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Verify image
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image: {str(e)}")
            raise HTTPException(400, detail="Invalid image format")

        # Get initial analysis
        analysis_response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama-3.2-11b-vision-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                    ]
                }],
                "max_tokens": 1000
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if analysis_response.status_code != 200:
            raise HTTPException(500, detail="Analysis failed")

        # Process analysis
        analysis_result = analysis_response.json()
        analysis_text = analysis_result["choices"][0]["message"]["content"]
        
        # Generate recommendations
        recommendations = generate_recommendations(analysis_text)

        return JSONResponse({
            "analysis": analysis_text,
            "recommendations": recommendations
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, detail="Processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
