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
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REQUIRED_KEYS = ["medications", "treatments", "precautions", "follow_up"]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def validate_recommendation_structure(data: dict) -> dict:
    """Ensure valid recommendation structure with fallback values"""
    return {
        "medications": [m for m in data.get("medications", []) if all(k in m for k in ["name", "dosage", "purpose"])],
        "treatments": data.get("treatments", []),
        "precautions": data.get("precautions", []),
        "follow_up": data.get("follow_up", [])
    }

def generate_recommendations(analysis_text: str) -> dict:
    """Generate medical recommendations with enhanced validation"""
    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama-3-70b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a medical recommendation system. "
                            "Generate structured treatment recommendations based on radiology findings. "
                            "Use valid JSON format with: medications (array of objects with name, dosage, purpose), "
                            "treatments (array of strings), precautions (array of strings), follow_up (array of strings)"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Radiology report: {analysis_text}\n\nGenerate recommendations in exact JSON format:"
                    }
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
            logger.error(f"Groq API error: {response.text}")
            return {key: [] for key in REQUIRED_KEYS}

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        try:
            recommendations = json.loads(content)
            return validate_recommendation_structure(recommendations)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid response format: {str(e)}")
            return {key: [] for key in REQUIRED_KEYS}

    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return {key: [] for key in REQUIRED_KEYS}

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Validate input
        if not image.content_type.startswith("image/"):
            raise HTTPException(400, detail="Invalid file type. Please upload an image")

        # Process image
        image_content = await image.read()
        
        # Verify image validity
        try:
            Image.open(io.BytesIO(image_content)).verify()
        except Exception as e:
            logger.error(f"Invalid image: {str(e)}")
            raise HTTPException(400, detail="Invalid or corrupted image file")

        # Get image analysis
        analysis_response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama-3.2-11b-vision-preview",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_content).decode('utf-8')}"
                        }}
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
            raise HTTPException(502, detail="Image analysis service unavailable")

        analysis_result = analysis_response.json()
        analysis_text = analysis_result["choices"][0]["message"]["content"]

        # Generate recommendations with fallback
        recommendations = generate_recommendations(analysis_text)

        return JSONResponse({
            "analysis": analysis_text,
            "recommendations": recommendations
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
