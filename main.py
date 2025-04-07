import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_recommendations(analysis_text: str) -> dict:
    """Generate real-time recommendations through Groq API"""
    recommendation_prompt = f"""
    Analyze this medical analysis and create treatment recommendations:
    {analysis_text}

    Respond in this JSON format:
    {{
        "medications": [{{"name": "", "dosage": "", "purpose": ""}}],
        "treatments": [],
        "precautions": [],
        "follow_up": []
    }}
    
    Guidelines:
    1. Use evidence-based medicine
    2. Consider drug interactions
    3. Prioritize patient safety
    4. Include dosage guidelines
    5. Suggest appropriate follow-up
    """

    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama-3-70b-8192",
                "messages": [
                    {"role": "system", "content": "Generate medical recommendations from analysis"},
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

        return json.loads(response.json()["choices"][0]["message"]["content"])

    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        return {}

def process_image(image_path: str, query: str) -> dict:
    try:
        with open(image_path, "rb") as f:
            image_content = f.read()
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Get analysis
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

        analysis_text = analysis_response.json()["choices"][0]["message"]["content"]
        
        # Generate recommendations
        recommendations = generate_recommendations(analysis_text)

        return {
            "analysis": analysis_text,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    result = process_image("test.png", "Analyze this medical image and suggest treatment")
    print("Analysis:\n", result["analysis"])
    print("\nRecommendations:\n", json.dumps(result["recommendations"], indent=2))import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_recommendations(analysis_text: str) -> dict:
    """Generate real-time recommendations through Groq API"""
    recommendation_prompt = f"""
    Analyze this medical analysis and create treatment recommendations:
    {analysis_text}

    Respond in this JSON format:
    {{
        "medications": [{{"name": "", "dosage": "", "purpose": ""}}],
        "treatments": [],
        "precautions": [],
        "follow_up": []
    }}
    
    Guidelines:
    1. Use evidence-based medicine
    2. Consider drug interactions
    3. Prioritize patient safety
    4. Include dosage guidelines
    5. Suggest appropriate follow-up
    """

    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "llama-3-70b-8192",
                "messages": [
                    {"role": "system", "content": "Generate medical recommendations from analysis"},
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

        return json.loads(response.json()["choices"][0]["message"]["content"])

    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        return {}

def process_image(image_path: str, query: str) -> dict:
    try:
        with open(image_path, "rb") as f:
            image_content = f.read()
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Get analysis
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

        analysis_text = analysis_response.json()["choices"][0]["message"]["content"]
        
        # Generate recommendations
        recommendations = generate_recommendations(analysis_text)

        return {
            "analysis": analysis_text,
            "recommendations": recommendations
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    result = process_image("test.png", "Analyze this medical image and suggest treatment")
    print("Analysis:\n", result["analysis"])
    print("\nRecommendations:\n", json.dumps(result["recommendations"], indent=2))