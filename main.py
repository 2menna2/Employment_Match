#!/usr/bin/env python3
"""
FastAPI application for Employment Match - Skill Extraction and Matching System
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import our existing modules
from extract_skills import extract_skills, load_esco_skills, load_embedder
from extract_cv_skills import extract_cv_skills, extract_cv_skills_from_text
from match_skills import match_skills as match_skills_func
import generate_embeddings
import convert_esco_to_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Employment Match API",
    description="AI-powered skill extraction and matching system for job candidates and requirements",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
class JobDescriptionRequest(BaseModel):
    job_description: str = Field(..., description="Job description text to extract skills from")
    similarity_threshold: Optional[float] = Field(0.6, description="Threshold for embedding-based matching (0.0-1.0)")
    fuzzy_threshold: Optional[int] = Field(90, description="Threshold for fuzzy matching (0-100)")

class CVTextRequest(BaseModel):
    cv_text: str = Field(..., description="CV text to extract skills from")
    similarity_threshold: Optional[float] = Field(0.4, description="Threshold for embedding-based matching (0.0-1.0)")
    fuzzy_threshold: Optional[int] = Field(90, description="Threshold for fuzzy matching (0-100)")

class SkillMatchRequest(BaseModel):
    cv_skills: List[str] = Field(..., description="List of skills from CV")
    job_skills: List[str] = Field(..., description="List of skills from job description")
    similarity_threshold: Optional[float] = Field(0.3, description="Threshold for embedding-based matching (0.0-1.0)")
    fuzzy_threshold: Optional[int] = Field(80, description="Threshold for fuzzy matching (0-100)")

class SkillsResponse(BaseModel):
    standardized: List[str] = Field(..., description="Standardized skills mapped to ESCO taxonomy")
    raw: List[str] = Field(..., description="Raw skills as extracted from input")

class MatchResponse(BaseModel):
    match_score: float = Field(..., description="Overall match score (0-100)")
    matched_skills: List[Dict[str, Any]] = Field(..., description="List of matched skills with details")
    missing_skills: List[str] = Field(..., description="Skills required but not found in CV")
    extra_skills: List[str] = Field(..., description="Skills in CV but not required for job")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    esco_skills_loaded: bool = Field(..., description="Whether ESCO skills are loaded")
    embeddings_loaded: bool = Field(..., description="Whether embeddings are loaded")
    gemini_configured: bool = Field(..., description="Whether Gemini API is configured")

# Global variables for loaded models and data
esco_skills = None
embedder = None
sentence_transformer_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    global esco_skills, embedder, sentence_transformer_model
    
    try:
        # Load ESCO skills
        esco_skills = load_esco_skills("data/esco_skills.json")
        logger.info(f"Loaded {len(esco_skills)} ESCO skills")
        
        # Load embedder models
        embedder = load_embedder()
        logger.info("Loaded embedder models")
        
        # Load sentence transformer for matching
        from sentence_transformers import SentenceTransformer
        sentence_transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loaded sentence transformer model")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Employment Match API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/ui")
async def serve_ui():
    """Serve the HTML frontend"""
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        esco_skills_loaded=esco_skills is not None and len(esco_skills) > 0,
        embeddings_loaded=os.path.exists("data/esco_embeddings.npy"),
        gemini_configured=bool(os.getenv("GEMINI_API_KEY"))
    )

@app.post("/extract-job-skills", response_model=SkillsResponse)
async def extract_job_skills(request: JobDescriptionRequest):
    """Extract skills from job description"""
    try:
        if not esco_skills or not embedder:
            raise HTTPException(status_code=500, detail="Models not properly loaded")
        
        # Temporarily update thresholds
        from extract_skills import SIMILARITY_THRESHOLD, FUZZY_THRESHOLD
        original_sim_threshold = SIMILARITY_THRESHOLD
        original_fuzzy_threshold = FUZZY_THRESHOLD
        
        # Update thresholds if provided
        if request.similarity_threshold is not None:
            import extract_skills as skills_module
            skills_module.SIMILARITY_THRESHOLD = request.similarity_threshold
        if request.fuzzy_threshold is not None:
            import extract_skills as skills_module
            skills_module.FUZZY_THRESHOLD = request.fuzzy_threshold
        
        try:
            skills = extract_skills(request.job_description, esco_skills, embedder)
        finally:
            # Restore original thresholds
            import extract_skills as skills_module
            skills_module.SIMILARITY_THRESHOLD = original_sim_threshold
            skills_module.FUZZY_THRESHOLD = original_fuzzy_threshold
        
        return SkillsResponse(**skills)
        
    except Exception as e:
        logger.error(f"Error extracting job skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-job-skills-string", response_model=SkillsResponse)
async def extract_job_skills_string(
    job_description: str = Form(..., description="Job description text to extract skills from"),
    similarity_threshold: Optional[float] = Form(0.6, description="Threshold for embedding-based matching (0.0-1.0)"),
    fuzzy_threshold: Optional[int] = Form(90, description="Threshold for fuzzy matching (0-100)")
):
    """Extract skills from job description using string input instead of JSON"""
    try:
        if not esco_skills or not embedder:
            raise HTTPException(status_code=500, detail="Models not properly loaded")
        
        # Temporarily update thresholds
        from extract_skills import SIMILARITY_THRESHOLD, FUZZY_THRESHOLD
        original_sim_threshold = SIMILARITY_THRESHOLD
        original_fuzzy_threshold = FUZZY_THRESHOLD
        
        # Update thresholds if provided
        if similarity_threshold is not None:
            import extract_skills as skills_module
            skills_module.SIMILARITY_THRESHOLD = similarity_threshold
        if fuzzy_threshold is not None:
            import extract_skills as skills_module
            skills_module.FUZZY_THRESHOLD = fuzzy_threshold
        
        try:
            skills = extract_skills(job_description, esco_skills, embedder)
        finally:
            # Restore original thresholds
            import extract_skills as skills_module
            skills_module.SIMILARITY_THRESHOLD = original_sim_threshold
            skills_module.FUZZY_THRESHOLD = original_fuzzy_threshold
        
        return SkillsResponse(**skills)
        
    except Exception as e:
        logger.error(f"Error extracting job skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-cv-skills-text", response_model=SkillsResponse)
async def extract_cv_skills_text(request: CVTextRequest):
    """Extract skills from CV text"""
    try:
        if not esco_skills or not embedder or not sentence_transformer_model:
            raise HTTPException(status_code=500, detail="Models not properly loaded")
        
        # Temporarily update thresholds
        from extract_cv_skills import SIMILARITY_THRESHOLD, FUZZY_THRESHOLD
        original_sim_threshold = SIMILARITY_THRESHOLD
        original_fuzzy_threshold = FUZZY_THRESHOLD
        
        # Update thresholds if provided
        if request.similarity_threshold is not None:
            import extract_cv_skills as cv_module
            cv_module.SIMILARITY_THRESHOLD = request.similarity_threshold
        if request.fuzzy_threshold is not None:
            import extract_cv_skills as cv_module
            cv_module.FUZZY_THRESHOLD = request.fuzzy_threshold
        
        try:
            skills = extract_cv_skills_from_text(request.cv_text, esco_skills, embedder)
        finally:
            # Restore original thresholds
            import extract_cv_skills as cv_module
            cv_module.SIMILARITY_THRESHOLD = original_sim_threshold
            cv_module.FUZZY_THRESHOLD = original_fuzzy_threshold
        
        return SkillsResponse(**skills)
        
    except Exception as e:
        logger.error(f"Error extracting CV skills from text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-cv-skills-pdf", response_model=SkillsResponse)
async def extract_cv_skills_pdf(
    file: UploadFile = File(..., description="PDF file to extract skills from"),
    similarity_threshold: Optional[float] = Form(0.4, description="Threshold for embedding-based matching (0.0-1.0)"),
    fuzzy_threshold: Optional[int] = Form(90, description="Threshold for fuzzy matching (0-100)")
):
    """Extract skills from CV PDF file"""
    try:
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        if not esco_skills or not embedder:
            raise HTTPException(status_code=500, detail="Models not properly loaded")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Temporarily update thresholds
            from extract_cv_skills import SIMILARITY_THRESHOLD, FUZZY_THRESHOLD
            original_sim_threshold = SIMILARITY_THRESHOLD
            original_fuzzy_threshold = FUZZY_THRESHOLD
            
            # Update thresholds if provided
            if similarity_threshold is not None:
                import extract_cv_skills as cv_module
                cv_module.SIMILARITY_THRESHOLD = similarity_threshold
            if fuzzy_threshold is not None:
                import extract_cv_skills as cv_module
                cv_module.FUZZY_THRESHOLD = fuzzy_threshold
            
            try:
                skills = extract_cv_skills(temp_path, esco_skills, embedder)
            finally:
                # Restore original thresholds
                import extract_cv_skills as cv_module
                cv_module.SIMILARITY_THRESHOLD = original_sim_threshold
                cv_module.FUZZY_THRESHOLD = original_fuzzy_threshold
            
            return SkillsResponse(**skills)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
        
    except Exception as e:
        logger.error(f"Error extracting CV skills from PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match-skills", response_model=MatchResponse)
async def match_skills(request: SkillMatchRequest):
    """Match CV skills against job skills"""
    try:
        if not sentence_transformer_model:
            raise HTTPException(status_code=500, detail="Sentence transformer model not loaded")
        
        # Temporarily update thresholds
        from match_skills import SIMILARITY_THRESHOLD, FUZZY_THRESHOLD
        original_sim_threshold = SIMILARITY_THRESHOLD
        original_fuzzy_threshold = FUZZY_THRESHOLD
        
        # Update thresholds if provided
        if request.similarity_threshold is not None:
            import match_skills as match_module
            match_module.SIMILARITY_THRESHOLD = request.similarity_threshold
        if request.fuzzy_threshold is not None:
            import match_skills as match_module
            match_module.FUZZY_THRESHOLD = request.fuzzy_threshold
        
        try:
            result = match_skills_func(request.cv_skills, request.job_skills, sentence_transformer_model)
        finally:
            # Restore original thresholds
            import match_skills as match_module
            match_module.SIMILARITY_THRESHOLD = original_sim_threshold
            match_module.FUZZY_THRESHOLD = original_fuzzy_threshold
        
        return MatchResponse(**result)
        
    except Exception as e:
        logger.error(f"Error matching skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete-matching")
async def complete_matching(
    job_description: str = Form(..., description="Job description text"),
    cv_text: Optional[str] = Form(None, description="CV text (if not uploading PDF)"),
    cv_file: Optional[UploadFile] = File(None, description="CV PDF file (if not providing text)"),
    job_similarity_threshold: Optional[float] = Form(0.6, description="Job skills extraction threshold"),
    cv_similarity_threshold: Optional[float] = Form(0.4, description="CV skills extraction threshold"),
    match_similarity_threshold: Optional[float] = Form(0.3, description="Skill matching threshold"),
    job_fuzzy_threshold: Optional[int] = Form(90, description="Job skills fuzzy matching threshold"),
    cv_fuzzy_threshold: Optional[int] = Form(90, description="CV skills fuzzy matching threshold"),
    match_fuzzy_threshold: Optional[int] = Form(80, description="Skill matching fuzzy threshold")
):
    """Complete workflow: extract skills from job and CV, then match them"""
    try:
        if not esco_skills or not embedder or not sentence_transformer_model:
            raise HTTPException(status_code=500, detail="Models not properly loaded")
        
        # Extract job skills
        job_skills_request = JobDescriptionRequest(
            job_description=job_description,
            similarity_threshold=job_similarity_threshold,
            fuzzy_threshold=job_fuzzy_threshold
        )
        job_skills_response = await extract_job_skills(job_skills_request)
        
        # Extract CV skills
        cv_skills_response = None
        if cv_file and cv_file.filename and cv_file.filename.lower().endswith('.pdf'):
            # Handle PDF upload
            cv_skills_response = await extract_cv_skills_pdf(
                file=cv_file,
                similarity_threshold=cv_similarity_threshold,
                fuzzy_threshold=cv_fuzzy_threshold
            )
        elif cv_text:
            # Handle text input
            cv_text_request = CVTextRequest(
                cv_text=cv_text,
                similarity_threshold=cv_similarity_threshold,
                fuzzy_threshold=cv_fuzzy_threshold
            )
            cv_skills_response = await extract_cv_skills_text(cv_text_request)
        else:
            raise HTTPException(status_code=400, detail="Either CV text or PDF file must be provided")
        
        # Match skills
        match_request = SkillMatchRequest(
            cv_skills=cv_skills_response.raw,
            job_skills=job_skills_response.raw,
            similarity_threshold=match_similarity_threshold,
            fuzzy_threshold=match_fuzzy_threshold
        )
        match_response = await match_skills(match_request)
        
        return {
            "job_skills": job_skills_response.dict(),
            "cv_skills": cv_skills_response.dict(),
            "matching_result": match_response.dict()
        }
        
    except Exception as e:
        logger.error(f"Error in complete matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup-data", response_model=Dict[str, str])
async def setup_data(background_tasks: BackgroundTasks):
    """Setup ESCO data and generate embeddings (runs in background)"""
    try:
        # Check if data files exist
        if not os.path.exists("data/skills_en.csv"):
            raise HTTPException(status_code=400, detail="skills_en.csv not found in data/ directory")
        
        # Run setup in background
        background_tasks.add_task(run_data_setup)
        
        return {"message": "Data setup started in background. Check logs for progress."}
        
    except Exception as e:
        logger.error(f"Error starting data setup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_data_setup():
    """Background task to setup ESCO data and embeddings"""
    try:
        logger.info("Starting data setup...")
        
        # Convert ESCO CSV to JSON
        if not os.path.exists("data/esco_skills.json"):
            logger.info("Converting ESCO CSV to JSON...")
            # Execute the conversion script
            import subprocess
            subprocess.run([sys.executable, "convert_esco_to_json.py"], check=True)
        
        # Generate embeddings
        if not os.path.exists("data/esco_embeddings.npy"):
            logger.info("Generating ESCO embeddings...")
            # Execute the embedding generation script
            import subprocess
            subprocess.run([sys.executable, "generate_embeddings.py"], check=True)
        
        logger.info("Data setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data setup: {e}")

@app.get("/esco-skills", response_model=Dict[str, Any])
async def get_esco_skills_info():
    """Get information about loaded ESCO skills"""
    try:
        if not esco_skills:
            raise HTTPException(status_code=500, detail="ESCO skills not loaded")
        
        return {
            "total_skills": len(esco_skills),
            "sample_skills": esco_skills[:5] if len(esco_skills) > 5 else esco_skills,
            "embeddings_available": os.path.exists("data/esco_embeddings.npy")
        }
        
    except Exception as e:
        logger.error(f"Error getting ESCO skills info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 