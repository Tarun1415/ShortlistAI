# main.py
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from res_pase import parse_job_description, process_candidates

BASE_DIR = Path(__file__).parent
ALLOWED = {".pdf", ".docx"}

app = FastAPI(title="Resume Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development ke liye; deploy par apna domain likhna
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    # Frontend page yahi se serve hota hai
    return FileResponse(BASE_DIR / "index.html")


@app.post("/analyze")
async def analyze(
    job_description: str = Form(...),
    resumes: list[UploadFile] = File(...),
):
    if not job_description.strip():
        raise HTTPException(400, "Job description khaali hai")

    tmp_dir = Path(tempfile.mkdtemp())
    saved = []

    try:
        for f in resumes:
            if Path(f.filename).suffix.lower() not in ALLOWED:
                continue
            dest = tmp_dir / Path(f.filename).name
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append((f.filename, str(dest)))

        if not saved:
            raise HTTPException(400, "Koi valid PDF/DOCX nahi mili")

        def run():
            job = parse_job_description(job_description)
            return {"job": job.model_dump(), **process_candidates(job, saved)}

        return await run_in_threadpool(run)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)