# res_pase.py
import os
import json
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document

# .env isi file ke folder se load hoga, chahe uvicorn kahin se bhi chalao
load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY missing hai. backend/.env me GROQ_API_KEY=... likho")

client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-120b"


# ---------- Schemas ----------
class JobD(BaseModel):
    role: str | None = None
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    minimum_experience: float | None = None
    education_requirements: list[str] = []
    responsibilities: list[str] = []


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


class MatchResult(BaseModel):
    candidate_name: str | None = None
    matching_skills: list[str] = []
    missing_skills: list[str] = []
    experience_met: bool | None = None
    score: float
    verdict: str = ""
    summary: str = ""
    experience_note: str = ""
    education_note: str = ""
    strengths: list[str] = []
    growth_areas: list[str] = []
    interview_focus: list[str] = []


# ---------- LLM helper (rate limit par retry ke saath) ----------
def call_llm(system_prompt: str, user_prompt: str, retries: int = 3) -> dict:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    raise last_error


# ---------- Job description ----------
def parse_job_description(job_description: str) -> JobD:
    system_prompt = f"""
You are an expert HR assistant.
Extract structured information from the job description.
Return ONLY valid JSON matching this schema:
{JobD.model_json_schema()}
Do NOT return the schema itself. Fill it with real data.
If minimum experience is missing, return null.
If a list has no info, return an empty list.
Do not invent information.
"""
    data = call_llm(system_prompt, f"Analyze this job description:\n\n{job_description}")
    return JobD(**data)


# ---------- Resume ----------
def parse_resume(resume_text: str) -> Resume:
    system_prompt = f"""
You are an expert resume parser.
Extract info based on meaning, not exact headings.
Return ONLY valid JSON matching this schema:
{Resume.model_json_schema()}
Rules:
1. Do not invent information.
2. Missing value -> null.
3. Empty list if no info.
4. Include internships in experiences.
5. Extract skills from the entire resume.
6. For each experience, copy the duration text exactly as written in the resume
   (for example "03/2025 - 05/2026" or "2-month internship").
7. total_experience_years: sum of all job and internship durations in years
   (6 months = 0.5). If the resume shows only education and projects, return 0.
   Use null only if you truly cannot tell.
"""
    data = call_llm(system_prompt, f"Parse this resume:\n\n{resume_text}")
    return Resume(**data)


def score_resume(job: JobD, resume: Resume) -> MatchResult:
    system_prompt = f"""
You are an HR recruiter. Compare the resume with the job description.
Return ONLY valid JSON matching this schema:
{MatchResult.model_json_schema()}
score must be a number from 0 to 100. Keep verdict short.
Ignore experience duration when judging. Experience is checked separately.
Focus on skills match and project relevance.
Also fill these fields, based ONLY on the resume and job data given:
- summary: 2 sentences at most.
- experience_note and education_note: 2 sentences at most each.
- strengths and growth_areas: up to 4 short items each.
- interview_focus: up to 3 short topics a recruiter should probe.
Never claim a skill is absent. If a skill is not mentioned, call it "not demonstrated".
"""
    user_prompt = f"""
JOB:
{job.model_dump_json(indent=2)}

RESUME:
{resume.model_dump_json(indent=2)}
"""
    data = call_llm(system_prompt, user_prompt)
    return MatchResult(**data)


# ---------- File reading ----------
def read_pdf(file_path) -> str:
    reader = PdfReader(file_path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def read_docx(file_path) -> str:
    doc = Document(file_path)
    text = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    text.append(cell.text)
    return "\n".join(text)


def read_resume(file_path) -> str | None:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(file_path)
    if suffix == ".docx":
        return read_docx(file_path)
    return None


# ---------- Experience (code se calculate, LLM se nahi) ----------
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _points(text: str):
    """'03/2025' ya 'Mar 2025' se (year, month) nikalta hai"""
    pts = []
    for m in re.finditer(r"(\d{1,2})/(\d{4})|([a-z]{3})[a-z]*\.?,?\s+(\d{4})", text):
        if m.group(1):
            pts.append((int(m.group(2)), int(m.group(1))))
        elif m.group(3) in MONTHS:
            pts.append((int(m.group(4)), MONTHS[m.group(3)]))
    return pts


def months_from_duration(duration: str | None) -> int:
    if not duration:
        return 0
    d = duration.lower()

    pts = _points(d)
    if pts:
        sy, sm = pts[0]
        if len(pts) >= 2:
            ey, em = pts[1]
        elif any(w in d for w in ("present", "current", "now", "ongoing")):
            today = date.today()
            ey, em = today.year, today.month
        else:
            return 0
        return max(0, (ey - sy) * 12 + (em - sm))

    n = re.search(r"(\d+)\s*-?\s*month", d)
    if n:
        return int(n.group(1))
    y = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*year", d)
    if y:
        return int(float(y.group(1)) * 12)
    return 0


def calc_experience_years(resume: Resume) -> float | None:
    months = sum(months_from_duration(e.duration) for e in resume.experiences)
    if months > 0:
        return round(months / 12, 1)
    return resume.total_experience_years      # dates nahi mili to LLM wali value


def check_experience(job: JobD, resume: Resume):
    required = job.minimum_experience
    if not required:                 # JD me experience nahi maanga
        return True
    have = calc_experience_years(resume)
    if have is None:
        return None                  # UI me "Experience not specified"
    return have >= required


# ---------- Main pipeline ----------
def process_candidates(job: JobD, resume_files: list[tuple[str, str]]) -> dict:
    """resume_files = [(filename, saved_path), ...]"""
    results, errors = [], []

    for filename, path in resume_files:
        try:
            text = read_resume(path)
            if not text or not text.strip():
                errors.append({"file": filename, "error": "Text extract nahi hua (scanned PDF ho sakti hai)"})
                continue

            resume = parse_resume(text)
            match = score_resume(job, resume)

            results.append({
                "file": filename,
                "name": resume.name or match.candidate_name,
                "email": resume.email,
                "score": match.score,
                "matching_skills": match.matching_skills,
                "missing_skills": match.missing_skills,
                "experience_met": check_experience(job, resume),
                "experience_years": calc_experience_years(resume),
                "verdict": match.verdict,
                "summary": match.summary,
                "experience_note": match.experience_note,
                "education_note": match.education_note,
                "strengths": match.strengths,
                "growth_areas": match.growth_areas,
                "interview_focus": match.interview_focus,
            })
        except Exception as e:
            errors.append({"file": filename, "error": str(e)})

    results.sort(key=lambda r: r["score"], reverse=True)

    return {
        "total_processed": len(results),
        "top_2": results[:2],
        "lowest_2": results[-2:] if len(results) > 2 else [],
        "all_results": results,
        "errors": errors,
    }
