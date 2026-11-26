
import os
import json
import re
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document


# ---------- Configuration ----------

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY missing hai. backend/.env me GROQ_API_KEY=... likho"
    )

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"


# ---------- Schemas ----------

class JobD(BaseModel):
    is_valid: bool = False
    validation_reason: str = ""

    role: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience: float | None = None
    education_requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    candidate_name: str | None = None
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    experience_met: bool | None = None
    score: float

    verdict: str = ""
    summary: str = ""
    experience_note: str = ""
    education_note: str = ""

    strengths: list[str] = Field(default_factory=list)
    growth_areas: list[str] = Field(default_factory=list)
    interview_focus: list[str] = Field(default_factory=list)


# ---------- LLM Helper ----------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    retries: int = 3
) -> dict:

    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("AI returned an empty response.")

            return json.loads(content)

        except Exception as e:
            last_error = e

            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"AI service failed after {retries} attempts."
    ) from last_error


# ---------- Job Description Validation ----------

def validate_job(job: JobD) -> None:
    """
    Check whether parsed job description contains
    sufficient job-related information.
    """

    has_job_details = any([
        job.required_skills,
        job.preferred_skills,
        job.responsibilities,
        job.education_requirements,
        job.minimum_experience is not None
    ])

    if not job.is_valid:
        raise ValueError(
            job.validation_reason or
            "Invalid Job Description. Please enter a proper job description."
        )

    if not job.role or not job.role.strip():
        raise ValueError(
            "Job role missing. Please mention the job position."
        )

    if not has_job_details:
        raise ValueError(
            "Job requirements missing. Please mention skills, "
            "responsibilities or experience."
        )


def parse_job_description(job_description: str) -> JobD:

    # Basic input validation
    if not job_description or not job_description.strip():
        raise ValueError(
            "Please enter a Job Description."
        )

    job_description = job_description.strip()

    # Reject very short input before calling the LLM
    if len(job_description) < 30:
        raise ValueError(
            "Invalid Job Description. Please enter a proper job description."
        )

    system_prompt = f"""
You are an expert HR assistant.

Your first task is to determine whether the given text
is a genuine Job Description.

A valid Job Description must:
1. Describe a real job role or position.
2. Contain job-related information.
3. Include at least one skill, responsibility,
   education requirement or experience requirement.

Invalid examples:
- General conversation.
- Greetings.
- Random text.
- Questions unrelated to hiring.
- Personal messages.
- "kya haal hai phle batao tum"
- Text that does not describe a job.

If the input is invalid:
- is_valid must be false.
- validation_reason must explain why it is invalid.
- Do not invent job information.

If the input is valid:
- is_valid must be true.
- Extract the actual job information.
- role should contain the job title.
- Extract skills and responsibilities.
- minimum_experience should be null if not mentioned.
- Do not invent information.

Return ONLY valid JSON matching this schema:

{JobD.model_json_schema()}
"""

    data = call_llm(
        system_prompt,
        f"Analyze this text:\n\n{job_description}"
    )

    job = JobD(**data)

    # Validate parsed result
    validate_job(job)

    return job


# ---------- Resume Parsing ----------

def parse_resume(resume_text: str) -> Resume:

    if not resume_text or len(resume_text.strip()) < 30:
        raise ValueError(
            "Resume content is too short or invalid."
        )

    system_prompt = f"""
You are an expert resume parser.

Extract information based on meaning, not exact headings.

Return ONLY valid JSON matching this schema:

{Resume.model_json_schema()}

Rules:
1. Do not invent information.
2. Missing value -> null.
3. Empty list if no information.
4. Include internships in experiences.
5. Extract skills from the entire resume.
6. For each experience, copy the duration text
   exactly as written in the resume.
7. total_experience_years: sum of all job and
   internship durations in years.
8. 6 months = 0.5 years.
9. If the resume shows only education and projects,
   return 0.
10. Use null only if you truly cannot tell.
"""

    data = call_llm(
        system_prompt,
        f"Parse this resume:\n\n{resume_text}"
    )

    return Resume(**data)


# ---------- Resume Scoring ----------

def score_resume(job: JobD, resume: Resume) -> MatchResult:

    # Never score against an invalid job description
    validate_job(job)

    system_prompt = f"""
You are an HR recruiter.

Compare the resume with the job description.

Return ONLY valid JSON matching this schema:

{MatchResult.model_json_schema()}

Scoring rules:
1. score must be a number from 0 to 100.
2. Keep verdict short.
3. Ignore experience duration when judging.
   Experience is checked separately.
4. Focus on skills match and project relevance.
5. Do not give a high score if relevant skills
   and projects are not demonstrated.
6. Never give 100 unless the resume strongly
   demonstrates all important job requirements.

Also fill these fields based ONLY on the
resume and job data given:

- summary: 2 sentences at most.
- experience_note: 2 sentences at most.
- education_note: 2 sentences at most.
- strengths: up to 4 short items.
- growth_areas: up to 4 short items.
- interview_focus: up to 3 short topics.

Never claim a skill is absent.
If a skill is not mentioned, call it
"not demonstrated".
"""

    user_prompt = f"""
JOB:
{job.model_dump_json(indent=2)}

RESUME:
{resume.model_dump_json(indent=2)}
"""

    data = call_llm(system_prompt, user_prompt)

    match = MatchResult(**data)

    # Validate score
    if not 0 <= match.score <= 100:
        raise ValueError(
            "AI returned an invalid resume score."
        )

    return match


# ---------- File Reading ----------

def read_pdf(file_path) -> str:

    try:
        reader = PdfReader(file_path)

        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if not result:
                    raise ValueError(
                        "PDF is password protected."
                    )
            except Exception:
                raise ValueError(
                    "PDF is password protected."
                )

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return text

    except ValueError:
        raise

    except Exception as e:
        raise ValueError(
            "Invalid or corrupted PDF file."
        ) from e


def read_docx(file_path) -> str:

    try:
        doc = Document(file_path)

        text = [
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        ]

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text.append(cell.text)

        return "\n".join(text)

    except Exception as e:
        raise ValueError(
            "Invalid or corrupted DOCX file."
        ) from e


def read_resume(file_path) -> str | None:

    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return read_pdf(file_path)

    if suffix == ".docx":
        return read_docx(file_path)

    raise ValueError(
        "Unsupported file format. Upload PDF or DOCX only."
    )


# ---------- Experience Calculation ----------

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec"
        ],
        1
    )
}


def _points(text: str):

    pts = []

    for m in re.finditer(
        r"(\d{1,2})/(\d{4})|([a-z]{3})[a-z]*\.?,?\s+(\d{4})",
        text
    ):

        if m.group(1):
            month = int(m.group(1))

            if 1 <= month <= 12:
                pts.append(
                    (int(m.group(2)), month)
                )

        elif m.group(3) in MONTHS:
            pts.append(
                (int(m.group(4)), MONTHS[m.group(3)])
            )

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

        elif any(
            w in d
            for w in ("present", "current", "now", "ongoing")
        ):
            today = date.today()
            ey, em = today.year, today.month

        else:
            return 0

        return max(
            0,
            (ey - sy) * 12 + (em - sm)
        )

    n = re.search(r"(\d+)\s*-?\s*month", d)

    if n:
        return int(n.group(1))

    y = re.search(
        r"(\d+(?:\.\d+)?)\s*-?\s*year",
        d
    )

    if y:
        return int(float(y.group(1)) * 12)

    return 0


def calc_experience_years(
    resume: Resume
) -> float | None:

    months = sum(
        months_from_duration(e.duration)
        for e in resume.experiences
    )

    if months > 0:
        return round(months / 12, 1)

    return resume.total_experience_years


def check_experience(
    job: JobD,
    resume: Resume
):

    required = job.minimum_experience

    if required is None:
        return True

    have = calc_experience_years(resume)

    if have is None:
        return None

    return have >= required


# ---------- Main Pipeline ----------

def process_candidates(
    job: JobD,
    resume_files: list[tuple[str, str]]
) -> dict:

    """
    resume_files = [(filename, saved_path), ...]
    """

    # Validate JD before processing any resume
    validate_job(job)

    results = []
    errors = []

    for filename, path in resume_files:

        try:

            # Read resume file
            text = read_resume(path)

            if not text or not text.strip():
                errors.append({
                    "file": filename,
                    "error": (
                        "Resume text extract nahi hua. "
                        "Scanned PDF ho sakti hai."
                    )
                })
                continue

            # Parse resume
            resume = parse_resume(text)

            # Score resume
            match = score_resume(job, resume)

            results.append({

                "file": filename,

                "name": (
                    resume.name or
                    match.candidate_name
                ),

                "email": resume.email,

                "score": match.score,

                "matching_skills": (
                    match.matching_skills
                ),

                "missing_skills": (
                    match.missing_skills
                ),

                "experience_met": (
                    check_experience(job, resume)
                ),

                "experience_years": (
                    calc_experience_years(resume)
                ),

                "verdict": match.verdict,

                "summary": match.summary,

                "experience_note": (
                    match.experience_note
                ),

                "education_note": (
                    match.education_note
                ),

                "strengths": match.strengths,

                "growth_areas": (
                    match.growth_areas
                ),

                "interview_focus": (
                    match.interview_focus
                ),
            })

        except ValueError as e:

            errors.append({
                "file": filename,
                "error": str(e)
            })

        except Exception:

            errors.append({
                "file": filename,
                "error": (
                    "Could not process this resume. "
                    "Please check the file and try again."
                )
            })

    # Highest score first
    results.sort(
        key=lambda r: r["score"],
        reverse=True
    )

    return {

        "total_processed": len(results),

        "top_2": results[:2],

        "lowest_2": (
            results[-2:]
            if len(results) > 2
            else []
        ),

        "all_results": results,

        "errors": errors,
    }
