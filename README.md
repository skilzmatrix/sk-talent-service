# Skillz Talent AI — Backend

FastAPI server that proxies Google Gemini AI and persists data to Supabase for the Skillz Talent AI frontend.

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | FastAPI |
| **AI** | Google Gemini 2.5 Flash (`google-genai`) |
| **Database** | Supabase (PostgreSQL) |
| **Validation** | Pydantic v2 |
| **Server** | Uvicorn |

## Project Structure

```
backend/
├── .env                  # Local environment variables (git-ignored)
├── .env.example          # Template for required env vars
├── requirements.txt      # Python dependencies
├── run.py                # Simple development entrypoint
└── app/
    ├── __init__.py
    ├── main.py                   # Thin FastAPI app factory / entrypoint
    ├── api/
    │   ├── router.py             # Top-level API router
    │   └── routes/
    │       ├── gemini.py         # Gemini proxy endpoint
    │       ├── health.py         # Health endpoint
    │       └── records.py        # Resume / job description / candidate endpoints
    ├── core/
    │   ├── config.py             # Env loading and app config helpers
    │   └── dependencies.py       # Shared route dependencies
    ├── schemas/
    │   ├── gemini.py             # Gemini request/response models
    │   └── records.py            # Persistence payload models
    ├── services/
    │   ├── gemini_service.py     # Gemini service wrapper
    │   └── persistence_service.py # Supabase service wrapper
    ├── gemini_operations.py      # Prompt templates and structured output schemas
    └── supabase_operations.py    # Low-level Supabase CRUD helpers
```

## Getting Started

### Prerequisites

- Python 3.12+
- A [Google Gemini API key](https://ai.google.dev/gemini-api/docs/api-key)
- (Optional) A [Supabase](https://supabase.com) project for data persistence

### Setup

1. **Create and activate a virtual environment**

   ```bash
   cd backend
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy the example and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `GEMINI_API_KEY` | Yes | Google Gemini API key |
   | `SUPABASE_URL` | No | Supabase project URL (enables persistence) |
   | `SUPABASE_SERVICE_ROLE_KEY` | No | Supabase service role key |
   | `CORS_ORIGINS` | No | Comma-separated allowed origins (defaults to `http://localhost:3000,http://127.0.0.1:3000`) |

   The server also reads `.env` and `.env.local` from the repo root, so you can keep keys there instead.

4. **Start the server**

   ```bash
   python run.py
   ```

   The API will be available at **http://127.0.0.1:8000**.

   If you want to start it from the repo root instead:

   ```bash
   npm run dev:backend
   ```

## API Reference

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |

### Gemini AI

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/gemini` | Run an AI operation via Gemini |

**Request body:**

```json
{
  "operation": "generateJobDescription",
  "payload": { "title": "Software Engineer", "responsibilities": "Build APIs, write tests" }
}
```

**Response:**

```json
{
  "error": null,
  "kind": "text",
  "value": "# Software Engineer\n\n..."
}
```

`kind` is either `"text"` (Markdown string) or `"json"` (structured object).

#### Available Operations

| Operation | Returns | Description |
|-----------|---------|-------------|
| `generateJobDescription` | text | Generate a job description from title and responsibilities |
| `optimizeJobAd` | text | Analyze and optimize a job ad for clarity, DEI, and engagement |
| `analyzeResume` | json | Parse a resume and extract structured data; optionally score against a JD |
| `createCandidateProfile` | json | Extract candidate profile fields from resume text |
| `compareCandidates` | text | Compare two candidates side-by-side against a job description |

### Supabase Persistence

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/resumes` | Save a parsed resume |
| `GET` | `/api/resumes` | List all saved resumes (newest first) |
| `POST` | `/api/job-descriptions` | Save a generated job description |
| `GET` | `/api/job-descriptions` | List all saved job descriptions |
| `POST` | `/api/candidates` | Save a candidate profile |
| `GET` | `/api/candidates` | List all saved candidates |

Supabase endpoints return `503` if the Supabase environment variables are not configured.

### Interactive Docs

Once the server is running, visit:

- **Swagger UI** — [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc** — [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
