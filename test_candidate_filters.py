import unittest
from unittest.mock import patch

from app.api.routes import records
from app import pinecone_operations
from app import supabase_operations
from app.schemas.records import CandidateProfileUpdate


class CandidateFiltersRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_build_candidate_filters_normalizes_values(self) -> None:
        filters = records._build_candidate_filters(
            work_authorization="  USC  ",
            experience=" 5 years ",
            location="   ",
            city=" Austin ",
            state=" TX ",
            preferred_location="New York",
            skills=["Python, FastAPI", "  SQL  ", ""],
        )

        self.assertEqual(
            filters,
            {
                "work_authorization": "USC",
                "experience": "5 years",
                "city": "Austin",
                "state": "TX",
                "preferred_location": "New York",
                "skills": ["Python", "FastAPI", "SQL"],
            },
        )

    async def test_get_candidates_passes_structured_filters(self) -> None:
        captured: dict[str, object] = {}

        async def fake_run_storage_call(fn, *args):
            captured["fn_name"] = fn.__name__
            captured["args"] = args
            return {
                "items": [],
                "page": 2,
                "page_size": 10,
                "total_items": 0,
                "total_pages": 0,
                "has_next": False,
                "has_previous": True,
            }

        with patch("app.api.routes.records._run_storage_call", new=fake_run_storage_call):
            response = await records.get_candidates(
                page=2,
                page_size=10,
                q="backend",
                work_authorization="H1B",
                experience="5 years",
                location="Austin",
                city="Austin",
                state="TX",
                linkedin_profile="linkedin.com/in/test",
                domain_industry="Fintech",
                preferred_location="Remote",
                open_to_relocation="Yes",
                expected_salary="150000",
                employment_type="Full-time",
                skills=["Python, FastAPI", "SQL"],
            )

        self.assertEqual(captured["fn_name"], "get_candidates_paginated")
        self.assertEqual(
            captured["args"],
            (
                2,
                10,
                "backend",
                {
                    "work_authorization": "H1B",
                    "experience": "5 years",
                    "location": "Austin",
                    "city": "Austin",
                    "state": "TX",
                    "linkedin_profile": "linkedin.com/in/test",
                    "domain_industry": "Fintech",
                    "preferred_location": "Remote",
                    "open_to_relocation": "Yes",
                    "expected_salary": "150000",
                    "employment_type": "Full-time",
                    "skills": ["Python", "FastAPI", "SQL"],
                },
            ),
        )
        self.assertEqual(response.page, 2)
        self.assertEqual(response.page_size, 10)
        self.assertTrue(response.has_previous)


class CandidateSkillsFilterTests(unittest.TestCase):
    def test_build_ilike_pattern_uses_postgrest_wildcards(self) -> None:
        self.assertEqual(supabase_operations._build_ilike_pattern("Austin"), "*Austin*")

    def test_normalize_linkedin_filter_strips_protocol_and_www(self) -> None:
        normalized = supabase_operations._normalize_linkedin_filter(
            "https://www.linkedin.com/in/jane-doe/"
        )

        self.assertEqual(normalized, "linkedin.com/in/jane-doe")

    def test_normalize_skill_filters_lowers_and_trims(self) -> None:
        normalized = supabase_operations._normalize_skill_filters(
            {"skills": [" Python ", "SQL", ""]}
        )

        self.assertEqual(normalized, ["python", "sql"])

    def test_candidate_matches_skills_supports_partial_case_insensitive_match(self) -> None:
        candidate = {"skills": ["Python", "FastAPI", "Data Analysis"]}

        self.assertTrue(supabase_operations._candidate_matches_skills(candidate, ["pyth"]))
        self.assertTrue(supabase_operations._candidate_matches_skills(candidate, ["fast"]))
        self.assertTrue(supabase_operations._candidate_matches_skills(candidate, ["analysis"]))
        self.assertFalse(supabase_operations._candidate_matches_skills(candidate, ["java"]))

    def test_normalize_location_fields_infers_city_and_state(self) -> None:
        location, city, state = supabase_operations._normalize_location_fields(
            {"location": "Austin, TX"}
        )

        self.assertEqual(location, "Austin, TX")
        self.assertEqual(city, "Austin")
        self.assertEqual(state, "TX")

    def test_normalize_location_fields_builds_location_from_city_state(self) -> None:
        location, city, state = supabase_operations._normalize_location_fields(
            {"city": "Dallas", "state": "TX"}
        )

        self.assertEqual(location, "Dallas, TX")
        self.assertEqual(city, "Dallas")
        self.assertEqual(state, "TX")

    def test_normalize_resume_storage_path_from_full_public_url(self) -> None:
        storage_key = supabase_operations._normalize_resume_storage_path(
            "https://demo.supabase.co/storage/v1/object/public/candidate_resumes/resumes/a1b2c3.pdf"
        )

        self.assertEqual(storage_key, "resumes/a1b2c3.pdf")

    def test_normalize_resume_storage_path_from_signed_url_with_query(self) -> None:
        storage_key = supabase_operations._normalize_resume_storage_path(
            "https://demo.supabase.co/storage/v1/object/sign/candidate_resumes/resumes/sample.docx?token=abc&download=1"
        )

        self.assertEqual(storage_key, "resumes/sample.docx")

    def test_normalize_resume_storage_path_from_bucket_prefixed_path(self) -> None:
        storage_key = supabase_operations._normalize_resume_storage_path(
            "candidate_resumes/resumes/sample.pdf"
        )

        self.assertEqual(storage_key, "resumes/sample.pdf")


class TalentSearchFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_talent_search_passes_metadata_filters_to_service(self) -> None:
        captured: dict[str, object] = {}

        def fake_run_talent_search(
            query: str,
            top_k: int,
            keyword_weight: float | None,
            gemini_client,
            metadata_filters: dict[str, object] | None,
            *,
            use_llm_rerank: bool = True,
        ):
            captured["query"] = query
            captured["top_k"] = top_k
            captured["keyword_weight"] = keyword_weight
            captured["metadata_filters"] = metadata_filters
            captured["use_llm_rerank"] = use_llm_rerank
            return [], {"semantic": 0.56, "keyword": 0.44}, {"llm_rerank": "disabled"}

        with patch(
            "app.api.routes.records.talent_search_service.run_talent_search",
            new=fake_run_talent_search,
        ):
            response = await records.talent_search(
                records.TalentSearchRequest(
                    query="Python backend",
                    top_k=3,
                    use_llm_rerank=False,
                    location="Austin",
                    work_authorization="H1B",
                    experience="5 years",
                    city="Austin",
                    state="TX",
                    skills=["Python", "FastAPI"],
                )
            )

        self.assertEqual(captured["query"], "Python backend")
        self.assertEqual(captured["top_k"], 3)
        self.assertEqual(captured["use_llm_rerank"], False)
        self.assertEqual(
            captured["metadata_filters"],
            {
                "work_authorization": "H1B",
                "experience": "5 years",
                "location": "Austin",
                "city": "Austin",
                "state": "TX",
            },
        )
        self.assertEqual(response["llm_rerank"], "disabled")

    def test_build_pinecone_metadata_filter_uses_normalized_ci_fields(self) -> None:
        out = pinecone_operations._build_pinecone_metadata_filter(
            {
                "location": "  Austin, TX  ",
                "work_authorization": "H1-B",
                "experience": "5 years",
                "full_name": "  Jane Doe  ",
                "job_role": "Backend Engineer",
                "skills": "Python",
            }
        )

        self.assertEqual(
            out,
            {
                "$and": [
                    {"location_ci": {"$eq": "austin, tx"}},
                    {"work_authorization_ci": {"$eq": "h1-b"}},
                    {"experience_ci": {"$eq": "5 years"}},
                    {
                        "$or": [
                            {"full_name_ci": {"$eq": "jane doe"}},
                            {"full_name": {"$eq": "Jane Doe"}},
                            {"full_name": {"$eq": "jane doe"}},
                        ]
                    },
                    {
                        "$or": [
                            {"job_role_ci": {"$eq": "backend engineer"}},
                            {"job_role": {"$eq": "Backend Engineer"}},
                            {"job_role": {"$eq": "backend engineer"}},
                        ]
                    },
                ]
            },
        )


class CandidateUpdateTests(unittest.TestCase):
    def test_candidate_profile_update_accepts_csv_skills(self) -> None:
        update = CandidateProfileUpdate.model_validate(
            {
                "skills": "Python, FastAPI, SQL",
            }
        )

        self.assertEqual(update.skills, ["Python", "FastAPI", "SQL"])

    def test_update_candidate_merges_with_existing_record(self) -> None:
        existing = {
            "id": "candidate-123",
            "full_name": "JAINAM SHAH",
            "job_role": "Backend Engineer",
            "email": "jainam@example.com",
            "phone": "555-0100",
            "location": "Austin, TX",
            "city": "Austin",
            "state": "TX",
            "linkedin_profile": "https://linkedin.com/in/jainam",
            "domain_industry": "Fintech",
            "work_authorization": "USC",
            "experience": "5 years",
            "preferred_location": "Remote",
            "open_to_relocation": "Yes",
            "expected_salary": "150000",
            "employment_type": "Full-time",
            "summary": "Experienced backend engineer.",
            "skills": ["Python", "FastAPI"],
            "experiences": [{"title": "Engineer"}],
            "projects": [{"name": "Payments"}],
            "certifications": [{"name": "AWS"}],
            "resume_url": "resumes/jainam.pdf",
        }

        payload = {"summary": "Updated summary", "skills": ["Python", "PostgreSQL"]}

        fake_update_result = {
            **existing,
            "summary": "Updated summary",
            "skills": ["Python", "PostgreSQL"],
        }

        update_query = unittest.mock.MagicMock()
        update_query.eq.return_value = update_query
        update_query.execute.return_value = unittest.mock.MagicMock(data=[fake_update_result])

        table = unittest.mock.MagicMock()
        table.update.return_value = update_query

        client = unittest.mock.MagicMock()
        client.table.return_value = table

        with patch("app.supabase_operations._client", return_value=client), patch(
            "app.supabase_operations.get_candidate_by_id", return_value=existing
        ):
            result = supabase_operations.update_candidate("candidate-123", payload)

        table.update.assert_called_once()
        updated_payload = table.update.call_args.args[0]
        self.assertEqual(updated_payload["full_name"], "JAINAM SHAH")
        self.assertEqual(updated_payload["experiences"], [{"title": "Engineer"}])
        self.assertEqual(updated_payload["projects"], [{"name": "Payments"}])
        self.assertEqual(updated_payload["certifications"], [{"name": "AWS"}])
        self.assertEqual(updated_payload["resume_url"], "resumes/jainam.pdf")
        self.assertEqual(result["summary"], "Updated summary")
        self.assertEqual(result["skills"], ["Python", "PostgreSQL"])


if __name__ == "__main__":
    unittest.main()
