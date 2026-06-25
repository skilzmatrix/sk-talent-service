import unittest
from unittest.mock import patch

from app.api.routes import records
from app import supabase_operations


class CandidateFiltersRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_build_candidate_filters_normalizes_values(self) -> None:
        filters = records._build_candidate_filters(
            work_authorization="  USC  ",
            location="   ",
            preferred_location="New York",
            skills=["Python, FastAPI", "  SQL  ", ""],
        )

        self.assertEqual(
            filters,
            {
                "work_authorization": "USC",
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
                location="Austin",
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
                    "location": "Austin",
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


if __name__ == "__main__":
    unittest.main()