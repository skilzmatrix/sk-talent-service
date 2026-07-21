import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.gemini_operations import (
    SchemaCandidateProfile,
    _extract_json_text,
    _generate_json,
    _resolve_num_questions,
)


class GeminiOperationsParseTests(unittest.TestCase):
    def test_extract_json_text_strips_code_fences(self) -> None:
        raw = "```json\n{\"full_name\": \"Jane Doe\", \"skills\": [\"Python\"]}\n```"

        self.assertEqual(
            _extract_json_text(raw),
            '{"full_name": "Jane Doe", "skills": ["Python"]}',
        )

    def test_extract_json_text_recovers_prefixed_json(self) -> None:
        raw = "Here is the parsed candidate profile:\n{\"full_name\": \"Jane Doe\"}"

        self.assertEqual(_extract_json_text(raw), '{"full_name": "Jane Doe"}')

    def test_generate_json_parses_cleaned_model_output(self) -> None:
        response = SimpleNamespace(text="```json\n{\"full_name\": \"Jane Doe\"}\n```")
        client = SimpleNamespace(models=SimpleNamespace(generate_content=MagicMock(return_value=response)))

        parsed = _generate_json(client, "prompt", {"type": "OBJECT"})

        self.assertEqual(parsed, {"full_name": "Jane Doe"})

    def test_candidate_profile_schema_requires_experience(self) -> None:
        self.assertIn("experience", SchemaCandidateProfile["required"])

    def test_resolve_num_questions_prefers_explicit_field(self) -> None:
        payload = {"numQuestions": "6"}

        self.assertEqual(_resolve_num_questions(payload), 6)

    def test_resolve_num_questions_supports_alternate_field_name(self) -> None:
        payload = {"questionCount": 7}

        self.assertEqual(_resolve_num_questions(payload), 7)

    def test_resolve_num_questions_parses_from_prompt_text(self) -> None:
        payload = {"prompt": "Please generate 5 interview questions for this candidate."}

        self.assertEqual(_resolve_num_questions(payload), 5)

    def test_resolve_num_questions_parses_number_words(self) -> None:
        payload = {"message": "Generate eight questions based on this resume"}

        self.assertEqual(_resolve_num_questions(payload), 8)

    def test_resolve_num_questions_defaults_to_ten(self) -> None:
        payload = {"jobDescription": "Need 7+ years of Python experience"}

        self.assertEqual(_resolve_num_questions(payload), 10)


if __name__ == "__main__":
    unittest.main()
