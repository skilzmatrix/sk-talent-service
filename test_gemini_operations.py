import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.gemini_operations import SchemaCandidateProfile, _extract_json_text, _generate_json


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


if __name__ == "__main__":
    unittest.main()
