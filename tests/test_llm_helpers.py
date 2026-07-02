"""Tests unitarios para llm_helpers.py — solo funciones puras, sin llamadas a APIs."""

import json

from simple_rag.utils.llm_helpers import strip_json_fences


class TestStripJsonFences:
    def test_plain_json_object(self):
        assert strip_json_fences('{"score": 5, "reason": "ok"}') == '{"score": 5, "reason": "ok"}'

    def test_plain_json_array(self):
        assert strip_json_fences('["a", "b"]') == '["a", "b"]'

    def test_markdown_fenced_json(self):
        raw = '```json\n{"score": 3, "reason": "partial"}\n```'
        assert json.loads(strip_json_fences(raw)) == {"score": 3, "reason": "partial"}

    def test_markdown_fenced_without_language_tag(self):
        raw = '```\n{"score": 3}\n```'
        assert json.loads(strip_json_fences(raw)) == {"score": 3}

    def test_nested_object(self):
        raw = '{"topics": ["yoga"], "filter_mode": "exact", "meta": {"confidence": 0.9}}'
        assert json.loads(strip_json_fences(raw)) == {
            "topics": ["yoga"],
            "filter_mode": "exact",
            "meta": {"confidence": 0.9},
        }

    def test_brace_inside_string_value_is_ignored(self):
        raw = '{"score": 4, "reason": "uses a brace like } in text"}'
        assert json.loads(strip_json_fences(raw)) == {
            "score": 4,
            "reason": "uses a brace like } in text",
        }

    def test_multiple_json_objects_only_first_is_returned(self):
        # Test de regresión: el judge de faithfulness a veces emite dos objetos
        # JSON uno después del otro; un match greedy de primer-a-último brace
        # concatenaría ambos y fallaría el parsing con "Extra data".
        raw = '{"score": 5, "reason": "first"}\n{"score": 5, "reason": "second"}'
        assert json.loads(strip_json_fences(raw)) == {"score": 5, "reason": "first"}

    def test_leading_and_trailing_prose_is_stripped(self):
        raw = 'Here is the result:\n{"score": 2, "reason": "weak"}\nThank you.'
        assert json.loads(strip_json_fences(raw)) == {"score": 2, "reason": "weak"}

    def test_no_json_returns_stripped_text(self):
        assert strip_json_fences("  no json here  ") == "no json here"

    def test_unterminated_json_returns_from_start(self):
        raw = '{"score": 5, "reason": "truncated'
        assert strip_json_fences(raw) == raw
