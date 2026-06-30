from simple_rag.utils.text_helpers import format_list_natural_language


class TestFormatListNaturalLanguage:
    def test_empty_returns_empty_string(self):
        assert format_list_natural_language([]) == ""

    def test_none_returns_empty_string(self):
        assert format_list_natural_language(None) == ""

    def test_single_item(self):
        assert format_list_natural_language(["uno"]) == "uno"

    def test_two_items_uses_and(self):
        assert format_list_natural_language(["uno", "dos"]) == "uno and dos"

    def test_three_items_oxford_comma(self):
        assert format_list_natural_language(["uno", "dos", "tres"]) == "uno, dos, and tres"

    def test_four_items(self):
        result = format_list_natural_language(["a", "b", "c", "d"])
        assert result == "a, b, c, and d"

    def test_duplicates_are_removed(self):
        assert format_list_natural_language(["a", "b", "a"]) == "a and b"

    def test_none_values_in_list_are_skipped(self):
        assert format_list_natural_language([None, "a", None]) == "a"

    def test_non_string_items_are_coerced(self):
        result = format_list_natural_language([1, 2, 3])
        assert result == "1, 2, and 3"
