"""Tests for the spec registry."""

import pytest


class TestJSONSpecRegistry:
    def test_loads_countries(self, spec_registry):
        """Should load all countries from the vendored JSON."""
        all_specs = spec_registry.get_all()
        assert len(all_specs) > 40  # We have 54 countries

    def test_get_by_code_us(self, spec_registry):
        """Should find US spec by code."""
        spec = spec_registry.get_by_code("US")
        assert spec is not None
        assert spec.code == "US"
        assert spec.name == "United States"
        assert len(spec.documents) >= 1

    def test_get_by_code_case_insensitive(self, spec_registry):
        """Should handle case-insensitive lookup."""
        spec = spec_registry.get_by_code("gb")
        assert spec is not None
        assert spec.code == "GB"

    def test_get_by_code_missing(self, spec_registry):
        """Should return None for unknown country."""
        spec = spec_registry.get_by_code("ZZZ")
        assert spec is None

    def test_search(self, spec_registry):
        """Should find countries by name search."""
        results = spec_registry.search("united")
        assert len(results) >= 2  # US + UK at minimum
        names = [r.name for r in results]
        assert "United States" in names

    def test_get_document_spec(self, spec_registry):
        """Should return document spec for passport."""
        doc = spec_registry.get_document_spec("US", "Passport")
        assert doc is not None
        assert doc.type == "Passport"
        assert doc.width == 51
        assert doc.height == 51
        assert doc.dpi == 600
        assert doc.bg_color == "#FFFFFF"
