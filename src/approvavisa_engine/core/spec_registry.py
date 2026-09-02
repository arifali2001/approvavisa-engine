"""Spec registry: loads and queries country/document specifications."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from approvavisa_engine.models.specs import CountrySpec, CountrySummary, DocumentSpec


class BaseSpecRegistry(ABC):
    """Abstract base for spec registries. Override to load from a different source."""

    @abstractmethod
    def get_all(self) -> List[CountrySpec]:
        ...

    @abstractmethod
    def get_by_code(self, code: str) -> Optional[CountrySpec]:
        ...

    @abstractmethod
    def search(self, query: str) -> List[CountrySummary]:
        ...

    @abstractmethod
    def get_document_spec(
        self, country_code: str, document_type: str = "Passport"
    ) -> Optional[DocumentSpec]:
        ...


class JSONSpecRegistry(BaseSpecRegistry):
    """Loads specs from the vendored countries.json file."""

    def __init__(self, json_path: Optional[Path] = None) -> None:
        if json_path is None:
            json_path = Path(__file__).parent.parent / "data" / "countries.json"

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._countries: Dict[str, CountrySpec] = {}
        for entry in raw.get("countries", []):
            docs = []
            for d in entry.get("documents", []):
                docs.append(
                    DocumentSpec(
                        type=d.get("type", "Passport"),
                        width=d.get("width", 51),
                        height=d.get("height", 51),
                        unit=d.get("unit", "mm"),
                        width_inches=d.get("widthInches", ""),
                        bg_color=d.get("bgColor", "#FFFFFF"),
                        bg_description=d.get("bgDescription", "Plain white"),
                        head_size_percent=d.get("headSizePercent", "50-69%"),
                        dpi=d.get("dpi", 600),
                        file_format=d.get("fileFormat", "JPEG"),
                        max_file_size=d.get("maxFileSize", "10MB"),
                        source_url=d.get("sourceUrl", ""),
                        last_verified=d.get("lastVerified", ""),
                    )
                )

            spec = CountrySpec(
                code=entry["code"],
                name=entry["name"],
                flag=entry.get("flag", ""),
                slug=entry.get("slug", ""),
                documents=docs,
                popular=entry.get("popular", False),
            )
            self._countries[spec.code.upper()] = spec

    def get_all(self) -> List[CountrySpec]:
        return list(self._countries.values())

    def get_by_code(self, code: str) -> Optional[CountrySpec]:
        return self._countries.get(code.upper())

    def search(self, query: str) -> List[CountrySummary]:
        query_lower = query.lower()
        results = []
        for spec in self._countries.values():
            if query_lower in spec.name.lower() or query_lower in spec.code.lower():
                results.append(
                    CountrySummary(
                        code=spec.code,
                        name=spec.name,
                        flag=spec.flag,
                        document_types=[d.type for d in spec.documents],
                        popular=spec.popular,
                    )
                )
        return results

    def get_document_spec(
        self, country_code: str, document_type: str = "Passport"
    ) -> Optional[DocumentSpec]:
        country = self.get_by_code(country_code)
        if not country:
            return None
        for doc in country.documents:
            if doc.type.lower() == document_type.lower():
                return doc
        # Fallback to first document
        return country.documents[0] if country.documents else None
