from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy import text
from database.connection import get_engine, get_session
from database.models import Citizen


REQUIRED_COLUMNS = {
    "DISTRICT_NAME_ENG",
    "ENROLLMENT_ID",
    "MEMBER_ID",
    "NAME_EN",
    "AGE",
    "GENDER",
}

DISTRICT_VALUE_NORMALIZATION = {
    "Balotara": "Balotra",
}


@dataclass(frozen=True)
class DatasetImportReport:
    source_name: str
    rows_loaded: int


def clean_caste(val) -> str | None:
    if val is None or pd.isna(val):
        return None
    cleaned = str(val).strip()
    # Strip leading numbers followed by optional spaces
    cleaned = re.sub(r"^\d+\s*", "", cleaned)
    # Title case to standardize mixed case entries in English
    return cleaned.title() if cleaned else None


def import_excel_dataset(
    source: str | Path | BinaryIO = "dummy_dataset/Dummy_Data_Set.xlsx",
    source_name: str | None = None,
    database_url: str | None = None,
) -> DatasetImportReport:
    # 1. Read dataset
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        if not source_path.is_absolute():
            # Resolve relative to project root
            from config.settings import PROJECT_ROOT
            source_path = PROJECT_ROOT / source_path
        
        if str(source_path).lower().endswith(".csv"):
            data = pd.read_csv(source_path)
        else:
            data = pd.read_excel(source_path)
    else:
        data = pd.read_excel(source)

    # Clean up entirely empty or metadata rows
    data = data.dropna(how="all")
    if data.empty:
        raise ValueError("The provided dataset is empty.")

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}.")

    # Clean row values to dictionary
    rows = data.where(pd.notna(data), None).to_dict(orient="records")
    
    citizens: list[Citizen] = []

    for row in rows:
        m_id = _integer(row.get("MEMBER_ID"))
        enrollment_id = _text(row.get("ENROLLMENT_ID"))
        if m_id is None or not enrollment_id:
            continue  # Skip rows missing crucial identifiers
        
        district = _text(row.get("DISTRICT_NAME_ENG"))
        if district:
            district = DISTRICT_VALUE_NORMALIZATION.get(district, district)
        
        # Gender canonicalization
        gender_raw = _text(row.get("GENDER"))
        gender = gender_raw.title() if gender_raw else "Unknown"

        citizens.append(
            Citizen(
                member_id=m_id,
                enrollment_id=enrollment_id,
                district_name_eng=district or "Unknown",
                is_rural=_integer(row.get("IS_RURAL")),
                block_name_eng=_text(row.get("BLOCK_NAME_ENG")),
                city_name_eng=_text(row.get("CITY_NAME_ENG")),
                ward_name_eng=_text(row.get("WARD_NAME_ENG")),
                gp_name_eng=_text(row.get("GP_NAME_ENG")),
                vill_name_eng=_text(row.get("VILL_NAME_ENG")),
                mem_type=_text(row.get("MEM_TYPE")),
                relation_with_hof=_text(row.get("RELATION_WITH_HOF")),
                name_en=str(row.get("NAME_EN")).strip(),
                father_name_en=_text(row.get("FATHER_NAME_EN")),
                mother_name_en=_text(row.get("MOTHER_NAME_EN")),
                marital_status=_text(row.get("MARITAL_STATUS")),
                spouce_name_en=_text(row.get("SPOUCE_NAME_EN")),
                dob=_date(row.get("DOB")),
                age=_integer(row.get("AGE")),
                gender=gender,
                caste_category=_text(row.get("CASTE_CATEGORY")),
                caste=clean_caste(row.get("CASTE")),
                bank=_text(row.get("BANK")),
                ifsc_code=_text(row.get("IFSC_CODE")),
                account_no=_text(row.get("ACCOUNT_NO")),
                mobile_no=_text(row.get("MOBILE_NO")),
                income=_integer(row.get("INCOME")),
                occupation=_text(row.get("OCCUPATION")),
                minority=_text(row.get("MINORITY")),
                education=_text(row.get("EDUCATION")),
            )
        )

    # 4. Truncate target table non-destructively
    engine = get_engine(database_url)
    from database.models import Base
    Base.metadata.create_all(engine)
    dialect_name = engine.dialect.name
    
    with engine.begin() as conn:
        if dialect_name == "sqlite":
            conn.execute(text("PRAGMA foreign_keys = OFF;"))
            conn.execute(text("DELETE FROM citizen;"))
            conn.execute(text("PRAGMA foreign_keys = ON;"))
        else:
            conn.execute(text("TRUNCATE TABLE citizen RESTART IDENTITY CASCADE;"))

    # 5. Bulk ingest data
    with get_session(database_url) as session:
        session.add_all(citizens)
        session.commit()

    resolved_name = source_name or getattr(source, "name", None) or str(source)
    return DatasetImportReport(
        source_name=Path(str(resolved_name)).name,
        rows_loaded=len(citizens),
    )


def _text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


def _integer(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _date(value):
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.date()
