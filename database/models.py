from __future__ import annotations

from sqlalchemy import Date, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Citizen(Base):
    __tablename__ = "citizen"

    member_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    enrollment_id: Mapped[str] = mapped_column(String(20), index=True)
    district_name_eng: Mapped[str] = mapped_column(String(80), index=True)
    is_rural: Mapped[int | None] = mapped_column(Integer, index=True)
    block_name_eng: Mapped[str | None] = mapped_column(String(80), index=True)
    city_name_eng: Mapped[str | None] = mapped_column(String(80), index=True)
    ward_name_eng: Mapped[str | None] = mapped_column(String(40), index=True)
    gp_name_eng: Mapped[str | None] = mapped_column(String(100), index=True)
    vill_name_eng: Mapped[str | None] = mapped_column(String(100), index=True)
    mem_type: Mapped[str | None] = mapped_column(String(20), index=True)
    relation_with_hof: Mapped[str | None] = mapped_column(String(40), index=True)
    name_en: Mapped[str] = mapped_column(String(120), index=True)
    father_name_en: Mapped[str | None] = mapped_column(String(120))
    mother_name_en: Mapped[str | None] = mapped_column(String(120))
    marital_status: Mapped[str | None] = mapped_column(String(32), index=True)
    spouce_name_en: Mapped[str | None] = mapped_column(String(120))
    dob: Mapped[Date | None] = mapped_column(Date)
    age: Mapped[int | None] = mapped_column(Integer, index=True)
    gender: Mapped[str] = mapped_column(String(16), index=True)
    caste_category: Mapped[str | None] = mapped_column(String(32), index=True)
    caste: Mapped[str | None] = mapped_column(String(180), index=True)
    bank: Mapped[str | None] = mapped_column(String(120), index=True)
    ifsc_code: Mapped[str | None] = mapped_column(String(16), index=True)
    account_no: Mapped[str | None] = mapped_column(String(32), index=True)
    mobile_no: Mapped[str | None] = mapped_column(String(16))
    income: Mapped[int | None] = mapped_column(Integer, index=True)
    occupation: Mapped[str | None] = mapped_column(String(80), index=True)
    minority: Mapped[str | None] = mapped_column(String(40), index=True)
    education: Mapped[str | None] = mapped_column(String(80), index=True)


Index("ix_citizen_geo", Citizen.district_name_eng, Citizen.block_name_eng, Citizen.gp_name_eng, Citizen.vill_name_eng)
Index("ix_citizen_demographics", Citizen.gender, Citizen.caste_category, Citizen.age)
Index("ix_citizen_enrollment", Citizen.enrollment_id)
