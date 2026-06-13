CREATE TABLE citizen (
    member_id INTEGER PRIMARY KEY,
    enrollment_id VARCHAR(20) NOT NULL,
    district_name_eng VARCHAR(80) NOT NULL,
    is_rural INTEGER,
    block_name_eng VARCHAR(80),
    city_name_eng VARCHAR(80),
    ward_name_eng VARCHAR(40),
    gp_name_eng VARCHAR(100),
    vill_name_eng VARCHAR(100),
    mem_type VARCHAR(20),
    relation_with_hof VARCHAR(40),
    name_en VARCHAR(120) NOT NULL,
    father_name_en VARCHAR(120),
    mother_name_en VARCHAR(120),
    marital_status VARCHAR(32),
    spouce_name_en VARCHAR(120),
    dob DATE,
    age INTEGER,
    gender VARCHAR(16) NOT NULL,
    caste_category VARCHAR(32),
    caste VARCHAR(180),
    bank VARCHAR(120),
    ifsc_code VARCHAR(16),
    account_no VARCHAR(32),
    mobile_no VARCHAR(16),
    income INTEGER,
    occupation VARCHAR(80),
    minority VARCHAR(40),
    education VARCHAR(80)
);

CREATE INDEX ix_citizen_geo ON citizen(district_name_eng, block_name_eng, gp_name_eng, vill_name_eng);
CREATE INDEX ix_citizen_demographics ON citizen(gender, caste_category, age);
CREATE INDEX ix_citizen_enrollment ON citizen(enrollment_id);
