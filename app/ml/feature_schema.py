"""
app/ml/feature_schema.py — Canonical 61-feature contract for the new XGBoost models.

Source of truth: programmatically verified from feature_names_in_ of both
  xgb_criticality_regressor.joblib (XGBRegressor, 61 features)
  xgb_priority_classifier.joblib  (XGBClassifier, 61 features, classes [0, 1, 2])

Both models share the same feature list in the same order.
This file is the ONLY authoritative definition of the ML feature schema.
"""

# Exact feature order as stored in model.feature_names_in_
# Do NOT reorder these — XGBoost uses positional feature alignment.
ML_MODEL_FEATURES: list[str] = [
    "FIPS",
    "ZIP",
    "LON",
    "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE",
    "INCOME",
    "encounter_count",
    "encounter_type_count",
    "unique_encounter_count",
    "condition_count",
    "unique_condition_count",
    "medication_count",
    "unique_medication_count",
    "procedure_count",
    "unique_procedure_count",
    "careplan_count",
    "unique_careplan_count",
    "allergy_count",
    "device_count",
    "unique_device_count",
    "immunization_count",
    "unique_immunization_count",
    "claim_count",
    "unique_claim_diagnosis_count",
    "PREFIX_Mrs.",
    "PREFIX_Ms.",
    "PREFIX_Unknown",
    "SUFFIX_MD",
    "SUFFIX_PhD",
    "SUFFIX_Unknown",
    "MARITAL_M",
    "MARITAL_S",
    "MARITAL_Unknown",
    "MARITAL_W",
    "RACE_black",
    "RACE_hawaiian",
    "RACE_native",
    "RACE_other",
    "RACE_white",
    "ETHNICITY_nonhispanic",
    "GENDER_M",
    "COUNTY_Berkshire County",
    "COUNTY_Bristol County",
    "COUNTY_Dukes County",
    "COUNTY_Essex County",
    "COUNTY_Franklin County",
    "COUNTY_Hampden County",
    "COUNTY_Hampshire County",
    "COUNTY_Middlesex County",
    "COUNTY_Nantucket County",
    "COUNTY_Norfolk County",
    "COUNTY_Plymouth County",
    "COUNTY_Suffolk County",
    "COUNTY_Worcester County",
    "CITY_FREQUENCY",
    "age",
    "coverage_expense_ratio",
    "medication_per_encounter",
    "procedure_per_encounter",
    "condition_per_encounter",
    "claim_per_encounter",
]

# Priority class label mapping — confirmed from model.classes_ = [0, 1, 2]
PRIORITY_CLASS_LABELS: dict[int, str] = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}
