"""
Mappings for the data transformations.
"""

FEATURE_NAME_MAPPING={
    "id": "id",
    "Age": "age",
    "Sex": "sex",
    "Chest pain type": "chest_pain_type",
    "BP": "blood_pressure",
    "Cholesterol": "cholesterol",
    "FBS over 120": "fbs_over_120",
    "EKG results": "ekg_results",
    "Max HR": "max_hr",
    "Exercise angina": "exercise_angina",
    "ST depression": "st_depression",
    "Slope of ST": "slope_st",
    "Number of vessels fluro": "number_vessels_fluro",
    "Thallium": "thallium",
    "Heart Disease": "heart_disease"
}

SEX_CAT_MAPPING={
    0: "female",
    1: "male"
}

CHEST_PAIN_CAT_MAPPING={
    1: "Typical angina",
    2: "Atypical angina",
    3: "Non-anginal pain",
    4: "Asymptomatic"
}

BLOOD_SUGAR_CAT_MAPPING={
    0: "no",
    1: "yes"
}

EKG_RESULT_CAT_MAPPING={
    0: "Normal",
    1: "ST-T wave abnormality",
    2: "Left ventricular hypertrophy"
}

EXERCISE_ANGINA_CAT_MAPPING={
    0: "no",
    1: "yes"
}

THALLIUM_TEST_CAT_MAPPING={
    3: "normal",
    6: "fixed effect",
    7: "reversible defect"
}

FEATURE_CAT_MAPPING={
    "sex": SEX_CAT_MAPPING,
    "chest_pain_type": CHEST_PAIN_CAT_MAPPING,
    "fbs_over_120": BLOOD_SUGAR_CAT_MAPPING,
    "ekg_results": EKG_RESULT_CAT_MAPPING,
    "exercise_angina": EXERCISE_ANGINA_CAT_MAPPING,
    "thallium": THALLIUM_TEST_CAT_MAPPING
}
