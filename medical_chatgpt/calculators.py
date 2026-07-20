from math import sqrt, log
from enum import Enum
from typing import NamedTuple

# enums

class PETwoTierRisk(str, Enum):
    LIKELY = "PE Likely"
    UNLIKELY = "PE Unlikely"

class PEThreeTierRisk(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"

class DVTTwoTierRisk(str, Enum):
    LIKELY = "DVT Likely"
    UNLIKELY = "DVT Unlikely"

class DVTThreeTierRisk(str, Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"

class DAS28DiseaseActivity(str, Enum):
    REMISSION = "Remission"
    LOW = "Low Disease Activity"
    MODERATE = "Moderate Disease Activity"
    HIGH = "High Disease Activity"

# return types

class PEResult(NamedTuple):
    score: float
    two_tier_risk: PETwoTierRisk
    three_tier_risk: PEThreeTierRisk

class DVTResult(NamedTuple):
    score: float
    two_tier_risk: DVTTwoTierRisk
    three_tier_risk: DVTThreeTierRisk

class DAS28Result(NamedTuple):
    score: float
    disease_activity: DAS28DiseaseActivity

# weights

PE_WEIGHTS = dict(
    clinical_signs_dvt = 3.0,
    alt_diagnosis_less_likely = 3.0,
    heart_rate_over_100 = 1.5,
    immobilization_or_surgery = 1.5,
    previous_dvt_or_pe = 1.5,
    hemoptysis = 1.0,
    active_malignancy = 1.0
)

DVT_WEIGHTS = dict(
    active_cancer = 1,
    paralysis_paresis_or_immobilization = 1,
    bedridden_or_surgery = 1,
    calf_swelling_gt_3cm = 1,
    collateral_superficial_veins = 1,
    entire_leg_swollen = 1,
    localized_tenderness = 1,
    pitting_edema = 1,
    previous_dvt = 1,
    alt_diagnosis_as_likely = -2
)

# functions

def wells_criteria_pe(**kwargs) -> PEResult:
    # pulmonary embolus

    score = sum(PE_WEIGHTS[k] for k, v in kwargs.items() if v and k in PE_WEIGHTS)

    two_tier = PETwoTierRisk.LIKELY if score > 4.0 else PETwoTierRisk.UNLIKELY

    three_tier = (
        PEThreeTierRisk.LOW if score < 2.0 else
        PEThreeTierRisk.MODERATE if score <= 6.0 else
        PEThreeTierRisk.HIGH
    )

    return PEResult(score, two_tier, three_tier)

def wells_criteria_dvt(**kwargs) -> DVTResult:
    # deep vein thrombosis

    score = sum(DVT_WEIGHTS[k] for k, v in kwargs.items() if v and k in DVT_WEIGHTS)

    two_tier = DVTTwoTierRisk.LIKELY if score >= 2 else DVTTwoTierRisk.UNLIKELY

    three_tier = (
        DVTThreeTierRisk.LOW if score <= 0 else
        DVTThreeTierRisk.MODERATE if score <= 2 else
        DVTThreeTierRisk.HIGH
    )

    return DVTResult(score, two_tier, three_tier)

# rheumatoid arthritis

def das28_activity(score: float) -> DAS28DiseaseActivity:
    return (
        DAS28DiseaseActivity.REMISSION if score <= 2.6 else
        DAS28DiseaseActivity.LOW if score <= 3.2 else
        DAS28DiseaseActivity.MODERATE if score <= 5.1 else
        DAS28DiseaseActivity.HIGH
    )

def das28_esr(tender_count: int, swollen_count: int, esr: float, global_health: float) -> DAS28Result:
    # esr
    score = (
        0.56 * sqrt(tender_count) +
        0.28 * sqrt(swollen_count) +
        0.70 * log(esr) +
        0.014 * global_health
    )
    return DAS28Result(score, das28_activity(score))

def das28_crp(tender_count: int, swollen_count: int, crp: float, global_health: float) -> DAS28Result:
    # crp
    score = (
        0.56 * sqrt(tender_count) +
        0.28 * sqrt(swollen_count) +
        0.36 * log(crp + 1) +
        0.014 * global_health +
        0.96
    )
    return DAS28Result(score, das28_activity(score))
