import pytest
import math
from clinical_calculator_tooluse.calculators import (
    wells_criteria_pe, wells_criteria_dvt,
    PETwoTierRisk, PEThreeTierRisk,
    DVTTwoTierRisk, DVTThreeTierRisk,
    das28_esr, das28_crp, DAS28DiseaseActivity, cdai,
    chads2_vasc, CHADSVASCRecommendation
)

def test_wells_criteria_pe_low_risk():
    result = wells_criteria_pe()
    assert result.score == 0.0
    assert result.two_tier_risk == PETwoTierRisk.UNLIKELY
    assert result.three_tier_risk == PEThreeTierRisk.LOW

def test_wells_criteria_pe_high_risk():
    result = wells_criteria_pe(
        clinical_signs_dvt = True,
        alt_diagnosis_less_likely = True,
        active_malignancy = True
    )
    assert result.score == 7.0
    assert result.two_tier_risk == PETwoTierRisk.LIKELY
    assert result.three_tier_risk == PEThreeTierRisk.HIGH

def test_wells_criteria_pe_moderate_risk():
    result = wells_criteria_pe(
        heart_rate_over_100 = True,
        hemoptysis = True
    )
    assert result.score == 2.5
    assert result.two_tier_risk == PETwoTierRisk.UNLIKELY
    assert result.three_tier_risk == PEThreeTierRisk.MODERATE

def test_wells_criteria_dvt_unlikely():
    result = wells_criteria_dvt(
        active_cancer = True,
        alt_diagnosis_as_likely = True
    )
    assert result.score == -1
    assert result.two_tier_risk == DVTTwoTierRisk.UNLIKELY
    assert result.three_tier_risk == DVTThreeTierRisk.LOW

def test_wells_criteria_dvt_likely():
    result = wells_criteria_dvt(
        active_cancer = True,
        entire_leg_swollen = True,
        previous_dvt = True
    )
    assert result.score == 3
    assert result.two_tier_risk == DVTTwoTierRisk.LIKELY
    assert result.three_tier_risk == DVTThreeTierRisk.HIGH

def test_wells_criteria_dvt_moderate():
    result = wells_criteria_dvt(
        active_cancer = True,
    )
    assert result.score == 1
    assert result.two_tier_risk == DVTTwoTierRisk.UNLIKELY
    assert result.three_tier_risk == DVTThreeTierRisk.MODERATE

def test_das28_esr():
    result = das28_esr(tender_count = 8, swollen_count = 4, esr = 30, global_health = 50)
    assert math.isclose(result.score, 5.22, abs_tol = 0.01)
    assert result.disease_activity == DAS28DiseaseActivity.HIGH

def test_das28_crp():
    result = das28_crp(tender_count = 8, swollen_count = 4, crp = 10, global_health = 50)
    assert math.isclose(result.score, 4.67, abs_tol = 0.01)
    assert result.disease_activity == DAS28DiseaseActivity.MODERATE

def test_cdai():
    result = cdai(tender_count = 4, swollen_count = 2, patient_global = 3.5, provider_global = 2.5)
    assert result.score == 12.0
    assert result.disease_activity == DAS28DiseaseActivity.MODERATE

def test_chads2_vasc_male_low_risk():
    result = chads2_vasc()
    assert result.score == 0
    assert result.recommendation == CHADSVASCRecommendation.NONE

def test_chads2_vasc_female_low_risk():
    result = chads2_vasc(female_sex = True)
    assert result.score == 1
    assert result.recommendation == CHADSVASCRecommendation.NONE

def test_chads2_vasc_male_consider():
    result = chads2_vasc(hypertension = True)
    assert result.score == 1
    assert result.recommendation == CHADSVASCRecommendation.CONSIDER

def test_chads2_vasc_female_consider():
    result = chads2_vasc(female_sex = True, hypertension = True)
    assert result.score == 2
    assert result.recommendation == CHADSVASCRecommendation.CONSIDER

def test_chads2_vasc_high_risk():
    result = chads2_vasc(
        stroke_or_tia = True,
        age_75_or_older = True,
        diabetes = True
    )
    assert result.score == 5
    assert result.recommendation == CHADSVASCRecommendation.RECOMMENDED
