import pytest
from medical_chatgpt.calculators import wells_criteria_pe, wells_criteria_dvt, PETwoTierRisk, PEThreeTierRisk, DVTTwoTierRisk, DVTThreeTierRisk

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
