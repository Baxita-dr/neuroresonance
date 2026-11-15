"""
Unit tests for BiometricProfile helper calculations.

Validates:
- Age- and wavelength-dependent approximate DPF calculation
- Fixed pacing reference
- Absence of unsupported PLV thresholds and age-based delay estimates
- Anthropometric blood volume (Nadler) and BMI equations
- Input validation and dataclass immutability
"""

import pytest
from src.domain.biometrics import BiometricProfile


def test_dpf_increases_with_age():
    young = BiometricProfile(age=20, height_cm=175.0, weight_kg=70.0, sex="M")
    elderly = BiometricProfile(age=70, height_cm=175.0, weight_kg=70.0, sex="M")

    dpf_young = young.compute_dpf(wavelength_nm=780)
    dpf_elderly = elderly.compute_dpf(wavelength_nm=780)

    # The implemented DPF approximation increases over this age comparison.
    assert dpf_young < dpf_elderly
    assert 5.0 <= dpf_young <= 7.0
    assert 6.0 <= dpf_elderly <= 8.5


def test_dpf_wavelength_dependency():
    profile = BiometricProfile(age=25, height_cm=180.0, weight_kg=75.0, sex="M")
    dpf_780 = profile.compute_dpf(wavelength_nm=780)
    dpf_850 = profile.compute_dpf(wavelength_nm=850)

    # The implementation returns wavelength-dependent approximate values.
    assert dpf_850 < dpf_780


def test_pacing_reference_does_not_depend_on_height():
    short_person = BiometricProfile(age=25, height_cm=155.0, weight_kg=55.0, sex="F")
    tall_person = BiometricProfile(age=25, height_cm=195.0, weight_kg=90.0, sex="M")

    freq_short = short_person.pacing_reference_hz()
    freq_tall = tall_person.pacing_reference_hz()

    # The fixed demonstration target does not vary with height or weight.
    assert freq_short == freq_tall == 0.1


def test_fixed_pacing_rate():
    for height in [150.0, 175.0, 200.0]:
        profile = BiometricProfile(age=20, height_cm=height, weight_kg=70.0, sex="M")
        bpm = profile.pacing_rate_bpm
        assert 4.5 <= bpm <= 7.5


def test_plv_thresholds_are_not_defined():
    young = BiometricProfile(age=22, height_cm=175.0, weight_kg=70.0, sex="M")
    elderly = BiometricProfile(age=68, height_cm=175.0, weight_kg=70.0, sex="M")

    th_young = young.plv_display_thresholds()
    th_elderly = elderly.plv_display_thresholds()

    assert th_young == th_elderly == {}


def test_blood_volume_male_vs_female():
    male = BiometricProfile(age=30, height_cm=175.0, weight_kg=70.0, sex="M")
    female = BiometricProfile(age=30, height_cm=175.0, weight_kg=70.0, sex="F")

    vol_m = male.estimated_blood_volume_l
    vol_f = female.estimated_blood_volume_l

    # Nadler's formula produces higher volume for male at identical height and weight
    assert vol_m > vol_f
    assert 4.0 <= vol_m <= 6.5
    assert 3.5 <= vol_f <= 5.5


def test_bmi_calculation():
    profile = BiometricProfile(age=25, height_cm=180.0, weight_kg=81.0, sex="M")
    expected_bmi = 81.0 / (1.80 ** 2)  # 25.0
    assert abs(profile.bmi - expected_bmi) < 1e-4


def test_expected_neurovascular_delay_age_scaling():
    young = BiometricProfile(age=18, height_cm=175.0, weight_kg=70.0, sex="M")
    older = BiometricProfile(age=60, height_cm=175.0, weight_kg=70.0, sex="M")

    delay_young = young.expected_neurovascular_delay_s()
    delay_older = older.expected_neurovascular_delay_s()

    # Age alone cannot estimate an individual's hemodynamic delay.
    assert delay_young != delay_young
    assert delay_older != delay_older


def test_invalid_parameters_raise_value_error():
    with pytest.raises(ValueError, match="Age"):
        BiometricProfile(age=150)

    with pytest.raises(ValueError, match="Age"):
        BiometricProfile(age=2)

    with pytest.raises(ValueError, match="Height"):
        BiometricProfile(height_cm=50.0)

    with pytest.raises(ValueError, match="Height"):
        BiometricProfile(height_cm=260.0)

    with pytest.raises(ValueError, match="Weight"):
        BiometricProfile(weight_kg=10.0)

    with pytest.raises(ValueError, match="Sex"):
        BiometricProfile(sex="X")


def test_biometric_profile_immutability():
    profile = BiometricProfile(age=25, height_cm=175.0, weight_kg=70.0, sex="M")
    with pytest.raises(Exception):
        profile.age = 26  # FrozenInstanceError expected
