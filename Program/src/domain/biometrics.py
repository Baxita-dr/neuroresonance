"""
src/domain/biometrics.py — Participant Profile Helpers

Provides basic derived profile values and an approximate fNIRS DPF estimate.
The profile does not calibrate sensors or determine an individual's physiology.

References:
- Scholkmann, F., & Wolf, M. (2013). General equation for the differential pathlength
  factor of the frontal head as a function of wavelength and age. Phys. Med. Biol.
- Vaschillo, E. G., et al. (2006). Characteristics of resonance in heart rate
  variability stimulated by biofeedback. Applied Psychophysiol. Biofeedback.
- Lehrer, P. M., & Gevirtz, R. (2014). Heart rate variability biofeedback: how and
  why does it work? Frontiers in Psychology.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BiometricProfile:
    """
    Subject biometric characteristics for personalized DSP normalization.

    Attributes:
        age: Used by the approximate DPF helper and the experimental delay helper.
        height_cm: Used only by derived BMI and blood-volume helpers.
        weight_kg: Used only by derived BMI and blood-volume helpers.
        sex: Used by the approximate blood-volume helper; it does not affect DPF.
    """
    age: int = 18
    height_cm: float = 178.0
    weight_kg: float = 70.0
    sex: str = "M"

    def __post_init__(self):
        if self.age < 5 or self.age > 110:
            raise ValueError(f"Age must be between 5 and 110 years, got {self.age}")
        if self.height_cm < 90.0 or self.height_cm > 230.0:
            raise ValueError(f"Height must be between 90 and 230 cm, got {self.height_cm}")
        if self.weight_kg < 25.0 or self.weight_kg > 250.0:
            raise ValueError(f"Weight must be between 25 and 250 kg, got {self.weight_kg}")
        if self.sex.upper() not in ("M", "F"):
            raise ValueError(f"Sex must be 'M' or 'F', got {self.sex}")

    @property
    def bmi(self) -> float:
        """Body Mass Index (kg/m^2)."""
        height_m = self.height_cm / 100.0
        return float(self.weight_kg / (height_m ** 2))

    @property
    def estimated_blood_volume_l(self) -> float:
        """Nadler's approximate circulating blood volume (liters)."""
        if self.sex.upper() == "M":
            return float(0.3669 * ((self.height_cm / 100) ** 3) + 0.03219 * self.weight_kg + 0.6041)
        else:
            return float(0.3561 * ((self.height_cm / 100) ** 3) + 0.03308 * self.weight_kg + 0.1833)

    def pacing_reference_hz(self) -> float:
        """
        Return the fixed slow-breathing demonstration target (0.1 Hz).

        Height and age alone are not a validated way to determine an
        individual's resonance frequency. In HRV biofeedback that frequency is
        usually assessed from physiological recordings (especially respiration
        and beat-to-beat heart rate). This project does not currently acquire
        heart-rate data, so it must not present an anthropometric estimate as a
        personalized measurement.
        """
        return 0.1

    def estimate_resonant_frequency_hz(self) -> float:
        """Deprecated compatibility wrapper for the fixed pacing reference."""
        return self.pacing_reference_hz()

    @property
    def pacing_rate_bpm(self) -> float:
        """Fixed demonstration pacing rate in breaths per minute."""
        return float(self.pacing_reference_hz() * 60.0)

    @property
    def resonant_breaths_per_minute(self) -> float:
        """Deprecated compatibility wrapper for the fixed pacing rate."""
        return self.pacing_rate_bpm

    def compute_dpf(self, wavelength_nm: int = 780) -> float:
        """
        Compute the age-dependent Differential Pathlength Factor (DPF) for fNIRS.
        General frontal-head equation from Scholkmann & Wolf (2013).

        DPF accounts for photon scattering pathlength through scalp, skull,
        CSF, and gray matter as a function of age.
        """
        # DPF(lambda, age) = alpha + beta*age**gamma + delta*lambda**3
        #                       + epsilon*lambda**2 + zeta*lambda
        # Coefficients are from the published fit; wavelength is in nm.
        age = float(self.age)
        wavelength = float(wavelength_nm)
        if not 690 <= wavelength <= 950:
            raise ValueError("wavelength_nm must be between 690 and 950 nm")
        dpf = (
            223.3
            + 0.05624 * age**0.8493
            - 5.723e-7 * wavelength**3
            + 0.001245 * wavelength**2
            - 0.9025 * wavelength
        )
        return float(dpf)

    def expected_neurovascular_delay_s(self) -> float:
        """Deprecated: this profile cannot estimate an individual's hemodynamic delay."""
        return float("nan")

    def plv_display_thresholds(self) -> dict:
        """
        No PLV cutoffs are defined because this prototype has no validated
        physiological or clinical thresholds.
        """
        return {}

    def personalized_plv_thresholds(self) -> dict:
        """Deprecated compatibility wrapper; thresholds are not personalized."""
        return self.plv_display_thresholds()
