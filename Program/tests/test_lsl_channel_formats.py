import pytest

from src.lsl_integration import (
    SelectedLSLSession,
    _core_metadata,
    is_numeric_lsl_channel_format,
    normalize_lsl_channel_format,
)


@pytest.mark.parametrize(
    ("value", "expected_name", "is_numeric"),
    [
        (1, "float32", True),
        ("2", "double64", True),
        ("cf_int32", "int32", True),
        (5, "int16", True),
        (6, "int8", True),
        (7, "int64", True),
        (3, "string", False),
        ("cf_string", "string", False),
        (0, "undefined", False),
        ("custom", "custom", False),
    ],
)
def test_lsl_channel_format_codes_and_names_are_classified(value, expected_name, is_numeric):
    assert normalize_lsl_channel_format(value) == expected_name
    assert is_numeric_lsl_channel_format(value) is is_numeric


def test_lsl_stream_metadata_displays_format_name_and_numeric_capability():
    class FakeInfo:
        def uid(self):
            return "test-uid"

        def name(self):
            return "test outlet"

        def type(self):
            return "EEG"

        def source_id(self):
            return "test-source"

        def hostname(self):
            return "localhost"

        def channel_count(self):
            return 2

        def nominal_srate(self):
            return 10.0

        def channel_format(self):
            return 1

    metadata = _core_metadata(FakeInfo())
    assert metadata["channel_format"] == "float32"
    assert metadata["channel_format_code"] == 1
    assert metadata["numeric_samples"] is True


def test_live_session_refuses_string_marker_outlet_before_recording(monkeypatch):
    class FakeInfo:
        def channel_format(self):
            return 3

    class FakeInlet:
        def __init__(self):
            self.closed = False

        def close_stream(self):
            self.closed = True

    inlet = FakeInlet()
    session = SelectedLSLSession(
        fnirs_selector="marker-stream",
        channel_mapping={"HbO2": 0, "HbR": 1},
        confirmed_derived_hb_channels=True,
    )
    monkeypatch.setattr(session, "_open_inlet", lambda _selector: (inlet, FakeInfo(), [{}, {}]))

    with pytest.raises(ValueError, match="string.*numeric LSL stream.*marker stream"):
        session.connect()

    assert inlet.closed
    assert session.is_connected is False
