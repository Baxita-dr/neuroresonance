from src.session_clock import duration_reached, elapsed_seconds


def test_session_timer_continues_past_179_seconds_to_300():
    started_at = 10_000.0

    assert elapsed_seconds(started_at, now=started_at + 179.0) == 179.0
    assert not duration_reached(started_at, 300.0, now=started_at + 179.0)
    assert elapsed_seconds(started_at, now=started_at + 300.0) == 300.0
    assert duration_reached(started_at, 300.0, now=started_at + 300.0)


def test_session_clock_never_reports_negative_elapsed_time():
    assert elapsed_seconds(started_at=200.0, now=199.0) == 0.0
