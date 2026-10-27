from __future__ import annotations

from cybertone.stitch import placements


class Row:
    def __init__(self, src_in: float) -> None:
        self.src_in = src_in


def test_flow_mode_uses_all_the_time():
    rows = [Row(0.0), Row(10.0), Row(20.0)]
    starts, total = placements(rows, [2.0, 2.0, 2.0], [0.1, 0.5, 0.5, 0.0], 7.2, 2.0, "flow")
    assert starts == [0.1, 2.6, 5.1]
    assert total == 7.2


def test_anchor_mode_pins_to_two_x_timestamps():
    rows = [Row(0.0), Row(10.0), Row(20.0)]
    starts, total = placements(rows, [2.0, 2.0, 2.0], [0, 0, 0, 0], 7.0, 2.0, "anchor")
    assert starts == [0.0, 5.0, 10.0]
    assert total >= 12.0


def test_anchor_never_overlaps_when_a_sentence_overruns():
    rows = [Row(0.0), Row(1.0)]
    starts, _total = placements(rows, [3.0, 1.0], [0, 0, 0], 5.0, 2.0, "anchor")
    assert starts[1] >= starts[0] + 3.0


def test_transcript_cache_key_depends_on_model(tmp_path):
    from cybertone.cli import _transcript_cache
    from cybertone.config import Settings

    f = tmp_path / "a.mp3"
    f.write_bytes(b"x" * 16)
    a = _transcript_cache(Settings(asr_model="small"), f)
    b = _transcript_cache(Settings(asr_model="large-v3"), f)
    assert a != b
