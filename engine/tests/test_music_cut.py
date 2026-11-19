"""The music-led director: cuts follow the song, shots follow its energy, no same-clip neighbours."""

from __future__ import annotations

from aieditor_engine.agent.video_planner import MediaInfo, VideoContext, _cut_times, plan_video


def _song(bpm: float = 120.0, seconds: float = 120.0) -> MediaInfo:
    beat = 60 / bpm
    beats = [round(i * beat, 3) for i in range(int(seconds / beat))]
    # quiet intro, build, loud middle, quiet outro
    energy = [0.2 if s < 16 else 0.5 if s < 28 else 0.9 if s < 44 else 0.3 for s in range(int(seconds))]
    return MediaInfo(9, "m9", "song.mp3", "audio", seconds, music={"bpm": bpm, "beats": beats, "energy": energy})


def _clip(n: int, motion: float) -> MediaInfo:
    secs = [0.6 + 0.01 * (i % 7) for i in range(60)]
    return MediaInfo(n, f"v{n}", f"clip{n}.mp4", "video", 60.0,
                     analysis={"seconds": secs, "motion_s": [motion] * 60, "motion": motion * 0.08, "notes": []})


def test_cut_times_vary_with_energy_and_land_on_beats() -> None:
    song = _song()
    beats = song.music["beats"]
    cuts, phrases = _cut_times(beats, song.music["energy"], 45.0, 4.0, song.duration)
    lengths = [b - a for a, b in zip(cuts, cuts[1:], strict=False)]
    assert cuts[-1] in {round(p, 3) for p in phrases}  # ends on a phrase
    assert all(any(abs(c - b) < 1e-6 for b in beats) for c in cuts[1:])  # every cut on a beat
    quiet = [d for a, d in zip(cuts, lengths, strict=False) if a < 14]
    loud = [d for a, d in zip(cuts, lengths, strict=False) if 28 <= a < 42]
    assert quiet and loud and min(quiet) > max(loud)  # long shots when calm, shorter at the peak


def test_music_led_program_is_not_uniform() -> None:
    clips = [_clip(i, m) for i, m in enumerate((0.1, 0.9, 0.5, 0.3), start=1)]
    ctx = VideoContext(media=[*clips, _song()])
    res = plan_video('cinematic travel film titled "Out", slow, with music', ctx, 1, use_llm=False)
    prog = res["variations"][0]["program"]
    shots = [ln for ln in prog.splitlines() if ln.startswith("clip ")]
    assert len(shots) >= 6
    media = [ln.split()[1] for ln in shots]
    assert all(a != b for a, b in zip(media, media[1:], strict=False))  # never the same clip twice in a row
    assert set(media) == {"1", "2", "3", "4"}
    lengths = {round(float(r.split("-")[1]) - float(r.split("-")[0]), 2) for r in (ln.split()[2] for ln in shots)}
    assert len(lengths) >= 3  # rhythm, not one shot length
    fades = [ln for ln in shots if "transition=" in ln]
    assert 0 < len(fades) < len(shots) / 2  # mostly hard cuts, dissolves only at phrase starts
    assert all("volume=0" in ln for ln in shots)
