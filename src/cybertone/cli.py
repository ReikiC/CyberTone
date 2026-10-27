"""CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

from . import asr, audio, mux, stitch, synth, triggers
from .config import Settings
from .models import Plan, Trigger, read_jsonl, write_jsonl
from .plan import allocate

app = typer.Typer(add_completion=False, help="2x playback with per-sentence voice cloning.")
console = Console()


# --------------------------------------------------------------------- helpers


def _cfg(config: Path | None, speed: float | None, backend: str | None) -> Settings:
    cfg = Settings.load(config)
    if speed is not None:
        cfg.speed = speed
    if backend is not None:
        cfg.backend = backend
    cfg.ensure_workdir()
    return cfg


def _load_triggers(cfg: Settings) -> list[Trigger]:
    path = cfg.work / "triggers.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"{path} missing; run `cybertone triggers` first")
    return read_jsonl(path, Trigger)


def _load_plan(cfg: Settings) -> Plan:
    path = cfg.work / "plan" / "plan.json"
    if not path.exists():
        raise typer.BadParameter(f"{path} missing; run `cybertone plan` first")
    return Plan.load(path)


def _report(plan: Plan, stats: dict) -> None:
    t = Table(title="budget", show_header=True, header_style="bold")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("total", f"{plan.total:.2f}s")
    t.add_row("target", f"{plan.target:.2f}s")
    t.add_row("speech", f"{plan.speech:.2f}s ({stats.get('speech_ratio', 0) * 100:.1f}%)")
    t.add_row("pause budget", f"{plan.gap_budget:.2f}s")
    t.add_row("needed rate b", f"{plan.b:.3f}x")
    t.add_row("feasible", "[green]yes[/green]" if plan.feasible else "[red]no[/red]")
    if plan.deficit > 0:
        t.add_row("deficit", f"[red]{plan.deficit:.2f}s[/red]")
    console.print(t)
    if not plan.feasible:
        console.print(
            "[yellow]b exceeds r_max: speech must be compressed harder than is comfortable. "
            "Expect audible strain; consider --speed 1.8 or trimming content.[/yellow]"
        )


# -------------------------------------------------------------------- commands


@app.command()
def inspect(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    speed: Annotated[float, typer.Option("--speed")] = 2.0,
    use_asr: Annotated[bool, typer.Option("--asr/--no-asr")] = False,
) -> None:
    """Measure the material before committing to a strategy."""
    cfg = _cfg(config, speed, None)
    x, sr = audio.load(input, cfg.sr)
    words = asr.transcribe(input, cfg, cfg.work / "transcript.json") if use_asr else None
    trs, stats = triggers.build(x, sr, cfg, words)
    plan = allocate(trs, x.size / sr, cfg)
    _report(plan, stats)
    console.print(f"[dim]{len(trs)} sentences, sr={sr}, words={'yes' if words else 'no'}[/dim]")


@app.command("triggers")
def triggers_cmd(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    speed: Annotated[float, typer.Option("--speed")] = 2.0,
    use_asr: Annotated[bool, typer.Option("--asr/--no-asr")] = False,
) -> None:
    """Segment the source into sentence nodes."""
    cfg = _cfg(config, speed, None)
    x, sr = audio.load(input, cfg.sr)
    words = asr.transcribe(input, cfg, cfg.work / "transcript.json") if use_asr else None
    trs, stats = triggers.build(x, sr, cfg, words)
    write_jsonl(cfg.work / "triggers.jsonl", trs)
    (cfg.work / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    console.print(f"[green]{len(trs)} triggers[/green] -> {cfg.work / 'triggers.jsonl'}")
    console.print(stats)


@app.command("plan")
def plan_cmd(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    speed: Annotated[float, typer.Option("--speed")] = 2.0,
) -> None:
    """Allocate a rate window to every sentence."""
    cfg = _cfg(config, speed, None)
    trs = _load_triggers(cfg)
    x, _sr = audio.load(input, cfg.sr)
    plan = allocate(trs, x.size / cfg.sr, cfg)
    plan.save(cfg.work / "plan" / "plan.json")
    stats = json.loads((cfg.work / "stats.json").read_text()) if (cfg.work / "stats.json").exists() else {}
    _report(plan, stats)


@app.command("synth")
def synth_cmd(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    backend: Annotated[str, typer.Option("--backend", "-b")] = "mock",
) -> None:
    """Render one stem per sentence."""
    cfg = _cfg(config, None, backend)
    trs = _load_triggers(cfg)
    plan = _load_plan(cfg)
    x, sr = audio.load(input, cfg.sr)
    stems = _synth_all(cfg, trs, plan, x, sr)
    synth.save_manifest(stems, cfg.work / "stems.json")
    console.print(f"[green]{len(stems)} stems[/green] -> {cfg.work / 'stems'}")


def _synth_all(cfg: Settings, trs: list[Trigger], plan: Plan, x, sr) -> list:
    if cfg.backend != "mock" and cfg.tts_ref is None:
        ref = synth.make_reference(x, sr, trs, 8.0)
        cfg.tts_ref = cfg.work / "ref.wav"
        audio.save(cfg.tts_ref, ref, sr)
        console.print(f"[dim]auto reference -> {cfg.tts_ref}[/dim]")
    backend = synth.make_backend(cfg.backend, cfg, x, sr)
    stems = []
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                  console=console) as bar:
        task = bar.add_task(f"synth ({cfg.backend})", total=len(plan.rows))
        for trig, row in zip(trs, plan.rows, strict=True):
            _wav, stem = synth.fit(backend, trig, row, cfg)
            stems.append(stem)
            bar.advance(task)
    return stems


@app.command("stitch")
def stitch_cmd(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Join stems into the final audio timeline."""
    cfg = _cfg(config, None, None)
    plan = _load_plan(cfg)
    stems_data = json.loads((cfg.work / "stems.json").read_text(encoding="utf-8"))
    from .models import Stem

    stems = [Stem(**s) for s in stems_data]
    x, sr = audio.load(input, cfg.sr)
    master, cues = stitch.assemble(stems, plan, cfg, x)
    audio.save(cfg.work / "out" / "master.wav", master, sr)
    (cfg.work / "out" / "cues.json").write_text(
        json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[green]master[/green] {master.size / sr:.2f}s -> {cfg.work / 'out/master.wav'}")


@app.command()
def run(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    speed: Annotated[float, typer.Option("--speed")] = 2.0,
    backend: Annotated[str, typer.Option("--backend", "-b")] = "mock",
    use_asr: Annotated[bool, typer.Option("--asr/--no-asr")] = False,
    baseline: Annotated[bool, typer.Option("--baseline/--no-baseline")] = False,
) -> None:
    """End-to-end: segment, plan, synthesise, stitch, mux."""
    cfg = _cfg(config, speed, backend)
    x, sr = audio.load(input, cfg.sr)

    words = asr.transcribe(input, cfg, cfg.work / "transcript.json") if use_asr else None
    trs, stats = triggers.build(x, sr, cfg, words)
    write_jsonl(cfg.work / "triggers.jsonl", trs)

    plan = allocate(trs, x.size / sr, cfg)
    plan.save(cfg.work / "plan" / "plan.json")
    _report(plan, stats)

    stems = _synth_all(cfg, trs, plan, x, sr)
    synth.save_manifest(stems, cfg.work / "stems.json")
    ok = sum(1 for s in stems if not s.degraded)
    console.print(f"stems: {len(stems)}  degraded: {len(stems) - ok}")

    master, cues = stitch.assemble(stems, plan, cfg, x)
    master_path = audio.save(cfg.work / "out" / "master.wav", master, sr)
    (cfg.work / "out" / "cues.json").write_text(
        json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"master: {master.size / sr:.2f}s (target {plan.target:.2f}s)")

    srt = mux.write_srt(cues, cfg.work / "out" / "out.srt")
    if srt:
        console.print(f"subtitles -> {srt}")

    out = output or (cfg.work / "out" / f"{input.stem}.{cfg.speed:g}x{input.suffix}")
    meta = mux.probe(input)
    if meta["has_video"]:
        out = out.with_suffix(".mp4")
    mux.render(input, master_path, out, cfg.speed, cfg)
    console.print(f"[bold green]output[/bold green] {out}  ({mux.probe(out)['duration']:.2f}s)")

    if baseline:
        b = mux.baseline(input, cfg.work / "out" / f"{input.stem}.baseline{out.suffix}", cfg.speed, cfg)
        console.print(f"[dim]baseline for A/B -> {b}[/dim]")


@app.command("baseline")
def baseline_cmd(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    speed: Annotated[float, typer.Option("--speed")] = 2.0,
) -> None:
    """Uniform atempo version, for A/B comparison only."""
    cfg = _cfg(config, speed, None)
    out = output or (cfg.work / "out" / f"{input.stem}.baseline{input.suffix}")
    mux.baseline(input, out, speed, cfg)
    console.print(f"[green]baseline[/green] {out}")


@app.command("ref")
def ref_cmd(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--out", "-o")] = Path("refs/ref.wav"),
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    duration: Annotated[float, typer.Option("--dur")] = 8.0,
) -> None:
    """Extract a clean reference clip for zero-shot cloning."""
    cfg = _cfg(config, None, None)
    x, sr = audio.load(input, cfg.sr)
    trs, _ = triggers.build(x, sr, cfg, None)
    seg = synth.make_reference(x, sr, trs, duration)
    audio.save(output, seg, sr)
    console.print(f"[green]reference[/green] {output} ({seg.size / sr:.2f}s, {audio.rms_db(seg):.1f} dBFS)")


@app.command()
def video(
    input: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    audio_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--out", "-o")] = Path("work/out/out.mp4"),
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    speed: Annotated[float, typer.Option("--speed")] = 2.0,
) -> None:
    """Mux an existing master track onto a retimed video."""
    cfg = _cfg(config, speed, None)
    mux.render(input, audio_path, output, cfg.speed, cfg)
    console.print(f"[green]video[/green] {output}")


