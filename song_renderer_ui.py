"""Tkinter-based UI for converting JSON song definitions into rendered audio.

This module provides a small workstation-style interface that lets users paste
JSON describing a song arrangement and export a fully rendered WAV file.  The
renderer contains a lightweight synthesis engine capable of layering multiple
instruments ranging from bright EDM leads to gentler acoustic-inspired voices,
along with optional tempo-synced delay and side-chain style dynamics.
"""

from __future__ import annotations

import json
import math
import os
import struct
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, Iterable, List


# ---------------------------------------------------------------------------
# Utility dataclasses and helpers
# ---------------------------------------------------------------------------


NoteName = str


def note_to_frequency(note: NoteName) -> float:
    """Convert scientific pitch notation (e.g. ``C#4``) to a frequency."""

    note = note.strip().upper()
    if not note:
        raise ValueError("Empty note name")

    letter = note[0]
    accidental = 0
    index = 1
    if len(note) > index and note[index] in {"#", "B"}:
        accidental = 1 if note[index] == "#" else -1
        index += 1

    try:
        octave = int(note[index:])
    except ValueError as exc:  # pragma: no cover - user input validation
        raise ValueError(f"Invalid octave in note '{note}'") from exc

    note_offsets = {
        "C": -9,
        "D": -7,
        "E": -5,
        "F": -4,
        "G": -2,
        "A": 0,
        "B": 2,
    }

    if letter not in note_offsets:
        raise ValueError(f"Invalid note letter '{letter}'")

    semitone_offset = note_offsets[letter] + accidental + (octave - 4) * 12
    return 440.0 * (2.0 ** (semitone_offset / 12.0))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# ---------------------------------------------------------------------------
# Core synthesis routines
# ---------------------------------------------------------------------------


class SongRenderer:
    """Render JSON song definitions into PCM audio samples."""

    def __init__(self, sample_rate: int = 44_100):
        self.sample_rate = sample_rate

    # ---------------------------- public API ---------------------------------

    def render(self, song: Dict, *, apply_delay: bool = True, apply_sidechain: bool = True) -> List[float]:
        tempo = float(song.get("tempo", 128))
        length_bars = float(song.get("length_bars", 4))
        beats_per_bar = float(song.get("beats_per_bar", 4))
        total_beats = length_bars * beats_per_bar
        seconds_per_beat = 60.0 / tempo
        total_seconds = total_beats * seconds_per_beat
        total_samples = int(total_seconds * self.sample_rate)

        mix = [0.0 for _ in range(total_samples)]

        instruments: Iterable[Dict] = song.get("instruments", [])
        for instrument in instruments:
            track = self._render_instrument(instrument, seconds_per_beat, total_samples)
            volume = float(instrument.get("volume", 0.8))
            for i in range(total_samples):
                mix[i] += track[i] * volume

        if apply_sidechain:
            self._apply_sidechain(mix, tempo)

        if apply_delay:
            self._apply_delay(mix)

        self._normalize(mix)
        return mix

    def save_wav(self, path: str, samples: List[float]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "wb") as f:
            num_channels = 2
            bits_per_sample = 16
            byte_rate = self.sample_rate * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8
            data = self._interleave_stereo(samples)

            # Write RIFF header manually to avoid external deps.
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(data)))
            f.write(b"WAVEfmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, self.sample_rate, byte_rate, block_align, bits_per_sample))
            f.write(b"data")
            f.write(struct.pack("<I", len(data)))
            f.write(data)

    def create_demo_song(self) -> Dict:
        """Return a feature-rich demo song JSON structure."""

        lead_bar = [
            {"note": "C5", "start": 0.0, "duration": 0.5, "velocity": 0.9},
            {"note": "D#5", "start": 0.5, "duration": 0.5, "velocity": 0.9},
            {"note": "G5", "start": 1.0, "duration": 0.5, "velocity": 1.0},
            {"note": "D#5", "start": 1.5, "duration": 0.5, "velocity": 0.9},
            {"note": "F5", "start": 2.0, "duration": 0.5, "velocity": 1.0},
            {"note": "G5", "start": 2.5, "duration": 0.5, "velocity": 0.9},
            {"note": "A#5", "start": 3.0, "duration": 0.75, "velocity": 1.0},
            {"note": "G5", "start": 3.75, "duration": 0.25, "velocity": 0.8},
        ]

        bass_bar = [
            {"note": "C2", "start": 0.0, "duration": 0.5, "velocity": 0.9},
            {"note": "C2", "start": 0.5, "duration": 0.5, "velocity": 0.8},
            {"note": "C2", "start": 1.0, "duration": 0.5, "velocity": 0.9},
            {"note": "G1", "start": 1.5, "duration": 0.5, "velocity": 0.9},
            {"note": "A#1", "start": 2.0, "duration": 0.5, "velocity": 1.0},
            {"note": "G1", "start": 2.5, "duration": 0.5, "velocity": 0.9},
            {"note": "F1", "start": 3.0, "duration": 0.5, "velocity": 1.0},
            {"note": "G1", "start": 3.5, "duration": 0.5, "velocity": 0.9},
        ]

        arp_sequence = [
            "C4",
            "D#4",
            "G4",
            "A#4",
            "G4",
            "D#4",
            "C4",
            "G3",
        ]

        return {
            "title": "Neon Skyline",
            "tempo": 126,
            "length_bars": 4,
            "beats_per_bar": 4,
            "scale": "C minor",
            "effects": {"delay": True, "sidechain": True},
            "instruments": [
                {
                    "name": "Analog Lead",
                    "type": "saw_lead",
                    "volume": 0.8,
                    "pattern": [
                        {
                            **note,
                            "start": note["start"] + bar * 4,
                        }
                        for bar in range(4)
                        for note in lead_bar
                    ],
                },
                {
                    "name": "Deep Bass",
                    "type": "bass",
                    "volume": 0.9,
                    "pattern": [
                        {
                            **note,
                            "start": note["start"] + bar * 4,
                        }
                        for bar in range(4)
                        for note in bass_bar
                    ],
                },
                {
                    "name": "Velvet Pad",
                    "type": "pad",
                    "volume": 0.65,
                    "pattern": [
                        {"note": "Cm", "start": 0.0, "duration": 4.0, "velocity": 0.7},
                        {"note": "A#maj", "start": 4.0, "duration": 4.0, "velocity": 0.7},
                        {"note": "Gm", "start": 8.0, "duration": 4.0, "velocity": 0.7},
                        {"note": "Fmaj", "start": 12.0, "duration": 4.0, "velocity": 0.7},
                    ],
                },
                {
                    "name": "Drums",
                    "type": "drumkit",
                    "volume": 1.0,
                    "pattern": [
                        {"hit": "kick", "start": beat, "duration": 0.25, "velocity": 1.0}
                        for beat in [b / 2 for b in range(0, 32, 2)]
                    ]
                    + [
                        {"hit": "snare", "start": beat + 0.5, "duration": 0.25, "velocity": 0.9}
                        for beat in [b for b in range(0, 16, 2)]
                    ]
                    + [
                        {"hit": "hihat", "start": beat + offset, "duration": 0.125, "velocity": 0.8}
                        for beat in [b / 2 for b in range(0, 32)]
                        for offset in [0.0, 0.25]
                    ],
                },
                {
                    "name": "Arp",
                    "type": "plucks",
                    "volume": 0.55,
                    "pattern": [
                        {
                            "note": arp_sequence[i % len(arp_sequence)],
                            "start": i * 0.25,
                            "duration": 0.25,
                            "velocity": 0.7,
                        }
                        for i in range(64)
                    ],
                },
            ],
        }

    # --------------------------- rendering logic ----------------------------

    def _render_instrument(self, instrument: Dict, seconds_per_beat: float, total_samples: int) -> List[float]:
        instrument_type = instrument.get("type", "saw_lead")
        pattern = instrument.get("pattern", [])

        track = [0.0 for _ in range(total_samples)]
        for entry in pattern:
            if instrument_type == "drumkit":
                start_beat = float(entry.get("start", 0.0))
                start_time = start_beat * seconds_per_beat
                duration = float(entry.get("duration", 0.25)) * seconds_per_beat
                velocity = clamp(float(entry.get("velocity", 1.0)), 0.0, 1.0)
                hit_type = entry.get("hit", "kick")
                samples = self._render_drum(hit_type, duration, velocity)
                self._add_samples(track, samples, start_time)
            else:
                note_name = entry.get("note", "C4")
                # Allow chord shorthand for pads (e.g. Cm or A#maj)
                if instrument_type == "pad" and any(ch in note_name for ch in ("m", "maj", "dim")):
                    chord_samples = self._render_chord(note_name, seconds_per_beat, entry)
                    self._add_samples(track, chord_samples, float(entry.get("start", 0.0)) * seconds_per_beat)
                    continue

                start = float(entry.get("start", 0.0)) * seconds_per_beat
                duration = float(entry.get("duration", 0.5)) * seconds_per_beat
                velocity = clamp(float(entry.get("velocity", 1.0)), 0.0, 1.0)
                samples = self._render_note(note_name, duration, velocity, instrument_type)
                self._add_samples(track, samples, start)

        return track

    def _render_note(self, note_name: str, duration: float, velocity: float, instrument_type: str) -> List[float]:
        frequency = note_to_frequency(note_name)
        waveform = self._select_waveform(instrument_type)
        envelope = self._select_envelope(instrument_type)
        modulation = self._select_modulation(instrument_type)

        samples = []
        for i in range(int(duration * self.sample_rate)):
            t = i / self.sample_rate
            amp = waveform(frequency, t)
            if modulation:
                amp = modulation(amp, t)
            amp *= envelope(t, duration)
            samples.append(amp * velocity)

        return samples

    def _render_chord(self, chord_name: str, seconds_per_beat: float, entry: Dict) -> List[float]:
        chord = chord_name.strip()
        chord_type = "major"
        if chord.endswith("maj"):
            root_note = chord[:-3]
            chord_type = "major"
        elif chord.endswith("dim"):
            root_note = chord[:-3]
            chord_type = "diminished"
        elif chord.endswith("m") and not chord.endswith("maj"):
            root_note = chord[:-1]
            chord_type = "minor"
        else:
            root_note = chord

        root_note = self._ensure_octave(root_note)

        intervals = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "diminished": [0, 3, 6],
        }[chord_type]

        duration = float(entry.get("duration", 1.0)) * seconds_per_beat
        velocity = clamp(float(entry.get("velocity", 0.8)), 0.0, 1.0)
        base_freq = note_to_frequency(root_note)

        voices = []
        for semitone in intervals:
            freq = base_freq * (2 ** (semitone / 12.0))
            voices.append(self._render_note_from_frequency(freq, duration, velocity, "pad"))

        combined = [0.0 for _ in range(max(len(v) for v in voices))]
        for voice in voices:
            for i, sample in enumerate(voice):
                combined[i] += sample / len(voices)

        return combined

    def _ensure_octave(self, root_note: str) -> str:
        root_note = root_note.strip().upper()
        if not root_note:
            return "C4"
        if root_note[-1].isdigit():
            return root_note
        if len(root_note) > 1 and root_note[-1] in {"#", "B"}:
            return f"{root_note}4"
        return f"{root_note}4"

    def _render_note_from_frequency(self, frequency: float, duration: float, velocity: float, instrument_type: str) -> List[float]:
        waveform = self._select_waveform(instrument_type)
        envelope = self._select_envelope(instrument_type)
        modulation = self._select_modulation(instrument_type)

        samples = []
        for i in range(int(duration * self.sample_rate)):
            t = i / self.sample_rate
            amp = waveform(frequency, t)
            if modulation:
                amp = modulation(amp, t)
            amp *= envelope(t, duration)
            samples.append(amp * velocity)

        return samples

    def _render_drum(self, hit_type: str, duration: float, velocity: float) -> List[float]:
        num_samples = int(duration * self.sample_rate)
        samples = []
        for i in range(num_samples):
            t = i / self.sample_rate
            if hit_type == "kick":
                envelope = math.exp(-6 * t)
                freq = 60 * (1 - t) + 40
                value = math.sin(2 * math.pi * freq * t) * envelope
            elif hit_type == "snare":
                envelope = math.exp(-12 * t)
                value = (2 * (i % 2) - 1) * envelope * 0.6 + (math.sin(2 * math.pi * 180 * t) * envelope * 0.4)
            else:  # hihat and fallback
                envelope = math.exp(-20 * t)
                value = ((hash((i, hit_type)) % 100) / 50.0 - 1.0) * envelope
            samples.append(value * velocity)

        return samples

    def _select_waveform(self, instrument_type: str):
        def sine(freq: float, t: float) -> float:
            return math.sin(2 * math.pi * freq * t)

        def saw(freq: float, t: float) -> float:
            period = 1.0 / freq
            position = (t % period) / period
            return 2.0 * position - 1.0

        def square(freq: float, t: float) -> float:
            return 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0

        def pluck(freq: float, t: float) -> float:
            return math.sin(2 * math.pi * freq * t) * math.exp(-3 * t)

        def piano(freq: float, t: float) -> float:
            fundamental = sine(freq, t)
            second = sine(freq * 2, t) * math.exp(-3.5 * t)
            third = sine(freq * 3, t) * math.exp(-5.5 * t)
            return (fundamental + 0.6 * second + 0.3 * third) / 1.9

        def accordion_wave(freq: float, t: float) -> float:
            fundamental = sine(freq, t)
            third = sine(freq * 3, t) * 0.35
            fifth = sine(freq * 5, t) * 0.18
            return (fundamental + third + fifth) / 1.53

        def violin_wave(freq: float, t: float) -> float:
            fundamental = sine(freq, t)
            harmonic = sine(freq * 2, t) * 0.45
            return (fundamental + harmonic) / 1.45

        def upright_bass_wave(freq: float, t: float) -> float:
            fundamental = sine(freq, t)
            overtone = sine(freq * 2, t) * math.exp(-4 * t) * 0.5
            return (fundamental + overtone) / 1.5

        mapping = {
            "saw_lead": saw,
            "bass": square,
            "pad": sine,
            "plucks": pluck,
            "piano": piano,
            "accordion": accordion_wave,
            "acoustic_guitar": pluck,
            "violin": violin_wave,
            "upright_bass": upright_bass_wave,
        }
        return mapping.get(instrument_type, sine)

    def _select_envelope(self, instrument_type: str):
        def fast_attack_release(t: float, duration: float) -> float:
            return math.exp(-6 * t) if t < duration else 0.0

        def long_pad(t: float, duration: float) -> float:
            sustain = clamp(duration * 0.7, 0.1, duration)
            if t < 0.2:
                return t / 0.2
            if t < sustain:
                return 1.0
            remaining = max(duration - sustain, 0.001)
            return max(0.0, 1.0 - (t - sustain) / remaining)

        def bass_envelope(t: float, duration: float) -> float:
            if t < 0.05:
                return t / 0.05
            if t > duration - 0.05:
                return max(0.0, 1.0 - (t - (duration - 0.05)) / 0.05)
            return 1.0

        def piano_envelope(t: float, duration: float) -> float:
            attack = max(min(duration * 0.1, 0.02), 0.005)
            if t < attack:
                return t / attack
            decay = max(duration * 0.6, 0.1)
            if t < decay:
                progress = (t - attack) / max(decay - attack, 0.001)
                return max(0.0, 1.0 - 0.3 * progress)
            tail = max(duration - decay, 0.001)
            return max(0.0, 0.7 * (1.0 - (t - decay) / tail))

        def accordion_envelope(t: float, duration: float) -> float:
            attack = min(0.15, duration * 0.2)
            if t < attack:
                return t / attack
            release = min(0.2, duration * 0.2)
            if t > duration - release:
                return max(0.0, 1.0 - (t - (duration - release)) / max(release, 0.001))
            return 1.0

        def guitar_envelope(t: float, duration: float) -> float:
            attack = max(min(duration * 0.1, 0.02), 0.003)
            if t < attack:
                return t / attack
            decay = max(duration * 0.4, 0.05)
            if t < decay:
                return max(0.0, 1.0 - 0.5 * (t - attack) / max(decay - attack, 0.001))
            tail = max(duration - decay, 0.001)
            return max(0.0, 0.5 * (1.0 - (t - decay) / tail))

        def violin_envelope(t: float, duration: float) -> float:
            attack = min(0.2, duration * 0.25)
            if t < attack:
                return t / attack
            release = min(0.25, duration * 0.25)
            if t > duration - release:
                return max(0.0, 1.0 - (t - (duration - release)) / max(release, 0.001))
            return 1.0

        def upright_bass_envelope(t: float, duration: float) -> float:
            attack = min(0.05, duration * 0.2)
            if t < attack:
                return t / attack
            release = min(0.2, duration * 0.3)
            if t > duration - release:
                return max(0.0, 1.0 - (t - (duration - release)) / max(release, 0.001))
            return 1.0

        mapping = {
            "saw_lead": fast_attack_release,
            "plucks": fast_attack_release,
            "pad": long_pad,
            "bass": bass_envelope,
            "piano": piano_envelope,
            "accordion": accordion_envelope,
            "acoustic_guitar": guitar_envelope,
            "violin": violin_envelope,
            "upright_bass": upright_bass_envelope,
        }
        return mapping.get(instrument_type, fast_attack_release)

    def _select_modulation(self, instrument_type: str):
        def vibrato(amp: float, t: float) -> float:
            return amp * (1.0 + 0.02 * math.sin(2 * math.pi * 5 * t))

        def bass_drive(amp: float, t: float) -> float:  # pragma: no cover - simple saturation
            return math.tanh(1.5 * amp)

        def pad_chorus(amp: float, t: float) -> float:
            return amp * (0.6 + 0.4 * math.sin(2 * math.pi * 0.25 * t + 1.2))

        def gentle_vibrato(amp: float, t: float) -> float:
            return amp * (1.0 + 0.035 * math.sin(2 * math.pi * 4.5 * t))

        def slow_vibrato(amp: float, t: float) -> float:
            return amp * (1.0 + 0.02 * math.sin(2 * math.pi * 3.0 * t))

        mapping = {
            "saw_lead": vibrato,
            "bass": bass_drive,
            "pad": pad_chorus,
            "accordion": gentle_vibrato,
            "violin": slow_vibrato,
        }
        return mapping.get(instrument_type)

    def _add_samples(self, track: List[float], samples: List[float], start_time: float) -> None:
        start_index = int(start_time * self.sample_rate)
        for i, value in enumerate(samples):
            index = start_index + i
            if 0 <= index < len(track):
                track[index] += value

    def _apply_sidechain(self, samples: List[float], tempo: float) -> None:
        beat_length = int((60.0 / tempo) * self.sample_rate)
        for i in range(len(samples)):
            position = i % max(beat_length, 1)
            pump = math.exp(-3.5 * position / max(beat_length, 1))
            samples[i] *= 0.4 + 0.6 * pump

    def _apply_delay(self, samples: List[float], delay_seconds: float = 0.3, feedback: float = 0.35, mix: float = 0.2) -> None:
        delay_samples = int(delay_seconds * self.sample_rate)
        if delay_samples <= 0:
            return
        for i in range(delay_samples, len(samples)):
            delayed = samples[i - delay_samples] * feedback
            samples[i] = samples[i] * (1.0 - mix) + delayed * mix

    def _normalize(self, samples: List[float]) -> None:
        peak = max((abs(s) for s in samples), default=0.0)
        if peak == 0:
            return
        gain = 0.98 / peak
        for i in range(len(samples)):
            samples[i] *= gain

    def _interleave_stereo(self, samples: List[float]) -> bytes:
        interleaved = bytearray()
        for sample in samples:
            value = int(clamp(sample, -1.0, 1.0) * 32767)
            interleaved.extend(struct.pack("<h", value))
            interleaved.extend(struct.pack("<h", value))
        return bytes(interleaved)


# ---------------------------------------------------------------------------
# Tkinter application
# ---------------------------------------------------------------------------


class RendererApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("JSON to Song Renderer")
        self.geometry("950x700")
        self.minsize(900, 650)

        self.renderer = SongRenderer()

        self.delay_var = tk.BooleanVar(value=True)
        self.sidechain_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Paste or load JSON to begin.")

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("TButton", font=("Helvetica", 11, "bold"))
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 15, "bold"))
        style.configure("Accent.TButton", background="#4A90E2", foreground="white")

        header = ttk.Frame(self)
        header.pack(fill="x", padx=16, pady=(16, 8))

        ttk.Label(header, text="JSON to Song Renderer", style="Header.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status_var, foreground="#666666").pack(side="right")

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True, padx=16, pady=8)

        editor_frame = ttk.Labelframe(content, text="Song JSON", padding=12)
        editor_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.editor = scrolledtext.ScrolledText(editor_frame, font=("Courier", 11), wrap="word")
        self.editor.pack(fill="both", expand=True)

        control_frame = ttk.Labelframe(content, text="Rendering", padding=12)
        control_frame.pack(side="right", fill="y")

        ttk.Label(control_frame, text="Effects", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Checkbutton(control_frame, text="Tempo-synced delay", variable=self.delay_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(control_frame, text="Sidechain pump", variable=self.sidechain_var).pack(anchor="w", pady=2)

        ttk.Label(control_frame, text="Actions", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(16, 6))

        ttk.Button(control_frame, text="Load JSON File", command=self._load_json).pack(fill="x", pady=4)
        ttk.Button(control_frame, text="Save as WAV", command=self._save_wav).pack(fill="x", pady=4)
        ttk.Button(control_frame, text="Insert Demo Song", command=self._insert_demo).pack(fill="x", pady=4)

        ttk.Label(control_frame, text="Tips", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(20, 6))
        tips = (
            "• Use beats for `start` times to stay in sync.\n"
            "• Instruments: piano, accordion, acoustic_guitar, violin, upright_bass,\n"
            "  plus saw_lead, bass, pad, plucks, drumkit.\n"
            "• Pad entries accept chord names like 'Cm' or 'A#maj'."
        )
        ttk.Label(control_frame, text=tips, justify="left", wraplength=250).pack(anchor="w")

        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=16, pady=(0, 16))
        ttk.Label(footer, text="Crafted for expressive electronic and acoustic textures.").pack(side="left")

    # ---------------------------- UI callbacks ------------------------------

    def _load_json(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.read()
            song_data = json.loads(data)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Error", f"Failed to load JSON: {exc}")
            return
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", data)
        self.status_var.set(f"Loaded {os.path.basename(file_path)}")
        self._apply_song_defaults(song_data)

    def _apply_song_defaults(self, song_data: Dict) -> None:
        effects = song_data.get("effects") if isinstance(song_data, dict) else None
        if isinstance(effects, dict):
            if "delay" in effects:
                self.delay_var.set(bool(effects["delay"]))
            if "sidechain" in effects:
                self.sidechain_var.set(bool(effects["sidechain"]))

    def _save_wav(self) -> None:
        try:
            song_data = json.loads(self.editor.get("1.0", tk.END))
        except json.JSONDecodeError as exc:
            messagebox.showerror("Invalid JSON", f"Please fix JSON errors first:\n{exc}")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not file_path:
            return

        self.status_var.set("Rendering...")
        self.update_idletasks()

        thread = threading.Thread(
            target=self._render_async,
            args=(song_data, file_path, self.delay_var.get(), self.sidechain_var.get()),
            daemon=True,
        )
        thread.start()

    def _render_async(self, song_data: Dict, file_path: str, apply_delay: bool, apply_sidechain: bool) -> None:
        started = time.time()
        try:
            samples = self.renderer.render(song_data, apply_delay=apply_delay, apply_sidechain=apply_sidechain)
            self.renderer.save_wav(file_path, samples)
        except Exception as exc:  # pragma: no cover - UI level catch
            self.after(0, lambda: messagebox.showerror("Rendering Failed", str(exc)))
            self.after(0, lambda: self.status_var.set("Rendering failed."))
            return

        elapsed = time.time() - started
        self.after(0, lambda: self.status_var.set(f"Saved to {os.path.basename(file_path)} in {elapsed:.1f}s"))
        self.after(0, lambda: messagebox.showinfo("Success", f"Song exported to {file_path}"))

    def _insert_demo(self) -> None:
        demo = self.renderer.create_demo_song()
        pretty = json.dumps(demo, indent=2)
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", pretty)
        self._apply_song_defaults(demo)
        self.status_var.set("Demo song inserted – tweak and render!")


def main() -> None:
    app = RendererApp()
    app.mainloop()


if __name__ == "__main__":
    main()
