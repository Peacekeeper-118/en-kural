# En Kural (என் குரல்) — convert any recording into *your* voice

![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-green.svg)

A local, privacy-first voice-conversion app built for Tamil content.
Record or upload clean samples of your own voice, upload a recording made by a
friend, and get the same words, timing, pauses and emotion back — spoken in your
voice.  Nothing leaves your computer.

```
My voice recordings → Voice profile → Friend's recording → Convert → Preview → Export (WAV / MP3)
```

## How it works

This is **not** a pitch shifter.  The default engine is
[Seed-VC](https://github.com/Plachtaa/seed-vc) v1, a zero-shot neural voice
converter that separates *what is said* from *who says it*:

```
Source audio ──► voice / background separation (Hybrid Demucs, only when music
                 or loud background is detected; the background is mixed back later)
                        │
                        ▼
                 Whisper-small encoder (multilingual speech content, keeps Tamil
                 pronunciation, rhythm, emotion)
                        │
                        ▼
              length regulator (same number of frames → timing preserved)
                        │
Your samples ──► CAMPPlus speaker embedding (averaged over every sample)
              └► clean reference clip (mel "prompt")        │
                        │                                   ▼
                        └────────────► DiT / flow-matching mel generator
                                                            │
                                                            ▼
                                                BigVGAN vocoder → waveform
                                                            │
                                                            ▼
                    loudness-contour restore · background remix · EBU R128 normalise
```

Voice samples go through the same separation before they are used, and
recordings that are still poor afterwards (noise, clipping, low volume) are
left out of the profile whenever the clean ones carry enough speech.  An
optional, separately consented **fine-tuning** step adapts the DiT to your
voice (see *Fine-tuning* below).

Because the content representation comes from a multilingual encoder and the
speaker identity is injected separately, Tamil, Tanglish and English words are
carried through unchanged — nothing is transcribed or translated.

The engine sits behind a small interface (`backend/enkural/vc/base.py`), so a
better model can be dropped in later without touching the API, pipeline or UI
(see *Adding an engine* below).  A second engine (Seed-VC v2) is included as an
example and can be selected under *Advanced*.

## Requirements

* Linux (tested on Ubuntu), 16 GB RAM recommended
* NVIDIA GPU with ≥ 6 GB VRAM strongly recommended (CPU works but is slow:
  roughly 1–3 minutes per 10 s of audio)
* Internet for the one-time setup (≈ 4 GB of Python packages and models)
* No sudo needed: Python, ffmpeg and the models are installed under this folder

## Setup

```bash
git clone https://github.com/Peacekeeper-118/en-kural.git
cd en-kural
./scripts/setup.sh        # once: installs uv + Python 3.11 venv, deps, static ffmpeg, Seed-VC, models
./start.sh                # every time: starts the app and opens it in your browser
```

`start.sh` starts the server (loading the voice model in the background),
waits until it answers, and opens your browser at the link that carries the
access token.  If the app is already running it just opens it again.  Keep the
terminal window open while you work; **Ctrl+C** or closing the window stops
the app.  Options: `--port 7870`, `--no-open`, and `--install-shortcut` (adds
*En Kural* to your applications menu so you can start it without a terminal).

`./run.sh` starts only the server and prints the link, e.g.
`http://127.0.0.1:7860/?token=…` — open **that** link.  The token protects
the API from other local processes and from web pages open in the same browser.

Optional: `./run.sh --preload` loads the model at start-up (first conversion is
then instant); `./scripts/download_models.py --all` also fetches the
pitch-control (44.1 kHz) and v2 models ahead of time.

## Using the app

The screen has two parts: **Your voice** on the left (set up once) and three
steps on the right (what you do for every video).

**Your voice** (once): click *Create my voice*, confirm the voice is yours or
you have permission, then upload or record 1–5 minutes of you talking — in
Tamil, the way you talk in your videos.  Each recording gets a quality note
(noise, low volume, clipping, background sound).  Click *Learn my voice*
(about a minute).  Audio saved from a Reel has its music removed
automatically; recordings that are still much noisier than the rest are marked
*Not used*.  The app copies the sound of your recordings into every result, so
a quiet room matters more than any setting.  Once learned, the panel folds
away; *How your voice sounds* and *Make it sound even more like you* (optional
training, see below) are underneath.

1. **Add the recording** — your friend's audio or video (MP3, M4A, WAV, MP4…).
2. **Choose how it sounds** — the defaults are natural; convert straight away
   or adjust:
   * *How much like you*: clearer words ↔ more like me.
   * *Quality*: quick preview / standard / best (8 / 16 / 30 sampler steps);
     the hint estimates the time for the selected recording.
   * *Background music*: keep it (voice lifted out, converted, music put back),
     remove it, or don't separate.
   * *More options*: pitch (±4 semitones; non-zero uses the pitch-following
     44.1 kHz model), speaking speed (0.85×–1.15×), emphasis (0.7×–1.3×),
     background-noise clean-up, the trained model, the engine.  All ranges are
     clamped server-side, so robotic extremes are impossible.
3. **Listen and download** — your friend's recording and the result in your
   voice, waveform over waveform, with a voice-match score.  *Download MP3*
   (320 kbps, for Reels and Shorts) or *Download WAV* (for editing); both
   44.1 kHz, loudness-normalised to −14 LUFS / −1.5 dBTP.

Everything is also usable by keyboard, works on a phone-sized screen, follows
the system's light / dark setting and respects reduced motion.

## Measured on this machine (RTX 4050 laptop, 6 GB)

`scripts/benchmark.py`, seed 0, voice isolated from a real Reel as the target,
a 21 s and a 50 s Tamil source (averages; per-conversion wall time including
source preparation).  *Before* = the original code path
(`ENKURAL_SAMPLER=legacy ENKURAL_COMPILE=0`); runs in `eval_runs/speed-*`.

| Quality | Before | After | DiT | Vocoder | Speaker sim (before → after) | CER | Drift from old *high* (dB) |
|---|---|---|---|---|---|---|---|
| Draft | 6.1 s | **4.0 s** | 2.15 → 1.46 s | 2.76 → 1.33 s | 0.964 → 0.963 | 0.070 → 0.065 | 2.8 → **1.8** |
| Standard (default) | 9.2 s | **5.3 s** | 5.19 → 2.82 s | 2.76 → 1.32 s | 0.962 → 0.960 | 0.077 → 0.070 | 1.0 → 1.1 |
| High | 14.4 s | **7.8 s** | 10.43 → 5.26 s | 2.76 → 1.32 s | 0.963 → 0.961 | 0.076 → 0.074 | 0 → 0.84 |

Standard now runs at ~6.5× real time (was 3.8×).  *Drift* is the log-mel
distance to the old 50-step output from the same noise; two different 50-step
solvers already differ by ~0.7 dB, so standard and high are within noise and
draft got *closer* to high quality.  Output duration always equals the source
duration (÷ speed).  The first conversion after start-up adds ~10–20 s for
loading the models (plus ~5 s once per machine to compile the vocoder kernels);
the pitch model loads separately the first time it is used.

## Speed

Where the time went (standard, 21 s clip, before): DiT 60 %, vocoder 27 %,
Whisper < 1 %.  What changed — all verified by `tests/test_fast_ops.py` and the
benchmark above:

* **Sampler** (`SAMPLER` in `vc/seedvc_v1.py`): F5-TTS "sway" time grid (16
  steps match uniform 25), guidance only while t < 0.75 (each guided step costs a
  second DiT pass; the late steps barely use it), steps draft 8 / standard 16 /
  high 30 (were 10 / 25 / 50).  With `ENKURAL_SAMPLER=legacy` the sampler is
  bit-identical to upstream.
* **Flash attention**: the DiT passed an all-true padding mask that ruled out
  the flash kernel; it is dropped for unpadded batches (−14 % per step).
* **Vocoder**: BigVGAN's anti-aliased Snake (upsample → Snake → downsample)
  rewritten in polyphase form and compiled with Triton — exact in float64,
  −52 % vocoder time, no nvcc needed.  (BigVGAN's own CUDA kernel needs the full
  CUDA toolkit, which pip does not ship and CUDA 12.8 does not support with GCC 15.)
* **Optional DiT compile** (`ENKURAL_COMPILE_DIT=1`): another ~25 % per DiT
  step for ~8 s of compiling per server start; worth it for long recordings.

Tried and rejected: fp16 vocoder (18 % waveform error), skipping guidance in the
first steps (drift 2.5–3 dB, worse CER), whole-model vocoder compile (slower and
45 s compile).  Seed-VC v2 is unchanged.

## Fine-tuning (optional, opt-in)

Zero-shot conversion *imitates* your reference clip.  Fine-tuning trains a copy
of the Seed-VC v1 DiT + length regulator on your clean speech (upstream's
trainer: timbre-perturbed content, random prompt split), which usually makes
the result sound more like you.

1. Build the profile, then under *Personalise the model* read and tick the
   training consent and click **Allow fine-tuning for this profile**.
2. Pick a length (quick 300 / standard 600 / thorough 1000 steps) and start.
   The job prepares clips from your usable samples, holds ~15 % out, trains in
   a separate process (the whole GPU is handed over; conversions wait), then
   checks candidates at ½, ¾ and the end of training.
3. Each candidate and the standard model convert recordings of *another*
   speaker (your uploaded sources, else the bundled test clip) and are scored
   against the held-out clips.  A candidate is kept only if it is more similar
   to your voice without worse clarity (CER), naturalness (UTMOS) or intonation
   (F0 correlation); otherwise the standard model stays in use.
4. Conversions then use the fine-tuned model automatically (*Advanced → Use my
   fine-tuned voice model* switches it off).  It applies to the standard
   22 kHz model; pitch-shifted conversions use the 44.1 kHz F0 model as before.

Needs ≥ 20 s of clean speech (1–5 minutes is better) and an NVIDIA GPU.  On
the RTX 4050 laptop (6 GB) training runs at ~0.47 s/step with a peak of
3.3 GB, so *quick* takes ~2.5 min plus ~1 min of checks.

## Measuring quality

The in-app *voice match* uses CAMPPlus, the embedding the engine is conditioned
on, so it is optimistic by construction.  `scripts/benchmark.py` measures with
judges the engine never sees and writes `report.md`, `results.csv/json` and the
audio under `eval_runs/`:

| metric | judge | meaning |
|---|---|---|
| speaker sim | WavLM-base-plus-SV | cosine to **held-out** recordings of the target voice |
| CER | Whisper (Tamil fine-tune) | character error rate vs. the source transcript, per phrase |
| UTMOS | UTMOS22 | predicted naturalness, 1–5 (compare settings, not absolute values) |
| F0 corr | WORLD | how well the source's intonation contour survives |
| RTF / stage times | – | whole pipeline and Whisper / DiT / vocoder seconds |

```bash
PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --consent \
    --samples me_1.wav me_2.wav me_3.wav --heldout me_other_day.wav \
    --sources friend_1.wav friend_2.wav \
    --grid quality=draft,standard,high --grid similarity=50,65,80 --listening
```

`--profile-id <id>` benchmarks an existing app profile (e.g. `--grid
use_finetuned=false,true`).  Every conversion reuses the same noise seed, so
settings are compared on equal terms.  `--listening` writes shuffled A/B/C
files plus `key.json` for a blind listen, which should make the final call.
Use your real recordings: the bundled TTS test voices are so clean that
similarity saturates.

## Privacy & safety

* Everything runs locally; the only network use is downloading models once.
* Data lives in `./data` with owner-only permissions (`0700` / `0600`), under
  random ids, and is never served by a static file handler.
* Every profile, sample, source and result has a delete button; *Delete ALL
  data* wipes everything.
* Voice samples are used for training **only** if you separately allow
  fine-tuning for that profile (a second, timestamped consent;
  `allow_training` is `false` until then).  Training runs on this computer, the
  cut training clips are deleted after each run, and the fine-tuned model lives
  in the profile folder: it is deleted with the profile, with *Delete fine-tuned
  model*, or when you withdraw the training consent.
* A profile can only be created after confirming the consent statement, and the
  timestamped consent is stored with the profile.  Conversion is refused for
  profiles without it.
* The server binds to `127.0.0.1`.  If you ever expose it, put it behind
  HTTPS (a reverse proxy) — voice data is sensitive.

## Command-line tools

```bash
# generate Tamil / Tanglish test audio (Microsoft neural voices, needs internet)
PYTHONPATH=backend .venv/bin/python scripts/make_test_audio.py --out tests/data

# objective checks of a result: speaker similarity, timing, optional Whisper ASR agreement
PYTHONPATH=backend .venv/bin/python scripts/evaluate.py \
    --source tests/data/friend.wav --converted data/jobs/<id>/output.wav \
    --reference data/profiles/<id>/reference.wav --asr

# benchmark settings / engines / fine-tuned vs standard (see "Measuring quality")
PYTHONPATH=backend .venv/bin/python scripts/benchmark.py --help

# run the automated tests
PYTHONPATH=backend .venv/bin/python -m pytest tests -q
```

## Adding an engine

1. Create `backend/enkural/vc/my_engine.py` with a class deriving from
   `VoiceConverter`, decorated with `@register`, implementing `load`,
   `build_target` (turn the user's reference + samples into whatever your model
   needs) and `convert` (source clip + target → converted clip, same duration).
2. Import it in `registry._ensure_builtin_registered()`.
3. Select it in the UI (*Advanced → Conversion engine*) or make it the default
   with `ENKURAL_ENGINE=my_engine ./run.sh`.

Profiles are built per engine on demand, so existing profiles keep working
when you switch.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ENKURAL_HOST` / `ENKURAL_PORT` | `127.0.0.1` / `7860` | bind address |
| `ENKURAL_ENGINE` | `seedvc_v1` | default engine |
| `ENKURAL_DEVICE` | `auto` | `cuda` or `cpu` |
| `ENKURAL_DATA_DIR` / `ENKURAL_MODELS_DIR` | `./data` / `./models` | storage |
| `ENKURAL_TARGET_LUFS` / `ENKURAL_TRUE_PEAK_DB` | `-14` / `-1.5` | export loudness |
| `ENKURAL_TOKEN` | generated | fixed access token |
| `ENKURAL_NO_AUTH` | unset | `1` disables the token (not recommended) |
| `ENKURAL_SEPARATION` | `auto` | `off` disables voice / background separation everywhere |
| `ENKURAL_SAMPLER` | tuned | `legacy` = upstream sampler (uniform steps, guidance on every step) |
| `ENKURAL_COMPILE` | `1` | `0` disables Triton-compiled vocoder kernels |
| `ENKURAL_COMPILE_DIT` | unset | `1` also compiles the DiT (faster steps, ~8 s per start) |
| `ENKURAL_EVAL_ASR_MODEL` | `vasista22/whisper-tamil-small` | ASR judge for CER |
| `ENKURAL_MIN_TRAINING_SPEECH_SECONDS` | `20` | clean speech required for fine-tuning |
| `ENKURAL_ALLOW_CPU_TRAINING` | unset | `1` allows fine-tuning without a GPU (very slow) |

## Project layout

```
backend/enkural/
  main.py, security.py, api/routes.py     FastAPI app, auth, endpoints
  jobs.py, store.py                       background queue, on-disk storage
  profiles.py, pipeline.py                profile building, conversion flow
  evaluation.py                           engine-independent quality judges
  audio/  io.py vad.py quality.py analysis.py enhance.py separate.py
  vc/     base.py registry.py hf.py seedvc_v1.py seedvc_v2.py
          fast_ops.py                     exact inference speed-ups (vocoder, attention)
          finetune.py finetune_worker.py  opt-in speaker fine-tuning
frontend/  index.html styles.css app.js   single-page UI (no build step)
           fonts/Catamaran (SIL OFL, Tamil + Latin, served locally)
scripts/   setup.sh download_models.py make_test_audio.py evaluate.py benchmark.py
vendor/seed-vc                            upstream model code (unmodified)
bin/ffmpeg, bin/ffprobe                   static ffmpeg build
```

## Responsible use

En Kural is meant for converting recordings into **your own voice**, or a voice
you have clear permission to use.  The app asks for a consent confirmation
before a profile can be created.  Do not use it to impersonate, deceive or
harass anyone, and follow the laws that apply to you.

## Contributing

Issues and pull requests are welcome.  Please run the tests
(`PYTHONPATH=backend .venv/bin/python -m pytest tests -q`) before opening a PR,
and never commit voice recordings, `data/`, `models/` or `.env` files.

## License

En Kural is released under the **GNU General Public License v3.0** — see
[`LICENSE`](LICENSE).  The default engine uses
[Seed-VC](https://github.com/Plachtaa/seed-vc), which is itself GPL-3.0, so the
combined work is distributed under the same terms.  Model weights and other
third-party components keep their own licences (see below).

## Credits

Seed-VC (Plachta), Whisper (OpenAI), BigVGAN (NVIDIA), CAMPPlus (FunASR),
RMVPE, Silero VAD, WORLD (pyworld), Hybrid Demucs (Meta, via torchaudio),
OpenVoice V2 (MyShell, fine-tuning), WavLM-SV (Microsoft), UTMOS22 (SpeechMOS),
whisper-tamil-small (vasista22), Catamaran typeface (The Catamaran Authors, SIL OFL,
licence in `frontend/fonts/OFL.txt`).  Please respect their licences.
