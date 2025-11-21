# 🎬 Morphogen.Video & Audio Encoding Specification v1.0

**A declarative video encoding, audio/video filtering, and sync correction dialect built on the Morphogen kernel.**

**Inspired by ffmpeg, DaVinci Resolve, and modern media processing pipelines.**

---

## 0. Overview

Morphogen.Video is a typed, declarative video and audio processing dialect layered on the Morphogen kernel. It provides deterministic semantics for video encoding, filtering, transcoding, audio leveling, and synchronization operations. It represents a natural extension of Morphogen's operator DAG paradigm to multimedia streams.

**Why Video Belongs in Morphogen:**

Video processing is fundamentally:
- **Stream-based** — continuous data flows through operator pipelines
- **Operator-based** — filters, codecs, and transformations as composable ops
- **Filter-based** — ffmpeg-style filter graphs map directly to Morphogen DAGs
- **Parameterizable** — every operation has typed parameters (CRF, bitrate, preset)
- **Batchable** — apply pipelines to multiple files in parallel
- **GPU-accelerable** — hardware-accelerated encoding/decoding fits naturally
- **Graph-representable** — video = operator DAG on AV streams

**This is literally Morphogen's native shape.**

```
Morphogen = operator DAG on structured data
Video = operator DAG on AV streams
```

Video fits Morphogen as naturally as audio, fields, or physics — perhaps **more naturally** than any of them, because ffmpeg already behaves like a domain-specific operator graph with streams, filters, and codecs.

---

## 1. Language Philosophy

| Principle | Meaning |
|-----------|---------|
| **Pipeline composition** | Video/audio operations compose as declarative pipelines. |
| **Deterministic processing** | Same input + same pipeline = same output (bitwise identical). |
| **Typed streams** | Video streams, audio streams, and metadata streams are typed. |
| **Multi-rate scheduling** | Handle variable frame rates, audio sample rates, and sync drift. |
| **GPU-aware execution** | Automatically leverage hardware encoders (NVENC, QuickSync, AMF). |
| **Cross-domain integration** | Video ↔ Audio ↔ Vision ↔ Geometry (overlay, 3D rendering). |
| **Unit safety** | Frame rates (fps), bitrates (kbps), time codes (ms, frames). |
| **Filter graph equivalence** | ffmpeg filter graphs map one-to-one to Morphogen pipelines. |

**Key insight:** Video processing pipelines are **typed operator DAGs** with **temporal constraints** (sync, frame rate, time alignment).

---

## 2. Core Types

All video/audio types are defined in the kernel's type system with explicit multimedia semantics.

| Type | Description | Units | Examples |
|------|-------------|-------|----------|
| `VideoStream` | Video data stream (frames) | fps, resolution | 1920×1080@30fps, 4K@60fps |
| `AudioStream` | Audio data stream (samples) | Hz, channels | 48kHz stereo, 44.1kHz mono |
| `Frame` | Single video frame | pixels, time | RGB frame, YUV420p frame |
| `AudioBuffer` | Audio sample buffer | samples, time | 1024 samples @ 48kHz |
| `Codec` | Video/audio codec configuration | bitrate, quality | H.264, H.265, ProRes, AAC |
| `Filter` | Video/audio filter operator | parameters | blur, sharpen, normalize |
| `TimeSeries<T>` | Time-aligned data | time, offset | Audio waveform, frame timestamps |
| `SyncMap` | Timing alignment function | ms, frames | Drift correction curve |
| `Pipeline` | Composition of operators | DAG | Decode → Filter → Encode |
| `Metadata` | Stream metadata | various | Color space, aspect ratio, LUFS |

**Temporal Units:**
- **Frame rate:** `fps` (frames per second)
- **Time:** `ms` (milliseconds), `frames`, `samples`, `timecode`
- **Bitrate:** `kbps` (kilobits per second), `Mbps`
- **Sample rate:** `Hz`, `kHz` (44.1kHz, 48kHz, 96kHz)
- **Audio level:** `dB`, `LUFS` (Loudness Units Full Scale)

**Type safety:** Prevents mixing incompatible streams (can't encode audio as video).

---

## 3. Proposed Domains

Morphogen.Video introduces four interconnected domains for comprehensive multimedia processing.

### 3.1 VideoDomain

**Purpose:** Structural operations on video streams (decoding, encoding, scaling, cropping, composition).

**Status:** 🔲 Planned

**Operators:**

**Decoding & Encoding:**
```morphogen
video.decode(path: String) -> VideoStream
video.encode(stream: VideoStream, codec: Codec, path: String) -> File
```

**Transformation:**
```morphogen
video.scale(stream: VideoStream, width: u32, height: u32) -> VideoStream
video.crop(stream: VideoStream, x: u32, y: u32, w: u32, h: u32) -> VideoStream
video.fps(stream: VideoStream, rate: f32) -> VideoStream
video.rotate(stream: VideoStream, degrees: f32) -> VideoStream
```

**Composition:**
```morphogen
video.concat(streams: List<VideoStream>) -> VideoStream
video.overlay(base: VideoStream, overlay: VideoStream, x: u32, y: u32) -> VideoStream
video.blend(a: VideoStream, b: VideoStream, mode: String, opacity: f32) -> VideoStream
```

**Text & Graphics:**
```morphogen
video.draw_text(stream: VideoStream, text: String, font: Font, pos: Vec2) -> VideoStream
video.draw_box(stream: VideoStream, rect: Rect, color: Color) -> VideoStream
```

**Conversion:**
```morphogen
video.to_audio(stream: VideoStream) -> AudioStream
video.from_frames(frames: List<Frame>) -> VideoStream
video.to_frames(stream: VideoStream) -> List<Frame>
video.color_convert(stream: VideoStream, format: String) -> VideoStream
```

**Example:**
```morphogen
# Decode, scale, crop, encode pipeline
pipeline:
  - input = video.decode("input.mp4")
  - scaled = video.scale(input, width=1920, height=1080)
  - cropped = video.crop(scaled, x=0, y=100, w=1920, h=880)
  - codec = codec.h264(crf=18, preset="fast")
  - video.encode(cropped, codec, "output.mp4")
```

---

### 3.2 AudioFilterDomain

**Purpose:** Audio processing operations commonly handled inside ffmpeg (normalization, leveling, delay, sync).

**Status:** 🔲 Planned

**Operators:**

**Loudness & Normalization:**
```morphogen
audio.normalize(stream: AudioStream, target: f32) -> AudioStream
audio.loudnorm(stream: AudioStream, lufs: f32 = -14.0) -> AudioStream  # EBU R128
audio.measure_loudness(stream: AudioStream) -> f32 [LUFS]
audio.match_loudness(stream: AudioStream, reference: AudioStream) -> AudioStream
```

**Dynamics:**
```morphogen
audio.compress(stream: AudioStream, ratio: f32, threshold: f32) -> AudioStream
audio.limiter(stream: AudioStream, threshold: f32) -> AudioStream
audio.gate(stream: AudioStream, threshold: f32, ratio: f32) -> AudioStream
```

**Timing:**
```morphogen
audio.delay(stream: AudioStream, ms: f32) -> AudioStream
audio.trim(stream: AudioStream, start: f32, end: f32) -> AudioStream
audio.fade_in(stream: AudioStream, duration: f32) -> AudioStream
audio.fade_out(stream: AudioStream, duration: f32) -> AudioStream
```

**Equalization:**
```morphogen
audio.equalize(stream: AudioStream, bands: List<EQBand>) -> AudioStream
audio.bass_boost(stream: AudioStream, gain: f32) -> AudioStream
audio.treble_boost(stream: AudioStream, gain: f32) -> AudioStream
```

**Conversion:**
```morphogen
audio.resample(stream: AudioStream, rate: f32) -> AudioStream
audio.channel_mix(stream: AudioStream, layout: String) -> AudioStream  # stereo→mono, 5.1→stereo
```

**Example:**
```morphogen
# Normalize and compress audio track
pipeline:
  - input = audio.decode("dialogue.wav")
  - normalized = audio.loudnorm(input, lufs=-16.0)
  - compressed = audio.compress(normalized, ratio=4.0, threshold=-20.0)
  - audio.encode(compressed, "dialogue_processed.wav")
```

---

### 3.3 FilterDomain

**Purpose:** Visual filters equivalent to ffmpeg's `-vf` stack (blur, sharpen, color correction, denoise, stabilize).

**Status:** 🔲 Planned

**Operators:**

**Spatial Filters:**
```morphogen
filter.blur(stream: VideoStream, sigma: f32) -> VideoStream
filter.sharpen(stream: VideoStream, amount: f32) -> VideoStream
filter.unsharp(stream: VideoStream, amount: f32) -> VideoStream
filter.denoise(stream: VideoStream, method: String = "nlmeans") -> VideoStream
```

**Color Correction:**
```morphogen
filter.brightness(stream: VideoStream, amount: f32) -> VideoStream
filter.contrast(stream: VideoStream, amount: f32) -> VideoStream
filter.saturation(stream: VideoStream, amount: f32) -> VideoStream
filter.gamma(stream: VideoStream, amount: f32) -> VideoStream
filter.colorgrade(stream: VideoStream, lut: LUT) -> VideoStream
filter.white_balance(stream: VideoStream, mode: String = "auto") -> VideoStream
```

**Artistic Effects:**
```morphogen
filter.vignette(stream: VideoStream, intensity: f32) -> VideoStream
filter.bloom(stream: VideoStream, threshold: f32, radius: f32) -> VideoStream
filter.chromatic_aberration(stream: VideoStream, amount: f32) -> VideoStream
```

**Temporal Effects:**
```morphogen
filter.time_blend(stream: VideoStream, mode: String = "average") -> VideoStream
filter.deflicker(stream: VideoStream) -> VideoStream
filter.stabilize(stream: VideoStream, smoothness: f32 = 10.0) -> VideoStream
```

**Quality:**
```morphogen
filter.deband(stream: VideoStream) -> VideoStream
filter.deinterlace(stream: VideoStream) -> VideoStream
filter.upscale(stream: VideoStream, factor: f32, model: String = "lanczos") -> VideoStream
```

**Example:**
```morphogen
# ffmpeg equivalent: -vf "scale=1920:-1, unsharp=5:5:1.5"
pipeline:
  - input = video.decode("raw.mp4")
  - scaled = video.scale(input, width=1920, height=-1)  # preserve aspect
  - sharpened = filter.unsharp(scaled, amount=1.5)
  - video.encode(sharpened, codec.h264(crf=18), "output.mp4")
```

---

### 3.4 CodecDomain

**Purpose:** Expose codecs as typed operators with quality/performance parameters.

**Status:** 🔲 Planned

**Operators:**

**Video Codecs:**
```morphogen
codec.h264(crf: f32 = 23, preset: String = "medium", profile: String = "high") -> Codec
codec.h265(crf: f32 = 28, preset: String = "medium", tune: String = "none") -> Codec
codec.av1(crf: f32 = 30, speed: u32 = 6) -> Codec
codec.vp9(crf: f32 = 31, speed: u32 = 1) -> Codec
codec.prores(profile: String = "standard") -> Codec  # proxy, lt, standard, hq, 4444
codec.dnxhd(profile: String = "1080p_36") -> Codec
```

**Image Codecs:**
```morphogen
codec.jpeg(quality: u32 = 90) -> Codec
codec.png(compression: u32 = 6) -> Codec
codec.webp(quality: u32 = 90, lossless: bool = false) -> Codec
codec.jpegxl(distance: f32 = 1.0, effort: u32 = 7) -> Codec
codec.gif(dither: String = "sierra2_4a") -> Codec
```

**Audio Codecs:**
```morphogen
codec.aac(bitrate: u32 = 192) -> Codec  # kbps
codec.opus(bitrate: u32 = 128) -> Codec
codec.mp3(bitrate: u32 = 320) -> Codec
codec.flac(compression: u32 = 5) -> Codec
```

**Hardware Acceleration:**
```morphogen
codec.h264_nvenc(crf: f32 = 23, preset: String = "p4") -> Codec  # Nvidia
codec.h265_nvenc(crf: f32 = 28, preset: String = "p4") -> Codec
codec.h264_qsv(crf: f32 = 23) -> Codec  # Intel QuickSync
codec.h264_amf(crf: f32 = 23) -> Codec  # AMD
```

**Example:**
```morphogen
# High-quality ProRes export
codec = codec.prores(profile="hq")
video.encode(stream, codec, "output.mov")

# GPU-accelerated H.265 with Nvidia
codec = codec.h265_nvenc(crf=20, preset="p7")  # p7 = slowest/best quality
video.encode(stream, codec, "output.mp4")
```

---

## 4. Audio/Video Synchronization (SyncDomain)

**Morphogen's Sweet Spot:** Time-domain alignment, signal processing, phase correction, and offset detection.

Morphogen already treats **time domains**, **signals**, **phases**, **offsets**, and **transforms** as first-class objects. This makes sync correction natural.

### 4.1 Common Sync Problems

**Problem 1: Constant Offset Drift**

Video lagging behind audio (or vice versa) by a fixed amount.

**Detection methods:**
- Audio onset vs. video event detection (flash, clapboard)
- Waveform correlation vs. visual mouth movement
- Cross-spectrum analysis

**Operators:**
```morphogen
sync.detect_constant_offset(video: VideoStream, audio: AudioStream) -> f32 [ms]
sync.apply_offset(stream: AudioStream, offset: f32 [ms]) -> AudioStream
```

**Example:**
```morphogen
# Detect and fix constant sync drift
offset = sync.detect_constant_offset(video, audio)  # Returns: +143ms
audio_fixed = sync.apply_offset(audio, offset)
```

---

**Problem 2: Variable Drift (Progressive Desync)**

Sync gets worse over time due to:
- Variable frame rate
- Incorrect sample rate
- Dropped frames
- Bad capture hardware

**Mathematical model:**
```
offset(t) = a*t + b  (linear drift)
or
offset(t) = spline(t)  (nonlinear drift)
```

**Operators:**
```morphogen
sync.detect_drift(video: VideoStream, audio: AudioStream) -> SyncMap
sync.timewarp(stream: AudioStream, map: SyncMap) -> AudioStream
sync.resample_with_drift_compensation(stream: AudioStream, map: SyncMap) -> AudioStream
```

**Example:**
```morphogen
# Detect and fix progressive drift
drift_map = sync.detect_drift(video, audio)  # Returns: SyncMap(linear, a=0.02, b=100)
audio_fixed = sync.timewarp(audio, drift_map)
```

---

**Problem 3: Clapboard Detection (Event Alignment)**

Automatically align video flash with audio clap.

**Operators:**
```morphogen
vision.detect_flash(video: VideoStream) -> f32 [frames]
audio.detect_clap(audio: AudioStream) -> f32 [samples]
sync.align_events(visual_event: f32, audio_event: f32) -> f32 [ms]
```

**Example:**
```morphogen
# Automatic clapboard sync
flash_frame = vision.detect_flash(video)
clap_sample = audio.detect_clap(audio)
offset = sync.align_events(flash_frame, clap_sample)
audio_synced = sync.apply_offset(audio, offset)
```

---

**Problem 4: Automatic Re-timing for Lip-Sync**

Detect mouth movement and align with audio envelope.

**Operators:**
```morphogen
vision.detect_mouth_open(video: VideoStream) -> TimeSeries<bool>
audio.envelope(audio: AudioStream) -> TimeSeries<f32>
sync.align_signals(visual: TimeSeries<T>, audio: TimeSeries<U>) -> SyncMap
```

**Example:**
```morphogen
# Lip-sync alignment
mouth_events = vision.detect_mouth_open(video)
audio_env = audio.envelope(audio)
sync_map = sync.align_signals(mouth_events, audio_env)
audio_synced = sync.timewarp(audio, sync_map)
```

---

### 4.2 Audio Level Matching / Loudness Correction

ffmpeg supports EBU R128 loudness normalization, but it's cumbersome. Morphogen makes it first-class.

**Operators:**
```morphogen
audio.measure_loudness(stream: AudioStream) -> f32 [LUFS]
audio.loudnorm_to(stream: AudioStream, target: f32 [LUFS]) -> AudioStream
audio.match_loudness(stream: AudioStream, reference: AudioStream) -> AudioStream
audio.compress(stream: AudioStream, ratio: f32, threshold: f32) -> AudioStream
audio.auto_mix(streams: List<AudioStream>) -> AudioStream
```

**Smart logic:**
```morphogen
# Detect quiet dialogue and boost speech frequencies
dialogue = audio.detect_speech_regions(stream)
boosted = audio.equalize(dialogue, bands=[
    {freq: 2000, gain: 3.0, q: 1.0},  # presence boost
    {freq: 200, gain: -2.0, q: 0.7}   # mud reduction
])

# Duck background music when dialogue is present
music_ducked = audio.duck(music, dialogue, threshold=-30.0, ratio=0.3)
```

**Example:**
```morphogen
# Normalize all audio tracks to -14 LUFS (broadcast standard)
dialogue = audio.loudnorm_to(dialogue_raw, -14.0)
music = audio.loudnorm_to(music_raw, -14.0)
sfx = audio.loudnorm_to(sfx_raw, -14.0)

# Mix with automatic level balancing
mixed = audio.auto_mix([dialogue, music, sfx])
```

---

## 5. Filter Graphs as Morphogen Pipelines

ffmpeg filter graphs map **one-to-one** to Morphogen pipelines.

### 5.1 ffmpeg → Morphogen Equivalence

**ffmpeg:**
```bash
ffmpeg -i input.mp4 \
  -vf "scale=1920:-1, unsharp=5:5:1.5, eq=brightness=0.1:contrast=1.2" \
  -c:v libx264 -crf 18 -preset fast \
  output.mp4
```

**Morphogen:**
```morphogen
pipeline:
  - input = video.decode("input.mp4")
  - scaled = video.scale(input, width=1920, height=-1)
  - sharpened = filter.unsharp(scaled, amount=1.5)
  - corrected = filter.brightness(sharpened, amount=0.1)
  - corrected = filter.contrast(corrected, amount=1.2)
  - codec = codec.h264(crf=18, preset="fast")
  - video.encode(corrected, codec, "output.mp4")
```

**Cleaner.** **Composable.** **GPU-aware.**

---

### 5.2 Complex Filter Graph Example

**ffmpeg:**
```bash
ffmpeg -i video.mp4 -i watermark.png \
  -filter_complex "[0:v]scale=1280:720[scaled]; \
                   [scaled][1:v]overlay=W-w-10:H-h-10[output]" \
  -map "[output]" output.mp4
```

**Morphogen:**
```morphogen
pipeline:
  - video = video.decode("video.mp4")
  - watermark = video.decode("watermark.png")
  - scaled = video.scale(video, width=1280, height=720)
  - output = video.overlay(scaled, watermark, x=-10, y=-10)  # relative to bottom-right
  - video.encode(output, codec.h264(crf=23), "output.mp4")
```

---

## 6. Batch Processing

Morphogen excels at batch pipelines with parallel execution.

### 6.1 Batch Operators

```morphogen
batch.apply_to_files(pattern: String, pipeline: Pipeline) -> List<File>
batch.parallel(n: u32, pipelines: List<Pipeline>) -> List<Result>
batch.map(files: List<File>, fn: (File) -> File) -> List<File>
```

### 6.2 Use Cases

**Encode entire folder:**
```morphogen
# Transcode all MP4s in a folder to H.265
batch.apply_to_files("videos/*.mp4", pipeline=[
    video.decode,
    video.encode(codec=codec.h265(crf=28), output="encoded/{name}.mp4")
])
```

**Re-sync all videos:**
```morphogen
# Detect and fix sync issues in all files
batch.map("footage/*.mp4", fn=(file) => {
    video = video.decode(file)
    audio = video.to_audio(video)
    offset = sync.detect_constant_offset(video, audio)
    audio_fixed = sync.apply_offset(audio, offset)
    video.encode(video, audio_fixed, "synced/{name}.mp4")
})
```

**Replace audio tracks:**
```morphogen
# Replace audio in all videos with processed versions
batch.parallel(n=8, [
    for file in glob("videos/*.mp4"):
        video = video.decode(file).strip_audio()
        audio = audio.decode("processed_audio/{name}.wav")
        combined = video.add_audio(video, audio)
        video.encode(combined, "output/{name}.mp4")
])
```

**Normalize all loudness:**
```morphogen
# Normalize all audio files to -16 LUFS
batch.apply_to_files("audio/*.wav", pipeline=[
    audio.decode,
    audio.loudnorm_to(-16.0),
    audio.encode(output="normalized/{name}.wav")
])
```

---

## 7. GPU Acceleration

Morphogen maps naturally to GPU-accelerated codecs.

### 7.1 GPU Operators

```morphogen
gpu.accelerate(codec: Codec, backend: String = "auto") -> Codec
gpu.filter(filter: Filter, backend: String = "auto") -> Filter
```

**Backends:**
- `"nvenc"` — Nvidia hardware encoding (H.264, H.265, AV1)
- `"qsv"` — Intel QuickSync
- `"amf"` — AMD Advanced Media Framework
- `"videotoolbox"` — Apple hardware encoding (macOS/iOS)
- `"auto"` — Detect available GPU and use best backend

### 7.2 Example

```morphogen
# Automatically use GPU if available
codec = codec.h265(crf=23, preset="medium")
codec_gpu = gpu.accelerate(codec, backend="auto")
video.encode(stream, codec_gpu, "output.mp4")

# Explicit Nvidia encoding
codec = codec.h264_nvenc(crf=20, preset="p7")
video.encode(stream, codec, "output.mp4")

# GPU-accelerated denoise filter
denoised = gpu.filter(filter.denoise(stream, method="nlmeans"))
```

---

## 8. Magic "Fix My Video" Operator

Morphogen can build a high-level convenience operator that automatically fixes common issues.

### 8.1 Operator

```morphogen
video.fix(input: String, output: String, options: FixOptions = {}) -> File
```

**FixOptions:**
```morphogen
struct FixOptions {
    detect_sync: bool = true
    detect_color_cast: bool = true
    denoise: bool = true
    stabilize: bool = true
    auto_white_balance: bool = true
    loudness_normalize: bool = true
    upscale_factor: f32 = 1.0
    upscale_model: String = "lanczos"
    output_codec: Codec = codec.h264(crf=18)
}
```

### 8.2 Implementation

```morphogen
fn video.fix(input: String, output: String, options: FixOptions) -> File {
    # Decode
    video = video.decode(input)
    audio = video.to_audio(video)

    # Detect and fix sync issues
    if options.detect_sync {
        offset = sync.detect_constant_offset(video, audio)
        audio = sync.apply_offset(audio, offset)
    }

    # Detect color cast
    if options.detect_color_cast {
        video = filter.auto_color_correct(video)
    }

    # Denoise
    if options.denoise {
        video = filter.denoise(video, method="nlmeans")
    }

    # Stabilize
    if options.stabilize {
        video = filter.stabilize(video, smoothness=10.0)
    }

    # Auto white balance
    if options.auto_white_balance {
        video = filter.white_balance(video, mode="auto")
    }

    # Loudness normalize
    if options.loudness_normalize {
        audio = audio.loudnorm_to(audio, -16.0)
    }

    # Upscale
    if options.upscale_factor > 1.0 {
        video = filter.upscale(video, factor=options.upscale_factor, model=options.upscale_model)
    }

    # Encode
    video = video.add_audio(video, audio)
    return video.encode(video, options.output_codec, output)
}
```

### 8.3 Usage

```morphogen
# One-liner to fix common issues
video.fix("raw_footage.mp4", "fixed_footage.mp4")

# Custom options
video.fix("raw_footage.mp4", "fixed_footage.mp4", options={
    denoise: true,
    stabilize: true,
    upscale_factor: 2.0,
    upscale_model: "esrgan",
    output_codec: codec.prores(profile="hq")
})
```

**Equivalent to DaVinci Resolve's auto-magic, but scripted and deterministic.**

---

## 9. What Morphogen Gains

### 9.1 New Major Domains

- **VideoDomain** — Structural video operations (decode, encode, scale, crop, concat)
- **AudioFilterDomain** — Audio processing (normalize, compress, delay, EQ)
- **FilterDomain** — Visual filters (blur, sharpen, color correction, denoise)
- **CodecDomain** — Codec configuration (H.264, H.265, ProRes, AAC)
- **SyncDomain** — Time alignment (offset detection, drift correction, event sync)
- **BatchDomain** — Parallel batch processing (encode folders, apply filters)
- **VisionDomain** — Computer vision for video (flash detection, mouth tracking) [future]

### 9.2 New Operator Categories

| Category | Operators |
|----------|-----------|
| **Encoding** | decode, encode, transcode, remux |
| **Decoding** | decode video, decode audio, extract frames, extract metadata |
| **Filtering** | blur, sharpen, denoise, stabilize, color correction |
| **Color Correction** | brightness, contrast, saturation, gamma, LUT, white balance |
| **Transformation** | scale, crop, rotate, flip, pad, trim |
| **Compositing** | overlay, blend, concat, transition, alpha compositing |
| **Audio Leveling** | normalize, loudnorm, compress, limiter, gate |
| **Time Alignment** | offset detection, drift correction, timewarp, event sync |
| **Stabilization** | motion analysis, smoothing, rolling shutter correction |
| **Upscaling** | Lanczos, bicubic, ESRGAN, Real-ESRGAN, Waifu2x |
| **Format Conversion** | color space conversion, frame rate conversion, aspect ratio |
| **Batch Processing** | parallel encoding, folder processing, pipeline mapping |
| **GPU Acceleration** | NVENC, QuickSync, AMF, VideoToolbox |

### 9.3 Cross-Domain Integration

Morphogen.Video naturally integrates with existing domains:

**Video ↔ Audio:**
```morphogen
audio = video.to_audio(video_stream)
video = video.add_audio(video_stream, audio_stream)
```

**Video ↔ Vision:**
```morphogen
frames = video.to_frames(video_stream)
analysis = vision.detect_objects(frames)
annotated = vision.draw_bboxes(frames, analysis)
video_out = video.from_frames(annotated)
```

**Video ↔ Geometry (3D rendering):**
```morphogen
# Render 3D scene to video frames
geometry = geometry.load("model.obj")
camera = camera.orbit(center=(0,0,0), radius=5.0, frames=300)
frames = render.frames(geometry, camera)
video = video.from_frames(frames)
video.encode(video, codec.h264(crf=18), "render.mp4")
```

**Video ↔ Fields (Fluid overlay):**
```morphogen
# Render fluid simulation as video overlay
@state vel : Field2D<Vec2<f32>> = zeros((1920, 1080))

flow(dt=0.01, steps=300) {
    vel = advect(vel, vel, dt)
    frame = visual.field_to_frame(vel, palette="viridis")
    output frame
}

video = video.from_frames(output_frames)
base = video.decode("background.mp4")
composited = video.overlay(base, video, x=0, y=0, opacity=0.5)
```

---

## 10. Implementation Priority

### Phase 1: Core Video Operations (MVP)
- **VideoDomain basics:** decode, encode, scale, crop
- **CodecDomain:** H.264, H.265, ProRes
- **FilterDomain basics:** blur, sharpen, brightness, contrast
- **Pipeline composition:** Chain operators into DAGs

### Phase 2: Audio Processing
- **AudioFilterDomain:** normalize, loudnorm, compress, delay
- **Audio-video muxing:** Combine audio + video streams
- **Basic sync:** Constant offset detection and correction

### Phase 3: Advanced Sync & Batch
- **SyncDomain:** Drift detection, timewarp, event alignment
- **BatchDomain:** Parallel processing, folder encoding
- **GPU acceleration:** NVENC, QuickSync integration

### Phase 4: Magic Operators & Vision
- **video.fix():** Auto-magic video correction
- **VisionDomain basics:** Flash detection, object tracking
- **Advanced filters:** Stabilization, upscaling (ESRGAN)

---

## 11. ffmpeg Integration Strategy

Morphogen doesn't need to reimplement ffmpeg — it can **orchestrate** ffmpeg as a backend.

### 11.1 Backend Architecture

```
Morphogen Pipeline → Graph IR → Backend Compiler → ffmpeg command
```

**Example:**

**Morphogen code:**
```morphogen
pipeline:
  - input = video.decode("input.mp4")
  - scaled = video.scale(input, width=1920, height=1080)
  - sharpened = filter.unsharp(scaled, amount=1.5)
  - video.encode(sharpened, codec.h264(crf=18), "output.mp4")
```

**Compiled ffmpeg command:**
```bash
ffmpeg -i input.mp4 \
  -vf "scale=1920:1080, unsharp=5:5:1.5" \
  -c:v libx264 -crf 18 \
  -y output.mp4
```

### 11.2 Advantages

- **No reimplementation:** Leverage ffmpeg's 20+ years of codec/filter development
- **Type safety:** Morphogen validates parameters at compile time
- **Composability:** Pipelines are first-class objects
- **Determinism:** Same Morphogen code → same ffmpeg command → same output
- **Optimization:** Morphogen can optimize filter graphs before compilation
- **GPU awareness:** Morphogen can auto-select hardware codecs based on system

### 11.3 Alternative Backends

For performance-critical or embedded use cases, Morphogen can also target:

- **Custom C++ backend:** Direct codec/filter implementation
- **GStreamer:** Alternative multimedia framework
- **GPU compute shaders:** Direct GPU video processing
- **Hardware APIs:** NVENC, VAAPI, VideoToolbox SDKs

---

## 12. Example Use Cases

### 12.1 YouTube Upload Pipeline

**Problem:** Prepare raw footage for YouTube upload (1080p, H.264, stereo audio, normalized loudness).

**Morphogen solution:**
```morphogen
pipeline:
  - video = video.decode("raw_footage.mov")
  - audio = video.to_audio(video)

  # Video processing
  - video = video.scale(video, width=1920, height=1080)
  - video = filter.denoise(video, method="nlmeans")
  - video = filter.sharpen(video, amount=1.2)

  # Audio processing
  - audio = audio.loudnorm_to(audio, -14.0)  # YouTube recommendation
  - audio = audio.compress(audio, ratio=3.0, threshold=-18.0)

  # Encode
  - video = video.add_audio(video, audio)
  - codec = codec.h264(crf=18, preset="slow", profile="high")
  - video.encode(video, codec, "youtube_upload.mp4")
```

---

### 12.2 Podcast Episode Processing

**Problem:** Normalize loudness, remove background noise, add intro/outro music.

**Morphogen solution:**
```morphogen
pipeline:
  - dialogue = audio.decode("raw_dialogue.wav")
  - intro = audio.decode("intro_music.wav")
  - outro = audio.decode("outro_music.wav")

  # Denoise dialogue
  - dialogue = audio.denoise(dialogue, method="spectral_subtraction")

  # Normalize loudness (podcast standard: -16 LUFS)
  - dialogue = audio.loudnorm_to(dialogue, -16.0)
  - intro = audio.loudnorm_to(intro, -16.0)
  - outro = audio.loudnorm_to(outro, -16.0)

  # Concat with fades
  - intro = audio.fade_out(intro, duration=2.0)
  - outro = audio.fade_in(outro, duration=2.0)
  - episode = audio.concat([intro, dialogue, outro])

  # Encode
  - audio.encode(episode, codec.aac(bitrate=192), "episode.m4a")
```

---

### 12.3 Multi-Camera Sync

**Problem:** Sync 3 camera angles from a concert (different start times, slight drift).

**Morphogen solution:**
```morphogen
pipeline:
  - cam1 = video.decode("cam1.mp4")
  - cam2 = video.decode("cam2.mp4")
  - cam3 = video.decode("cam3.mp4")

  # Detect flash event (light cue at start)
  - flash1 = vision.detect_flash(cam1)
  - flash2 = vision.detect_flash(cam2)
  - flash3 = vision.detect_flash(cam3)

  # Align to cam1 as reference
  - offset2 = flash2 - flash1
  - offset3 = flash3 - flash1

  - cam2 = sync.apply_offset(cam2, offset2)
  - cam3 = sync.apply_offset(cam3, offset3)

  # Detect and fix progressive drift
  - drift2 = sync.detect_drift(cam1, cam2)
  - drift3 = sync.detect_drift(cam1, cam3)

  - cam2 = sync.timewarp(cam2, drift2)
  - cam3 = sync.timewarp(cam3, drift3)

  # Encode synced videos
  - video.encode(cam1, codec.prores(profile="hq"), "cam1_synced.mov")
  - video.encode(cam2, codec.prores(profile="hq"), "cam2_synced.mov")
  - video.encode(cam3, codec.prores(profile="hq"), "cam3_synced.mov")
```

---

### 12.4 Batch Transcode for Archive

**Problem:** Convert 500 old MOV files (ProRes) to modern H.265 (HEVC) for storage.

**Morphogen solution:**
```morphogen
# Parallel batch processing (8 concurrent encodes)
batch.parallel(n=8,
  batch.map("archive/*.mov", fn=(file) => {
    video = video.decode(file)
    codec = codec.h265(crf=28, preset="slow")
    video.encode(video, codec, "h265_archive/{name}.mp4")
  })
)
```

---

### 12.5 AI Upscaling Pipeline

**Problem:** Upscale 720p footage to 4K using ESRGAN model.

**Morphogen solution:**
```morphogen
pipeline:
  - video = video.decode("720p_source.mp4")
  - frames = video.to_frames(video)

  # AI upscale (4x)
  - upscaled_frames = frames.map(|frame| {
      filter.upscale(frame, factor=4.0, model="realesrgan")
  })

  - upscaled_video = video.from_frames(upscaled_frames)
  - codec = codec.h265(crf=18, preset="slow")
  - video.encode(upscaled_video, codec, "4k_upscaled.mp4")
```

---

## 13. Performance Characteristics

### 13.1 Determinism Guarantees

| Operation | Determinism Level | Notes |
|-----------|-------------------|-------|
| Decode | Bitwise identical | Same file → same frames |
| Encode (lossless) | Bitwise identical | Same input → same output |
| Encode (lossy) | Deterministic* | Same parameters → same bitstream (if encoder is deterministic) |
| Filters (spatial) | Bitwise identical | Same input → same output |
| Filters (temporal) | Bitwise identical | Deterministic frame order |
| Sync detection | Reproducible | May vary with algorithm parameters |
| Batch processing | Order-independent | Parallel execution, deterministic results |

\* **Note:** Some encoders (e.g., x264, x265) are deterministic if run single-threaded. Multi-threaded encoding may introduce non-determinism. Morphogen can enforce single-threaded mode for strict determinism.

### 13.2 Performance Optimization

**Pipeline fusion:**
```morphogen
# Before fusion (3 passes):
scaled = video.scale(input, width=1920, height=1080)
sharpened = filter.unsharp(scaled, amount=1.5)
brightened = filter.brightness(sharpened, amount=0.1)

# After fusion (1 pass):
# Morphogen optimizer merges filters into single pass
output = video.apply_filters(input, [
    scale(1920, 1080),
    unsharp(1.5),
    brightness(0.1)
])
```

**GPU offloading:**
```morphogen
# Automatically detect GPU and offload heavy operations
config = gpu.auto_detect()  # Returns: {backend: "nvenc", available: true}

if config.available {
    codec = codec.h265_nvenc(crf=23)
} else {
    codec = codec.h265(crf=23)
}
```

**Parallel batch processing:**
```morphogen
# Process 100 videos using all CPU cores
batch.parallel(n=cpu.cores(),
    batch.map("videos/*.mp4", encode_pipeline)
)
```

---

## 14. Integration with Existing Morphogen Domains

### 14.1 Audio Domain

**Already implemented in v0.5.0 and v0.6.0!**

Morphogen.Video extends the existing Audio domain with filtering and sync operations.

**Existing operators:**
- `audio.play()` — Real-time playback
- `audio.save()` — WAV/FLAC export
- `audio.load()` — Load audio files
- `audio.record()` — Microphone recording

**New operators (Morphogen.Video):**
- `audio.loudnorm()` — EBU R128 loudness normalization
- `audio.compress()` — Dynamics compression
- `audio.delay()` — Time delay
- `audio.sync_to()` — Sync to video stream

**Cross-domain example:**
```morphogen
# Load video, process audio with existing Audio domain
video = video.decode("concert.mp4")
audio = video.to_audio(video)

# Use Morphogen.Audio operators
audio = audio |> reverb(mix=0.2) |> limiter(threshold=-1.0)

# Add back to video
video = video.add_audio(video, audio)
video.encode(video, codec.h264(crf=18), "concert_processed.mp4")
```

---

### 14.2 Visual Domain

**Already implemented in v0.6.0!**

Morphogen.Video extends the Visual domain to export video instead of static images.

**Existing operators:**
- `visual.save()` — PNG/JPEG export
- `visual.show()` — Interactive display
- `visual.video()` — MP4/GIF export (NEW in v0.6.0!)

**New operators (Morphogen.Video):**
- `visual.to_video_stream()` — Convert frame generator to VideoStream
- `visual.from_video_stream()` — Convert VideoStream to frames

**Cross-domain example:**
```morphogen
# Render field simulation as video
@state temp : Field2D<f32> = random_normal(seed=42, shape=(512, 512))

flow(dt=0.01, steps=300) {
    temp = diffuse(temp, rate=0.1, dt)
    frame = colorize(temp, palette="fire")
    output frame
}

# Export as video (existing v0.6.0 feature)
visual.video(output_frames, "heat_diffusion.mp4", fps=30)

# Or use new Morphogen.Video operators
video = visual.to_video_stream(output_frames, fps=30)
video = filter.sharpen(video, amount=1.2)  # Apply video filter
video.encode(video, codec.h265(crf=20), "heat_diffusion_hq.mp4")
```

---

### 14.3 Transform Domain

**Already implemented!**

Morphogen.Video can use Transform domain for audio/video analysis.

**Cross-domain example:**
```morphogen
# Detect sync using cross-correlation (FFT-based)
audio1_fft = fft(audio1.samples)
audio2_fft = fft(audio2.samples)
cross_corr = ifft(audio1_fft * conj(audio2_fft))
offset = argmax(cross_corr)  # Peak = offset in samples

audio2_synced = sync.apply_offset(audio2, offset)
```

---

### 14.4 Agent Domain (Future)

**Example: Particle overlay on video**

```morphogen
# Simulate particles and render onto video
@state particles : Agents<Particle> = alloc(count=1000, init=spawn_particle)

flow(dt=0.01, steps=300) {
    particles = particles.map(|p| {
        vel: p.vel + gravity * dt,
        pos: p.pos + p.vel * dt
    })

    # Render particles to frame
    frame = visual.agents(particles, width=1920, height=1080, size=3.0)
    output frame
}

# Composite onto video
base_video = video.decode("background.mp4")
particle_video = visual.to_video_stream(output_frames, fps=30)
composited = video.overlay(base_video, particle_video, x=0, y=0, opacity=0.8)
video.encode(composited, codec.h264(crf=18), "particles_overlay.mp4")
```

---

## 15. Why This Matters

**Morphogen becomes the only platform that unifies:**

✅ **Audio synthesis** (oscillators, filters, effects, physical modeling)
✅ **Video encoding** (codecs, filters, transcoding, batch processing)
✅ **Audio/video sync** (drift correction, event alignment, lip-sync)
✅ **Field simulation** (fluids, reaction-diffusion, heat transfer)
✅ **Agent simulation** (particles, boids, N-body)
✅ **Geometry** (parametric CAD, mesh operations)
✅ **Circuit simulation** (analog audio, PCB layout)
✅ **Optimization** (design discovery, parameter tuning)

**All domains share the same:**
- Type system
- Scheduler
- MLIR compilation
- Deterministic execution model
- Cross-domain operators

**This positions Morphogen as:**

🎬 **Universal multimedia processing platform** (ffmpeg + DaVinci Resolve + Audacity)
🎛️ **Creative computation kernel** (generative art, music, video)
🔬 **Multi-physics simulation engine** (engineering, research, education)
🎨 **Parametric design system** (CAD, PCB, 3D modeling)

**No other platform offers this level of integration.**

---

## 16. Roadmap Summary

| Phase | Features | Timeline |
|-------|----------|----------|
| **Phase 1: MVP** | VideoDomain basics (decode, encode, scale, crop), CodecDomain (H.264, H.265), FilterDomain basics (blur, sharpen, color correction) | Q1 2026 |
| **Phase 2: Audio** | AudioFilterDomain (normalize, compress, delay), audio-video muxing, basic sync (constant offset) | Q2 2026 |
| **Phase 3: Advanced** | SyncDomain (drift detection, timewarp), BatchDomain (parallel processing), GPU acceleration (NVENC, QuickSync) | Q3 2026 |
| **Phase 4: Magic** | video.fix() auto-magic operator, VisionDomain basics (flash detection), advanced filters (stabilization, AI upscaling) | Q4 2026 |

**Dependencies:**
- **MLIR integration** (v0.7.0) ✅ Complete
- **Audio domain** (v0.5.0, v0.6.0) ✅ Complete
- **Visual domain** (v0.6.0) ✅ Complete
- **Transform domain** ✅ Complete
- **Geometry domain** (v0.9.0+) 🔲 Planned

---

## 17. Conclusion

**Video encoding, audio/video filtering, sync correction, and ffmpeg-style pipelines fit Morphogen perfectly.**

In fact, they map onto Morphogen's architecture **better than audio or physics**, because ffmpeg already behaves like a domain-specific operator graph with streams, filters, and codecs.

**Morphogen = operator DAG on structured data**
**Video = operator DAG on AV streams**

By adding VideoDomain, AudioFilterDomain, FilterDomain, CodecDomain, SyncDomain, and BatchDomain, Morphogen becomes:

✅ **Cleaner than ffmpeg** (typed operators, composable pipelines)
✅ **More powerful than ffmpeg** (GPU-aware, cross-domain integration, AI upscaling)
✅ **More deterministic than ffmpeg** (same code → same output, always)
✅ **More accessible than DaVinci Resolve** (scripted, batchable, version-controllable)

**This is a huge new slice of capability — but one that fits perfectly with Morphogen's core architecture.**

**Video belongs in Morphogen. Let's build it.**

---

**Version:** 1.0
**Status:** Specification (Ready for Implementation)
**Last Updated:** 2025-11-15
**Author:** Morphogen Architecture Team
**Related Specs:** transform.md, circuit.md, timbre-extraction.md, ../architecture/domain-architecture.md
