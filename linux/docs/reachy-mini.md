# Talk to Muse through Reachy Mini

Reachy Mini captures your voice, recognizes your words, sends them through Muse's
encrypted connection, and speaks Muse's answer through its own speaker.
Its head and antennas respond to listening, thinking, and speech. Muse can also
request an expression or a bounded pose through registered robot commands.

The integration needs a running Reachy Mini daemon with SDK 1.9 or later, Python
3.10 or later, an SDK token, and pairing with the Muse phone app. The SDK token
alone cannot authenticate a conversation.

## Install on a Wireless robot

Running on the robot uses its native microphone and speaker. Reachy's official
[development guide](https://huggingface.co/docs/reachy_mini/en/platforms/reachy_mini/development_workflow)
explains SSH access and its preinstalled `/venvs/apps_venv` environment. Stop any
other robot app before starting this integration.

Copy this repository to `/home/pollen/muse-gadget-sdk`. To keep the existing
Reachy apps environment intact, create a separate environment that can use its
installed SDK and media dependencies:

```sh
/venvs/apps_venv/bin/python -m venv --system-site-packages ~/muse-reachy-venv
printf '%s\n' /venvs/apps_venv/lib/python3.12/site-packages \
  > ~/muse-reachy-venv/lib/python3.12/site-packages/reachy_apps.pth
~/muse-reachy-venv/bin/pip install --no-deps -e ~/muse-gadget-sdk/linux
~/muse-reachy-venv/bin/pip install webrtcvad-wheels
~/muse-reachy-venv/bin/pip install piper-tts
~/muse-reachy-venv/bin/pip install faster-whisper==1.2.1
~/muse-reachy-venv/bin/python -m piper.download_voices en_US-lessac-medium \
  --data-dir ~/.local/share/musegadget-reachy/voices
~/muse-reachy-venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Systran/faster-whisper-tiny.en", local_dir="/home/pollen/.local/share/musegadget-reachy/whisper-tiny.en", token=False)'
```

Adjust `python3.12` if the robot uses another Python version. If the SDK is not
preinstalled, install `~/muse-gadget-sdk/linux[reachy]` in the environment instead.
Reachy's local audio backend needs its platform-specific GStreamer setup.

Bluetooth setup uses `dbus-python` and `gi`. They must be importable by this
interpreter. On an environment that cannot import the distro's D-Bus binding:

```sh
sudo apt-get install libdbus-1-dev libglib2.0-dev
~/muse-reachy-venv/bin/pip install dbus-python
```

Android pairing also needs `[GATT] ExchangeMTU = 256` in
`/etc/bluetooth/main.conf`. Restart Bluetooth after changing it, before opening
Muse pairing. This is the same requirement as the Linux gadget installer.

## Pair with Muse

Store your SDK token in `~/.local/share/musegadget-reachy/sdk_token`, with its
parent directory mode `0700` and the file mode `0600`. Keep it out of shell
arguments and source control. The default state directory holds the identity
and device credentials too.

```sh
~/muse-reachy-venv/bin/muse-reachy pair
```

In the Muse phone app, enable Settings > Devices > Developer mode. Add the
`MuseGadgetXXXXXX` device printed by the command. Select the current network
when prompted. Reachy is already online, so no Wi-Fi password is needed.

If BlueZ denies setup to the normal account, use sudo with the same explicit
state directory, then return ownership to the service account:

```sh
sudo ~/muse-reachy-venv/bin/muse-reachy \
  --state-dir /home/pollen/.local/share/musegadget-reachy pair
sudo chown -R pollen:pollen /home/pollen/.local/share/musegadget-reachy
```

## Choose a mode

`muse-reachy run --mode` decides who hears you and who speaks.

- `muse-voice` is the default. Reachy records your turn and sends the audio to
  Muse. Muse writes the answer and speaks it. The robot needs no speech models.
  The tested Muse VM does not answer reliably this way yet; see
  [Muse response compatibility](#muse-response-compatibility).
- `on-robot` keeps speech on the robot. Whisper turns your words into text, and
  Piper speaks Muse's answer. This mode needs `--stt-model` and `--tts-model`.
  `--stream-replies` and wake-word invocation work only in this mode.

Reachy checks the flags before it starts. A flag that does not fit the mode
stops startup with a message such as `--stt-model needs --mode on-robot`.

## Check and start

```sh
~/muse-reachy-venv/bin/muse-reachy doctor
~/muse-reachy-venv/bin/muse-reachy doctor --exercise
~/muse-reachy-venv/bin/muse-reachy run --mode on-robot --stream-replies --tts-model \
  ~/.local/share/musegadget-reachy/voices/en_US-lessac-medium.onnx --stt-model \
  ~/.local/share/musegadget-reachy/whisper-tiny.en
```

The exercise captures three seconds of microphone input, queues a quiet tone
on the robot speaker, and moves through conversation poses. Speak during the
capture and listen for the tone. Reported samples and queued audio prove the
software path; your hearing confirms physical sound.

Speak naturally and pause for about two seconds to send your turn. Reachy keeps
reading its microphone independently of recognition and Muse's response.
The default endpoint is two seconds of silence, so shorter pauses keep sentences
in one request. `--silence 1.0` sends turns sooner. Each recording is bounded at
60 seconds; remaining speech continues into the next turn while the microphone
stays open.

Up to eight completed follow-up turns wait in RAM while Muse answers. With
`--stt-backend sherpa-streaming`, recognition consumes authorized microphone
audio during capture; the endpoint seals the existing transcript. The default
Whisper and legacy Moonshine backends start decoding after recording ends.
Requests go to the same side chat in order, with one active Muse request.
Once Muse finishes that request, the next one can start while previous speech
is still buffered. One speaker owner preserves answer order and each sentence's
expression. Speaking again creates a later request; it does not amend or cancel
work already sent to Muse. If the queue fills, saved turns remain and Reachy asks you to repeat
the newest question after it finishes. Audio and transcripts are never saved.

On Reachy Mini Wireless, the adapter verifies the actual local ALSA route and
reads the microphone board's DSP controls before enabling capture through
speaker playback. The XVF3800 hardware echo canceller removes the speaker's
reference from the processed microphone channels. This permits simultaneous
speaking and listening, although room acoustics and residual echo still affect
recognition. Unknown or unqualified audio routes pause input during actual
playback and its echo tail; they continue listening during recognition and
backend work. The adapter does not change the DSP settings.
On the verified duplex route, sustained user speech holds new TTS and pauses
speech already playing. Reachy shows its listening pose, keeps transcribing
follow-ups, and resumes the retained audio after the configured silence interval
(two seconds by default). Playback uses 40 ms chunks and at most 80 ms of software
queue lead; device buffering and voice detection add some interruption latency.
Pausing retains unsent samples instead of flushing or restarting the microphone
pipeline. Only the current sentence and one lookahead sentence prepare audio,
with a bounded FIFO of sentence text behind them.
If Muse finishes without answer text, Reachy announces the empty reply and
returns to listening. Empty recognition does not send a message to Muse.
With local speech enabled, startup loads the voice model once. Without wake-word
invocation, it introduces the expression channel to Muse and speaks "Ready to
talk" with a nod. The loaded worker remains ready for later turns.
Reachy chooses a short acknowledgment after recognizing your request. Weather,
lookup, and reasoning requests receive different acknowledgments; greetings and
unmatched requests skip this extra phrase. The adapter selects these locally
without an additional model request. Startup caches their audio. Empty
recognition does not send a message to Muse. A timed wake opens question capture
without transcribing the wake phrase. Untimed wake results retain the local
"Yes?" retry cue when recognition cannot confirm the phrase; other empty
recordings stay quiet.
Local recognition still takes several seconds on the tested Pi. The
acknowledgment covers Muse's generation time; it does not remove that work.
Piper synthesizes a complete sentence before yielding its audio, so long
opening sentences also delay playback. The conversation instructions ask Muse
to answer naturally and fully with useful details, and to check time-sensitive
facts with available tools. They define
the sentence and expression format without word or sentence quotas. Robot
capabilities and active movement limitations stay in the context; microphone
timing and capture mechanics stay in the adapter. Each streamed sentence
carries its own expression. Reachy
prepares the next sentence during playback to reduce pauses between sentences,
keeping one sentence of lookahead and at most 4 MiB of queued float32 audio.
Waiting for Muse has a separate 180-second budget that starts after speech
recognition. Each speech segment has its own 180-second timeout excluding time
held for user speech, so a late reply can finish speaking while stalled generation
or playback still times out. Microphone processing uses a dedicated worker and
batches already available audio without waiting to fill a batch. A capture
overflow discards the incomplete backlog and resumes with fresh audio; completed
questions retain their place in the request queue.
Logs report recognition, Muse's first assistant event, first text, completed
reply messages, and speaker startup separately. Use those timestamps to
distinguish Muse generation from recognition and synthesis. A later message
may wait behind earlier speech even after its text has arrived.

## Spoken progress for long requests

With local speech enabled, Reachy can give a short progress update when an
accepted request takes longer to answer. A fresh, explicit public action or
public progress frame can provide the first update immediately. A generic
phase or waiting update waits 20 seconds. Later updates are at least 20 seconds
apart while the request remains active.
The progress worker runs separately from answer playback and is canceled when
the answer arrives, so a progress cue does not hold up the reply.

Reachy first speaks the latest unspoken public progress message supplied by Muse.
It retains one such message until spoken, superseded, or the answer arrives.
Reports more than ten seconds old are introduced with "Earlier from Muse:" so
the original statement is attributed to an earlier point in the request.
Public backend phases also distinguish working through a request from preparing
a reply. A generic phase change preserves the last specific reported step.
For example, "Muse is preparing a reply. Its last reported step was searching
the web." If only a generic phase is available, Reachy reports that phase;
it does not infer a search from the question. If neither phase nor milestone
is available, it says that no detailed progress update has arrived.
Fixed cues are synthesized and cached at startup. A specific stage requires an
explicit public update belonging to this turn; older stages are stated as history.
Reachy's dedicated side chat can attribute a status update with its exact chat
ID to the current acknowledged request. Main chats and explicitly selected
shared chats require message ancestry. Generic activity labels such as
"is working" or "is responding" do not establish a web or connector operation.
Reachy does not infer website names, connector calls, or completed work from
the user's question, tool arguments, or private payloads.

The first audio of a progress update starts a smooth, playful 1.4-second antenna
flourish, then the normal thinking rhythm resumes. The head keeps its rare-glance
cadence; updates do not add a head gesture. One motion writer applies the flourish
and the normal score. With `--antenna-mode left`, the right antenna stays at its
measured startup position throughout.
Spoken answers take over with their selected expressions. These updates cover
the wait; they do not speed up Muse's processing. A read-only official-documentation
lookup in an isolated side chat produced public backend stages "Searching web"
at 8.2 seconds and "Searching sources" at 22.2 seconds. Its model-authored progress
frame arrived at 25.1 seconds, shortly before completion at 26.4 seconds.
Those measurements establish backend web-search status updates. They do not
establish connector-specific telemetry or show that response frames arrive
before tool execution.

## Wake-word invocation

Enable the [Sherpa offline keyword model](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html#sherpa-onnx-kws-zipformer-gigaspeech-3-3m-2024-01-01-english)
to open the conversation with "Hey Muse". While
waiting for that phrase, the microphone supplies audio only to the local keyword
detector. Reachy does not run Whisper or send that audio to Muse. A bounded
three-second audio buffer preserves a question spoken immediately after the
wake phrase; it stays in memory. For supported Sherpa models at 16 kHz, a bounded
fresh keyword replay verifies acoustic timing and discards the buffered audio
before the final wake token. Question capture then opens directly, without
running speech recognition on the wake phrase. If timing cannot be verified,
the first recording must contain the exact wake phrase in its transcription.
Otherwise Reachy asks "Yes?" and waits for a fresh question instead of forwarding
the buffered words. After a confirmed wake, a capture gap discards the incomplete
recording and keeps the listening window open for a fresh question.

Capture runs independently of keyword inference and speech gating, including
processing the buffered wake audio. A three-second audio queue absorbs processing
bursts without losing parts of the wake phrase. An overflow
resets the speech detectors and recording rather than joining audio across the
missing interval. Gap logs distinguish a capture stall from a queue overflow.
Logs report sample coverage and inference timing every ten
seconds for the first minute, then once a minute during wake listening, without
recording words or audio.

The `--wake-model` directory also accepts the official
[phonetic Chinese/English 3M model from December 2025](https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html).
Extract that release and pass its directory to the same option. Reachy selects
its chunk-16 int8 encoder and joiner, fp32 decoder, and `en.phone` pronunciation
lexicon automatically. English wake phrases use that lexicon instead of the
older model's SentencePiece vocabulary. The phrase must have a pronunciation
for every word. This model requires no custom training; compare recognition on
actual microphone attempts before choosing a model for deployment.

On detection, Reachy raises its head, tilts, and gives a short antenna wiggle
while continuing to capture your question. Once awake, it accepts questions and
follow-ups. It returns to wake-word
detection after 10 seconds without speech. Thinking and speaker playback pause
that timer. A question accepted by Muse or a confirmed wake cue starts a fresh
window when its turn finishes. Empty recognition does not extend the window.
An active recording can finish before the window closes. Say "Hey Muse", wait
for its wake gesture, and ask a question, or say "Hey Muse, what's the weather?"
in one breath. A verified timed wake alone needs no spoken confirmation. This
version keeps capturing authorized follow-ups during question recognition and
thinking, and during speaker playback when hardware echo cancellation is
verified. The timer stays paused while queued or active requests finish. Empty
recognition does not renew authorization; an expired window closes admission
while previously accepted turns remain queued. `--wake-timeout 0` requires a new
wake phrase for each request, including requests made while Muse is busy.

Install the optional runtime and official English model:

```sh
~/muse-reachy-venv/bin/pip install -e "$HOME/muse-gadget-sdk/linux[reachy-wake]"
mkdir -p ~/.local/share/musegadget-reachy
curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2 \
  -o /tmp/muse-reachy-keywords.tar.bz2
tar -xjf /tmp/muse-reachy-keywords.tar.bz2 -C ~/.local/share/musegadget-reachy
~/muse-reachy-venv/bin/muse-reachy run --mode on-robot --stream-replies \
  --wake-model ~/.local/share/musegadget-reachy/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 \
  --wake-phrase 'hey muse' --wake-timeout 10 \
  --tts-model ~/.local/share/musegadget-reachy/voices/en_US-lessac-medium.onnx \
  --stt-model ~/.local/share/musegadget-reachy/whisper-tiny.en
```

The detector uses the int8 encoder, decoder, and joiner, plus `tokens.txt` and
`bpe.model`. These files total about 5.2 MiB. It runs on one CPU thread and
tokenizes the configured phrase locally. The keyword threshold is 0.25, with
keyword score 1.0 and four decoder paths. Installed wheels must support the
device's Python version and architecture.
Wake-word invocation needs `--mode on-robot`. Startup remains quiet.
The dedicated chat receives expression instructions during initialization;
later user messages contain only the recognized request. Explicit main-chat or
existing-chat modes retain their initial instruction prefix.
`--wake-timeout` accepts 1 to 300 seconds.

Sherpa remains the default wake backend. To evaluate Vosk, install its optional
runtime alongside the Reachy dependencies:

```sh
~/muse-reachy-venv/bin/pip install -e "$HOME/muse-gadget-sdk/linux[reachy,reachy-wake-vosk]"
```

Extract the official [small US English Vosk model](https://alphacephei.com/vosk/models)
into `~/.local/share/musegadget-reachy/wake-vosk-en-us`, with `am`, `conf`, and
`graph` directly inside that directory. Use the existing Piper and Whisper models:

```sh
~/muse-reachy-venv/bin/muse-reachy run --mode on-robot --stream-replies \
  --wake-backend vosk \
  --wake-model ~/.local/share/musegadget-reachy/wake-vosk-en-us \
  --wake-phrase 'hey muse' --wake-timeout 10 \
  --tts-model ~/.local/share/musegadget-reachy/voices/en_US-lessac-medium.onnx \
  --stt-model ~/.local/share/musegadget-reachy/whisper-tiny.en
```

The Vosk backend supports "Hey Muse" and waits for an utterance endpoint before
accepting the phrase. Its local grammar includes similar phrases so that the
recognizer can distinguish "Hey Muse" from "Hey news", "Hey music", "Hey moose",
and "Hey mouse". This backend remains experimental. A native comparison caught
six synthetic wake examples and rejected 24 negative examples, but a repeat run
missed a long request starting with "Hey Muse". The three-second pre-roll can
also lose the wake phrase if recognition arrives after a long question.
For evaluation, say "Hey Muse", wait for Reachy's acknowledgment, then ask your
question. This model has not demonstrated better live reliability than Sherpa.
Experimental comparisons also evaluate a partial result stable for 0.5 seconds.
A partial result can change after more speech arrives and can cause an early
false wake. The CLI uses endpoint results only.
`--wake-backend vosk` requires `--wake-model`. Vosk loads only when selected, so
Sherpa and ordinary conversations do not require its optional runtime.

By default Reachy uses its own Muse side chat. It saves the chat ID in
`reachy-chat.json` inside `--state-dir`, separately for each Muse VM and bound to
the robot's identity. Reconnects and restarts resume the same chat.
The first startup creates it with a visible setup message containing the robot's
capabilities and sentence/expression format. The setup asks Muse to perform no
actions and echo a one-time initialization challenge. Reachy consumes that
response without speaking it, before accepting microphone input. A digest of
the applied instructions avoids repeating setup on unchanged reconnects.
Capability or format changes refresh setup in the same chat; interrupted setup
gets a fresh challenge because the event subscription does not replay history.
Existing state without an instruction digest is migrated on its original chat ID.
If a saved chat no longer exists, Reachy recreates that same side-chat ID.
Creation must confirm the requested ID and `is_thread: true`; errors never
redirect speech to the main chat. Preserve this state file with the pairing
and identity files when moving or reinstalling the service.

`run --main-chat` explicitly selects your main Muse conversation.
`run --session-id UUID` selects an existing chat and leaves Reachy's default
saved selection intact. These options cannot be combined. Muse sometimes sends
assistant events without a parent message ID, which limits attribution when
other clients speak in the same explicitly selected chat during Reachy's turn.

For startup after reboot, adapt and install
[`../examples/muse-reachy.service`](../examples/muse-reachy.service):

```sh
sudo cp ~/muse-gadget-sdk/linux/examples/muse-reachy.{service,path} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now muse-reachy.service muse-reachy.path
sudo journalctl -u muse-reachy -f
```

The path unit starts the voice service when pairing credentials appear.
To keep a single-antenna configuration after reboot or source updates, add a
systemd drop-in with `sudo systemctl edit muse-reachy.service`. Reset `ExecStart`
and repeat the service's command with `--antenna-mode left` or `right`:

```ini
[Service]
ExecStart=
ExecStart=/home/pollen/muse-reachy-venv/bin/muse-reachy run --mode on-robot --stream-replies --antenna-mode left --tts-model /home/pollen/.local/share/musegadget-reachy/voices/en_US-lessac-medium.onnx --stt-model /home/pollen/.local/share/musegadget-reachy/whisper-tiny.en
```

Run `sudo systemctl daemon-reload` and `sudo systemctl restart muse-reachy.service`
after saving. Installing the example unit again retains an existing drop-in.
Stop both with `sudo systemctl stop muse-reachy.path muse-reachy.service`. The controller parks the robot and
restores the motor mode it found at startup. Terminal robot connection errors
exit so systemd can recreate the hardware controller. Network failures reconnect and
rotate Muse credentials through the existing Linux service.

## Muse response compatibility

The tested Muse VM handled some isolated `output_modality: "voice"` greetings
quickly, but robot-aware consecutive requests returned internal delegation
controls instead of answers. That route is not qualified for this integration.
Muse's TTS endpoint only speaks messages generated in voice mode.

Reachy has two modes today. In `--mode muse-voice`, Muse hears the recorded
WAV and speaks its own reply. In `--mode on-robot`, Reachy transcribes and
speaks locally. Muse then receives only text.

`--mode on-robot` needs both `--stt-model` and `--tts-model`. The first uses a
local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) worker.
The provided tiny.en model recognizes English. The second uses a local
[Piper neural voice](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md).
Empty speech does not start a conversation.
The words still come from Muse. A Piper voice with Muse transcription is not
supported.

The Pi keeps a warmed recognition worker. Streaming Silero speech detection
controls recording endpoints for local recognition; it uses the bundled model
in the tested `faster-whisper==1.2.1` installation. Silence detection also gates
recognition in the worker;
the original waveform is retained so softer words are not removed by cropping.
Short-turn decoding rejects repetitive or invalid output before falling back
to the ordinary decoder. Larger local models were slower on the tested Pi and
are not selected automatically.

### Incremental local recognition

This backend is experimental. The tested local streaming models have not met
the accuracy and latency checks on Reachy Mini, so Whisper remains the default.

The optional `reachy-stt-streaming` extra pins Sherpa ONNX 1.13.8. Install the
[English streaming Zipformer model](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26)
separately and select its directory with `--stt-model`. It requires the int8
encoder, float decoder, int8 joiner with `chunk-16-left-128` filenames, and
`tokens.txt`. The adapter uses one inference thread and modified beam search
with four active paths; it does not download assets automatically.

```sh
pip install 'musegadget[reachy-stt-streaming]'
muse-reachy run --mode on-robot --stt-backend sherpa-streaming \
  --stt-model /path/to/zipformer-en-2023-06-26 \
  --tts-model /path/to/voice.onnx --stream-replies
```

The model stays in a child process. Only audio admitted after wake authorization
and speech detection enters recognition. Partial text stays in RAM; native
endpoint detection is disabled. Reachy's two-second silence
boundary finalizes the transcript, then the request can be submitted. An earlier
active Muse request still holds later requests in order. Playback does not hold
the request queue. Missing audio or a recognition overflow invalidates the
affected utterance rather than submitting an incomplete transcript.

Before selecting this backend on a device, measure it under the active capture,
face-following, VAD, and synthesis workload. It must keep up with incoming audio,
retain the first and last words, and finalize shortly after the endpoint. The
three-second queue bound is a failure limit, not a target latency. A synthetic
speech benchmark cannot establish accuracy for a particular speaker or room.

### Measure conversation delays

Separate endpoint-to-transcript, waiting for an earlier request, Muse's first
text, synthesis, and playback when diagnosing latency. Sentence streaming only
helps after the server emits text; a prompt cannot guarantee that tools, the
model, or an HTTP intermediary flush updates promptly. Public progress frames
must describe actions reported for the current turn.

`--audio-diagnostics` adds content-free playback timing counters. They distinguish
hardware-call dispatch delay, synchronous writes, event-loop return delay,
scheduler lateness, and gaps between PCM feeds. The counters contain no audio,
transcripts, or chat IDs. Producer gaps are not measurements of speaker underruns;
confirm audible continuity on the physical device before claiming a playback fix.

Piper is a separate GPL-licensed program, invoked as a subprocess. Its model is
not bundled with this SDK. Use `--mode muse-voice` to request Muse's native
voice output and stream its MP3 response once your VM supports it.

## Local face following

Add `--face-follow` to the existing `muse-reachy run` command to let Reachy
follow a face while listening, thinking and speaking. The small bundled
OpenCV YuNet model runs locally at five frames per second. It uses Reachy's
camera calibration and current head pose, smooths gaze changes, and limits
combined gaze and expression targets to 20 degrees of yaw and 12 degrees of
pitch. The main motion thread combines gaze with expressions; the detector
never writes motor targets.

Tracking pauses during explicit robot commands and while waiting for the wake
phrase. When a face disappears, normal expressions resume after 0.8 seconds.
The detector initially selects the largest face and stays with that face while
it remains nearby. It does not identify people or distinguish which person is
speaking in a group. Camera frames remain in memory on Reachy and are not sent
to Muse or saved. Muse knows that local tracking is active but cannot describe
the camera scene through this feature.

The model's MIT license, pinned source and checksum are packaged alongside it.
`--face-follow-model PATH` selects another compatible YuNet model and requires
`--face-follow`. Missing calibration or a model startup error prevents startup;
a later camera error disables tracking while voice stays available.

The local SDK 1.9 speaker path receives the official SDK 1.10 shell equalization
and limiter in float32 at 48 kHz, then converts back to the device's native
output format and rate. Other media backends and SDKs with their own speaker
correction retain their own audio path. The voice remains Piper Lessac medium;
synthesis of Lessac high was too slow on the tested Pi.

## Expressivity

[Reachy's SDK](https://huggingface.co/docs/reachy_mini/en/SDK/quickstart) supplies
speech wobbling that adds audio-driven head offsets while the primary pose
continues. The controller uses it during spoken replies. One motion thread
owns the primary head, body, and antenna targets.
Thinking is led by the antennas. Each 3.6-second beat has a smooth 0.45-second
attack, 1.2 seconds of lively movement, a 0.6-second release, and a 1.35-second
pause. The antennas oscillate at 1.6 Hz with up to 30-degree amplitude around
their resting positions. The pause gives each burst a distinct beginning and end.
The head stays centered at a three-degree attentive pitch and 4 mm lift.
After nine seconds, a 1.8-second curious glance moves out and back. Further
glances occur every 30 seconds, alternating sides. Each stays within six degrees
of yaw and roll, with an extra 2 mm of lift. The tilt remains visible while face
tracking controls the gaze direction. The head is stationary
for more than 90 percent of a long thinking sequence; it does not sway or bob.
The same rhythm continues under spoken acknowledgments. At the first audio of
a progress update, a 1.4-second playful antenna flourish briefly takes over,
then the normal score resumes. It does not restart the head-glance cadence or
add a head gesture. SDK audio wobbling is disabled during thinking, so these
phrases cannot add extra head motion.
The motion writer enables it only for spoken answers.
While speaking, Muse's selected emotion rises, holds briefly, and settles through
head lift, tilt, and antenna movement. Face following preserves eye contact while
these gestures play. A nod or shake begins with the spoken chunk and holds the
gaze direction for its 1.6-second gesture so tracking does not erase it. Eased
travel and brief held peaks give the head time to reach each pose: nods use up
to eight degrees around the gaze direction, and shakes use up to ten. Absolute
head-angle limits still apply near the edges of the tracking range.
A reply without an emotion marker uses a neutral speaking pose. Motion smoothing
uses elapsed time so changes in the update rate do not slow the transitions.

The expressive playbook gives each conversational state a recognizable purpose:

| Moment | Expression | Meaning |
| --- | --- | --- |
| "Hey Muse" | One warm head lift, tilt, and antenna wiggle | "I'm here." |
| Listening | Attentive pose with still antennas | "You have my attention." |
| Thinking | Quick antenna bursts with short pauses; rare small head glances | "I'm working on it." |
| Long-request progress | A brief antenna flourish at audio onset, then the thinking score | Waiting for this answer, or a confirmed current step. |
| Speaking | Muse's emotion for each chunk, with speech-driven head movement | The motion follows the answer's tone. |
| Spoken nod or shake | One deliberate gesture starting with the spoken chunk | A clear acknowledgment. |
| Quiet waiting | Resting antennas and subtle idle breathing | Ready for another wake phrase. |
| Error | Gentle lowered pose | A problem needs attention. |

Long waits repeat the same calm score without becoming more frantic or implying
that an answer is almost ready. A new spoken answer takes over immediately.
Partial antenna mode applies this score to the working antenna and keeps the
other target at its measured startup position.

The device registers two commands:

| Command | Behavior |
| --- | --- |
| `reachy.expression` | Plays neutral, happy, sad, surprised, curious, nod, shake, listening, thinking, or error. An exact name from Pollen's official recorded emotions library loads that move on demand. Sound effects stay off during Muse speech. |
| `reachy.move` | Moves head translation, rotation, body yaw, and antennas smoothly within advertised limits. Invalid or nonfinite values are rejected. |

Try asking Muse, "Give me a happy nod when you finish answering."
Some Muse runtimes accept device registration but do not expose custom commands
to their conversational planner. The local speech adapter gives Muse an
expression channel without requiring tool discovery.

With `--stream-replies`, Muse sends one JSON object per spoken chunk:

```json
{"text":"Wow!","expression":"surprised"}
{"text":"That is wonderful news.","expression":"happy"}
{"text":"Let me think through the next step.","expression":"thinking"}
```

Each chunk can be a sentence, phrase, or single word. Reachy applies its
expression at the first audio, including nod and shake gestures.
It speaks complete incoming frames in order while Muse continues
generating. Partial frames stay buffered, and JSON controls are never spoken.
Completed transcripts are reconciled with already queued frames so they do
not repeat the same speech. If Muse revises a frame already committed to
playback, the adapter stops that message instead of replaying conflicting text.
Streaming needs `--mode on-robot`. It shortens the wait after text arrives;
it cannot shorten Muse's delay before its first text.

The optional field `kind` distinguishes public progress from answer chunks:

```json
{"kind":"progress","text":"I am writing a short greeting.","expression":"thinking"}
{"text":"Hello there!","expression":"happy"}
```

Omit `kind`, or set it to `answer`, for normal speech. A progress frame uses
`kind: "progress"`, fits the 240-byte UTF-8 transport limit, and sets
`expression` to `thinking` or `null`. It describes a real action that started or
a milestone that completed. It must not contain hidden reasoning, raw tool
arguments, or invented results. Muse can send a frame whenever a real stage
changes; the adapter spaces updates at least twenty seconds apart. It speaks
each reported activity or phase once per turn. Without new information it stays
quiet instead of repeating the same milestone; a generic waiting cue can play
once when no useful progress is available.
Progress frames are excluded from final-answer reconciliation, so they are not
replayed with the completed answer. Local backend phase updates use the same
progress timing and thinking motion.

Without streaming, Muse appends a marker such as `[reachy:nod]` to its answer.
The adapter removes it before speaking and executes the expression through
the same bounded Reachy controller. Both channels accept the built-in names
listed above, excluding error.
Recorded moves remain available through the registered command path.
Recorded moves come from
[`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library).
The first recorded expression needs internet access to download the library.
The built-in expressions work without that download.

### An antenna does not follow the animation

Check measured motor positions and the daemon log. A successful `set_target`
call and an overall `enabled` mode do not prove that each motor moved.
The tested daemon logs individual overload faults without setting its
aggregate `error` field:

```sh
journalctl -u reachy-mini-daemon --since '10 minutes ago' --no-pager
```

If a motor reports `Overload Error`, stop the application and disable motion.
Check for an obstruction before trying to move it again. A shutdown fault
requires a motor reboot; enabling torque alone cannot clear it. The daemon's
motor-controller initialization reboots faulted motors when restarted.
Follow [Reachy's motor diagnosis guide](https://huggingface.co/docs/reachy_mini/troubleshooting/motors_diagnosis).
Use `run --no-motion` to retain microphone and speaker access during diagnosis.
Avoid automatically retrying motion against a repeated hardware fault.
If one antenna is unavailable, `run --antenna-mode left` or
`run --antenna-mode right` keeps the head and working antenna expressive.
The other antenna's target remains at its measured startup position, including
during wake and shutdown. Requests to move it explicitly return an error.
`--antenna-mode none` retains head motion while keeping both antenna targets
fixed. This option controls targets; individual torque is not exposed by the
daemon's public motor API.

The SDK 1.9 local GStreamer capture queue is reduced to two buffers so busy
speech synthesis cannot leave seconds-old microphone audio for the next turn.
This uses its `_appsink_audio` element because SDK 1.9 has no public input queue
setting. Other media backends retain their own buffering behavior.

## Lite and remote clients

For Lite, run on the computer attached to its USB audio board and daemon. Use
`run --media-backend local`. Pairing currently uses BlueZ, so setup requires
Linux or existing device credentials.

A client running away from a Wireless robot needs Reachy's WebRTC media
backend. Use `run --host ROBOT_IP --media-backend webrtc` after installing the
Reachy SDK's remote media dependencies. Connection to the motion API alone
does not provide access to the robot's microphone and speaker.

## Develop

```sh
cd linux
uv run --with pytest --with '.[reachy]' pytest -q
```

The tests cover the real Noise protocol, voice-note uploads, split subscription
events, streaming MP3 decoding, turn attribution, audio formats, and bounded
motor commands. No robot or Muse account is needed for those tests. Use
`doctor --exercise` and a spoken Muse exchange for physical verification.

The robot command list contains only the Reachy commands. It does not advertise
the generic Linux shell or file commands. State files and credentials retain
the Linux SDK's owner-only storage policy.
