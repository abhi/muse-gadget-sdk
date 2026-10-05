# Gadget API

This page lists the parts of the `musegadget` package that a gadget uses to
talk to Muse. Reachy Mini is one such gadget. Everything else in the package
is internal and can change.

## Identity and state

- `identity.load_or_create()` returns the device `Identity`. It creates one on
  first use and keeps it across unpairing.
- `Identity.node_id` is the `homelink-xxxxxx` ID that Muse knows the device by.
  `Identity.ble_name` is the name the Muse app shows during pairing.
- `config.state_dir()` is the directory for device state. Set
  `MUSEGADGET_STATE_DIR` to move it.
- `config.load_json(name)`, `config.save_json(name, data)` and
  `config.delete_json(name)` read, write and remove one JSON file in the state
  directory. `save_json` writes atomically and makes the file readable only by
  its owner.
- `config.sdk_token()` returns the SDK token from `MUSEGADGET_SDK_TOKEN` or the
  state directory, or None. It raises `ValueError` for a token that
  gadgets.muse.ai could not have issued.
- `config.PAIRING_FILE` is the file that holds the pairing.

## Pairing

- `pair.pair(on_open=..., force=False, timeout_s=600)` opens Bluetooth setup
  so that the Muse app can pair the device. It prints nothing.
- Just before setup opens, it calls `on_open` with a `SetupWindow`. Use it to
  tell the user what to do. `ble_name` is the device to choose in the Muse
  app, `timeout_s` is how long setup stays open, and `has_sdk_token` is false
  when the state directory holds no SDK token.
- It returns a `PairResult`. `PairResult.outcome` is a `PairOutcome`:
  `PAIRED`, `ALREADY_PAIRED`, `INVALID_SDK_TOKEN` or `WINDOW_CLOSED`. For
  `INVALID_SDK_TOKEN`, `PairResult.sdk_token_problem` says what is wrong.
- A paired device is left alone unless `force` is true.

## Connecting

- `Service` keeps a paired device connected to its Muse. It fetches the
  device's VMs, connects, reconnects with backoff and rotates the device token.
- `Service.run()` runs until `Service.stop()` is called.
- To describe your own device, or to run your own work while it is connected,
  subclass `Service` and override any of these three methods. `Service` calls
  them for each session.
- `describe()` returns the `DeviceDescription` to register with Muse.
- `run_command(name, params, timeout_ms)` runs one command from Muse. By
  default it calls the `Executor`.
- `serve(session, vm, stop)` runs one `LinkSession` and returns its `Outcome`.
  By default it runs `session.run(stop)`, and returns `CLOSED` if the session
  fails. Override it to run your own work beside the session, and return
  when `stop` is set. The local message socket forwards to `session` while
  `serve` runs.
- `DeviceDescription(node_id, display_name, version, commands)` describes the
  device to Muse. `commands` maps each command name to its spec: a
  `description`, `required` and `optional` parameters, and an optional
  `timeout_ms`.
- `Outcome` says how a session ended: `CLOSED`, `AUTH_REJECTED`, `FORBIDDEN`,
  `UNPAIRED` or `STOPPED`.

## LinkSession

`LinkSession(noise_host=..., vm_id=..., vm_auth_token=..., device=...,
run_command=...)` is one encrypted session with a Muse VM.

- `run(stop)` connects, registers the device and serves Muse's commands until
  `stop` is set or the link ends. It returns an `Outcome`.
- `run_command(name, params, timeout_ms)` is your function. Muse calls it for
  each command and it returns the result as a dict.
- `registered` is an `asyncio.Event` that is set once Muse accepts the device.
- `send_chat(message, session_id=None, *, output_modality="text")` sends a
  user message as coming from this device. Without `session_id` it goes to
  the main chat.
- `send_voice(wav_bytes, session_id=None, *, message="", output_modality=None)`
  sends a WAV voice note of at most 2 MiB.
- `link_client.acknowledgement(result)` reads Muse's acknowledgement from the
  result of `send_chat` or `send_voice`. It raises `RequestRejected` when
  Muse refused the request.
- `open_side_chat(session_id)` looks up a side chat and returns a `SideChat`.
  `SideChat.exists` is false for an ID that Muse does not know yet. The first
  `send_chat` to that ID creates the chat.
- `subscribe_chat(session_id=None)` yields Muse's chat events.
  `chat_subscribed` is set while a subscription is open.
- `stream_tts(message_id)` yields MP3 audio for a reply that Muse gave in voice
  mode.
- A refused HTTP request raises `HttpStreamError`, which has the `status`.

## Following one request

- `muse_turn.MuseTurn` follows one request through Muse's chat events.
- Create it before you send the request. Pass every chat event to `feed()`.
  Pass the acknowledgement to `acknowledge()`.
- It returns `AgentStatus` for public progress, `Reply` each time an answer
  message changes, and `TaskFinished` when Muse's task ends.
- `complete(now)` says when Muse is done answering.
- `ReplayScope` drops events from earlier turns.

## The Reachy Mini contract

`musegadget.reachy` holds what Muse and Reachy Mini agree on. It uses only the
standard library.

- `reachy.commands.COMMAND_SPECS` lists the commands Muse can call on Reachy:
  `reachy.expression` and `reachy.move`.
- `reachy.commands.parse_command(name, params, *, antenna_mode, timeout_ms)`
  checks one call and returns a `Command` with the defaults filled in. It
  raises `ValueError` that names the first problem.
- `reachy.expressions.Expression` lists the expressions that Muse can name in
  a reply. `Expression.parse(name)` returns None for an unknown name.
- `reachy.prompts.voice_context(style, ...)` describes Reachy and the reply
  format to Muse. It also records whether motion, each antenna and face
  tracking are on.
- `reachy.prompts` also holds every other text Reachy sends Muse: the setup
  text for Reachy's side chat and for each reply style, the stop request, and
  the text that introduces a spoken request.
- `reachy.replies.ReplyStyle` names the reply formats.
- `reachy.replies.spoken_reply(text)` splits a reply into speech and the
  expression from its last `[reachy:NAME]` marker.
- `reachy.replies.SentenceStream` reads a reply as it streams, one JSON
  sentence frame per line or plain sentences, and returns `SpokenSentence`
  values. It raises `ReplyProtocolError` for a malformed reply and
  `ReplyRevisionError` when the final reply changes speech already given.
