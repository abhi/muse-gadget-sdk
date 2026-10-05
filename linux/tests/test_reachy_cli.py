# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
from contextlib import asynccontextmanager
import json
import sys
from types import SimpleNamespace
import uuid
import pytest

from musegadget import config
from musegadget import reachy_cli
from musegadget.reachy_capabilities import ReplyStyle


SETUP_NONCE = "a" * 32


@pytest.fixture(autouse=True)
def fixed_setup_nonce(monkeypatch):
    monkeypatch.setattr(reachy_cli.secrets, "token_hex", lambda size: SETUP_NONCE)


def setup_wire_message(message=reachy_cli.CHAT_SETUP_MESSAGE):
    return f"{message}\n\nInitialization challenge: reply exactly Ready {SETUP_NONCE}"


def first_setup_hash():
    return reachy_cli.ReachySideChat("reachy-1").setup_sha256


def test_setup_challenge_explicitly_overrides_later_spoken_reply_format():
    assert "ignore the later spoken-response format" in reachy_cli.CHAT_SETUP_SUFFIX
    assert setup_wire_message().endswith(f"reply exactly Ready {SETUP_NONCE}")


@pytest.mark.parametrize('options, enabled', [([], False), (['--face-follow'], True),
                                            (['--face-follow', '--face-follow-model', '/tmp/yunet.onnx'], True)])
def test_camera_following_is_explicit_and_passes_local_model_to_motion_owner(monkeypatch, options, enabled):
    from musegadget.local_face_tracking import default_face_model
    captured = {}

    def hardware(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr('musegadget.reachy_hardware.ReachyController', hardware)
    args = reachy_cli.parser().parse_args(['run', *options])
    reachy_cli._hardware(args)
    expected = args.face_follow_model or default_face_model() if enabled else None
    assert captured['face_follow_model'] == expected
    assert captured['enable_motion'] is True


def test_doctor_does_not_activate_the_camera_detector(monkeypatch):
    captured = {}
    monkeypatch.setattr('musegadget.reachy_hardware.ReachyController',
                        lambda **kwargs: captured.update(kwargs))
    reachy_cli._hardware(reachy_cli.parser().parse_args(['doctor']))
    assert captured['face_follow_model'] is None


@pytest.mark.parametrize('options, error', [
    (['--face-follow-model', '/tmp/yunet.onnx'], '--face-follow-model requires --face-follow'),
    (['--face-follow', '--no-motion'], '--face-follow requires motion'),
])
def test_invalid_camera_configuration_fails_before_hardware_start(tmp_path, monkeypatch, options, error):
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: pytest.fail('hardware must not start'))
    args = reachy_cli.parser().parse_args(['--state-dir', str(tmp_path), 'run', *options])
    with pytest.raises(ValueError, match=error):
        asyncio.run(reachy_cli.run(args))


class ChatSession:
    def __init__(self, status=404, *, ack=None, events=None):
        self.registered = asyncio.Event()
        self.registered.set()
        self.status = status
        self.ack = ack
        self.events = events
        self.requests = []
        self.posts = []
        self.subscription_closed = False
        self.chat_subscribed = asyncio.Event()
        self.posted = asyncio.Event()
        self.timeline = []

    @asynccontextmanager
    async def stream_http(self, method, path, body, *, headers):
        self.requests.append((method, path, json.loads(body)))
        yield SimpleNamespace(status=self.status)

    async def send_chat(self, message, session_id):
        self.timeline.append('post')
        self.posts.append((message, session_id))
        self.posted.set()
        return self.ack if self.ack is not None else {
            'ok': True, 'status': 200,
            'response': {'session_id': session_id, 'is_thread': True,
                         'channel': 'main', 'message_id': 'setup-user',
                         'reply_to_message_id': 'setup-user'},
        }

    async def subscribe_chat(self, session_id):
        try:
            self.timeline.append('subscribe')
            self.chat_subscribed.set()
            await self.posted.wait()
            events = self.events if self.events is not None else [
                {'event': 'task.status', 'payload': {'session_id': session_id,
                                                    'status': 'running', 'task_id': 'setup-task'}},
                {'event': 'delta.message_start', 'payload': {'session_id': session_id,
                                                            'message_id': 'setup-assistant'}},
                {'event': 'delta.text_append', 'payload': {'session_id': session_id,
                                                          'message_id': 'setup-assistant',
                                                          'parent_message_id': 'setup-assistant',
                                                          'text': f'Ready {SETUP_NONCE}'}},
                {'event': 'task.status', 'payload': {'session_id': session_id,
                                                    'status': 'completed',
                                                    'task_id': 'setup-task'}},
                {'event': 'delta.message_done', 'payload': {'session_id': session_id,
                                                          'message_id': 'setup-assistant',
                                                          'parent_message_id': 'setup-assistant',
                                                          'display_text': f'Ready {SETUP_NONCE}'}},
            ]
            for event in events:
                yield event
            await asyncio.Event().wait()
        finally:
            self.chat_subscribed.clear()
            self.subscription_closed = True


def test_dedicated_chat_is_created_once_and_resumed_after_restart(tmp_path):
    async def check():
        first = ChatSession()
        selected = await reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(first, 'vm-1')
        assert str(uuid.UUID(selected)) == selected
        assert first.requests == [('POST', '/chat/subscribe', {'session_id': selected})]
        assert first.posts == [(setup_wire_message(), selected)]
        assert first.subscription_closed
        saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
        assert saved == {'version': 2, 'device_id': 'reachy-1',
                         'chats': {'vm-1': {'session_id': selected, 'initialized': True,
                                          'applied_setup_sha256': first_setup_hash()}}}
        assert (tmp_path / reachy_cli.CHAT_STATE_FILE).stat().st_mode & 0o777 == 0o600
        reconnect = ChatSession(200)
        assert await reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(reconnect, 'vm-1') == selected
        assert reconnect.posts == []
        second_vm = ChatSession()
        other = await reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(second_vm, 'vm-2')
        assert other != selected
        assert await reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(ChatSession(200), 'vm-1') == selected
    asyncio.run(check())


def test_v1_initialized_chat_migrates_in_place_with_current_setup(tmp_path):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    config.save_json(reachy_cli.CHAT_STATE_FILE, {
        'version': 1, 'device_id': 'reachy-1',
        'chats': {'vm-1': {'session_id': selected, 'initialized': True}},
    }, tmp_path)
    session = ChatSession(200)
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(
        session, 'vm-1')) == selected
    assert session.posts == [(setup_wire_message(), selected)]
    saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
    assert saved['version'] == 2
    assert saved['chats']['vm-1'] == {
        'session_id': selected, 'initialized': True,
        'applied_setup_sha256': first_setup_hash(),
    }


def test_changed_setup_refreshes_same_chat_once(tmp_path):
    async def check():
        original = reachy_cli.ReachySideChat('reachy-1', tmp_path, setup_message='first setup')
        selected = await original.prepare(ChatSession(), 'vm-1')
        changed_session = ChatSession(200)
        changed = reachy_cli.ReachySideChat('reachy-1', tmp_path, setup_message='second setup')
        assert await changed.prepare(changed_session, 'vm-1') == selected
        assert changed_session.posts == [(setup_wire_message('second setup'), selected)]
        assert changed_session.timeline[:2] == ['subscribe', 'post']
        reconnect = ChatSession(200)
        assert await changed.prepare(reconnect, 'vm-1') == selected
        assert reconnect.posts == []
    asyncio.run(check())


def test_pending_newer_setup_prevents_old_applied_digest_from_skipping(tmp_path):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    old = reachy_cli.ReachySideChat('reachy-1', tmp_path, setup_message='old setup')
    config.save_json(reachy_cli.CHAT_STATE_FILE, {
        'version': 2, 'device_id': 'reachy-1', 'chats': {'vm-1': {
            'session_id': selected,
            'initialized': False,
            'applied_setup_sha256': old.setup_sha256,
            'pending_setup_sha256': 'f' * 64,
        }}}, tmp_path)
    session = ChatSession(200)
    assert asyncio.run(old.prepare(session, 'vm-1')) == selected
    assert session.posts == [(setup_wire_message('old setup'), selected)]


def test_dedicated_chat_waits_for_registration_before_any_http_or_state_write(tmp_path):
    async def check():
        session = ChatSession()
        session.registered.clear()
        pending = asyncio.create_task(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
        await asyncio.sleep(0)
        assert session.requests == []
        assert not (tmp_path / reachy_cli.CHAT_STATE_FILE).exists()
        session.registered.set()
        await pending
    asyncio.run(check())


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_chat_lookup_failure_never_posts_or_falls_back_to_main(tmp_path, status):
    from musegadget.link_client import HttpStreamError
    session = ChatSession(status)
    with pytest.raises(HttpStreamError):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    assert session.posts == []
    assert session.requests[0][2]['session_id']


@pytest.mark.parametrize('ack', [
    {'ok': False, 'status': 500, 'response': {}},
    {'ok': True, 'response': {'session_id': 'wrong-chat', 'is_thread': True}},
    {'ok': True, 'response': {'session_id': 'wrong-chat', 'is_thread': False}},
    {'ok': True, 'response': {'result': None}},
])
def test_failed_creation_retains_same_id_for_safe_retry(tmp_path, ack):
    session = ChatSession(ack=ack)
    with pytest.raises(RuntimeError):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    saved_id = session.posts[0][1]
    resumed = ChatSession(200)
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(resumed, 'vm-1')) == saved_id
    assert resumed.posts == [(setup_wire_message(), saved_id)]


@pytest.mark.parametrize('contents', ['not-json', '[]', '{"version": 1, "device_id": "another-device", "chats": {}}'])
def test_invalid_chat_state_stops_before_http(tmp_path, contents):
    (tmp_path / reachy_cli.CHAT_STATE_FILE).write_text(contents)
    session = ChatSession()
    with pytest.raises(ValueError):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    assert session.requests == session.posts == []


def test_setup_drain_ignores_other_chats_and_user_messages(tmp_path, monkeypatch):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    monkeypatch.setattr(reachy_cli.uuid, 'uuid4', lambda: uuid.UUID(selected))
    session = ChatSession(events=[
        {'event': 'task.status', 'payload': {'session_id': 'other-chat', 'status': 'completed'}},
        {'event': 'delta.message_done', 'payload': {'session_id': selected, 'message_id': 'setup-user'}},
        {'event': 'task.status', 'payload': {'session_id': selected, 'status': 'running',
                                            'task_id': 'setup-task'}},
        {'event': 'delta.message_start', 'payload': {'session_id': selected,
                                                    'message_id': 'setup-assistant'}},
        {'event': 'delta.text_append', 'payload': {'session_id': selected,
                                                  'message_id': 'setup-assistant',
                                                  'parent_message_id': 'setup-assistant',
                                                  'text': f'Ready {SETUP_NONCE}'}},
        {'event': 'task.status', 'payload': {'session_id': selected, 'status': 'completed',
                                            'task_id': 'setup-task'}},
    ])
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1')) == selected
    assert session.subscription_closed


def test_setup_drain_rejects_completed_turn_with_stale_challenge(tmp_path, monkeypatch):
    monkeypatch.setattr(reachy_cli, 'CHAT_SETUP_TIMEOUT_S', 0.01)
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    monkeypatch.setattr(reachy_cli.uuid, 'uuid4', lambda: uuid.UUID(selected))
    session = ChatSession(events=[
        {'event': 'task.status', 'payload': {'session_id': selected, 'status': 'running',
                                            'task_id': 'old-task'}},
        {'event': 'delta.message_start', 'payload': {'session_id': selected,
                                                    'message_id': 'old-assistant'}},
        {'event': 'delta.text_append', 'payload': {'session_id': selected,
                                                  'message_id': 'old-assistant',
                                                  'text': 'Ready stale-nonce'}},
        {'event': 'task.status', 'payload': {'session_id': selected, 'status': 'completed',
                                            'task_id': 'old-task'}},
    ])
    with pytest.raises(RuntimeError, match='did not finish'):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
    assert 'applied_setup_sha256' not in saved['chats']['vm-1']


def test_exact_challenge_in_final_response_succeeds_without_task_metadata(tmp_path):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    config.save_json(reachy_cli.CHAT_STATE_FILE, {
        'version': 1, 'device_id': 'reachy-1',
        'chats': {'vm-1': {'session_id': selected, 'initialized': True}},
    }, tmp_path)
    session = ChatSession(200, events=[
        {'event': 'message.assistant', 'payload': {
            'session_id': selected, 'message_id': 'setup-assistant',
            'display_text': f'Ready {SETUP_NONCE}',
        }},
    ])
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(
        session, 'vm-1')) == selected


@pytest.mark.parametrize('completion', ['root-task', 'assistant-final'])
def test_setup_child_completion_cannot_release_a_still_running_root(completion):
    async def check():
        selected, challenge = 'public-side-chat', 'Ready public-challenge'
        child_done, finish = asyncio.Event(), asyncio.Event()
        armed = asyncio.Event()
        armed.set()

        async def events():
            for task_id in ('root-task', 'child-task'):
                yield {'event': 'task.status', 'payload': {
                    'session_id': selected, 'task_id': task_id, 'status': 'running'}}
            yield {'event': 'delta.message_start', 'payload': {
                'session_id': selected, 'message_id': 'setup-assistant'}}
            yield {'event': 'delta.text_append', 'payload': {
                'session_id': selected, 'message_id': 'setup-assistant', 'text': challenge}}
            yield {'event': 'task.status', 'payload': {
                'session_id': selected, 'task_id': 'child-task', 'status': 'completed'}}
            child_done.set()
            await finish.wait()
            if completion == 'root-task':
                yield {'event': 'task.status', 'payload': {
                    'session_id': selected, 'task_id': 'root-task', 'status': 'completed'}}
            else:
                yield {'event': 'delta.message_done', 'payload': {
                    'session_id': selected, 'message_id': 'setup-assistant',
                    'display_text': challenge}}

        drain = asyncio.create_task(reachy_cli.ReachySideChat._drain_setup(
            events(), selected, challenge, armed))
        try:
            await asyncio.wait_for(child_done.wait(), 1)
            assert not drain.done()
            finish.set()
            await asyncio.wait_for(drain, 1)
        finally:
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)

    asyncio.run(check())


@pytest.mark.parametrize('status', ['failed', 'errored', 'cancelled', 'canceled'])
def test_failed_setup_task_is_never_marked_applied(tmp_path, status):
    session = ChatSession(events=[
        {'event': 'task.status', 'payload': {
            'session_id': 'unused', 'status': 'running', 'task_id': 'setup-task'}},
        {'event': 'task.status', 'payload': {
            'session_id': 'unused', 'status': status, 'task_id': 'setup-task'}},
    ])
    # Fill the generated chat ID into the test event at subscription time.
    original_subscribe = session.subscribe_chat
    async def subscribe(session_id):
        for event in session.events:
            event['payload']['session_id'] = session_id
        async for event in original_subscribe(session_id):
            yield event
    session.subscribe_chat = subscribe
    with pytest.raises(RuntimeError, match='did not apply'):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
    assert 'applied_setup_sha256' not in saved['chats']['vm-1']


def test_setup_without_terminal_event_times_out_and_closes_subscription(tmp_path, monkeypatch):
    monkeypatch.setattr(reachy_cli, 'CHAT_SETUP_TIMEOUT_S', 0.01)
    session = ChatSession(events=[])
    with pytest.raises(RuntimeError, match='did not finish'):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    assert session.subscription_closed


def test_setup_completion_without_fresh_challenge_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(reachy_cli, 'CHAT_SETUP_TIMEOUT_S', 0.01)
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    monkeypatch.setattr(reachy_cli.uuid, 'uuid4', lambda: uuid.UUID(selected))
    session = ChatSession(events=[
        {'event': 'task.status', 'payload': {'session_id': selected, 'status': 'running',
                                            'task_id': 'setup-task'}},
        {'event': 'task.status', 'payload': {'session_id': selected, 'status': 'completed',
                                            'task_id': 'setup-task'}},
    ])
    with pytest.raises(RuntimeError, match='did not finish'):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    assert session.subscription_closed


def test_cancelled_setup_closes_subscription_and_keeps_persisted_id(tmp_path):
    async def check():
        session = ChatSession(events=[])
        pending = asyncio.create_task(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
        while not session.posts:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert session.subscription_closed
        saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
        assert saved['chats']['vm-1'] == {
            'session_id': session.posts[0][1], 'initialized': False,
            'pending_setup_sha256': first_setup_hash(),
        }
        resumed = ChatSession(200)
        assert await reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(resumed, 'vm-1') == session.posts[0][1]
        assert resumed.subscription_closed
        assert resumed.posts == [(setup_wire_message(), session.posts[0][1])]
    asyncio.run(check())


def test_existing_uninitialized_chat_reapplies_setup_before_accepting_completion(tmp_path):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    config.save_json(reachy_cli.CHAT_STATE_FILE, {
        'version': 2, 'device_id': 'reachy-1', 'chats': {
            'vm-1': {'session_id': selected, 'initialized': False,
                     'pending_setup_sha256': first_setup_hash(),
                     'setup_ack': {'message_id': 'setup-user'}},
        },
    }, tmp_path)
    session = ChatSession(200, events=[
        {'event': 'delta.message_start', 'payload': {'session_id': selected,
                                                    'message_id': 'setup-assistant'}},
        {'event': 'message.assistant', 'payload': {'session_id': selected,
                                                  'message_id': 'setup-assistant',
                                                  'parent_message_id': 'setup-assistant',
                                                  'display_text': f'Ready {SETUP_NONCE}'}},
    ])
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1')) == selected
    assert session.posts == [(setup_wire_message(), selected)]
    assert session.subscription_closed
    saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
    assert saved['chats']['vm-1']['initialized'] is True


def test_existing_chat_after_lost_ack_reposts_before_marking_setup_applied(tmp_path):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    config.save_json(reachy_cli.CHAT_STATE_FILE, {
        'version': 1, 'device_id': 'reachy-1',
        'chats': {'vm-1': {'session_id': selected, 'initialized': False}},
    }, tmp_path)
    session = ChatSession(200)
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1')) == selected
    assert session.posts == [(setup_wire_message(), selected)]
    assert session.subscription_closed


def test_setup_eof_without_completion_keeps_initialization_pending(tmp_path, monkeypatch):
    class EndedSession(ChatSession):
        async def subscribe_chat(self, session_id):
            self.chat_subscribed.set()
            if False:
                yield {}

    monkeypatch.setattr(reachy_cli, 'CHAT_SETUP_TIMEOUT_S', 0.01)
    session = EndedSession()
    with pytest.raises(ConnectionError, match='subscription ended'):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))
    saved = json.loads((tmp_path / reachy_cli.CHAT_STATE_FILE).read_text())
    assert saved['chats']['vm-1']['initialized'] is False


def test_existing_chat_subscription_failure_is_reported_without_waiting_for_ready(tmp_path):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    config.save_json(reachy_cli.CHAT_STATE_FILE, {
        'version': 1, 'device_id': 'reachy-1',
        'chats': {'vm-1': {'session_id': selected, 'initialized': True}},
    }, tmp_path)

    class ClosedBeforeReady(ChatSession):
        async def subscribe_chat(self, session_id):
            if False:
                yield {}

    with pytest.raises(ConnectionError, match='subscription ended'):
        asyncio.run(asyncio.wait_for(
            reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(
                ClosedBeforeReady(200), 'vm-1'),
            .1,
        ))


def test_setup_message_count_bounds_final_events_without_start():
    async def events():
        for index in range(33):
            yield {'event': 'message.assistant', 'payload': {
                'session_id': 'side-chat', 'message_id': f'assistant-{index}',
                'display_text': 'not the challenge',
            }}

    armed = asyncio.Event()
    armed.set()
    with pytest.raises(RuntimeError, match='too many side-chat setup messages'):
        asyncio.run(reachy_cli.ReachySideChat._drain_setup(
            events(), 'side-chat', f'Ready {SETUP_NONCE}', armed))


@pytest.mark.parametrize('thread_value', [False, None, 'true'])
def test_creation_requires_explicit_thread_confirmation(tmp_path, monkeypatch, thread_value):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    monkeypatch.setattr(reachy_cli.uuid, 'uuid4', lambda: uuid.UUID(selected))
    session = ChatSession(ack={'ok': True, 'response': {'session_id': selected,
                                                      'is_thread': thread_value}})
    with pytest.raises(RuntimeError, match='did not confirm'):
        asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1'))


def test_nested_creation_ack_is_accepted(tmp_path, monkeypatch):
    selected = '9dc752fa-9df0-4e28-a554-dc737e29cc74'
    monkeypatch.setattr(reachy_cli.uuid, 'uuid4', lambda: uuid.UUID(selected))
    session = ChatSession(ack={'ok': True, 'response': {'result': {'session_id': selected,
                                                                 'is_thread': True,
                                                                 'message_id': 'setup-user'}}})
    assert asyncio.run(reachy_cli.ReachySideChat('reachy-1', tmp_path).prepare(session, 'vm-1')) == selected


@pytest.mark.parametrize('selection, expected_session, owned', [
    ([], None, True), (['--main-chat'], None, False),
    (['--session-id', 'existing-side-chat'], 'existing-side-chat', False),
])
def test_cli_selects_owned_side_chat_by_default_and_preserves_explicit_routes(
        tmp_path, monkeypatch, selection, expected_session, owned):
    from musegadget.identity import Identity

    captured = {}
    class Hardware:
        def start(self):
            pass

        def close(self):
            pass

    class Service:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def stop(self):
            pass

        async def run(self):
            pass

    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli.identity, 'load_or_create', lambda: Identity('02:00:00:ab:cd:ef'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: Hardware())
    monkeypatch.setattr('musegadget.reachy_voice.ReachyService', Service)
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', *selection]) == 0
    assert captured['session_id'] == expected_session
    assert captured['owns_chat'] is owned
    prepare = captured['prepare_chat']
    assert (prepare is not None) is owned
    if owned:
        selected = asyncio.run(prepare(ChatSession(), 'vm-1'))
        reconnect = ChatSession(200)
        assert asyncio.run(prepare(reconnect, 'vm-1')) == selected
        assert reconnect.posts == []
        assert "one JSON object per line" not in prepare.__self__.setup_message
        assert "Both antennas are enabled" in prepare.__self__.setup_message
    else:
        assert not (tmp_path / reachy_cli.CHAT_STATE_FILE).exists()


def test_main_and_explicit_session_conflict_before_hardware(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: (_ for _ in ()).throw(
        AssertionError('hardware started with conflicting chat options')))
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--main-chat',
                           '--session-id', 'existing-side-chat']) == 1
    assert 'choose either --main-chat or --session-id' in capsys.readouterr().err


def test_native_media_connects_to_local_daemon(monkeypatch):
    captured = {}
    monkeypatch.setattr('musegadget.reachy_hardware.ReachyController',
                        lambda **kwargs: captured.update(kwargs))
    reachy_cli._hardware(reachy_cli.parser().parse_args(['run']))
    assert captured == {'host': 'localhost', 'port': 8000,
                        'connection_mode': 'localhost_only', 'media_backend': 'local',
                        'enable_motion': True, 'antenna_mode': 'both', 'face_follow_model': None}


def test_remote_media_uses_supported_sdk_connection_mode(monkeypatch):
    captured = {}
    monkeypatch.setattr('musegadget.reachy_hardware.ReachyController',
                        lambda **kwargs: captured.update(kwargs))
    reachy_cli._hardware(reachy_cli.parser().parse_args(
        ['run', '--host', '192.0.2.1', '--media-backend', 'webrtc']))
    assert captured['connection_mode'] == 'network'
    assert captured['media_backend'] == 'webrtc'


def test_unpaired_run_does_not_enable_hardware(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(config.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: (_ for _ in ()).throw(
        AssertionError('hardware started before pairing')))
    assert asyncio.run(reachy_cli.run(reachy_cli.parser().parse_args(['run']))) == 1
    assert 'not paired with Muse' in capsys.readouterr().err


def test_invalid_token_is_rejected_without_printing_it(tmp_path, monkeypatch, capsys):
    token = tmp_path / 'token'
    token.write_text('secret-invalid-value')
    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    assert reachy_cli.main(['--state-dir', str(tmp_path), '--sdk-token-file', str(token),
                           'doctor']) == 1
    assert 'secret-invalid-value' not in capsys.readouterr().err


def test_pair_uses_explicit_state_and_existing_cli(tmp_path, monkeypatch):
    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    captured = {}
    def pair(args):
        captured.update(state=config.state_dir(), force=args.force, timeout=args.timeout)
        return 0
    monkeypatch.setattr('musegadget.cli.cmd_pair', pair)
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'pair', '--timeout', '90']) == 0
    assert captured == {'state': tmp_path, 'force': False, 'timeout': 90}


def test_streaming_option_reaches_the_conversation_service(tmp_path, monkeypatch):
    from musegadget.identity import Identity

    calls = []
    captured = {}

    class Hardware:
        def start(self):
            calls.append('hardware started')

        def close(self):
            calls.append('hardware closed')

    class Speech:
        def __init__(self, name):
            self.name = name

        async def start(self):
            calls.append(self.name + ' started')

        async def close(self):
            calls.append(self.name + ' closed')

    class Service:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def stop(self):
            pass

        async def run(self):
            calls.append('service ran')

    speech = Speech('speech')
    progress_speech = Speech('progress speech')
    speech_backends = iter((speech, progress_speech))
    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli.identity, 'load_or_create', lambda: Identity('02:00:00:ab:cd:ef'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: Hardware())
    monkeypatch.setattr('musegadget.local_speech.PiperSpeech', lambda model: next(speech_backends))
    monkeypatch.setattr('musegadget.speech_gate.SileroSpeechGate', object)
    monkeypatch.setattr('musegadget.local_transcription.WhisperTranscriber', lambda model: Speech('transcriber'))
    monkeypatch.setattr('musegadget.reachy_voice.ReachyService', Service)
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--mode', 'on-robot',
                           '--stt-model', str(tmp_path / 'whisper'), '--tts-model',
                           str(tmp_path / 'voice.onnx'), '--stream-replies']) == 0
    backends = captured['backends'](object())
    assert backends.reply_style() is ReplyStyle.EXPRESSIVE_JSON
    assert backends.voice.speech is speech
    assert backends.progress_voice.speech is progress_speech
    assert calls == ['hardware started', 'speech started', 'progress speech started', 'transcriber started',
                     'service ran', 'hardware closed', 'speech closed', 'progress speech closed',
                     'transcriber closed']


@pytest.mark.parametrize('timeout', ['0', '-1', '301', 'nan', 'inf'])
def test_invalid_wake_timeout_is_rejected_before_hardware(tmp_path, monkeypatch, capsys, timeout):
    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: (_ for _ in ()).throw(
        AssertionError('hardware started with an invalid wake timeout')))
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--wake-timeout', timeout]) == 1
    assert '--wake-timeout must be between 1 and 300 seconds' in capsys.readouterr().err


@pytest.mark.parametrize('backend', [None, 'sherpa', 'vosk'])
@pytest.mark.parametrize('stt_backend', ['whisper', 'moonshine', 'sherpa-streaming'])
def test_wake_model_phrase_and_idle_timeout_reach_the_service(tmp_path, monkeypatch, caplog, backend, stt_backend):
    from musegadget.identity import Identity

    captured = {}
    calls = []

    class Hardware:
        def start(self):
            calls.append('start')

        def close(self):
            calls.append('close')

    class Transcriber:
        async def start(self):
            pass

        async def close(self):
            pass

    class Service:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def stop(self):
            pass

        async def run(self):
            calls.append('run')

    class Detector:
        closed = False
        timing_supported = backend != 'vosk'

        def close(self):
            self.closed = True

    detector = Detector()
    def wake(model, *, phrase):
        captured['model'] = model
        captured['phrase'] = phrase
        return detector

    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli.identity, 'load_or_create', lambda: Identity('02:00:00:ab:cd:ef'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: Hardware())
    transcriber = Transcriber()
    def recognize(model):
        captured['stt_model'] = model
        return transcriber
    factories = {
        'whisper': 'musegadget.local_transcription.WhisperTranscriber',
        'moonshine': 'musegadget.local_transcription.MoonshineTranscriber',
        'sherpa-streaming': 'musegadget.streaming_transcription.SherpaStreamingTranscriber',
    }
    for name, factory in factories.items():
        monkeypatch.setattr(factory, recognize if name == stt_backend else
                            lambda model: pytest.fail('wrong transcription backend selected'))
    monkeypatch.setattr('musegadget.local_speech.PiperSpeech', lambda model: Transcriber())
    speech_gate = object()
    monkeypatch.setattr('musegadget.speech_gate.SileroSpeechGate', lambda: speech_gate)
    if backend == 'vosk':
        monkeypatch.setitem(sys.modules, 'musegadget.vosk_wake', SimpleNamespace(VoskWakeWordDetector=wake))
        monkeypatch.setitem(sys.modules, 'musegadget.wake_word', None)
    else:
        monkeypatch.setitem(sys.modules, 'musegadget.wake_word', SimpleNamespace(WakeWordDetector=wake))
        monkeypatch.setitem(sys.modules, 'musegadget.vosk_wake', None)
    monkeypatch.setattr('musegadget.reachy_voice.ReachyService', Service)
    caplog.set_level('INFO', logger='musegadget.reachy_cli')
    model = tmp_path / 'keywords'
    backend_args = ['--wake-backend', backend] if backend is not None else []
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--mode', 'on-robot', '--stt-model',
                           str(tmp_path / 'whisper'), '--wake-model', str(model),
                           '--tts-model', str(tmp_path / 'voice.onnx'),
                           '--wake-phrase', 'hey muse', '--wake-timeout', '10',
                           '--stt-backend', stt_backend, *backend_args]) == 0
    assert captured['wake_detector'] is detector
    assert captured['backends'](object()).hearing.transcriber is transcriber
    assert captured['speech_gate'] is speech_gate
    assert captured['stt_model'] == tmp_path / 'whisper'
    assert (captured['model'], captured['phrase'], captured['wake_timeout_s']) == (model, 'hey muse', 10.0)
    assert calls == ['start', 'run', 'close']
    assert detector.closed
    assert ('Reachy timed wake handoff ready' in caplog.text) is detector.timing_supported


def test_vosk_backend_requires_a_model_before_hardware(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setitem(sys.modules, 'musegadget.vosk_wake', None)
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: (_ for _ in ()).throw(
        AssertionError('hardware started without a Vosk model')))
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--wake-backend', 'vosk']) == 1
    assert '--wake-backend vosk requires --wake-model' in capsys.readouterr().err


@pytest.mark.parametrize('operation, failed_resource', [
    ('setup', 'service'),
    ('start', 'hardware'), ('start', 'speech'), ('start', 'progress'), ('start', 'transcriber'),
    ('close', 'hardware'), ('close', 'speech'), ('close', 'progress'),
    ('close', 'transcriber'), ('close', 'wake'),
])
@pytest.mark.parametrize('stt_backend', ['whisper', 'moonshine', 'sherpa-streaming'])
def test_startup_and_cleanup_failures_attempt_all_resource_cleanup(
        tmp_path, monkeypatch, capsys, operation, failed_resource, stt_backend):
    from musegadget.identity import Identity
    monkeypatch.setattr('musegadget.speech_gate.SileroSpeechGate', object)

    class Resource:
        def __init__(self, name):
            self.name = name
            self.close_attempted = False

        def start_resource(self):
            if operation == 'start' and self.name == failed_resource:
                raise RuntimeError(self.name + ' start failed')

        def close_resource(self):
            self.close_attempted = True
            if operation == 'close' and self.name == failed_resource:
                raise RuntimeError(self.name + ' close failed')

    class Hardware(Resource):
        def start(self):
            self.start_resource()

        def close(self):
            self.close_resource()

    class Backend(Resource):
        async def start(self):
            self.start_resource()

        async def close(self):
            self.close_resource()

    class Detector(Resource):
        def close(self):
            self.close_resource()

    hardware = Hardware('hardware')
    speech = Backend('speech')
    progress_speech = Backend('progress')
    transcriber = Backend('transcriber')
    detector = Detector('wake')
    speech_backends = iter((speech, progress_speech))
    service_ran = []

    class Service:
        def __init__(self, **kwargs):
            backends = kwargs['backends'](object())
            assert backends.voice.speech is speech
            assert backends.progress_voice.speech is progress_speech
            if operation == 'setup':
                raise RuntimeError('service setup failed')

        def stop(self):
            pass

        async def run(self):
            service_ran.append(True)

    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli.identity, 'load_or_create', lambda: Identity('02:00:00:ab:cd:ef'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: hardware)
    monkeypatch.setattr('musegadget.local_speech.PiperSpeech', lambda model: next(speech_backends))
    factory = {
        'whisper': 'musegadget.local_transcription.WhisperTranscriber',
        'moonshine': 'musegadget.local_transcription.MoonshineTranscriber',
        'sherpa-streaming': 'musegadget.streaming_transcription.SherpaStreamingTranscriber',
    }[stt_backend]
    monkeypatch.setattr(factory, lambda model: transcriber)
    monkeypatch.setitem(sys.modules, 'musegadget.wake_word', SimpleNamespace(
        WakeWordDetector=lambda model, phrase: detector))
    monkeypatch.setattr('musegadget.reachy_voice.ReachyService', Service)
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--mode', 'on-robot',
                           '--tts-model', str(tmp_path / 'voice.onnx'),
                           '--stt-model', str(tmp_path / 'whisper'),
                           '--wake-model', str(tmp_path / 'wake'), '--stt-backend', stt_backend]) == 1
    assert failed_resource + ' ' + operation + ' failed' in capsys.readouterr().err
    assert all(resource.close_attempted for resource in
               (hardware, speech, progress_speech, transcriber, detector))
    assert service_ran == ([True] if operation == 'close' else [])


def test_missing_speech_gate_stops_before_capture_and_closes_prepared_resources(
        tmp_path, monkeypatch, capsys):
    from musegadget.identity import Identity

    closed = []

    class Hardware:
        def start(self):
            raise AssertionError('capture started without a speech gate')

        def close(self):
            closed.append('hardware')

    class Speech:
        async def close(self):
            closed.append('speech')

    def missing_gate():
        raise RuntimeError('The faster-whisper Silero VAD asset is missing')

    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli.identity, 'load_or_create', lambda: Identity('02:00:00:ab:cd:ef'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: Hardware())
    monkeypatch.setattr('musegadget.local_speech.PiperSpeech', lambda model: Speech())
    monkeypatch.setattr('musegadget.speech_gate.SileroSpeechGate', missing_gate)
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', '--mode', 'on-robot',
                           '--tts-model', str(tmp_path / 'voice.onnx'),
                           '--stt-model', str(tmp_path / 'whisper')]) == 1
    assert 'Silero VAD asset is missing' in capsys.readouterr().err
    assert closed == ['hardware', 'speech', 'speech']


@pytest.mark.parametrize('argv, errors', [
    (['--stt-model', 'whisper'], ['--stt-model needs --mode on-robot']),
    (['--stt-model', 'whisper', '--tts-model', 'voice.onnx'],
     ['--stt-model needs --mode on-robot', '--tts-model needs --mode on-robot']),
    (['--mode', 'muse-voice', '--tts-model', 'voice.onnx', '--stream-replies'],
     ['--tts-model needs --mode on-robot', '--stream-replies needs --mode on-robot']),
    (['--stt-backend', 'moonshine'], ['--stt-backend needs --mode on-robot']),
    (['--wake-model', 'keywords'], ['--wake-model needs --mode on-robot']),
    (['--mode', 'on-robot'], ['--mode on-robot needs --stt-model', '--mode on-robot needs --tts-model']),
    (['--mode', 'on-robot', '--stt-model', 'whisper', '--wake-model', 'keywords'],
     ['--mode on-robot needs --tts-model']),
])
def test_each_mode_rejects_missing_and_foreign_flags_before_anything_starts(
        tmp_path, monkeypatch, capsys, argv, errors):
    monkeypatch.setattr(config, 'load_json', lambda *args: pytest.fail('pairing read with invalid flags'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: pytest.fail('hardware started with invalid flags'))
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', *argv]) == 2
    assert capsys.readouterr().err == ''.join(f'Reachy: {error}\n' for error in errors)


def test_companion_mode_is_not_offered_yet(capsys):
    with pytest.raises(SystemExit) as raised:
        reachy_cli.main(['run', '--mode', 'companion'])
    assert raised.value.code == 2
    assert "argument --mode: invalid choice: 'companion'" in capsys.readouterr().err


@pytest.mark.parametrize('argv, style, voice', [
    ([], ReplyStyle.MUSE_VOICE, 'MuseVoice'),
    (['--mode', 'muse-voice'], ReplyStyle.MUSE_VOICE, 'MuseVoice'),
    (['--mode', 'on-robot', '--stt-model', 'whisper', '--tts-model', 'voice.onnx'],
     ReplyStyle.MARKER, 'LocalVoice'),
    (['--mode', 'on-robot', '--stt-model', 'whisper', '--tts-model', 'voice.onnx', '--stream-replies'],
     ReplyStyle.EXPRESSIVE_JSON, 'LocalVoice'),
])
def test_each_mode_starts_the_conversation_with_its_reply_style_and_voice(tmp_path, monkeypatch, argv, style, voice):
    from musegadget.identity import Identity

    captured = {}

    class Hardware:
        def start(self):
            pass

        def close(self):
            pass

    class Model:
        async def start(self):
            pass

        async def close(self):
            pass

    class Service:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def stop(self):
            pass

        async def run(self):
            pass

    monkeypatch.delenv(config.SDK_TOKEN_ENV, raising=False)
    monkeypatch.setattr(config, 'load_json', lambda *args: {'paired': True})
    monkeypatch.setattr(reachy_cli.identity, 'load_or_create', lambda: Identity('02:00:00:ab:cd:ef'))
    monkeypatch.setattr(reachy_cli, '_hardware', lambda *args: Hardware())
    monkeypatch.setattr('musegadget.local_speech.PiperSpeech', lambda model: Model())
    monkeypatch.setattr('musegadget.local_transcription.WhisperTranscriber', lambda model: Model())
    monkeypatch.setattr('musegadget.speech_gate.SileroSpeechGate', object)
    monkeypatch.setattr('musegadget.reachy_voice.ReachyService', Service)
    assert reachy_cli.main(['--state-dir', str(tmp_path), 'run', *argv]) == 0
    backends = captured['backends'](object())
    assert (backends.reply_style(), type(backends.voice).__name__) == (style, voice)
