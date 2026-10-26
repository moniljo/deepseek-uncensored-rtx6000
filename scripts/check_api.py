"""Default-dry, single-attempt API checks. Never manages a model or executes tools.

Protocol checks derive from the earlier retained production verifier. This
portable version has no deployment hostname, private path, or embedded key.
Only --run sends requests and creates one exclusive, compact JSON receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOCAL = 'http://127.0.0.1:8080'
MAX_CONTEXT = 1048576
MAX_RESPONSE = 2 << 20
MAX_PROMPT = 12 << 20
PATHS = {'/v1/models', '/metrics', '/health', '/status', '/v1/chat/completions', '/v1/messages'}
TOOL_SCHEMA = {'type': 'object', 'properties': {'value': {'type': 'integer', 'enum': [7]}},
               'required': ['value'], 'additionalProperties': False}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(text):
    def invalid(_):
        raise RuntimeError('Non-finite JSON is forbidden')
    return json.loads(text, parse_constant=invalid)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def redacted(value, key):
    if isinstance(value, dict):
        return {name: ('[REDACTED]' if name.lower() in
                       {'authorization', 'api_key', 'api_keys', 'x-api-key'}
                       else redacted(item, key)) for name, item in value.items()}
    if isinstance(value, list):
        return [redacted(item, key) for item in value]
    if isinstance(value, str) and key:
        return value.replace(key, '[REDACTED]')
    return value


def origin(value):
    need(isinstance(value, str) and value and not any(c.isspace() for c in value),
         'A single HTTP origin is required')
    p = urlsplit(value)
    need(p.scheme in {'http', 'https'} and p.hostname and not p.username and not p.password
         and not p.query and not p.fragment and p.path in {'', '/'},
         'Base must be an origin without credentials, path, query, or fragment')
    port = p.port  # Reject malformed or out-of-range ports.
    need(port is None or 1 <= port <= 65535, 'Invalid endpoint port')
    need(p.scheme == 'https' or p.hostname == '127.0.0.1',
         'Plain HTTP is restricted to the loopback IPv4 address')
    need('\\' not in value and all(ord(c) < 128 for c in value), 'Use an ASCII HTTPS hostname')
    return p.scheme + '://' + p.netloc


def natural(body, expected):
    choices = body.get('choices', [])
    need(len(choices) == 1, 'Exactly one OpenAI choice required')
    c = choices[0]
    text = c.get('message', {}).get('content')
    need(c.get('finish_reason') == 'stop' and isinstance(text, str)
         and text.strip() == expected and not c.get('message', {}).get('tool_calls'),
         'Natural completion or content differs')
    return {'natural_stop': True, 'answer_sha256': sha(text.strip().encode())}


def synthetic_arguments(value):
    need(isinstance(value, dict) and set(value) == {'value'}
         and type(value['value']) is int and value['value'] == 7,
         'Synthetic arguments must contain only the integer value 7')


def bounded_bytes(chunks, tick=lambda: None, limit=MAX_RESPONSE):
    body = bytearray()
    for chunk in chunks:
        tick()
        need(len(body) + len(chunk) <= limit, 'Response exceeds bounded size')
        body.extend(chunk)
    return bytes(body)


def bounded_lines(chunks, tick=lambda: None, limit=MAX_RESPONSE):
    """Check bytes/deadline before buffering a line, including malformed no-newline input."""
    pending, total = bytearray(), 0
    for chunk in chunks:
        tick()
        total += len(chunk)
        need(total <= limit, 'SSE response exceeds bounded size')
        for byte in chunk:
            if byte == 10:
                yield bytes(pending).removesuffix(b'\r')
                pending.clear()
            else:
                pending.append(byte)
    if pending:
        yield bytes(pending).removesuffix(b'\r')


def openai_tool(body):
    choices = body.get('choices', [])
    need(len(choices) == 1 and choices[0].get('finish_reason') == 'tool_calls',
         'OpenAI forced-tool finish differs')
    calls = choices[0].get('message', {}).get('tool_calls', [])
    need(len(calls) == 1 and calls[0].get('type') == 'function'
         and isinstance(calls[0].get('id'), str) and calls[0]['id'],
         'One identified synthetic function call required')
    f = calls[0].get('function', {})
    need(f.get('name') == 'report_result' and isinstance(f.get('arguments'), str),
         'Synthetic function binding differs')
    synthetic_arguments(strict_json(f['arguments']))


def anthropic_tool(body):
    calls = [x for x in body.get('content', []) if x.get('type') == 'tool_use']
    need(body.get('stop_reason') == 'tool_use' and len(calls) == 1
         and calls[0].get('name') == 'report_result'
         and isinstance(calls[0].get('id'), str) and calls[0]['id'],
         'Anthropic synthetic tool call differs')
    synthetic_arguments(calls[0].get('input'))


def sse_values(lines, stats, tick=lambda: None):
    """Decode SSE data frames; comments never become output/token observations."""
    data = []
    size = 0
    for line in lines:
        tick()
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        size += len(line.encode('utf-8'))
        need(size <= MAX_RESPONSE, 'SSE response exceeds the bounded size')
        if line.startswith(':'):
            stats['heartbeat_comments'] += 1
            continue
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            raw = '\n'.join(data)
            data = []
            stats['data_frames'] += 1
            yield '[DONE]' if raw == '[DONE]' else strict_json(raw)
    need(not data, 'Incomplete final SSE frame')


def stream_result(values, expected, *, require_long=False, expected_prompt_tokens=None, diagnostic=None):
    content, finishes, usage = [], [], []
    done = 0
    frames = 0
    for event in values:
        need(done == 0, 'Data after terminal SSE DONE')
        if event == '[DONE]':
            done += 1
            continue
        need(isinstance(event, dict) and 'error' not in event, 'SSE error or non-object frame')
        if event.get('usage') is not None:
            need(isinstance(event['usage'], dict), 'Typed usage required')
            usage.append(event['usage'])
        choices = event.get('choices', [])
        need(isinstance(choices, list) and len(choices) <= 1, 'At most one streamed choice required')
        for choice in choices:
            need(choice.get('index', 0) == 0, 'Unexpected choice index')
            delta = choice.get('delta', {})
            need(isinstance(delta, dict) and not delta.get('tool_calls'), 'Unexpected streamed tool call')
            text = delta.get('content')
            if text is not None:
                need(isinstance(text, str), 'Nontext SSE content')
                content.append(text)
                if text:
                    frames += 1
            if choice.get('finish_reason') is not None:
                finishes.append(choice['finish_reason'])
    text = ''.join(content).strip()
    if diagnostic is not None:
        # Save only metadata BEFORE the unchanged assertions. A failed first
        # result must remain inspectable without saving its plaintext content.
        def numeric(value):
            return value if value is None or type(value) in (int, float, bool) else {'type': type(value).__name__}
        usage_numbers = []
        for item in usage:
            details = item.get('prompt_tokens_details')
            usage_numbers.append({'prompt_tokens': numeric(item.get('prompt_tokens')),
                'completion_tokens': numeric(item.get('completion_tokens')),
                'cached_tokens': numeric(details.get('cached_tokens') if isinstance(details, dict) else None),
                'total_tokens': numeric(item.get('total_tokens'))})
        diagnostic.update(answer_sha256=sha(text.encode()), answer_characters=len(text),
            answer_UTF8_bytes=len(text.encode()), exact_expected_match=text == expected,
            expected_answer_sha256=sha(expected.encode()), DONE_count=done,
            finish_reasons=[reason if reason in ('stop', 'length', 'tool_calls', 'content_filter')
                            else {'type': type(reason).__name__} for reason in finishes],
            usage_count=len(usage), usage_numbers=usage_numbers,
            content_frames_not_tokens=frames, plaintext_saved=False)
    need(done == 1 and finishes == ['stop'] and text == expected,
         'Complete natural SSE stop, exact answer and DONE required')
    result = {'natural_stop': True, 'DONE_count': done, 'finish_reasons': finishes,
              'content_frames_not_tokens': frames, 'answer_sha256': sha(text.encode()),
              'reasoning_and_heartbeat_not_counted_as_tokens': True}
    if require_long:
        need(len(usage) == 1, 'One final long-request usage object required')
        u = usage[0]
        prompt, completion = u.get('prompt_tokens'), u.get('completion_tokens')
        cached = u.get('prompt_tokens_details', {}).get('cached_tokens')
        need(integer(prompt) and 900000 <= prompt <= MAX_CONTEXT and integer(completion, 1)
             and integer(cached) and cached == 0, 'Actual near-1M uncached token accounting required')
        if expected_prompt_tokens is not None:
            need(prompt == expected_prompt_tokens, 'Long prompt token count differs')
        if 'total_tokens' in u:
            need(integer(u['total_tokens']) and u['total_tokens'] == prompt + completion,
                 'Long usage total differs')
        result['usage'] = {'prompt_tokens': prompt, 'completion_tokens': completion,
                           'cached_tokens': cached, 'total_tokens': u.get('total_tokens')}
    return result


def chat(model, prompt, stream=False, maximum=128):
    value = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
             'temperature': 0, 'seed': 12345, 'reasoning_effort': 'none',
             'max_tokens': maximum, 'stream': stream}
    if stream:
        value['stream_options'] = {'include_usage': True}
    return value


def long_chat(model, prompt, preset='none'):
    """Explicit FIRST-long client settings; every short/tool request stays unchanged.

    DeepSeek high renders effort 75. That is a prompt level, not a token budget.
    The separate bounded thinking budget here is 256 tokens, with 1024 total.
    """
    need(preset in ('none', 'high-budget256'), 'Unknown long-only thinking preset')
    if preset == 'none':
        return chat(model, prompt, True, 256)
    value = chat(model, prompt, True, 1024)
    value.update(reasoning_effort='high', reasoning_budget_tokens=256)
    return value


class API:
    def __init__(self, key, base, report, seconds):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False
        self.key, self.base, self.report = key, base, report
        self.deadline = time.monotonic() + seconds

    def remaining(self):
        left = self.deadline - time.monotonic()
        need(left > 0, 'Whole verification deadline exceeded')
        return left

    def request(self, path, *, payload=None, auth=True, anthropic=False, stream=False):
        need(path in PATHS, 'Endpoint is not permitted')
        headers = {'Accept': 'text/event-stream' if stream else 'application/json',
                   'Cache-Control': 'no-cache'}
        if auth:
            headers.update({'x-api-key': self.key, 'anthropic-version': '2023-06-01'}
                           if anthropic else {'Authorization': 'Bearer ' + self.key})
        left = self.remaining()
        # Explicit CLI origin only; no inherited proxies or redirect credential forwarding.
        return self.session.request('POST' if payload is not None else 'GET', self.base + path,
                                    headers=headers, json=payload, stream=True,
                                    allow_redirects=False, timeout=(min(5, left), min(20, left)))

    def row(self, name, path, auth=True):
        row = {'name': name, 'path': path, 'authenticated': auth, 'passed': False}
        self.report['checks'].append(row)
        return row

    def json(self, name, path, *, payload=None, auth=True, anthropic=False, status=200):
        row = self.row(name, path, auth)
        if payload is not None:
            self.report['inference_order'].append(name)
        with self.request(path, payload=payload, auth=auth, anthropic=anthropic) as response:
            row.update(status=response.status_code)
            # A one-byte transport chunk exposes byte-dribble progress to the deadline.
            raw = bounded_bytes(response.iter_content(chunk_size=1), self.remaining)
            row.update(response_bytes=len(raw), response_sha256=sha(raw))
            need(response.status_code == status, 'HTTP status differs')
            need('json' in response.headers.get('Content-Type', '').lower(), 'JSON content type required')
            body = strict_json(raw.decode('utf-8'))
            need(isinstance(body, dict), 'JSON object required')
            return row, body

    def values(self, response, stats):
        need(response.status_code == 200, 'SSE HTTP status differs')
        need('text/event-stream' in response.headers.get('Content-Type', '').lower(),
             'SSE content type required')
        return sse_values(bounded_lines(response.iter_content(chunk_size=1), self.remaining),
                          stats, self.remaining)

    def complete_stream(self, name, payload, expected, *, require_long=False, expected_prompt_tokens=None):
        row = self.row(name, '/v1/chat/completions')
        self.report['inference_order'].append(name)
        stats = {'heartbeat_comments': 0, 'data_frames': 0}
        row['stream_counters'] = stats  # Keep bounded parser progress even if validation fails.
        row['stream_summary_before_assertions'] = {}
        began, first = time.monotonic(), None
        with self.request('/v1/chat/completions', payload=payload, stream=True) as response:
            row['status'] = response.status_code
            def observed():
                nonlocal first
                for event in self.values(response, stats):
                    if isinstance(event, dict) and any(c.get('delta', {}).get('content')
                                                      for c in event.get('choices', [])):
                        if first is None:
                            first = time.monotonic() - began
                    yield event
            row['validation'] = stream_result(observed(), expected, require_long=require_long,
                                             expected_prompt_tokens=expected_prompt_tokens,
                                             diagnostic=row['stream_summary_before_assertions'])
        row.update(passed=True, stream_counters=stats, first_content_seconds=first,
                   total_seconds=time.monotonic() - began)
        return row


def run(a):
    config, output = a.config.resolve(), a.output.resolve()
    need(output.suffix.lower() == '.json' and output.parent.is_dir() and not output.exists()
         and output != config, 'Use one new JSON output in an existing directory')
    # Reserve exclusively before sending anything; a pre-existing receipt is never overwritten.
    receipt = output.open('x', encoding='utf-8')
    key, api, config_hash, rc = '', None, None, 1
    began = time.monotonic()
    report = {'schema': 'portable-api-check-v1', 'passed': False, 'finished': False,
              'checks': [], 'inference_order': [], 'single_attempt': True,
              'credentials_printed': False, 'raw_prompts_responses_or_logs_saved': False,
              'actual_tool_execution': False, 'model_process_service_config_actions': False,
              'identity_scope': 'API-reported model/context; not executable identity attestation',
              'origin_sha256': sha(a.base.encode()), 'HTTPS_remote': a.base.startswith('https://'),
              'deadline_seconds': a.deadline}
    try:
        raw = config.read_bytes()
        config_hash = sha(raw)
        if a.expected_config_sha256:
            need(config_hash == a.expected_config_sha256, 'Configuration pin differs')
        cfg = strict_json(raw.decode('utf-8-sig'))
        key, model = cfg.get('api_key'), cfg.get('model_name')
        need(isinstance(key, str) and key and not any(c in key for c in '\r\n'),
             'Private configuration must have an API key')
        need(isinstance(model, str) and model, 'Private configuration must have a model ID')
        args = cfg.get('args', [])
        need(args.count('--max-context') == 1 and args[args.index('--max-context') + 1] == str(MAX_CONTEXT),
             'Private configuration must request the original 1M context')
        report.update(config_sha256=config_hash, model=model)
        long_text = None
        if a.long_prompt:
            need(a.long_prompt.is_file() and a.long_prompt.stat().st_size <= MAX_PROMPT,
                 'Long prompt file exceeds bounded input size')
            with a.long_prompt.open('rb') as source:
                raw_prompt = source.read(MAX_PROMPT + 1)  # Also bound a file that grows after stat.
            need(0 < len(raw_prompt) <= MAX_PROMPT, 'Long prompt file exceeds bounded input size')
            long_text = raw_prompt.decode('utf-8')
            report['long_prompt'] = {'bytes': len(raw_prompt), 'sha256': sha(raw_prompt),
                                     'expected_answer_sha256': sha(a.expected_long_answer.encode()),
                                     'expected_prompt_tokens': a.expected_long_prompt_tokens,
                                     'plaintext_saved': False}
        api = API(key, a.base, report, a.deadline)
        row, _ = api.json('models-missing-auth', '/v1/models', auth=False, status=401)
        row['passed'] = True
        row, body = api.json('models-auth', '/v1/models')
        need(any(x.get('id') == model for x in body.get('data', [])), 'Expected model ID missing')
        row['passed'] = True
        row, _ = api.json('metrics-missing-auth', '/metrics', auth=False, status=401)
        row['passed'] = True
        row, body = api.json('metrics-auth-ready-context', '/metrics')
        need(body.get('engine', {}).get('model') == model
             and type(body.get('engine', {}).get('max_context')) is int
             and body['engine']['max_context'] == MAX_CONTEXT and body.get('live', {}).get('state') == 'idle',
             'API metrics must report the idle native 1M model')
        row.update(passed=True, model_matched=True, native_context=MAX_CONTEXT, state='idle')
        row, body = api.json('health-loaded-context', '/health')
        need(body.get('status') == 'ok' and body.get('loaded') is True
             and body.get('model') == model and type(body.get('max_context')) is int
             and body['max_context'] == MAX_CONTEXT, 'Loaded 1M health required')
        row.update(passed=True, loaded=True, max_context=MAX_CONTEXT)
        row, body = api.json('status-idle-before-FIRST', '/status')
        need(body.get('busy') is False, 'Service must be idle before FIRST inference')
        row.update(passed=True, busy=False)
        if long_text is not None:
            preset = getattr(a, 'long_thinking_preset', 'none')
            payload = long_chat(model, long_text, preset)
            report['long_client_settings'] = {'preset': preset,
                'reasoning_effort': payload['reasoning_effort'],
                'reasoning_budget_tokens': payload.get('reasoning_budget_tokens'),
                'max_tokens': payload['max_tokens'], 'scope': 'FIRST long request only; configuration unchanged'}
            api.complete_stream('FIRST-long-uncached-near-1M', payload,
                                a.expected_long_answer, require_long=True,
                                expected_prompt_tokens=a.expected_long_prompt_tokens)
            report['long_is_FIRST_inference'] = report['inference_order'][0] == 'FIRST-long-uncached-near-1M'
            need(report['long_is_FIRST_inference'], 'Long request must be FIRST inference')
            del long_text
        row, body = api.json('natural-arithmetic', '/v1/chat/completions',
                             payload=chat(model, 'Reply with only the integer result of 3 + 4.'))
        row.update(passed=True, validation=natural(body, '7'))
        api.complete_stream('complete-SSE', chat(model, 'Reply with only the integer result of 11 + 6.', True), '17')
        payload = chat(model, 'Call report_result exactly once with value 7. Do not add text.')
        payload.update(tools=[{'type': 'function', 'function': {'name': 'report_result',
                       'description': 'Synthetic validation only; no side effects.',
                       'parameters': TOOL_SCHEMA, 'strict': True}}],
                       tool_choice={'type': 'function', 'function': {'name': 'report_result'}})
        row, body = api.json('OpenAI-forced-synthetic-tool', '/v1/chat/completions', payload=payload)
        openai_tool(body)
        row.update(passed=True, tool_name='report_result', arguments_validated=True, executed=False)
        payload = {'model': model, 'max_tokens': 128, 'temperature': 0,
                   'thinking': {'type': 'disabled'},
                   'messages': [{'role': 'user', 'content': 'Call report_result exactly once with value 7.'}],
                   'tools': [{'name': 'report_result', 'description': 'Synthetic validation only; no side effects.',
                              'input_schema': TOOL_SCHEMA}], 'tool_choice': {'type': 'tool', 'name': 'report_result'}}
        row, body = api.json('Anthropic-forced-synthetic-tool', '/v1/messages', payload=payload, anthropic=True)
        anthropic_tool(body)
        row.update(passed=True, tool_name='report_result', arguments_validated=True, executed=False)
        row = api.row('active-content-disconnect-to-idle', '/v1/chat/completions')
        report['inference_order'].append(row['name'])
        stats, observed = {'heartbeat_comments': 0, 'data_frames': 0}, False
        row['stream_counters'] = stats
        with api.request('/v1/chat/completions', payload=chat(model,
                'Count upward from 1 to 2000 in order, one number per line. Continue without explanation.', True, 512), stream=True) as response:
            row['status'] = response.status_code
            for event in api.values(response, stats):
                need(event != '[DONE]' and isinstance(event, dict) and 'error' not in event,
                     'Disconnect request finished or failed before coverage')
                if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                    statusrow, body = api.json('status-busy-before-client-close', '/status')
                    need(body.get('busy') is True, 'Actual content and busy state required before disconnect')
                    statusrow.update(passed=True, busy=True)
                    observed = True
                    break
            need(observed, 'No actual content observed before disconnect')
        row.update(client_response_closed=True, actual_content_seen=True, busy_before_close=True)
        until, polls = min(api.deadline, time.monotonic() + 120), 0
        while True:
            statusrow, body = api.json('status-after-disconnect', '/status')
            need(type(body.get('busy')) is bool, 'Typed busy status required')
            statusrow.update(passed=True, busy=body['busy'])
            polls += 1
            if body['busy'] is False:
                break
            need(time.monotonic() < until, 'Disconnect did not return service to idle')
            time.sleep(min(.25, api.remaining()))
        row.update(passed=True, idle_after_disconnect=True, status_poll_count=polls, stream_counters=stats)
        row, body = api.json('natural-after-disconnect', '/v1/chat/completions',
                             payload=chat(model, 'Reply with only the integer result of 2 + 3.'))
        row.update(passed=True, validation=natural(body, '5'))
        need(all(x['passed'] for x in report['checks']), 'A required check failed')
        report['passed'], rc = True, 0
    except BaseException as error:
        report.update(passed=False, error_type=type(error).__name__,
                      failure='Single-attempt validation failed; last incomplete check retained, no retry',
                      error_reason=redacted(str(error), key if isinstance(key, str) else '').replace('\r', ' ').replace('\n', ' ')[:2048])
        rc = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if api is not None:
            api.session.close()
        try:
            report['configuration_unchanged'] = config_hash is not None and sha(config.read_bytes()) == config_hash
        except OSError:
            report['configuration_unchanged'] = False
        if not report['configuration_unchanged']:
            report['passed'], rc = False, 1
        report.update(finished=True, elapsed_seconds=time.monotonic() - began)
        try:
            json.dump(redacted(report, key), receipt, indent=2, allow_nan=False)
            receipt.write('\n')
        finally:
            receipt.close()
        print(json.dumps(redacted({'receipt': str(output), 'passed': report['passed'], 'keys_printed': False}, key)))
    return rc


def self_test():
    checks = 0
    def yes(value):
        nonlocal checks
        need(value, 'CPU regression failed')
        checks += 1
    def refuses(fn):
        nonlocal checks
        try:
            fn()
        except (RuntimeError, ValueError, TypeError, KeyError):
            checks += 1
            return
        raise RuntimeError('Bad parser input accepted')
    good = [{'choices': [{'index': 0, 'delta': {'reasoning_content': 'ignored'}}]},
            {'choices': [{'delta': {'content': '17'}}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}, '[DONE]']
    yes(stream_result(iter(good), '17')['DONE_count'] == 1)
    refuses(lambda: stream_result(good[:-1], '17'))
    refuses(lambda: stream_result(good + ['[DONE]'], '17'))
    refuses(lambda: stream_result(good + [{'choices': []}], '17'))
    refuses(lambda: stream_result([{'choices': [{'delta': {'content': '17'}, 'finish_reason': 'length'}]}, '[DONE]'], '17'))
    refuses(lambda: stream_result([{'choices': [{'delta': {'reasoning_content': '17'}, 'finish_reason': 'stop'}]}, '[DONE]'], '17'))
    refuses(lambda: stream_result([{'error': {'message': 'synthetic'}}], '17'))
    stats = {'heartbeat_comments': 0, 'data_frames': 0}
    frames = list(sse_values([b': heartbeat', b'', b'data: {"choices":[]}', b'', b'data: [DONE]', b''], stats))
    yes(stats == {'heartbeat_comments': 1, 'data_frames': 2} and frames[-1] == '[DONE]')
    yes(list(bounded_lines([b'data: x\r', b'\n\n'], limit=32)) == [b'data: x', b''])
    refuses(lambda: list(bounded_lines([b'x'] * 17, limit=16)))
    refuses(lambda: bounded_bytes([b'x'] * 17, limit=16))
    yes(strict_json(bounded_bytes([b'{', b'"x":1', b'}']).decode()) == {'x': 1})
    ticks = [0]
    def expire():
        ticks[0] += 1
        need(ticks[0] < 4, 'Synthetic deadline expired')
    refuses(lambda: list(bounded_lines([b'x'] * 20, expire, limit=32)))
    yes(ticks[0] == 4)
    refuses(lambda: list(sse_values(['data: {"choices":[]}'], {'heartbeat_comments': 0, 'data_frames': 0})))
    refuses(lambda: strict_json('{"value": NaN}'))
    natural({'choices': [{'finish_reason': 'stop', 'message': {'content': '7\n'}}]}, '7'); checks += 1
    refuses(lambda: natural({'choices': [{'finish_reason': 'length', 'message': {'content': '7'}}]}, '7'))
    o = {'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': [
        {'id': 's', 'type': 'function', 'function': {'name': 'report_result', 'arguments': '{"value":7}'}}]}}]}
    openai_tool(o); checks += 1
    bad_float = strict_json(json.dumps(o))
    bad_float['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"value":7.0}'
    refuses(lambda: openai_tool(bad_float))
    refuses(lambda: openai_tool({'choices': [{'finish_reason': 'stop', 'message': {'tool_calls': []}}]}))
    anthropic_tool({'stop_reason': 'tool_use', 'content': [{'id': 's', 'type': 'tool_use', 'name': 'report_result', 'input': {'value': 7}}]}); checks += 1
    refuses(lambda: anthropic_tool({'stop_reason': 'tool_use', 'content': [{'id': 's', 'type': 'tool_use', 'name': 'report_result', 'input': {'value': 7.0}}]}))
    refuses(lambda: synthetic_arguments({'value': 7, 'extra': 0}))
    refuses(lambda: anthropic_tool({'stop_reason': 'end_turn', 'content': []}))
    long = good[:-1] + [{'usage': {'prompt_tokens': 999900, 'completion_tokens': 2,
                                 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 999902}}] + ['[DONE]']
    yes(stream_result(long, '17', require_long=True, expected_prompt_tokens=999900)['usage']['cached_tokens'] == 0)
    for u in ({'prompt_tokens': 999900, 'completion_tokens': 2},
              {'prompt_tokens': 999900, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 1}},
              {'prompt_tokens': True, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 0}},
              {'prompt_tokens': 100, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 0}},
              {'prompt_tokens': 999900, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': False}},
              {'prompt_tokens': 999900, 'completion_tokens': 2, 'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 1}):
        refuses(lambda u=u: stream_result(good[:-1] + [{'usage': u}, '[DONE]'], '17', require_long=True))
    refuses(lambda: stream_result(long, '17', require_long=True, expected_prompt_tokens=999901))
    yes(origin(LOCAL) == LOCAL and origin('https://example.invalid/') == 'https://example.invalid')
    for bad in ('http://example.invalid', 'https://key@example.invalid', 'https://example.invalid/?key=x',
                'https://example.invalid/#x', 'https://example.invalid/path', 'https://example.invalid\\evil',
                'https://example.invalid:99999', 'https://example.invalid\n'):
        refuses(lambda bad=bad: origin(bad))
    sentinel = 'synthetic-credential'
    value = redacted({'api_key': sentinel, 'nested': [{'note': 'Bearer ' + sentinel}], 'x-api-key': sentinel}, sentinel)
    yes(sentinel not in json.dumps(value) and value['api_key'] == '[REDACTED]')
    summary = {}
    wrong = [{'choices': [{'delta': {'content': '18'}, 'finish_reason': 'stop'}]},
             {'usage': {'prompt_tokens': 999771, 'completion_tokens': 11,
                        'prompt_tokens_details': {'cached_tokens': 0}, 'total_tokens': 999782}}, '[DONE]']
    refuses(lambda: stream_result(wrong, '17', require_long=True, diagnostic=summary))
    yes(summary['exact_expected_match'] is False and summary['answer_characters'] == 2
        and summary['answer_sha256'] == sha(b'18') and summary['DONE_count'] == 1
        and summary['finish_reasons'] == ['stop'] and summary['usage_count'] == 1
        and summary['usage_numbers'][0]['cached_tokens'] == 0
        and summary['plaintext_saved'] is False and 'answer_text' not in summary)
    summary = {}
    refuses(lambda: stream_result(good[:-1], '17', diagnostic=summary))
    yes(summary['DONE_count'] == 0 and summary['exact_expected_match'] is True)
    summary = {}
    bad_cache = good[:-1] + [{'usage': {'prompt_tokens': 999900, 'completion_tokens': 2,
                 'prompt_tokens_details': {'cached_tokens': 1}, 'total_tokens': 999902}}, '[DONE]']
    refuses(lambda: stream_result(bad_cache, '17', require_long=True, diagnostic=summary))
    yes(summary['usage_numbers'][0]['cached_tokens'] == 1 and summary['exact_expected_match'] is True)
    yes(sentinel not in redacted('fixed reason ' + sentinel, sentinel))
    yes(long_chat('synthetic', 'same input') == chat('synthetic', 'same input', True, 256))
    enabled = long_chat('synthetic', 'same input', 'high-budget256')
    yes(enabled['messages'] == [{'role': 'user', 'content': 'same input'}]
        and enabled['reasoning_effort'] == 'high' and enabled['reasoning_budget_tokens'] == 256
        and enabled['max_tokens'] == 1024 and enabled['temperature'] == 0 and enabled['seed'] == 12345
        and enabled['stream'] is True and enabled['stream_options'] == {'include_usage': True})
    yes(chat('synthetic', 'short')['reasoning_effort'] == 'none'
        and chat('synthetic', 'short')['max_tokens'] == 128)
    refuses(lambda: long_chat('synthetic', 'same input', 'unrecognized'))
    return {'passed': True, 'checks': checks, 'HTTP_network_model_process_calls': 0, 'files_written': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--config', type=Path, default=ROOT / '.work/local/server.json')
    p.add_argument('--base', default=LOCAL, help='Loopback HTTP or explicitly selected HTTPS origin')
    p.add_argument('--output', type=Path, help='One exclusive new JSON receipt; required with --run')
    p.add_argument('--deadline', type=int, default=1200, help='Whole verification deadline: 600..1800 seconds')
    p.add_argument('--expected-config-sha256')
    p.add_argument('--long-prompt', type=Path, help='UTF-8 synthetic near-1M request, never copied into receipt')
    p.add_argument('--expected-long-answer')
    p.add_argument('--expected-long-prompt-tokens', type=int)
    p.add_argument('--long-thinking-preset', choices=('none', 'high-budget256'), default='none',
                   help='FIRST long only: high effort (75), separate 256-token thinking budget, 1024 total cap')
    a = p.parse_args()
    need(not (a.run and a.self_test) and 600 <= a.deadline <= 1800, 'Invalid mode or deadline')
    a.base = origin(a.base)
    need(bool(a.long_prompt) == bool(a.expected_long_answer), 'Long prompt and expected answer must be paired')
    need(a.expected_long_prompt_tokens is None or
         (a.long_prompt is not None and 900000 <= a.expected_long_prompt_tokens <= MAX_CONTEXT),
         'Expected long token count requires the paired near-1M input')
    need(not a.expected_config_sha256 or re.fullmatch('[0-9a-f]{64}', a.expected_config_sha256),
         'Expected config SHA must be lowercase hex')
    if a.self_test:
        print(json.dumps(self_test(), indent=2)); return 0
    if not a.run:
        print(json.dumps({'default_dry': True, 'HTTP_network_model_process_calls': 0, 'files_written': 0,
                          'context_required': MAX_CONTEXT, 'long_request_FIRST_if_supplied': bool(a.long_prompt),
                          'required': 'auth models/metrics; loaded native context; natural/SSE; forced synthetic tools; active disconnect/idle/recovery',
                          'redirects': False, 'inherited_proxies': False, 'actual_tool_execution': False})); return 0
    need(a.output is not None, '--run requires one exclusive --output JSON')
    return run(a)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        # Parser/refusal messages contain only fixed validation text, never config or response bodies.
        print(json.dumps({'refused': True, 'error_type': type(error).__name__, 'keys_printed': False}))
        raise SystemExit(2)
