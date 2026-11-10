"""OpenAI-compatible client with per-attempt usage and timing records."""
import asyncio
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
from openai import APIConnectionError, APIStatusError, AsyncOpenAI


@dataclass
class CallUsage:
    role: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    retry_wait_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Completion:
    text: str
    usage: CallUsage
    attempts: tuple[CallUsage, ...] = ()

    @property
    def attempt_usage(self):
        return self.attempts or (self.usage,)


def _retryable(exc):
    return (isinstance(exc, APIConnectionError)
            or isinstance(exc, APIStatusError)
            and (exc.status_code in {408, 429} or 500 <= exc.status_code < 600))


def _usage(role, raw, elapsed, retry_wait):
    def field(name):
        return raw.get(name) if isinstance(raw, Mapping) else getattr(raw, name, None)
    prompt, completion, total = (field(k) for k in
                                 ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return CallUsage(role, prompt, completion, total, elapsed, retry_wait)


def _error_usage(exc):
    body = getattr(exc, 'body', None)
    if isinstance(body, Mapping) and isinstance(body.get('usage'), Mapping):
        return body['usage']
    response = getattr(exc, 'response', None)
    if response is not None:
        try:
            payload = response.json()
        except (ValueError, TypeError):
            return None
        if isinstance(payload, Mapping):
            return payload.get('usage')
    return None


class LLMClient:
    def __init__(self, *, model, base_url, api_key, seed=None, timeout=180.0,
                 log_path=None, max_attempts=3, vllm_parameters=True):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError('max_attempts must be an integer from 1 to 3')
        self.model, self.seed = model, seed
        self.max_attempts = max_attempts
        self.vllm_parameters = vllm_parameters
        self.request_index = 0
        self.log_path = Path(log_path) if log_path else None
        self.example_id = None
        self.records = []
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def _log(self, record):
        record.update(example_id=self.example_id, model=self.model, seed=self.seed)
        self.records.append(record)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False)+'\n')

    async def complete(self, *, role, system, user, temperature, max_tokens):
        self.request_index += 1
        common = dict(role=role, system=system, user=user, temperature=temperature,
                      max_tokens=max_tokens, request_index=self.request_index,
                      max_attempts=self.max_attempts)
        attempts = []
        retry_wait = 0.0
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            raw_usage = None
            record = dict(common, attempt=attempt)
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role':'system','content':system},{'role':'user','content':user}],
                    temperature=temperature, max_tokens=max_tokens, top_p=1.0, seed=self.seed,
                    **({'extra_body': {'top_k': -1}} if self.vllm_parameters else {}))
                raw_usage = response.usage
                record.update(reported_model=getattr(response, 'model', None),
                              system_fingerprint=getattr(response, 'system_fingerprint', None),
                              response_id=getattr(response, 'id', None),
                              request_id=getattr(response, '_request_id', None))
                text = response.choices[0].message.content or ''
            except asyncio.CancelledError:
                usage = _usage(role, raw_usage, time.perf_counter()-started, retry_wait)
                self._log(dict(record, status='cancelled', usage=usage.to_dict()))
                raise
            except Exception as exc:
                usage = _usage(role, raw_usage if raw_usage is not None else _error_usage(exc),
                               time.perf_counter()-started, retry_wait)
                attempts.append(usage)
                retryable = _retryable(exc)
                retry = retryable and attempt < self.max_attempts
                delay = float(2 ** (attempt - 1)) if retry else 0.0
                self._log(dict(record, status='error', error_type=type(exc).__name__,
                               http_status=getattr(exc, 'status_code', None),
                               retryable=retryable, will_retry=retry,
                               next_retry_delay_seconds=delay, usage=usage.to_dict()))
                if not retry:
                    raise
                wait_started = time.perf_counter()
                await asyncio.sleep(delay)
                retry_wait = time.perf_counter() - wait_started
                continue
            usage = _usage(role, raw_usage, time.perf_counter()-started, retry_wait)
            attempts.append(usage)
            self._log(dict(record, status='success', output=text, usage=usage.to_dict()))
            return Completion(text, usage, tuple(attempts))

    async def close(self):
        await self.client.close()


def sum_usage(calls):
    totals = {}
    for name in ['prompt_tokens','completion_tokens','total_tokens']:
        values = [getattr(c,name) for c in calls]
        totals[name] = sum(values) if all(v is not None for v in values) else None
        totals['known_'+name] = sum(v for v in values if v is not None)
    latency = sum(c.latency_seconds for c in calls)
    retry_wait = sum(c.retry_wait_seconds for c in calls)
    return dict(totals, call_count=len(calls), latency_seconds=latency,
                retry_wait_seconds=retry_wait, total_request_seconds=latency+retry_wait,
                calls=[c.to_dict() for c in calls])
