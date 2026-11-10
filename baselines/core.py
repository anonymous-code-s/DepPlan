"""Logged model calls and question-local method state."""
import asyncio
import json
import time

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from .retrieval import PassageIndex


def parse_json(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in '{[':
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError('Model response contains no JSON object or array')


class Tokenizer:
    def __init__(self, name, revision=None):
        self.name, self.revision, self.instance = name, revision, None

    def _get(self):
        if self.instance is None:
            from transformers import AutoTokenizer
            self.instance = AutoTokenizer.from_pretrained(self.name, revision=self.revision)
        return self.instance

    def __getattr__(self, key):
        return getattr(self._get(), key)


class ModelClient:
    def __init__(self, *, model, base_url, api_key, seed, config, tokenizer,
                 log_path=None, client=None):
        self.model, self.seed = model, seed
        self.config, self.tokenizer = config, tokenizer
        self.records, self.example_id = [], None
        self.log_path = log_path
        self.client = client or AsyncOpenAI(
            base_url=base_url, api_key=api_key,
            timeout=float(config.get('request_timeout_seconds', 180)), max_retries=0)

    def _log(self, row):
        row.update(example_id=self.example_id, model=self.model, seed=self.seed)
        self.records.append(row)
        if self.log_path:
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    async def generate(self, *, role, system='', user='', prompt=None,
                       temperature=None, max_tokens=None, stop=None, include_stop=False):
        temperature = self.config.get('temperature', 0) if temperature is None else temperature
        max_tokens = int(self.config.get('max_tokens', 512) if max_tokens is None else max_tokens)
        common = dict(model=self.model, temperature=temperature, max_tokens=max_tokens,
                      top_p=self.config.get('top_p', 1.0), seed=self.seed)
        if stop:
            common['stop'] = stop
        if prompt is None:
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': user}]
            request = dict(common, messages=messages)
            generate = self.client.chat.completions.create
        else:
            request = dict(common, prompt=prompt)
            generate = self.client.completions.create
        extra = dict(self.config.get('generation_extra_body', {}))
        if include_stop:
            extra['include_stop_str_in_output'] = True
        if extra:
            request['extra_body'] = extra
        limit = self.config.get('max_context_tokens')
        if limit:
            encoded = (self.tokenizer.encode(prompt, add_special_tokens=False)
                       if prompt is not None else self.tokenizer.apply_chat_template(
                           messages, tokenize=True, add_generation_prompt=True))
            if len(encoded) + max_tokens > int(limit):
                raise ValueError(f'{role}: input plus output budget exceeds context limit {limit}')
        attempts = int(self.config.get('api_attempts', 3))
        wait_seconds = 0.0
        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            record = dict(role=role, attempt=attempt, temperature=temperature,
                          max_tokens=max_tokens, top_p=common['top_p'],
                          stop=stop, retry_wait_seconds=wait_seconds)
            record.update({'prompt': prompt} if prompt is not None
                          else {'system': system, 'user': user})
            try:
                response = await generate(**request)
                choice = response.choices[0]
                text = choice.text if prompt is not None else choice.message.content
                text = text or ''
                matched = getattr(choice, 'stop_reason', None) or getattr(choice, 'matched_stop', None)
                if include_stop and isinstance(matched, str) and not text.endswith(matched):
                    text += matched
                usage = response.usage
                token_usage = {key: getattr(usage, key, None) for key in
                               ['prompt_tokens', 'completion_tokens', 'total_tokens']}
                if token_usage['total_tokens'] is None and all(
                        token_usage[k] is not None for k in ['prompt_tokens', 'completion_tokens']):
                    token_usage['total_tokens'] = token_usage['prompt_tokens'] + token_usage['completion_tokens']
                self._log(dict(record, status='success', output=text,
                               response_model=getattr(response, 'model', None),
                               response_id=getattr(response, 'id', None),
                               response_created=getattr(response, 'created', None),
                               system_fingerprint=getattr(response, 'system_fingerprint', None),
                               request_id=getattr(response, '_request_id', None),
                               finish_reason=getattr(choice, 'finish_reason', None),
                               latency_seconds=time.perf_counter() - start, **token_usage))
                return text
            except asyncio.CancelledError:
                self._log(dict(record, status='cancelled', prompt_tokens=None,
                               completion_tokens=None, total_tokens=None,
                               latency_seconds=time.perf_counter() - start))
                raise
            except Exception as exc:
                retryable = (isinstance(exc, APIConnectionError) or
                             isinstance(exc, APIStatusError) and
                             (exc.status_code in {408, 429} or exc.status_code >= 500))
                body = getattr(exc, 'body', None)
                usage = body.get('usage', {}) if isinstance(body, dict) else {}
                self._log(dict(record, status='error', error_type=type(exc).__name__,
                               http_status=getattr(exc, 'status_code', None),
                               retryable=retryable, latency_seconds=time.perf_counter() - start,
                               **{k: usage.get(k) for k in
                                  ['prompt_tokens', 'completion_tokens', 'total_tokens']}))
                if not retryable or attempt == attempts:
                    raise
                before = time.perf_counter()
                await asyncio.sleep(2 ** (attempt - 1))
                wait_seconds = time.perf_counter() - before

    async def close(self):
        await self.client.close()


class MethodContext:
    def __init__(self, question, passages, dataset, config, seed, client, encoder, tokenizer):
        self.question, self.passages, self.dataset = question, passages, dataset
        self.config, self.seed, self.client = config, seed, client
        self.tokenizer = tokenizer
        self.retriever = PassageIndex(passages, encoder, mode=config.get('retriever', 'dense'),
                                      k1=config.get('bm25_k1', 1.2), b=config.get('bm25_b', 0.75),
                                      rrf_k=config.get('rrf_k', 60))
        self.trace, self.steps, self.stop = [], 0, ''

    async def ask(self, role, system, user, temperature=None, max_tokens=None, **kwargs):
        if not isinstance(user, str):
            user = json.dumps(user, ensure_ascii=False)
        return await self.client.generate(role=role, system=system, user=user,
                                          temperature=temperature, max_tokens=max_tokens, **kwargs)

    async def complete(self, role, prompt, temperature=None, max_tokens=None,
                       stop=None, include_stop=False):
        return await self.client.generate(role=role, prompt=prompt, temperature=temperature,
                                          max_tokens=max_tokens, stop=stop, include_stop=include_stop)

    def retrieve(self, query, k=None, mode=None):
        results = self.retriever.rank(query, self.config.get('top_k', 5) if k is None else k, mode)
        self.event('retrieval', query=query, mode=mode or self.retriever.mode,
                   passage_ids=[p['id'] for p in results])
        return results

    def rank_texts(self, query, texts, k=None, mode='dense'):
        passages = [dict(id=str(i), title='', text=text) for i, text in enumerate(texts)]
        return PassageIndex(passages, self.retriever.encoder, mode=mode,
                            rrf_k=self.config.get('rrf_k', 60)).rank(query, k, mode)

    def embed(self, texts):
        return self.retriever.encode(texts)

    @staticmethod
    def context(passages):
        return '\n\n'.join(f"[{p.get('id', i)}] {p.get('title', '')}\n{p['text']}".strip()
                           for i, p in enumerate(passages))

    @staticmethod
    def json(text):
        return parse_json(text)

    def event(self, kind, **fields):
        self.trace.append(dict(kind=kind, **fields))
