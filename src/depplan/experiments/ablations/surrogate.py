"""Question-text controls with the same numbered-list layout and token budget."""
import re

from ...prompts import planning_context


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _fit(tokenizer, prefix, question, target):
    text = question
    while token_count(tokenizer, prefix + text) < target + 8:
        text += ' ' + question
    ids = tokenizer.encode(prefix + text, add_special_tokens=False)
    candidates = []
    for size in range(max(1, target - 8), min(len(ids), target + 8) + 1):
        decoded = tokenizer.decode(ids[:size], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded.startswith(prefix):
            body = decoded[len(prefix):].strip()
            if body:
                candidates.append(body)
    if not candidates:
        raise ValueError('Tokenizer cannot preserve the surrogate list prefix')
    return min(candidates, key=lambda s: abs(token_count(tokenizer, prefix + s) - target))


def build_text_surrogate(question, reference_units, tokenizer, *, strategy=None,
                         indexed_units=True):
    question = ' '.join(question.split())
    if not question:
        raise ValueError('A text surrogate requires a nonempty question')
    if not reference_units:
        count = token_count(tokenizer, planning_context(
            question, [], strategy, indexed_units=indexed_units))
        return [], {'source': 'original_question_text',
                    'construction': 'repeat_and_truncate_question_text',
                    'unit_count': 0, 'reference_context_tokens': count,
                    'surrogate_context_tokens': count, 'absolute_token_difference': 0,
                    'tokenizer': getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)}
    controls = []
    prefixes = []
    for i, unit in enumerate(reference_units, 1):
        match = re.match(r'^(U\d+: )', unit)
        prefix = match.group(1) if match else f'U{i}: '
        prefixes.append(prefix)
        body = _fit(tokenizer, '- ' + prefix, question,
                    token_count(tokenizer, '- ' + unit))
        controls.append(prefix + body)

    def count(units):
        return token_count(tokenizer, planning_context(
            question, units, strategy, indexed_units=indexed_units))

    target = count(reference_units)
    best = controls
    error = abs(count(best) - target)
    last_target = token_count(tokenizer, '- ' + controls[-1]) + target - count(controls)
    for offset in range(-8, 9):
        if error == 0:
            break
        body = _fit(tokenizer, '- ' + prefixes[-1], question, max(1, last_target + offset))
        candidate = controls[:-1] + [prefixes[-1] + body]
        distance = abs(count(candidate) - target)
        if distance < error:
            best, error = candidate, distance
    if error > 1:
        raise ValueError('The text-control prompt differs by more than one tokenizer token')
    return best, {
        'source': 'original_question_text',
        'construction': 'repeat_and_truncate_question_text',
        'unit_count': len(best),
        'reference_context_tokens': target,
        'surrogate_context_tokens': count(best),
        'absolute_token_difference': error,
        'tokenizer': getattr(tokenizer, 'name_or_path', type(tokenizer).__name__),
    }
