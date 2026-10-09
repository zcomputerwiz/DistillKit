# Assisted-by: Codex
import importlib.util
from pathlib import Path


def load_audit():
    path = Path(__file__).parents[1]/'scratch/dense_gr/code_failure_audit.py'
    spec = importlib.util.spec_from_file_location('code_failure_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_channel_excludes_draft_and_keeps_unfenced_source():
    audit = load_audit()
    raw = 'Draft:\n```python\nreturn 1\n```\n</think>\n```python\ndef f():\n    return 2\n```'
    assert audit.final_answer_code(raw) == 'def f():\n    return 2'
    assert audit.final_answer_code('draft prose\n</think>\ndef f():\n    return 2') == 'def f():\n    return 2'


def test_unclosed_thinking_is_not_repaired_or_fabricated():
    audit = load_audit()
    assert audit.final_answer_code('draft prose only') == 'draft prose only'
    assert audit.final_answer_code('draft\n```python\ndef f():\n    return') == 'def f():\n    return'
    assert audit.final_answer_code('draft\n</think>\n') == ''
