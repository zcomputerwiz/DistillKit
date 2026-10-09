# Assisted-by: Codex
"""Conservative task confirmation must distinguish copied and modified contracts."""
import importlib.util
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / 'scratch/dense_gr'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('code_replay_review', SCRIPTS / 'code_replay_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_import_reordering_preserves_a_complete_contract_match():
    problem = dict(prompt='import math\n\ndef drain(grid, capacity):\n    """Return the number of bucket trips needed to empty every well."""\n',
                   entry_point='drain')
    text = 'def drain(grid, capacity):\n    import math\n    """Return the number of bucket trips needed to empty every well."""\n'
    assert review.match_evidence(text, problem, 'humaneval')['confirmed']


def test_modified_contract_is_retained_despite_shared_function_and_phrases():
    problem = dict(prompt='def choose(values):\n    """Return every third entry from the list, preserving order."""\n', entry_point='choose')
    text = 'def choose(values):\n    """Return every second entry from the list, preserving order."""\n'
    assert not review.match_evidence(text, problem, 'humaneval')['confirmed']


def test_generic_stem_requires_matching_interface_and_concrete_tests():
    problem = dict(prompt='Write a function to sort a list of elements.', code='def comb_sort(values):\n    return sorted(values)',
                   test_list=['assert comb_sort([2, 1]) == [1, 2]', 'assert comb_sort([]) == []'])
    other = 'Write a function to sort a list of elements.\nassert pancake_sort([2, 1]) == [1, 2]\nassert pancake_sort([]) == []'
    assert not review.match_evidence(other, problem, 'mbpp')['confirmed']
    copied = other.replace('pancake_sort', 'comb_sort')
    assert review.match_evidence(copied, problem, 'mbpp')['confirmed']
