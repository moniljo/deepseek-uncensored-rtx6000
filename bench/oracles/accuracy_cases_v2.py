"""Versioned dictionary-input clarification; original questions/golds stay frozen.

Only the account-balance prompt changes. Its oracle, input vectors, bounded AST
policy and grading rules are the original functions, with no relaxed checks.
Results using this module belong to v2 and must not replace the original11/12.
"""
from copy import deepcopy
import accuracy_cases as original

BENCHMARK_VERSION = 'dictionary-contract-v2'
CORRECTED_CASE_ID = 'code-account-balance'
_OLD_SENTENCE = 'Each event has account,posted,type,amount_cents.'
_DICT_SENTENCE = (
    'events is a finite list of Python dictionaries. Each event dictionary has '
    'the string keys "account", "posted", "type", and "amount_cents". '
    'Read these fields with dictionary indexing, such as e["account"], '
    'rather than object attribute access.'
)


def _clarified(case):
    result = deepcopy(case)
    if result['id'] == CORRECTED_CASE_ID:
        if result['prompt'].count(_OLD_SENTENCE) != 1:
            raise RuntimeError('Original account-balance prompt changed')
        result['prompt'] = result['prompt'].replace(_OLD_SENTENCE, _DICT_SENTENCE)
    return result


CASES = tuple(_clarified(case) for case in original.CASES)
code_tests = original.code_tests
extract_code = original.extract_code
grade = original.grade
REFERENCE_CODE = original.REFERENCE_CODE
