from __future__ import annotations

import re


def normalize(code: str) -> str:
    if not code:
        return code
    code = _clean_negative_literals(code)
    code = _recover_modulo(code)
    code = _infer_undefined_types(code)
    code = _remove_dead_null_assignments(code)
    return code


def _clean_negative_literals(code: str) -> str:
    return re.sub(r'\+ -(0x[0-9a-fA-F]+|\d+)', r'- \1', code)


def _recover_modulo(code: str) -> str:
    _NUM = r'(?:0x[0-9a-fA-F]+|\d+)'

    # Find (EXPR / N) * -N or (EXPR / N) * N patterns, then verify
    # the preceding text contains EXPR with + or - operator.
    # Strategy: locate the division-multiply tail, extract the inner
    # expression, then search backwards for a matching leading expression.

    tail_pattern = re.compile(
        r'\(('           # open paren + start capture of inner EXPR
        r'[^()]*'        # simple content
        r'(?:\([^()]*\)[^()]*)*'  # allow one level of nested parens
        r')'             # end capture of inner EXPR
        r' / (' + _NUM + r')'  # / N
        r'\)'            # close paren
        r' \* -?(' + _NUM + r')'  # * N or * -N
    )

    code = _recover_modulo_cast_wrapped(code, _NUM)

    changed = True
    while changed:
        changed = False
        for match in tail_pattern.finditer(code):
            inner_expr = match.group(1)
            divisor = match.group(2)
            multiplier = match.group(3)

            if divisor != multiplier:
                continue

            tail_start = match.start()
            prefix = code[:tail_start]

            # Check for EXPR - (EXPR / N) * N form
            suffix_sub = ' - '
            if prefix.endswith(suffix_sub) and prefix[:-len(suffix_sub)].endswith(inner_expr):
                expr_start = tail_start - len(suffix_sub) - len(inner_expr)
                code = code[:expr_start] + inner_expr + ' % ' + divisor + code[match.end():]
                changed = True
                break

            # Check for EXPR + (EXPR / N) * -N form
            suffix_add = ' + '
            has_neg_mult = match.group(0).find('* -') != -1
            if has_neg_mult and prefix.endswith(suffix_add) and prefix[:-len(suffix_add)].endswith(inner_expr):
                expr_start = tail_start - len(suffix_add) - len(inner_expr)
                code = code[:expr_start] + inner_expr + ' % ' + divisor + code[match.end():]
                changed = True
                break

    return code


def _recover_modulo_cast_wrapped(code: str, _NUM: str) -> str:
    # Handles: (CAST)(EXPR) + (CAST)((EXPR) / N) * -N -> (CAST)((EXPR) % N)
    # Also:    (CAST)(EXPR) - (CAST)((EXPR) / N) * N  -> (CAST)((EXPR) % N)
    cast_tail = re.compile(
        r'\((\w+)\)'                           # (CAST) before division group
        r'\(\('                                # ((
        r'([^()]*(?:\([^()]*\)[^()]*)*)'       # EXPR inside
        r'\) / (' + _NUM + r')\)'              # ) / N)
        r' \* (-?' + _NUM + r')'               # * N or * -N
    )

    changed = True
    while changed:
        changed = False
        for match in cast_tail.finditer(code):
            cast = match.group(1)
            inner_expr = match.group(2)
            divisor = match.group(3)
            raw_multiplier = match.group(4)

            multiplier = raw_multiplier.lstrip('-')
            if divisor != multiplier:
                continue

            is_neg_mult = raw_multiplier.startswith('-')

            tail_start = match.start()
            prefix = code[:tail_start]

            # Look for (CAST)(EXPR) + or (CAST)(EXPR) - before
            leading_with_add = f'({cast})({inner_expr}) + '
            leading_with_sub = f'({cast})({inner_expr}) - '

            if is_neg_mult and prefix.endswith(leading_with_add):
                expr_start = tail_start - len(leading_with_add)
                replacement = f'({cast})(({inner_expr}) % {divisor})'
                code = code[:expr_start] + replacement + code[match.end():]
                changed = True
                break
            elif not is_neg_mult and prefix.endswith(leading_with_sub):
                expr_start = tail_start - len(leading_with_sub)
                replacement = f'({cast})(({inner_expr}) % {divisor})'
                code = code[:expr_start] + replacement + code[match.end():]
                changed = True
                break

    return code


def _infer_undefined_types(code: str) -> str:
    undef4_vars = re.findall(r'undefined4\s+(\w+)\s*[;,]', code)
    undef8_vars = re.findall(r'undefined8\s+(\w+)\s*[;,]', code)

    for var in undef4_vars:
        if _is_loop_counter(code, var) or _is_accumulator(code, var):
            code = re.sub(
                r'undefined4(\s+' + re.escape(var) + r')',
                r'int\1',
                code,
            )

    for var in undef8_vars:
        if _is_pointer_like(code, var):
            code = re.sub(
                r'undefined8(\s+' + re.escape(var) + r')',
                r'long\1',
                code,
            )

    return code


def _is_loop_counter(code: str, var: str) -> bool:
    v = re.escape(var)
    loop_pattern = re.compile(
        r'for\s*\(\s*' + v + r'\s*=\s*0\s*;\s*' + v + r'\s*<\s*\w+\s*;\s*'
        + v + r'\s*(?:=\s*' + v + r'\s*\+\s*1|\+\+)\s*\)'
    )
    return bool(loop_pattern.search(code))


def _is_accumulator(code: str, var: str) -> bool:
    v = re.escape(var)
    init_zero = re.compile(v + r'\s*=\s*0\s*;')
    add_assign = re.compile(v + r'\s*\+=\s*\w+\s*;')
    add_self = re.compile(v + r'\s*=\s*' + v + r'\s*\+\s*\w+\s*;')
    return bool(init_zero.search(code) and (add_assign.search(code) or add_self.search(code)))


def _is_pointer_like(code: str, var: str) -> bool:
    v = re.escape(var)
    null_cmp = re.compile(v + r'\s*==\s*(?:0\b|NULL\b)')
    dat_assign = re.compile(v + r'\s*=\s*DAT_\w+')
    ptr_cast_deref = re.compile(r'\*\s*\([^)]*\*\s*\)\s*' + v)
    return bool(null_cmp.search(code) or dat_assign.search(code) or ptr_cast_deref.search(code))


def _remove_dead_null_assignments(code: str) -> str:
    def _remove_dead_in_block(m: re.Match[str]) -> str:
        var = m.group(1)
        type_cast = m.group(2)
        body = m.group(3)
        assignment_pattern = re.compile(
            r'\s*' + re.escape(var) + r'\s*=\s*\(' + re.escape(type_cast) + r'\)0x0\s*;\n?'
        )
        cleaned_body = assignment_pattern.sub('', body)
        return f'if ({var} == ({type_cast})0x0) {{{cleaned_body}}}'

    result = re.sub(
        r'if\s*\((\w+)\s*==\s*\(([^)]+\*)\)0x0\)\s*\{([^}]*)\}',
        _remove_dead_in_block,
        code,
    )
    return result
