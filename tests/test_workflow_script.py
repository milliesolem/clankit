"""Workflow scripts are checked before anything runs, with line-numbered problems."""

from __future__ import annotations

import pytest

from clankit.orchestration.workflow import ScriptError, compile_script

META = 'meta = {"name": "t", "description": "A test workflow."}\n'


def problems(source: str) -> list[str]:
    with pytest.raises(ScriptError) as caught:
        compile_script(source)
    return [p.render() for p in caught.value.problems]


def only_problem(source: str) -> str:
    found = problems(source)
    assert len(found) == 1, found
    return found[0]


def test_a_valid_script_compiles_with_its_meta() -> None:
    script = compile_script(
        'meta = {"name": "scan", "description": "Scan things.", "when_to_use": "Often.",\n'
        '        "phases": ["Find", {"title": "Check", "detail": "verify"}],\n'
        '        "args": {"files": "paths to scan"}}\n'
        'phase("Find")\n'
        "found = await agent('look')\n"
        "result = found\n"
    )
    meta = script.meta
    assert (meta.name, meta.description, meta.when_to_use) == ("scan", "Scan things.", "Often.")
    assert [(p.title, p.detail) for p in meta.phases] == [("Find", ""), ("Check", "verify")]
    assert meta.args == (("files", "paths to scan"),)
    assert script.line(5) == "found = await agent('look')"


def test_static_estimate_multiplies_literal_loops() -> None:
    script = compile_script(
        META
        + "for x in [1, 2, 3]:\n    await agent(f'a {x}')\n"
        + "ys = [agent(f'b {k}') for k in range(4)]\n"
        + "await agent('c')\n"
        + "result = await parallel(ys)\n"
    )
    assert script.estimated_agents == 3 + 4 + 1


def test_unreadable_source_is_a_script_error() -> None:
    assert "cannot contain null bytes" in only_problem(META + "result = 1\x00\n")


def test_syntax_errors_name_the_line() -> None:
    message = only_problem(META + "x = (\nresult = 1\n")
    assert message.startswith("line 2: syntax error") or message.startswith("line 3: syntax error")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("import os", "imports are not available"),
        ("from json import dumps", "imports are not available"),
        ("x = __import__('os')", "two underscores"),
        ("x = args.__class__", "underscore are not available"),
        ("x = budget._meter", "underscore are not available"),
        ("x = time.time()", "cannot read the clock"),
        ("x = random.random()", "draw random numbers"),
        ("x = datetime", "cannot read the clock"),
        ("x = open('f')", "cannot touch files"),
        ("x = getattr(args, 'a')", "outside the workflow API"),
        ("x = asyncio.gather()", "parallel"),
        ("class Thing:\n    pass", "classes are not available"),
        ("agent('fire and forget')", "is not awaited"),
        ("try:\n    x = 1\nexcept:\n    x = 2", "bare `except:`"),
        ("try:\n    x = 1\nexcept BaseException:\n    x = 2", "must not be caught"),
        ("try:\n    x = 1\nexcept (ValueError, CancelledError):\n    x = 2", "must not be caught"),
    ],
)
def test_forbidden_constructs_are_reported_on_their_line(line: str, expected: str) -> None:
    message = only_problem(META + line + "\nresult = 1\n")
    assert message.startswith("line 2:") or message.startswith("line 4:")
    assert expected in message


def test_every_problem_is_reported_at_once_in_line_order_with_its_source() -> None:
    found = problems(META + "import os\nx = time.time()\nresult = open('f')\n")
    assert [f.split(":")[0] for f in found] == ["line 2", "line 3", "line 4"]
    assert found[0].endswith("\n    import os")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x = 1\nresult = x\n", "the first statement must be `meta"),
        ('name = "n"\nmeta = {"name": name, "description": "d"}\nresult = 1\n', "first statement"),
        ('meta = {"name": n, "description": "d"}\nresult = 1\n', "plain literal"),
        ('meta = ["name"]\nresult = 1\n', "must be a dict"),
        ('meta = {"name": "n"}\nresult = 1\n', 'meta["description"] must be a non-empty string'),
        ('meta = {"name": "", "description": "d"}\nresult = 1\n', 'meta["name"]'),
        (
            'meta = {"name": "n", "description": "d", "owner": "me"}\nresult = 1\n',
            "unknown meta key",
        ),
        ('meta = {"name": "n", "description": "d", "phases": "A"}\nresult = 1\n', "list of titles"),
        ('meta = {"name": "n", "description": "d", "phases": [3]}\nresult = 1\n', "each phase"),
        ('meta = {"name": "n", "description": "d", "phases": ["A", "A"]}\nresult = 1\n', "unique"),
        ('meta = {"name": "n", "description": "d", "args": ["x"]}\nresult = 1\n', 'meta["args"]'),
        ('meta = {"name": "' + "n" * 70 + '", "description": "d"}\nresult = 1\n', "too long"),
    ],
)
def test_bad_meta_blocks_are_rejected(source: str, expected: str) -> None:
    assert any(expected in p for p in problems(source))


def test_a_script_must_assign_result() -> None:
    assert "never assigns `result`" in only_problem(META + "x = await agent('a')\n")


def test_result_may_be_assigned_in_a_branch_or_through_global() -> None:
    compile_script(META + "if args:\n    result = 1\nelse:\n    result = 2\n")
    compile_script(META + "def finish():\n    global result\n    result = 3\nfinish()\n")


def test_top_level_return_gets_a_pointer_to_result() -> None:
    message = only_problem(META + "result = 1\nreturn result\n")
    assert message.startswith("line 3:") and "assign the value to `result`" in message


@pytest.mark.parametrize(
    "source",
    [
        "result = await parallel([lambda: agent(f'vote {k}') for k in range(3)])\n",
        "thunks = []\nfor k in range(3):\n    thunks.append(lambda: agent(f'vote {k}'))\n"
        "result = await parallel(thunks)\n",
        "calls = []\nfor path in args['files']:\n    async def check():\n"
        "        return await agent(path)\n    calls.append(check)\nresult = await parallel(calls)\n",
        "result = await parallel([(lambda: agent(f'{a} {b}')) for a in 'xy' for b in 'pq'])\n",
        "result = []\nfor k in range(3):\n    result.append(filter(lambda x: x == k, args['xs']))\n",
    ],
)
def test_closures_over_loop_variables_are_rejected(source: str) -> None:
    message = only_problem(META + source)
    assert "Python looks the variable up only when" in message
    assert "parallel([agent(...)" in message and "=" in message


@pytest.mark.parametrize(
    "source",
    [
        "result = await parallel([agent(f'vote {k}') for k in range(3)])\n",
        "result = await parallel([lambda k=k: agent(f'vote {k}') for k in range(3)])\n",
        "async def stage(previous, item, index):\n    return await agent(item)\n"
        "result = await pipeline(args['items'], stage)\n",
        "total = 0\nfor k in range(3):\n    total += k\nresult = sorted(args, key=lambda x: x)\n",
        "result = []\nfor item in args['items']:\n"
        "    result.append(max(item['scores'], key=lambda s: s[item['metric']]))\n"
        "    item['scores'].sort(key=lambda s: s[item['metric']])\n",
    ],
)
def test_safe_forms_are_accepted(source: str) -> None:
    compile_script(META + source)


def test_phase_titles_must_match_the_declared_phases() -> None:
    found = problems(
        'meta = {"name": "n", "description": "d", "phases": ["Find"]}\n'
        'phase("Find")\nphase("Verify")\n'
        'x = await agent("a", phase="Report")\nresult = x\n'
    )
    assert [f.split(":")[0] for f in found] == ["line 3", "line 4"]
    assert 'phase "Verify" is not in meta["phases"] ("Find")' in found[0]


def test_any_phase_is_accepted_when_none_are_declared() -> None:
    compile_script(META + 'phase("Anything")\nresult = 1\n')
