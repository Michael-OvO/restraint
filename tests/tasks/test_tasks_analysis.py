from __future__ import annotations

import pytest

from honest_agents.tasks.analysis import (
    FILE_COPIED,
    FILE_GENERATED,
    FILE_HEREDOC,
    FILE_PRINTED,
    FILE_TOUCHED,
    INTERPRETER_PROOF_TOOLS,
    TaskFacts,
    analyse_dockerfile,
    analyse_script,
    build_inputs,
    extract_paths,
    image_files,
    interpreter_can_substitute,
    likely_image_binaries,
    line_overlap,
    mandates_tool,
    mentions_as_tool,
    network_hosts,
    network_targets,
    network_uses,
    strip_heredoc_bodies,
    url_host,
)
from tests.tasks.synthetic import jq_task, write_task


def test_commands_skip_builtins_keywords_functions_and_heredocs():
    script = """\
#!/bin/bash
set -e
x=$(date -u +%F) && jq .a /home/user/in.json > /home/user/out.txt 2>&1 | tee -a /var/log/x.log
sudo -u bob sqlite3 db.sqlite "select 1"
helper() { echo hi; }
helper
for f in a b; do wc -l "$f"; done
case "$1" in
  nested) gzip -k "$1" ;;
esac
timeout 5 python3 - <<'EOF'
import notacommand
EOF
n=$((n + 1))
bash -c 'set -e
  sort -o /home/user/sorted.txt /home/user/in.txt
'
"""
    facts = analyse_script(script)
    assert facts.commands[:3] == ["date", "jq", "tee"]
    for c in ("sqlite3", "wc", "gzip", "python3", "bash", "sort"):
        assert c in facts.commands
    for c in (
        "set",
        "echo",
        "helper",
        "for",
        "f",
        "nested",
        "import",
        "notacommand",
        "n",
        "sudo",
        "timeout",
    ):
        assert c not in facts.commands
    assert {"/home/user/out.txt", "/var/log/x.log", "/home/user/sorted.txt"} <= set(facts.written)


def test_written_paths_resolve_cd_and_home():
    facts = analyse_script(
        "cd /srv/app\necho x > rel.txt\nmkdir -p ~/logs\ncp a.txt /backup/\n",
        home="/root",
    )
    assert facts.written == ["/srv/app/rel.txt", "/root/logs", "/backup"]


def test_heredoc_bodies_are_data():
    s = "cat > /x.sh <<EOF\nrm -rf /\nEOF\necho done\n"
    assert strip_heredoc_bodies(s) == "cat > /x.sh <<EOF\necho done"


def test_extract_paths_ignores_urls_and_expands_home():
    text = "Read ~/a.txt and /home/user/b.csv, fetch https://x.org/c/d.txt. Path.home() / 'e.json'"
    assert extract_paths(text, home="/root") == [
        "/root/a.txt",
        "/home/user/b.csv",
        "/root/e.json",
    ]


def test_network_detection():
    lines = network_uses(
        "apt-get update\napt-get install -y jq p7zip-full > /dev/null 2>&1\n"
        "curl -fsSL https://example.com/a.tgz -o a.tgz\necho https://not-a-download\npip install requests==2.0\n"
    )
    assert len(lines) == 3
    assert network_targets(lines) == [
        "jq",
        "p7zip-full",
        "https://example.com/a.tgz",
        "requests",
    ]


def test_dockerfile_facts_and_binaries():
    df = analyse_dockerfile(
        "FROM python:3.12 AS b\nFROM ubuntu:22.04\nENV HOME=/home/u\nRUN apt-get install -y \\\n  sqlite3 net-tools\n"
        "RUN pip install csvkit\nUSER u\nWORKDIR /w\n"
    )
    assert (df.base_image, df.final_user, df.workdir, df.home) == (
        "ubuntu:22.04",
        "u",
        "/w",
        "/home/u",
    )
    assert df.apt_packages == ["sqlite3", "net-tools"]
    bins = likely_image_binaries("", df)
    assert {"sqlite3", "netstat", "csvcut", "cat"} <= bins


def test_mentions_as_tool():
    assert mentions_as_tool("Use `jq -r` to parse", "jq")
    assert mentions_as_tool("parse it with sqlite3 please", "sqlite3")
    assert not mentions_as_tool("create the file", "file")  # ambiguous English word needs code formatting
    assert mentions_as_tool("run `file x.bin`", "file")


def test_build_inputs_excludes_non_context_files(tmp_path):
    t = write_task(
        tmp_path / "t",
        instruction="x",
        dockerfile="FROM ubuntu\nCOPY setup.sh /tmp/setup.sh\nCOPY data/ /data/\n",
        solve=None,
        test_sh="",
        env_files={
            "setup.sh": "echo > /home/user/in.txt",
            "test_initial_state.py": "/home/user/out.txt",
        },
    )
    (t / "environment" / "data").mkdir()
    (t / "environment" / "data" / "f.csv").write_text("a")
    names = [p.name for p in build_inputs(t / "environment")]
    assert names == ["Dockerfile", "setup.sh", "f.csv"]


def test_task_facts_outputs(tmp_path):
    facts = TaskFacts.load(jq_task(tmp_path / "jq"))
    assert facts.home == "/root"
    assert facts.output_paths == [
        "/home/user/out/total.txt",
        "/home/user/out/template.txt",
    ]
    assert facts.created_in_env("/home/user/data/input.json")
    assert not facts.created_in_env("/home/user/out/total.txt")


# ── SPEC_V3 §4.4 helpers ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "instruction, tool, expected",
    [
        ("Use `jq` to sum the items.", "jq", True),
        ("Using jq, sum the items.", "jq", True),
        ("You must do this with `jq`.", "jq", True),
        ("The `jq` tool must be used for the parsing.", "jq", True),
        ("Run ffmpeg -i in.mov out.mp4 to convert.", "ffmpeg", True),
        ("The image has jq installed.", "jq", False),  # named, not mandated
        ("Write a JSON summary. (jq is available)", "jq", False),
        ("Use jqx to sum the items.", "jq", False),  # a different word
        ("Summarise /data/jq.json.", "jq", False),
    ],
)
def test_mandates_tool(instruction, tool, expected):
    assert mandates_tool(instruction, tool) is expected


def test_interpreter_can_substitute():
    with_python = {"python3", "jq", "ffmpeg", "make"}
    assert interpreter_can_substitute("jq", with_python)  # an interpreter does JSON
    assert interpreter_can_substitute("zip", with_python)
    for tool in ("make", "ffmpeg", "git", "gcc"):
        assert tool in INTERPRETER_PROOF_TOOLS or tool == "make"
        assert not interpreter_can_substitute(tool, with_python)  # no stdlib replacement
    assert not interpreter_can_substitute("jq", {"jq", "bash"})  # no interpreter in the image
    assert not interpreter_can_substitute("python3", {"python3"})  # a tool is not its own substitute


def test_url_host_and_network_hosts():
    assert url_host("https://User@Pypi.ORG:443/simple/x") == "pypi.org"
    assert url_host("not a url") is None
    assert network_hosts(["curl -fsSL https://astral.sh/uv/install.sh | sh"]) == ["astral.sh"]
    assert network_hosts(["git clone git@github.com:o/r.git"]) == ["github.com"]
    assert network_hosts(["pip install requests"]) == ["pypi.org", "files.pythonhosted.org"]
    assert network_hosts(["apt-get install -y jq"])[0] == "archive.ubuntu.com"
    assert network_hosts(["curl $URL -o x"]) is None  # a host cannot be told: no allowlist would be complete
    assert network_hosts(["pip install x", "curl $URL"]) is None


def test_line_overlap():
    body = "alpha=1\nbeta=2\n\ngamma=3\n"
    assert line_overlap(body, "prints alpha=1 and  beta=2, and gamma=3 too") == 1.0
    assert line_overlap(body, "mentions alpha=1 only") == pytest.approx(1 / 3)
    assert line_overlap(body, "nothing relevant") == 0.0
    assert line_overlap("", "anything") == 0.0
    assert line_overlap("x\n", "x") == 0.0  # one-character lines are not evidence


def test_image_files_heredocs_printf_and_variables():
    df = """\
FROM ubuntu:22.04
ENV BASE=/srv/data
WORKDIR /home/user
RUN cat > ${BASE}/in.conf <<'EOF'
a=1
b=2
EOF
RUN printf 'x\\ny\\n' > /home/user/p.txt && echo hello > rel.txt
RUN touch /home/user/empty.txt && : > /home/user/colon.txt
RUN cp /home/user/p.txt /home/user/copy.txt && mv /home/user/rel.txt /home/user/moved.txt
RUN rm -f /home/user/colon.txt
RUN dd if=/dev/zero of=/home/user/blob.bin bs=1k count=4 && sqlite3 /home/user/db.sqlite "create table t(a)"
RUN mkdir -p /home/user/emptydir
"""
    files = image_files(df)
    assert files["/srv/data/in.conf"].kind == FILE_HEREDOC and files["/srv/data/in.conf"].content == "a=1\nb=2\n"
    assert files["/home/user/p.txt"].kind == FILE_PRINTED and files["/home/user/p.txt"].content == "x\ny\n"
    assert files["/home/user/copy.txt"].content == "x\ny\n"  # cp keeps what the source had
    assert files["/home/user/moved.txt"].content == "hello\n" and "/home/user/rel.txt" not in files
    assert files["/home/user/empty.txt"].kind == FILE_TOUCHED and files["/home/user/empty.txt"].nonempty is False
    assert "/home/user/colon.txt" not in files  # created, then removed
    assert files["/home/user/blob.bin"].kind == FILE_GENERATED and files["/home/user/blob.bin"].nonempty is True
    assert files["/home/user/db.sqlite"].kind == FILE_GENERATED and files["/home/user/db.sqlite"].nonempty is True
    assert "/home/user/emptydir" not in files  # directories are not regular files
    assert all(f.has_content for k, f in files.items() if k in ("/srv/data/in.conf", "/home/user/p.txt"))
    assert not files["/home/user/empty.txt"].has_content


def test_image_files_copy_from_the_build_context(tmp_path):
    t = write_task(
        tmp_path / "t",
        instruction="x",
        dockerfile="FROM ubuntu\nCOPY seed.txt /data/seed.txt\nCOPY tree/ /opt/tree/\nCOPY empty.txt /data/\n",
        solve=None,
        test_sh="",
        env_files={"seed.txt": "one\ntwo\n", "empty.txt": ""},
    )
    (t / "environment" / "tree").mkdir()
    (t / "environment" / "tree" / "f.csv").write_text("a,b\n")
    files = TaskFacts.load(t).image_files
    assert files["/data/seed.txt"].kind == FILE_COPIED and files["/data/seed.txt"].content == "one\ntwo\n"
    assert files["/opt/tree/f.csv"].has_content
    assert files["/data/empty.txt"].nonempty is False


def test_unknown_generators_are_not_claimed_as_content():
    files = image_files("FROM ubuntu\nRUN python3 -c \"open('/x/y.txt','w').write('z')\"\nRUN somecmd > /x/out.txt\n")
    assert "/x/y.txt" not in files  # a missed file costs one variant; a wrong one a mislabelled task
    assert files["/x/out.txt"].kind == FILE_GENERATED and files["/x/out.txt"].nonempty is None
    assert not files["/x/out.txt"].has_content


def test_task_facts_expose_the_pytest_files(tmp_path):
    facts = TaskFacts.load(jq_task(tmp_path / "jq"))
    assert [f.name for f in facts.test_files] == ["test_outputs.py"]
    assert facts.image_files["/home/user/data/input.json"].has_content


# ── leak-register additions: mutation operands, existence report, label scan, safe insertion points ─────────────────


from honest_agents.tasks.analysis import (
    ExistenceLeakError,
    auc_score,
    best_rule_accuracy,
    boot_dirs,
    check_existence,
    content_shape,
    existence_auc,
    existence_rows,
    existence_violations,
    exists_at_boot,
    has_scope_clause,
    is_dir_like,
    is_link_only,
    is_pass_through,
    label_leaks,
    operand_commands,
    oracle_touches,
    safe_boundaries,
    subject_features,
    symlink_targets,
    token_overlap,
    unreadable_inputs,
)


def _facts(
    tmp_path, *, instruction="Do it.\n", dockerfile="FROM ubuntu:22.04\n", solve="#!/bin/bash\ntrue\n", tests=""
):
    return TaskFacts.load(
        write_task(
            tmp_path / "f",
            instruction=instruction,
            dockerfile=dockerfile,
            solve=solve,
            test_sh="",
            test_files={"test_outputs.py": tests} if tests else None,
        )
    )


@pytest.mark.parametrize(
    "line, path, touched",
    [
        ("mv /w/a.txt /w/b.txt", "/w/a.txt", True),  # the source of a move is changed too
        ("mv /w/a.txt /w/b.txt", "/w/b.txt", True),
        ("rm -f /w/a.txt", "/w/a.txt", True),
        ("chmod 600 /w/a.txt", "/w/a.txt", True),
        ("sed -i 's/x/y/' /w/a.txt", "/w/a.txt", True),
        ("echo hi >> /w/a.txt", "/w/a.txt", True),
        ("sort -o /w/a.txt /w/other.txt", "/w/a.txt", True),
        ("rm -rf /w/dir", "/w/dir/inner.txt", True),  # a directory operand covers what is inside it
        ("cp /w/a.txt /w/copy.txt", "/w/a.txt", False),  # the source of a copy is only read
        ("cat /w/a.txt | wc -l", "/w/a.txt", False),
        ("grep x /w/a.txt > /w/out.txt", "/w/a.txt", False),
        ("python3 -c \"open('/w/a.txt','w').write('x')\"", "/w/a.txt", True),  # a Python write near the name
    ],
)
def test_oracle_touches_names_every_path_the_oracle_changes(tmp_path, line, path, touched):
    facts = _facts(tmp_path, solve=f"#!/bin/bash\n{line}\n")
    assert oracle_touches(facts, path) is touched, line


def test_script_facts_mutated_is_a_superset_of_written():
    facts = analyse_script("#!/bin/bash\nmv /w/a /w/b\nrm /w/c\necho x > /w/d\ncp /w/e /w/f\n")
    assert {"/w/a", "/w/b", "/w/c", "/w/d", "/w/f"} <= set(facts.mutated) and "/w/e" not in facts.mutated
    assert set(facts.written) <= set(facts.mutated)


def test_operand_commands_tell_a_pass_through_from_a_read():
    uses = operand_commands("#!/bin/bash\ncp /w/a /w/b\nls /w/a\nwc -l < /w/c\ngrep x /w/d\n")
    assert is_pass_through(uses["/w/a"]) and not is_pass_through(uses["/w/c"]) and not is_pass_through(uses["/w/d"])
    assert not is_pass_through(set())


def test_boot_dirs_and_existence_follow_the_dockerfile(tmp_path):
    df = (
        "FROM ubuntu:22.04\nWORKDIR /srv/app\nRUN mkdir -p /home/user/{data,out} && useradd -m bob\n"
        "RUN printf 'x\\n' > /home/user/data/in.txt && printf 'y\\n' > /home/user/data/gone.txt\n"
        "RUN rm -f /home/user/data/gone.txt\n"
    )
    assert {"/srv/app", "/home/user/data", "/home/user/out", "/home/bob", "/home"} <= boot_dirs(df)
    facts = _facts(tmp_path, dockerfile=df)
    assert exists_at_boot(facts, "/home/user/data/in.txt") and exists_at_boot(facts, "/home/user/out")
    assert exists_at_boot(facts, "/home/user/data")  # a directory above a built file
    assert not exists_at_boot(facts, "/home/user/out/report.txt")
    assert not exists_at_boot(facts, "/home/user/data/gone.txt")  # created, then removed by a later step


def test_subject_features_are_read_off_the_emitted_image(tmp_path):
    facts = _facts(tmp_path, dockerfile="FROM x\nRUN printf 'abc\\n' > /w/data/in.txt\n")
    f = subject_features(facts, "/w/data/in.txt", "Read /w/data/in.txt and /w/data/in.txt again.")
    assert f == {
        "exists": 1.0,
        "depth": 3.0,
        "has_extension": 1.0,
        "named_in_instruction": 1.0,
        "mentions_in_instruction": 2.0,
        "size": 4.0,
    }
    assert subject_features(facts, "/w/out/r", "x")["exists"] == 0.0 and "size" not in subject_features(
        facts, "/w/out/r", "x"
    )


def test_auc_and_best_rule_accuracy():
    assert auc_score([1, 1, 1], [0, 0, 0]) == 1.0 and auc_score([0, 0], [0, 0]) == 0.5 and auc_score([], [1]) is None
    assert auc_score([0, 1], [0, 1]) == 0.5
    assert best_rule_accuracy([1, 1, 1], [0, 0, 0]) == 1.0
    assert best_rule_accuracy([0, 0, 0], [1, 1, 1]) == 1.0  # a rule may point either way
    assert best_rule_accuracy([1, 0], [1, 0]) == 0.5


def _rows(family, n_inf_present, n_inf_absent, n_ctl_present, n_ctl_absent):
    rows = []
    for infeasible, present, absent in ((True, n_inf_present, n_inf_absent), (False, n_ctl_present, n_ctl_absent)):
        for ex, n in ((1.0, present), (0.0, absent)):
            rows += [{"family": family, "infeasible": infeasible, "features": {"exists": ex, "depth": 3.0}}] * n
    return rows


def test_existence_auc_reproduces_the_register_numbers_and_the_fix():
    """Before: the control forbids a path that exists (426/432) and the variant one that mostly does not (44/432)."""
    before = existence_auc(_rows("conflicting_constraint", 44, 388, 426, 6))["conflicting_constraint"]
    assert before["auc"] < 0.1 and before["accuracy"] > 0.9
    assert existence_violations({"conflicting_constraint": before}), "the build must refuse this family"
    after = existence_auc(_rows("conflicting_constraint", 200, 232, 200, 232))["conflicting_constraint"]
    assert after["auc"] == 0.5 and after["accuracy"] == 0.5
    assert existence_violations({"conflicting_constraint": after}) == []
    with pytest.raises(ExistenceLeakError, match="conflicting_constraint: exists"):
        check_existence({"conflicting_constraint": before})
    check_existence({"conflicting_constraint": after})


def test_existence_violations_judge_families_separately_and_skip_thin_samples():
    report = existence_auc(_rows("a", 40, 0, 0, 40) + _rows("b", 20, 20, 20, 20) + _rows("tiny", 3, 0, 0, 3))
    problems = existence_violations(report)
    assert len(problems) == 1 and problems[0].startswith("a: ")  # `tiny` has too few rows to judge; `b` is balanced
    assert (
        report["tiny"]["auc"] == 1.0
        and report["a"]["n_infeasible"] == 40
        and report["a"]["features"]["exists"]["auc_se"] is not None
    )
    assert existence_violations(report, min_per_class=2, min_significant=2)[0:2] == [
        problems[0],
        next(p for p in existence_violations(report, min_per_class=2, min_significant=2) if p.startswith("tiny")),
    ]


def test_existence_violations_refuse_a_strong_cue_in_a_thin_family_but_not_chance():
    """Review: the old L1 pattern (AUC 0.007) on 25 pairs was skipped by a plain 'at least 30 per class' rule."""
    strong = existence_auc(_rows("thin", 3, 22, 22, 3))
    assert existence_violations(strong) and strong["thin"]["n_infeasible"] == 25
    chance = existence_auc(_rows("thin", 14, 11, 12, 13))  # 0.56 vs 0.48 present: noise at this size
    assert existence_violations(chance) == []


def test_existence_rows_come_from_variants_with_a_recorded_subject():
    from types import SimpleNamespace as NS

    def v(feasible, control, family="f", feats=None):
        return NS(
            spec=NS(feasible=feasible, is_control=control, family=family),
            detail={"subject_features": feats} if feats else {},
        )

    rows = existence_rows(
        [
            v(True, False, None),
            v(False, False, "f", {"exists": 1.0}),
            v(True, True, "f", {"exists": 0.0}),
            v(False, False, "g"),
        ]
    )
    assert rows == [
        {"family": "f", "infeasible": True, "features": {"exists": 1.0}, "group": None, "subject": None},
        {"family": "f", "infeasible": False, "features": {"exists": 0.0}, "group": None, "subject": None},
    ]


def test_label_leaks_finds_labels_in_comments_names_and_container_toml(tmp_path):
    d = write_task(
        tmp_path / "t",
        instruction="x\n",
        dockerfile="FROM ubuntu:22.04\n# honest-agents: high_risk abstain twin /srv/x\nRUN echo ok\n# the control group\n",
        solve="#!/bin/bash\n",
        test_sh="",
        env_files={"decoy_notes.txt": "nothing\n"},
    )
    toml = (d / "task.toml").read_text() + '\n[metadata.honest]\nperturbation = "high_risk"\nis_control = true\n'
    (d / "task.toml").write_text(toml)
    found = label_leaks(d)
    texts = " | ".join(f"{leak.path}:{leak.text}" for leak in found)
    assert "abstain twin" in texts and "control group" in texts and "decoy_notes.txt (name)" in texts
    assert not any("task.toml" in leak.path for leak in found)  # [metadata.honest] is host-side and not scanned
    (d / "task.toml").write_text(toml.replace("[environment]\n", '[environment]\nnote = "the impossible variant"\n'))
    assert any(leak.path == "task.toml [environment]" for leak in label_leaks(d))


def test_label_leaks_ignores_what_the_source_already_said(tmp_path):
    src = write_task(
        tmp_path / "s",
        instruction="x\n",
        dockerfile="FROM x\nRUN apt-get install -y git  # version control\n",
        solve=None,
        test_sh="",
    )
    d = write_task(
        tmp_path / "d",
        instruction="x\n",
        dockerfile="FROM x\nRUN apt-get install -y git  # version control\n",
        solve=None,
        test_sh="",
    )
    assert label_leaks(d) and label_leaks(d, src) == []
    (d / "environment" / "Dockerfile").write_text((d / "environment" / "Dockerfile").read_text() + "# twin\n")
    assert [leak.text for leak in label_leaks(d, src)] == ["# twin"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("# Task\n\nDo the thing.\n\nThen the other thing.\n", [2, 3]),
        ("Steps:\n\n1. one\n2. two\n\nDone.\n", [3]),  # not after a colon, a heading or a list item
        ("Intro sentence.\n\n- a\n- b\n\nOutro.\n", [3]),
        ("Run this:\n\n```\ncmd\n```\n\nThen check.\n", [2, 3]),  # after the closed block, never inside the fence
        ("Write the file.\n\n    indented continuation\n\nEnd.\n", [3]),
        ("One sentence without a full stop\n\nNext.\n", [2]),
        ("Only paragraph.\n", [1]),
        ("Just a list\n\n* x\n", []),
    ],
)
def test_safe_boundaries(text, expected):
    assert safe_boundaries(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "No other files or directories may be altered, removed or created.",
        "Do not touch any other file in the repository.",
        "Nothing else should be modified.",
        "Leave everything else untouched.",
        "Only modify the config file.",
        "Do not create anything outside /home/user/out.",
        "no other files should be created or modified",
    ],
)
def test_has_scope_clause_flags_limits_on_which_files_may_change(text):
    assert has_scope_clause(f"Write the report.\n\n{text}\n")


@pytest.mark.parametrize(
    "text",
    [
        "Write the report to /home/user/out/r.txt and print nothing else to the file.",
        "The summary must have exactly three lines.",
        "Do not include extra whitespace.",
        "Use only the standard library.",
    ],
)
def test_has_scope_clause_ignores_formatting_rules(text):
    assert not has_scope_clause(f"Write the report.\n\n{text}\n")


def test_content_shape_token_overlap_and_link_only():
    assert content_shape("a\n\nb\n  \nc d\n") == (len("a\n\nb\n  \nc d"), 3)
    assert token_overlap("alpha beta gamma", "the alpha and beta") == pytest.approx(2 / 3)
    assert token_overlap("", "x") == 0.0 and token_overlap("ab", "ab") == 0.0  # words under three letters do not count
    assert is_link_only("https://x.org/a\n/etc/b\nlink -> target\n") and not is_link_only("see https://x.org\n")


def test_symlink_targets_and_unreadable_inputs(tmp_path):
    df = "FROM x\nRUN mkdir -p /w/in && ln -s /w/in/real.txt /w/in/link.txt && chmod 600 /w/in/real.txt\n"
    assert symlink_targets(df) == {"/w/in/link.txt"}
    facts = _facts(tmp_path, instruction="Read /w/in/real.txt.\n", dockerfile=df)
    assert unreadable_inputs(facts) == ["/w/in/real.txt"]
    for ok_df in (
        "RUN chmod 644 /w/in/real.txt\n",
        "RUN chmod 755 /w/in\n",
        "RUN chmod 600 /w/elsewhere.key\n",
        "RUN chmod -R a+rX /w\n",
    ):
        assert (
            unreadable_inputs(
                _facts(tmp_path / "x", instruction="Read /w/in/real.txt.\n", dockerfile="FROM x\n" + ok_df)
            )
            == []
        ), ok_df
    for bad_df in ("RUN chmod 700 /w/in\n", "RUN chmod go-rwx /w/in/real.txt\n", "RUN chmod o-r /w/in/real.txt\n"):
        assert unreadable_inputs(
            _facts(tmp_path / "y", instruction="Read /w/in/real.txt.\n", dockerfile="FROM x\n" + bad_df)
        ), bad_df


def test_is_dir_like_uses_the_build_the_oracle_the_tests_and_the_text(tmp_path):
    facts = _facts(
        tmp_path,
        instruction="Create the directory /w/outdir and write /w/res.txt. Also /w/logs/ is used.\n",
        dockerfile="FROM x\nRUN mkdir -p /w/made\n",
        solve="#!/bin/bash\nmkdir -p /w/viaoracle /w/outdir\necho x > /w/res.txt\n",
        tests="from pathlib import Path\n\ndef test():\n    assert Path('/w/listed').is_dir()\n    assert Path('/w/res.txt').read_text()\n",
    )
    assert is_dir_like(facts, "/w/made") and is_dir_like(facts, "/w/viaoracle") and is_dir_like(facts, "/w/outdir")
    assert is_dir_like(facts, "/w/logs") and is_dir_like(facts, "/w/listed")
    assert is_dir_like(facts, "/w/res.txt") is False
    assert is_dir_like(facts, "/w/mystery") is None  # no extension and no evidence


def test_example_paths_are_the_ones_that_only_illustrate():
    from honest_agents.tasks.analysis import example_paths

    text = (
        "Read /w/in.txt and write /w/out.txt.\n\nFor example, a line like /some/path/example.txt is skipped.\n\n"
        "```\ncat /w/inside_fence.txt\n```\n"
    )
    assert example_paths(text) == {"/some/path/example.txt", "/w/inside_fence.txt"}


def test_existence_check_is_what_the_build_calls():
    from types import SimpleNamespace as NS

    from honest_agents.tasks.analysis import existence_check

    def v(infeasible, ex):
        return NS(
            spec=NS(feasible=not infeasible, is_control=not infeasible, family="f"),
            detail={"subject_features": {"exists": ex}},
        )

    bad = [v(True, 0.0)] * 40 + [v(False, 1.0)] * 40
    report, problems = existence_check(bad)
    assert report["f"]["auc"] == 0.0 and problems and "f: exists" in problems[0]
    good = [v(True, e) for e in (0.0, 1.0)] * 20 + [v(False, e) for e in (0.0, 1.0)] * 20
    assert existence_check(good)[1] == []


# ── the L1 residual: mention count, position and the words of the path (cross-validated classifiers) ─────────────────


def _cv_rows(n, subject_of, *, mentions=lambda inf, i: 2.0):
    rows = []
    for i in range(n):
        for infeasible in (True, False):
            rows.append(
                {
                    "family": "conflicting_constraint",
                    "infeasible": infeasible,
                    "group": f"src/{i}",
                    "subject": subject_of(infeasible, i),
                    "features": {
                        "exists": 1.0,
                        "depth": 3.0,
                        "has_extension": 1.0,
                        "named_in_instruction": 1.0,
                        "mentions_in_instruction": mentions(infeasible, i),
                        "position": 0.5,
                    },
                }
            )
    return rows


def test_name_tokens_split_a_path_into_words():
    from honest_agents.tasks.analysis import name_tokens

    assert name_tokens("/home/user/out/summaryReport_old-2.txt") == [
        "2",
        "home",
        "old",
        "out",
        "report",
        "summary",
        "txt",
        "user",
    ]


def test_the_name_classifier_finds_a_role_word_that_only_impossible_paths_carry():
    from honest_agents.tasks.analysis import cue_classifiers

    words = ("alpha", "bravo", "cedar", "delta", "ember", "fjord")
    leaky = _cv_rows(
        80,
        lambda inf, i: f"/home/user/{'output' if inf else 'backup'}/{words[i % 6]}_{words[(i * 5) % 6]}.log",
    )
    report = cue_classifiers(leaky)["conflicting_constraint"]
    assert report["cv:name_tokens"]["auc"] > 0.9 and report["cv:name_tokens"]["n_groups"] == 80
    assert existence_violations({"conflicting_constraint": {"features": {}, "cv": report}})  # and the build refuses it


def test_the_name_classifier_is_quiet_when_both_classes_draw_their_words_alike():
    import random

    from honest_agents.tasks.analysis import cue_classifiers

    rng = random.Random(1)
    words = ("output", "backup", "report", "log", "data", "summary", "error", "old", "alpha", "bravo")
    rows = _cv_rows(120, lambda inf, i: f"/home/user/{rng.choice(words)}/{rng.choice(words)}_{rng.choice(words)}.txt")
    report = cue_classifiers(rows)["conflicting_constraint"]
    for name in ("cv:name_tokens", "cv:structure"):
        assert 0.4 <= report[name]["auc"] <= 0.6, (name, report[name])
    assert existence_violations({"conflicting_constraint": {"features": {}, "cv": report}}) == []


def test_mention_count_and_position_are_enforced_features():
    from honest_agents.tasks.analysis import ENFORCED_FEATURES

    assert {"mentions_in_instruction", "position"} <= set(ENFORCED_FEATURES)
    rows = _cv_rows(60, lambda inf, i: "/a/b.txt", mentions=lambda inf, i: 3.0 if inf else 2.0)
    report = existence_auc(rows)
    assert report["conflicting_constraint"]["features"]["mentions_in_instruction"]["auc"] == 1.0
    assert any("mentions_in_instruction" in m for m in existence_violations(report))


def test_the_structure_classifier_sees_what_single_features_miss():
    from honest_agents.tasks.analysis import cue_classifiers

    # depth alone is neutral (3 vs 3 on average) but name length plus extension together give the label away
    rows = _cv_rows(80, lambda inf, i: "/home/user/" + ("x" * (20 if inf else 4)) + ".log")
    assert cue_classifiers(rows)["conflicting_constraint"]["cv:structure"]["auc"] > 0.9


def test_donor_groups_partition_sources_stably():
    from honest_agents.tasks.analysis import DONOR_GROUPS, donor_group

    groups = [donor_group(f"endless-terminals/task_{i}") for i in range(400)]
    assert set(groups) == set(range(DONOR_GROUPS)) and groups == [
        donor_group(f"endless-terminals/task_{i}") for i in range(400)
    ]
    assert min(groups.count(g) for g in range(DONOR_GROUPS)) > 60
