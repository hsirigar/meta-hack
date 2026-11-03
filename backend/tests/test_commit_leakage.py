"""SPEC §8 leakage guards + commitment integrity."""
import json

import numpy as np
import pytest

from ghostline import pipeline as P
from ghostline import storage
from ghostline.commit import canonical_json, sha256_hex, verify_commitment
from ghostline.schemas import ValidityReport

from conftest import make_array_run


def test_canonical_json_is_deterministic():
    a = {"b": 1.0000000001, "a": [0.1, 2], "c": {"z": 1e-12, "y": "s"}}
    b = {"c": {"y": "s", "z": 1.0e-12}, "a": [0.1, 2.0], "b": 1.0}
    assert canonical_json(a) == canonical_json(b)
    assert " " not in canonical_json(a)
    assert sha256_hex(canonical_json(a)) == sha256_hex(canonical_json(b))


def test_mutating_holdout_does_not_change_fit():
    e, r = make_array_run("pendulum", 8.0)
    f1 = P.fit_run(r.id, bootstrap=0)
    r = storage.load_run(r.id)
    ts = r.series["theta"]
    t = np.asarray(ts.t)
    y = np.asarray(ts.y)
    y[t > r.split.train_end_t] += np.random.default_rng(1).normal(0, 0.5, (t > r.split.train_end_t).sum())
    ts.y = y.tolist()
    storage.save_run(r)
    f2 = P.fit_run(r.id, bootstrap=0)
    for a, b in zip(f1, f2):
        assert a.params == b.params
        assert a.train_rmse == b.train_rmse
        assert a.holdout_rmse != b.holdout_rmse  # holdout scoring did see the change


def test_split_is_immutable():
    e, r = make_array_run("pendulum", 8.0)
    with pytest.raises(Exception):
        r.split.train_end_t = 1.0


def test_reveal_runs_are_never_refit():
    e, r = make_array_run("pendulum", 8.0)
    P.fit_run(r.id, bootstrap=0)
    P.design(e.id)
    c = P.commit(e.id)
    _, rv = make_array_run("pendulum", 70.0, role="reveal", exp_id=e.id, commitment_id=c.id, seed=2)
    with pytest.raises(PermissionError):
        P.fit_run(rv.id)


def _committed():
    e, r = make_array_run("pendulum", 8.0)
    P.fit_run(r.id, bootstrap=0)
    P.design(e.id)
    return e, P.commit(e.id)


def test_tampered_commitment_fails_verification():
    e, c = _committed()
    assert verify_commitment(c)
    content = json.loads(c.canonical_json)
    content["models"][0]["params"]["omega0"] *= 1.01
    forged = c.model_copy(update={"canonical_json": canonical_json(content)})
    assert not verify_commitment(forged)  # hash no longer matches
    forged2 = c.model_copy(deep=True)
    forged2.models[0].params["omega0"] *= 1.01  # structured copy edited, hashed JSON untouched
    assert not verify_commitment(forged2)
    _, rv = make_array_run("pendulum", 70.0, role="reveal", exp_id=e.id, commitment_id=c.id, seed=2)
    from ghostline.commit import evaluate_commitment
    ts = rv.series["theta"]
    v, _ = evaluate_commitment(forged, rv.id, np.asarray(ts.t), np.asarray(ts.y), np.ones(len(ts.t), bool),
                               ts.noise_sigma, ValidityReport(status="valid", checks=[]))
    assert v.hash_verified is False and v.winner is None and v.per_model == []


def test_commitment_is_write_once():
    e, c = _committed()
    with pytest.raises(storage.ImmutableError):
        storage.save_commitment(c)


def test_commitment_bytes_unchanged_by_reveal():
    e, c = _committed()
    path = storage.data_dir() / "commits" / f"{c.id}.json"
    before = path.read_bytes()
    _, rv = make_array_run("pendulum", 70.0, role="reveal", exp_id=e.id, commitment_id=c.id, seed=2)
    P.check_validity  # validity needs tracks; score directly with a valid report
    rv = storage.load_run(rv.id)
    rv.validity = ValidityReport(status="valid", checks=[])
    storage.save_run(rv)
    v = P.evaluate(c.id, rv.id)
    assert v.hash_verified and v.winner == "pendulum_nonlinear"
    assert path.read_bytes() == before
    assert storage.load_commitment(c.id).canonical_json == c.canonical_json


def test_invalid_run_is_not_scored():
    e, c = _committed()
    _, rv = make_array_run("pendulum", 70.0, role="reveal", exp_id=e.id, commitment_id=c.id, seed=2)
    rv = storage.load_run(rv.id)
    from ghostline.schemas import ValidityCheck
    rv.validity = ValidityReport(status="invalid", checks=[ValidityCheck(name="camera_jump", status="invalid", reason="bumped")])
    storage.save_run(rv)
    v = P.evaluate(c.id, rv.id)
    assert v.winner is None and v.per_model == [] and "INVALID" in v.summary_facts["refused"]
