import numpy as np
import pandas as pd

from src.a4_ortho import OrthoRouterConfig, ortho_route, rank01, validate_submission_frame


LABELS = [f"L{i}" for i in range(12)]


def test_rank01_shape_and_range():
    x = np.arange(60, dtype=float).reshape(5, 12)
    r = rank01(x)
    assert r.shape == x.shape
    assert np.isfinite(r).all()
    assert ((r > 0) & (r <= 1)).all()


def test_ortho_route_is_bounded_and_deterministic():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(32, 12))
    b = rng.normal(size=(32, 12))
    c = b + rng.normal(scale=0.03, size=(32, 12))
    y1, audit1 = ortho_route(a, b, c, OrthoRouterConfig(displacement_cap=0.2))
    y2, audit2 = ortho_route(a, b, c, OrthoRouterConfig(displacement_cap=0.2))
    assert np.array_equal(y1, y2)
    assert y1.shape == (32, 12)
    assert ((y1 > 0) & (y1 <= 1)).all()
    assert audit1["gate"].shape == y1.shape
    assert np.array_equal(audit1["gate"], audit2["gate"])


def test_submission_validator():
    uids = ["a", "b", "c"]
    df = pd.DataFrame(np.full((3, 12), 0.5), columns=LABELS)
    df.insert(0, "StudyInstanceUID", uids)
    validate_submission_frame(df, LABELS, uids)
