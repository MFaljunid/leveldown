import numpy as np
import pandas as pd

from leveldown.analysis import decompose, matched_gap, ols_gap
from leveldown.data import activity_groups, load, temporal_loo
from leveldown.models import EASE, IALS, ItemPop, evaluate, rank_of_target


def synth_udata(path, n_users=300, n_items=200, seed=0):
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(n_users, 6)), rng.normal(size=(n_items, 6))
    pop = rng.normal(size=n_items)
    rows = []
    for u in range(n_users):
        s = U[u] @ V.T + 1.5 * pop
        p = np.exp(s - s.max()); p /= p.sum()
        n = int(rng.integers(8, 120))
        for t, it in enumerate(rng.choice(n_items, n, replace=False, p=p)):
            rows.append((u + 1, int(it) + 1, 4, t))
    pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False)


def test_split_and_groups(tmp_path):
    synth_udata(tmp_path / "u.data")
    s = load("ml-100k", tmp_path)
    for u in range(s.n_users):
        assert s.val[u] not in s.train[u] and s.test[u] not in s.train[u] and s.val[u] != s.test[u]
    g, n = activity_groups(s.train)
    assert set(g) == {0, 1, 2, 3} and n[g == 3].mean() > n[g == 0].mean()
    assert abs(np.bincount(g).max() - np.bincount(g).min()) <= 1


def test_temporal_order():
    df = pd.DataFrame({"user": [1] * 5, "item": [10, 11, 12, 13, 14], "rating": 4, "ts": [5, 1, 4, 2, 3]})
    s = temporal_loo(df, min_interactions=5)
    # by time: 11(1) 13(2) 14(3) 12(4) 10(5) -> test 10, val 12
    inv = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14}
    assert inv[int(s.test[0])] == 10 and inv[int(s.val[0])] == 12


def test_rank_of_target():
    s = np.array([5., 4., 3., 2., 1.])
    assert rank_of_target(s, 2, [0]) == 1 and rank_of_target(s, 0, []) == 0
    assert rank_of_target(np.array([1., 1., 1.]), 0, []) == 2   # pessimistic ties


def test_models_beat_itempop(tmp_path):
    synth_udata(tmp_path / "u.data", n_users=400)
    s = load("ml-100k", tmp_path)
    _, nd_pop = evaluate(ItemPop().fit(s.train, s.n_items), s, "val")
    _, nd_als = evaluate(IALS(dim=16, alpha=5, reg=10, iters=8).fit(s.train, s.n_items), s, "val")
    _, nd_ease = evaluate(EASE(lam=50).fit(s.train, s.n_items), s, "val")
    assert nd_als.mean() > nd_pop.mean() and nd_ease.mean() > nd_pop.mean()


def test_decompose_planted():
    rng = np.random.default_rng(0)
    n = 3000
    g = rng.integers(0, 4, n)
    c = -g + rng.normal(size=n)            # covariate shifts with group
    y = 0.5 * c + rng.normal(size=n) * 0.1
    d = decompose(y, g, c, 0, 3, B=100)
    assert d["share"] > 0.9 and abs(d["matched"]) < abs(d["raw"]) / 3
    assert abs(ols_gap(y, g, [], 0, 3)) > 1
    mg, cov = matched_gap(y, g, np.zeros(n, int), 0, 3)
    assert np.isclose(mg, y[g == 3].mean() - y[g == 0].mean()) and cov == 1.0


def test_weighted_models_reduce_to_standard_and_change_with_weights(tmp_path):
    synth_udata(tmp_path / "u.data", n_users=200, n_items=150)
    s = load("ml-100k", tmp_path)
    w1 = np.ones(s.n_users)
    a = EASE(lam=50).fit(s.train, s.n_items)
    b = EASE(lam=50, user_weights=w1).fit(s.train, s.n_items)
    assert np.allclose(a.B, b.B)
    c = IALS(dim=8, iters=3, seed=1).fit(s.train, s.n_items)
    d = IALS(dim=8, iters=3, seed=1, user_weights=w1).fit(s.train, s.n_items)
    assert np.allclose(c.U, d.U) and np.allclose(c.V, d.V)
    w = np.where(np.arange(s.n_users) % 2 == 0, 4.0, 1.0)
    e = EASE(lam=50, user_weights=w).fit(s.train, s.n_items)
    assert not np.allclose(a.B, e.B)


def test_gender_groups_align_with_split(tmp_path):
    synth_udata(tmp_path / "u.data", n_users=60, n_items=150)
    rows = [f"{u}|30|{'F' if u % 3 == 0 else 'M'}|other|00000" for u in range(1, 61)]
    (tmp_path / "u.user").write_text("\n".join(rows) + "\n")
    from leveldown.data import group_labels
    s = load("ml-100k", tmp_path)
    lab, names = group_labels(s, "gender", "ml-100k", tmp_path)
    assert names == ["F", "M"] and len(lab) == s.n_users
    for u in range(s.n_users):
        assert lab[u] == (0 if int(s.users[u]) % 3 == 0 else 1)


def test_itemknn_and_sparse_ease(tmp_path):
    from leveldown.models import ItemKNN, interaction_matrix
    synth_udata(tmp_path / "u.data", n_users=300, n_items=200)
    s = load("ml-100k", tmp_path)
    _, nd_pop = evaluate(ItemPop().fit(s.train, s.n_items), s, "val")
    knn = ItemKNN(k=50).fit(s.train, s.n_items)
    _, nd_knn = evaluate(knn, s, "val")
    assert nd_knn.mean() > nd_pop.mean()
    knn1 = ItemKNN(k=50, user_weights=np.ones(s.n_users)).fit(s.train, s.n_items)
    assert np.allclose(knn.S.toarray(), knn1.S.toarray())
    # sparse EASE equals the dense closed form
    X = interaction_matrix(s.train, s.n_items)
    P = np.linalg.inv(X.T @ X + 50 * np.eye(s.n_items))
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0.0)
    assert np.allclose(EASE(lam=50).fit(s.train, s.n_items).B, B)


def test_gowalla_loader_dedupe_and_kcore(tmp_path):
    from leveldown.data import k_core, read_ratings
    lines = []
    for u in range(30):
        for t, it in enumerate(range(u % 5, u % 5 + 12)):
            lines.append(f"{u}\t2010-10-{10 + t:02d}T10:00:00Z\t0.0\t0.0\t{it}")
    lines.append("0\t2010-11-30T10:00:00Z\t0.0\t0.0\t0")            # repeat visit: dropped (first kept)
    lines.append("99\t2010-10-01T10:00:00Z\t0.0\t0.0\t500")         # sparse user/item: removed by k-core
    (tmp_path / "loc-gowalla_totalCheckins.txt").write_text("\n".join(lines) + "\n")
    df = read_ratings("gowalla", tmp_path)
    assert not df.duplicated(["user", "item"]).any()
    first = df[(df.user == 0) & (df.item == 0)]["ts"].iloc[0]
    assert first == pd.Timestamp("2010-10-10T10:00:00Z").value // 10**9
    core = k_core(df, 10, 10)
    assert 99 not in set(core.user) and core.groupby("user").size().min() >= 10
    s = load("gowalla", tmp_path, core=10)
    assert s.n_users == 30


def test_bprmf_learns_and_gap_penalty_moves_training_losses(tmp_path):
    from leveldown.models import BPRMF
    synth_udata(tmp_path / "u.data", n_users=300, n_items=200)
    s = load("ml-100k", tmp_path)
    _, nd_pop = evaluate(ItemPop().fit(s.train, s.n_items), s, "val")
    m0 = BPRMF(dim=16, epochs=15, batch=512, seed=1).fit(s.train, s.n_items)
    _, nd = evaluate(m0, s, "val")
    assert nd.mean() > nd_pop.mean()
    m0b = BPRMF(dim=16, epochs=15, batch=512, seed=1).fit(s.train, s.n_items)
    assert np.allclose(m0.U, m0b.U)                                    # deterministic

    n = np.array([len(t) for t in s.train])
    lab = (n > np.median(n)).astype(int)                                # activity halves

    def inter_loss(m, g):                                               # the quantity the penalty targets:
        tot = cnt = 0.0                                                 # mean BPR loss per training interaction
        for u in np.where(lab == g)[0]:
            sc = m.scores(np.array([u]))[0]
            pos = s.train[u]
            neg = np.setdiff1d(np.arange(s.n_items), pos)
            tot += np.logaddexp(0, -(sc[pos][:, None] - sc[neg][None, :])).mean(axis=1).sum()
            cnt += len(pos)
        return tot / cnt
    gap0 = abs(inter_loss(m0, 0) - inter_loss(m0, 1))
    m1 = BPRMF(dim=16, epochs=15, batch=512, seed=1, gap_lambda=0.1, gap_groups=(lab, 0, 1)).fit(s.train, s.n_items)
    assert abs(inter_loss(m1, 0) - inter_loss(m1, 1)) < gap0


def test_sampled_evaluation_is_easier_than_full(tmp_path):
    from leveldown.models import evaluate_sampled
    synth_udata(tmp_path / "u.data", n_users=200, n_items=300)
    s = load("ml-100k", tmp_path, tie_break="random")
    m = IALS(dim=16, iters=5).fit(s.train, s.n_items)
    _, full = evaluate(m, s, "test")
    _, samp = evaluate_sampled(m, s, "test", n_neg=99)
    assert samp.mean() >= full.mean()