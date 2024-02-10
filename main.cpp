#include <bits/stdc++.h>
#include <random>

using namespace std;

using ll = long long;

ll next_long(mt19937 &rnd, ll l, ll u) {
    uniform_int_distribution<ll> dist(l, u - 1);
    return dist(rnd);
}

struct P {
    ll i, j;
    P(ll i, ll j) : i(i), j(j) {}
};

struct stamp {
    ll h, w;
    vector<P> ps;

    stamp(ll h, ll w, const vector<P> &ps) : h(h), w(w), ps(ps) {}

    ll size() {
        return ps.size();
    }
};

double calc_ent(const ll &N, const vector<vector<double>> &prob) {
    double mx_ent = 0.0;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            mx_ent = max(mx_ent, min(prob[i][j], 1 - prob[i][j]));
        }
    }
    return mx_ent;
}

ll naive_matcher(
        const ll &n, const ll &m, const double &e,
        vector<stamp> &s,
        vector<vector<ll>> &field,
        vector<vector<double>> &prob,
        ll k,
        vector<vector<ll>> &f2) {
    if (k == m) {
        for (ll i = 0; i < n; i++) {
            for (ll j = 0; j < n; j++) {
                if (field[i][j] >= 0 && f2[i][j] != field[i][j]) {
                    return 0;
                }
            }
        }
        for (ll i = 0; i < n; i++) {
            for (ll j = 0; j < n; j++) {
                if (f2[i][j] > 0) {
                    prob[i][j] += 1;
                }
            }
        }
        return 1LL;
    }

    ll cnt = 0;
    for (ll si = 0; si + s[k].h <= n; si++) {
        for (ll sj = 0; sj + s[k].w <= n; sj++) {
            bool ok = true;
            for (ll l = 0; l < s[k].size(); l++) {
                ll i = s[k].ps[l].i;
                ll j = s[k].ps[l].j;
                f2[si + i][sj + j] += 1;
                if (field[si + i][sj + j] >= 0 && f2[si + i][sj + j] > field[si + i][sj + j]) {
                    ok = false;
                }
            }
            if (ok) {
                cnt += naive_matcher(n, m, e, s, field, prob, k + 1, f2);
            }
            for (ll l = 0; l < s[k].size(); l++) {
                ll i = s[k].ps[l].i;
                ll j = s[k].ps[l].j;
                f2[si + i][sj + j] -= 1;
            }
        }
    }

    return cnt;
}

void prob_naive(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<ll>> field(N, vector<ll>(N, -1));

    ll remaining = 0;
    for (ll i = 0; i < M; i++) {
        remaining += s[i].size();
    }

    while (remaining > 0) {
        const ll trial = 2000;
        ll cnt = 0;

        vector<vector<double>> prob(N, vector<double>(N, 0));
        for (ll t = 0; t < trial; t++) {
            vector<vector<ll>> f2(N, vector<ll>(N, 0));
            for (ll k = 0; k < M; k++) {
                ll si = next_long(rnd, 0, N - s[k].h + 1);
                ll sj = next_long(rnd, 0, N - s[k].w + 1);
                for (ll l = 0; l < s[k].size(); l++) {
                    ll i = s[k].ps[l].i;
                    ll j = s[k].ps[l].j;
                    f2[si + i][sj + j] += 1;
                }
            }
            bool ok = true;
            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    if (field[i][j] >= 0 && field[i][j] != f2[i][j]) {
                        ok = false;
                    }
                }
            }
            if (ok) {
                cnt++;
                for (ll i = 0; i < N; i++) {
                    for (ll j = 0; j < N; j++) {
                        if (f2[i][j] > 0) {
                            prob[i][j] += 1;
                        }
                    }
                }
            }
        }

        if (cnt == 0) {
            vector<vector<ll>> f2(N, vector<ll>(N, 0));
            ll c = naive_matcher(N, M, e, s, field, prob, 0, f2);

            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    prob[i][j] /= c;
                }
            }

            double mx_ent = calc_ent(N, prob);
            if (mx_ent < 1e-9) {
                for (ll i = 0; i < N; i++) {
                    for (ll j = 0; j < N; j++) {
                        if (prob[i][j] > 0) {
                            field[i][j] = 1;
                        }
                    }
                }
                break;
            }
        } else {
            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    prob[i][j] /= cnt;
                }
            }
        }

        double mx_ent = calc_ent(N, prob);
        if (mx_ent < 1e-9) {
            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    prob[i][j] = 0;
                }
            }
            ll c = naive_matcher(N, M, e, s, field, prob, 0, field);
            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    prob[i][j] /= c;
                }
            }
            mx_ent = calc_ent(N, prob);
            if (mx_ent < 1e-9) {
                for (ll i = 0; i < N; i++) {
                    for (ll j = 0; j < N; j++) {
                        if (prob[i][j] > 0) {
                            field[i][j] = 1;
                        }
                    }
                }
                break;
            }
        }

        ll gi = -1, gj = -1;
        for (ll i = 0; i < N; i++) {
            for (ll j = 0; j < N; j++) {
                if (field[i][j] < 0 && (gi < 0 || abs(prob[i][j] - 0.5) < abs(prob[gi][gj] - 0.5))) {
                    gi = i;
                    gj = j;
                }
            }
        }
        cout << "q 1 " << gi << " " << gj << endl;
        flush(cout);
        ll v;
        cin >> v;
        field[gi][gj] = v;
        remaining -= v;
    }

    vector<P> result;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] > 0) {
                result.emplace_back(i, j);
            }
        }
    }

    cout << "a " << result.size();
    for (auto r: result) {
        cout << " " << r.i << " " << r.j;
    }
}

void naive_solver(ll N, ll M, double e, vector<stamp> &s, mt19937 &rnd) {
    ll sum = 0;
    for (ll i = 0; i < M; i++) {
        sum += s[i].size();
    }

    vector<P> result;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            cout << "q 1 " << i << " " << j << endl;
            flush(cout);
            ll v;
            cin >> v;
            if (v > 0) {
                result.emplace_back(i, j);
            }
            sum -= v;
            if (sum == 0) {
                break;
            }
        }
        if (sum == 0) {
            break;
        }
    }

    cout << "a " << result.size();
    for (auto r: result) {
        cout << " " << r.i << " " << r.j;
    }
    cout << endl;
    flush(cout);
    ll res;
    cin >> res;
}

int main() {
    mt19937 rnd(123);

    ll N, M;
    double e;
    cin >> N >> M >> e;
    vector<vector<P>> p(M);
    vector<stamp> s;
    for (ll i = 0; i < M; i++) {
        ll d;
        cin >> d;

        ll li = 0, lj = 0, ui = 0, uj = 0;
        for (ll j = 0; j < d; j++) {
            ll a, b;
            cin >> a >> b;
            p[i].emplace_back(a, b);
            ui = max(ui, a);
            uj = max(uj, b);
        }
        stamp st(ui + 1, uj + 1, p[i]);
        s.push_back(st);
    }

    if (N <= 15 && M <= 2) {
        prob_naive(N, M, e, s, rnd);
    } else {
        naive_solver(N, M, e, s, rnd);
    }

    return 0;
}
