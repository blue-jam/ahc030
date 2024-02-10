#include <bits/stdc++.h>
#include <random>

static const double EPS = 1e-15;
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
            mx_ent = max(mx_ent, min(prob[i][j], abs(1 - prob[i][j])));
        }
    }
    return mx_ent;
}

ll dfs(
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
                cnt += dfs(n, m, e, s, field, prob, k + 1, f2);
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

ll naive_matcher(
        const ll &n, const ll &m, const double &e,
        vector<stamp> &s,
        vector<vector<ll>> &field,
        vector<vector<double>> &prob) {
    for (ll i = 0; i < n; i++) {
        for (ll j = 0; j < n; j++) {
            prob[i][j] = 0;
        }
    }
    vector<vector<ll>> f2(n, vector<ll>(n, 0));
    ll c = dfs(n, m, e, s, field, prob, 0, f2);
    for (ll i = 0; i < n; i++) {
        for (ll j = 0; j < n; j++) {
            prob[i][j] /= c;
        }
    }
    return c;
}

void prob_naive2(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<ll>> field(N, vector<ll>(N, -1));

    const ll SN = min(sqrt(N), 3.0);
    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    for (ll si = 0; si < N; si += SN) {
        for (ll sj = 0; sj < N; sj += SN) {
            vector<P> ps;
            for (ll i = 0; si + i < min(si + SN, N); i++) {
                for (ll j = 0; sj + j < min(sj + SN, N); j++) {
                    ps.emplace_back(si + i, sj + j);
                }
            }
            cout << "q " << ps.size();
            for (auto p: ps) {
                cout << " " << p.i << " " << p.j;
            }
            cout << endl;
            flush(cout);
            ll v;
            cin >> v;
            if (v == 0) {
                for (ll i = 0; si + i < min(si + SN, N); i++) {
                    for (ll j = 0; sj + j < min(sj + SN, N); j++) {
                        printf("#c %lld %lld yellow\n", si + i, sj + j);
                    }
                }
                for (ll i = 0; si + i < min(si + SN, N); i++) {
                    for (ll j = 0; sj + j < min(sj + SN, N); j++) {
                        init_prob[si + i][sj + j] = 0.01;
                    }
                }
            }
        }
    }

    ll remaining = 0;
    for (ll i = 0; i < M; i++) {
        remaining += s[i].size();
    }

    while (remaining > 0) {
        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            ll cnt = 0;
            for (ll si = 0; si + s[k].h <= N; si++) {
                for (ll sj = 0; sj + s[k].w <= N; sj++) {
                    bool ok = true;
                    for (ll l = 0; l < s[k].size(); l++) {
                        ll i = s[k].ps[l].i;
                        ll j = s[k].ps[l].j;
                        if (field[si + i][sj + j] == 0) {
                            ok = false;
                        }
                    }
                    if (ok) {
                        cnt++;
                        for (ll l = 0; l < s[k].size(); l++) {
                            ll i = s[k].ps[l].i;
                            ll j = s[k].ps[l].j;
                            prob_each[k][si + i][sj + j] += 1;
                        }
                    }
                }
            }
            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    prob_each[k][i][j] /= cnt;
                    prob_each[k][i][j] *= init_prob[i][j];
                }
            }
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        for (ll i = 0; i < N; i++) {
            for (ll j = 0; j < N; j++) {
                double p = 1.0;
                for (ll k = 0; k < M; k++) {
                    p *= 1 - prob_each[k][i][j];
                }
                prob[i][j] = (1 - p);
                if (field[i][j] == 0) {
                    prob[i][j] = 0;
                } else if (field[i][j] > 0) {
                    prob[i][j] = 1;
                }
            }
        }

        double mx_ent = calc_ent(N, prob);
        if (mx_ent < EPS) {
            ll c = naive_matcher(N, M, e, s, field, prob);
            for (ll i = 0; i < N; i++) {
                for (ll j = 0; j < N; j++) {
                    if (field[i][j] < 0) {
                        prob[i][j] *= init_prob[i][j];
                    }
                }
            }
            double m = calc_ent(N, prob);
            if (m < EPS) {
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

        ll ui = 0, uj = 0;
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

    prob_naive2(N, M, e, s, rnd);

    return 0;
}
