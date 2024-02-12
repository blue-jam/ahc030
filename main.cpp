#include <bits/stdc++.h>
#include <random>

static const double EPS = 1e-6;
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

    ll size() const {
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

void color_probability(const ll &N, const vector<vector<double>> &prob) {
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            double p = prob[i][j];
            ll v = (ll) (abs(0.5 - p) * 100) + 155;
            // output color in #c i j #00GGBB
            cout << "#c " << i << " " << j << " #00" << hex << v << v << dec << endl;
        }
    }
}

const double PP = 0.01;

void calc_init_prob(const ll &N, vector<vector<double>> &init_prob) {
    const ll SN = min(sqrt(N), 3.0);
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
                        init_prob[si + i][sj + j] = PP;
                    }
                }
            }
        }
    }
}

ll calc_remaining(const ll &M, vector<stamp> &s) {
    ll remaining= 0;
    for (ll i = 0; i < M; i++) {
        remaining += s[i].size();
    }
    return remaining;
}

void calc_prob_each(const ll &N, const vector<vector<ll>> &field, const vector<vector<double>> &init_prob, ll k,
                    vector<stamp> &s, vector<vector<vector<double>>> &prob_each) {
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

void calc_prob(const ll &N, const ll &M, const vector<vector<ll>> &field, const vector<vector<vector<double>>> &prob_each,
          vector<vector<double>> &prob) {
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
}

ll sense_high_ent_cell(const ll &N, const vector<vector<double>> &prob, vector<vector<ll>> &field) {
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

    return v;
}

void prob_naive(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<ll>> field(N, vector<ll>(N, -1));

    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    calc_init_prob(N, init_prob);
    ll remaining = calc_remaining(M, s);

    while (remaining > 0) {
        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            calc_prob_each(N, field, init_prob, k, s, prob_each);
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        calc_prob(N, M, field, prob_each, prob);

        color_probability(N, prob);

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

        ll v = sense_high_ent_cell(N, prob, field);
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

double calc_score(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<ll>> &field, const vector<vector<double>> &prob, const vector<P> &solution) {
    vector<vector<ll>> f2(N, vector<ll>(N, 0));
    for (ll k = 0; k < M; k++) {
        ll si = solution[k].i;
        ll sj = solution[k].j;
        for (ll l = 0; l < s[k].size(); l++) {
            ll i = s[k].ps[l].i;
            ll j = s[k].ps[l].j;
            f2[si + i][sj + j] += 1;
        }
    }
    double score = 0.0;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] >= 0) {
                score += abs(field[i][j] - f2[i][j]);
            }
        }
    }
    return score;
}

double update_score(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<ll>> &field, const vector<vector<double>> &prob, const vector<P> &solution, const vector<P> &prev_solution, const vector<vector<ll>> &prev_f2, const double prev_score) {
    map<pair<ll, ll>, ll> delta;

    for (ll k = 0; k < M; k++) {
        if (solution[k].i != prev_solution[k].i || solution[k].j != prev_solution[k].j) {
            ll pi = prev_solution[k].i;
            ll pj = prev_solution[k].j;
            ll ni = solution[k].i;
            ll nj = solution[k].j;
            for (ll l = 0; l < s[k].size(); l++) {
                ll i = s[k].ps[l].i;
                ll j = s[k].ps[l].j;
                delta[make_pair(pi + i, pj + j)] -= 1;
                delta[make_pair(ni + i, nj + j)] += 1;
            }
        }
    }

    double score = prev_score;

    for (auto p: delta) {
        ll i = p.first.first;
        ll j = p.first.second;
        ll v = p.second;

        if (field[i][j] < 0) {
            continue;
        }

        score -= abs(field[i][j] - prev_f2[i][j]);
        ll nv = prev_f2[i][j] + v;
        score += abs(field[i][j] - nv);
    }

    return score;
}

void print_solution(const ll &N, const ll &M, vector<stamp> &s, const vector<P> &solution) {
    for (ll k = 0; k < M; k++) {
        for (ll l = 0; l < s[k].size(); l++) {
            ll i = s[k].ps[l].i;
            ll j = s[k].ps[l].j;
            cout << "#c " << solution[k].i + i << " " << solution[k].j + j << " yellow" << endl;
        }
    }
}

void calc_field_status(const ll &N, const ll &M, const vector<stamp> &s, const vector<P> &solution, vector<vector<ll>> &f) {
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            f[i][j] = 0;
        }
    }
    for (ll k = 0; k < M; k++) {
        ll si = solution[k].i;
        ll sj = solution[k].j;
        for (ll l = 0; l < s[k].size(); l++) {
            ll i = s[k].ps[l].i;
            ll j = s[k].ps[l].j;
            f[si + i][sj + j] += 1;
        }
    }
}

bool try_hc_solution(const ll &N, const ll &M, const double &e, vector<stamp> &s, vector<vector<ll>> &field, vector<vector<double>> &prob, mt19937 &rnd) {
    vector<P> solution;

    for (ll k = 0; k < M; k++) {
        ll si = next_long(rnd, 0, N - s[k].h + 1);
        ll sj = next_long(rnd, 0, N - s[k].w + 1);
        solution.emplace_back(si, sj);
    }

    vector<vector<ll>> f2(N, vector<ll>(N, 0));
    calc_field_status(N, M, s, solution, f2);

    double currentScore = calc_score(N, M, s, field, prob, solution);
    bool updated;
    do {
        updated = false;
        // 1-opt
        for (ll k = 0; k < M; k++) {
            for (ll i = 0; i < N - s[k].h + 1; i++) {
                for (ll j = 0; j < N - s[k].w + 1; j++) {
                    vector<P> newSolution = solution;
                    newSolution[k] = P(i, j);
                    double newScore = update_score(N, M, s, field, prob, newSolution, solution, f2, currentScore);
                    if (newScore < currentScore) {
                        solution = newSolution;
                        currentScore = newScore;
                        calc_field_status(N, M, s, solution, f2);
                        updated = true;
                    }
                }
            }
        }
        // 2-opt
        for (ll k = 0; k < M; k++) {
            for (ll l = k + 1; l < M; l++) {
                for (ll t = 0; t < 50; t++) {
                    ll i = next_long(rnd, 0, N - s[k].h + 1);
                    ll j = next_long(rnd, 0, N - s[k].w + 1);
                    ll ni = next_long(rnd, 0, N - s[l].h + 1);
                    ll nj = next_long(rnd, 0, N - s[l].w + 1);
                    vector<P> newSolution = solution;
                    newSolution[k] = P(i, j);
                    newSolution[l] = P(ni, nj);
                    double newScore = update_score(N, M, s, field, prob, newSolution, solution, f2, currentScore);
                    if (newScore < currentScore) {
                        solution = newSolution;
                        currentScore = newScore;
                        calc_field_status(N, M, s, solution, f2);
                        updated = true;
                    }
                }
            }
        }
    } while(updated);

    print_solution(N, M, s, solution);

    if (currentScore > EPS) {
        return false;
    }

    calc_field_status(N, M, s, solution, f2);
    vector<P> result;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (f2[i][j] > 0) {
                result.emplace_back(i, j);
            }
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

    return res == 1;
}

void prob_hc(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<ll>> field(N, vector<ll>(N, -1));

    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    calc_init_prob(N, init_prob);
    ll remaining = calc_remaining(M, s);
    ll total = remaining;

    while (remaining > 0) {
        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            calc_prob_each(N, field, init_prob, k, s, prob_each);
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        calc_prob(N, M, field, prob_each, prob);

        color_probability(N, prob);

        double mx_ent = calc_ent(N, prob);
        double ratio = (double) remaining / total;
        if (mx_ent < 0.01 || ratio < 0.05) {
            bool ok = try_hc_solution(N, M, e, s, field, prob, rnd);
            if (ok) {
                return;
            }
        }

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

        ll v = sense_high_ent_cell(N, prob, field);
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
    ios_base::sync_with_stdio(false);

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

    prob_hc(N, M, e, s, rnd);

    return 0;
}
