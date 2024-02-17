#include <bits/stdc++.h>
#include <random>

static const double EPS = 1e-6;
using namespace std;

using ll = long long;

ll next_long(mt19937 &rnd, ll l, ll u) {
    uniform_int_distribution<ll> dist(l, u - 1);
    return dist(rnd);
}

// generate a random number in double range [l, r)
double next_double(mt19937 &rnd, double l, double r) {
    uniform_real_distribution<double> dist(l, r);
    return dist(rnd);
}

// generate a random number in range [0,1)
double next_prob(mt19937 &rnd) {
    return next_double(rnd, 0, 1);
}

struct P {
    short i, j;
    P(short i, short j) : i(i), j(j) {}

    bool operator<(const P &p) const {
        return i < p.i || (i == p.i && j < p.j);
    }
};

struct stamp {
    short h, w;
    vector<P> ps;

    stamp(short h, short w, const vector<P> &ps) : h(h), w(w), ps(ps) {}

    ll size() const {
        return ps.size();
    }
};

double prob0[] = {
                       1.0,
/*e=0.01, k=2, x=0, */ 0.999000999000999,
/*e=0.02, k=2, x=0, */ 0.9909365558912386,
/*e=0.03, k=2, x=0, */ 0.9886831275720165,
/*e=0.04, k=2, x=0, */ 0.9669421487603306,
/*e=0.05, k=2, x=0, */ 0.946875,
/*e=0.06, k=2, x=0, */ 0.9347593582887701,
/*e=0.07, k=2, x=0, */ 0.9035369774919614,
/*e=0.08, k=2, x=0, */ 0.8823529411764706,
/*e=0.09, k=2, x=0, */ 0.8824833702882483,
/*e=0.1,  k=2, x=0, */ 0.8819362455726092,
/*e=0.11, k=2, x=0, */ 0.8452380952380952,
/*e=0.12, k=2, x=0, */ 0.8325471698113207,
/*e=0.13, k=2, x=0, */ 0.8238153098420413,
/*e=0.14, k=2, x=0, */ 0.805952380952381,
/*e=0.15, k=2, x=0, */ 0.784841075794621,
/*e=0.16, k=2, x=0, */ 0.7632911392405063,
/*e=0.17, k=2, x=0, */ 0.7518427518427518,
/*e=0.18, k=2, x=0, */ 0.7525510204081632,
/*e=0.19, k=2, x=0, */ 0.7392900856793145,
/*e=0.2,  k=2, x=0, */ 0.7269129287598944
};

void calc_field_status(const ll &N, const ll &M, const vector<stamp> &s, const vector<P> &solution, vector<vector<short>> &f) {
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
        vector<vector<short>> &field,
        vector<vector<double>> &prob,
        ll k,
        vector<vector<short>> &f2) {
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

    const ll remaining = m - k;
    for (ll i = 0; i < n; i++) {
        for (ll j = 0; j < n; j++) {
            if (field[i][j] >= 0 && f2[i][j] + remaining < field[i][j]) {
                return 0;
            }
        }
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
        vector<vector<short>> &field,
        vector<vector<double>> &prob) {
    for (ll i = 0; i < n; i++) {
        for (ll j = 0; j < n; j++) {
            prob[i][j] = 0;
        }
    }
    vector<vector<short>> f2(n, vector<short>(n, 0));
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
            if (p < 0.5) {
                // output color in #c i j #00GGBB
                cout << "#c " << i << " " << j << " #00" << hex << v << v << dec << endl;
            } else {
                cout << "#c " << i << " " << j << " #" << hex << v << "00" << v << dec << endl;
            }
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
            double a = min(SN, N - si) * min(SN, N - sj);
            for (ll i = 0; si + i < min(si + SN, N); i++) {
                for (ll j = 0; sj + j < min(sj + SN, N); j++) {
                    init_prob[si + i][sj + j] = min((double)v / a + PP, 0.99);
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

void calc_prob_each(const ll &N, const vector<vector<short>> &field, const vector<vector<double>> &init_prob, ll k,
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

void calc_prob(const ll &N, const ll &M, const vector<vector<short>> &field, const vector<vector<vector<double>>> &prob_each,
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

P find_high_ent_cell(const ll &N, const vector<vector<double>> &prob, vector<vector<short>> &field) {
    ll gi = -1, gj = -1;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] < 0 && (gi < 0 || abs(prob[i][j] - 0.5) < abs(prob[gi][gj] - 0.5))) {
                gi = i;
                gj = j;
            }
        }
    }

    return P(gi, gj);
}

ll sense_high_ent_cell(const ll &N, const vector<vector<double>> &prob, vector<vector<short>> &field) {
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

ll sense_high_prod_cell(
        const ll &N,
        const vector<vector<double>> &prob,
        vector<vector<short>> &field,
        const vector<stamp> &s,
        const vector<vector<P>> solutions
        ) {
    vector<vector<double>> used_prob(N, vector<double>(N, 0));

    vector<vector<short>> f2(N, vector<short>(N, 0));
    for (ll k = 0; k < solutions.size(); k++) {
        calc_field_status(N, s.size(), s, solutions[k], f2);
        for (ll i = 0; i < N; i++) {
            for (ll j = 0; j < N; j++) {
                used_prob[i][j] += f2[i][j] > 0 ? 1 : 0;
            }
        }
    }
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            used_prob[i][j] /= solutions.size();
        }
    }
    ll gi = -1, gj = -1;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] < 0 && (gi < 0 || used_prob[i][j] > used_prob[gi][gj])) {
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
    vector<vector<short>> field(N, vector<short>(N, -1));

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

double calc_score(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<short>> &field, const vector<vector<double>> &prob, const vector<P> &solution) {
    vector<vector<short>> f2(N, vector<short>(N, 0));
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

double calc_diff_score(const double expect, const double actual) {
    const double diff = expect - actual;
    double result = diff * diff * 0.1;
    if (diff < 0) {
        result += -diff;
    } else {
        result += diff / 2.0;
    }
    return result;
}

double calc_cell_score(const ll e, const double v, const double p) {
    double result = 0.0;
    if (e >= 0) {
        result += calc_diff_score(e, v);
        result -= v * p;
    } else {
        result -= (p - 0.5) * v;
    }
    return result;
}

double calc_prob_score(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<short>> &field, const vector<vector<double>> &prob, const vector<P> &solution) {
    vector<vector<short>> f2(N, vector<short>(N, 0));
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
            score += calc_cell_score(field[i][j], f2[i][j], prob[i][j]);
        }
    }
    return score;
}

double calc_cell_score2(const ll e, const double v, const double p) {
    double result = 0.0;
    if (e >= 0) {
        result += calc_diff_score(e, v);
    } else {
        result -= (p - 0.5) * v;
    }
    return result;
}

double calc_prob_score2(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<short>> &field, const vector<vector<double>> &prob, const vector<P> &solution) {
    vector<vector<short>> f2(N, vector<short>(N, 0));
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
            score += calc_cell_score2(field[i][j], f2[i][j], prob[i][j]);
        }
    }
    return score;
}

double update_prob_score(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<short>> &field, const vector<vector<double>> &prob, const vector<P> &solution, const vector<P> &prev_solution, const vector<vector<short>> &prev_f2, const double prev_score) {
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

        ll nv = prev_f2[i][j] + v;
        score -= calc_cell_score(field[i][j], prev_f2[i][j], prob[i][j]);
        score += calc_cell_score(field[i][j], nv, prob[i][j]);
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

bool try_hc_solution(const ll &N, const ll &M, const double &e, vector<stamp> &s, vector<vector<short>> &field, vector<vector<double>> &prob, mt19937 &rnd) {
    vector<P> solution;

    for (ll k = 0; k < M; k++) {
        ll si = next_long(rnd, 0, N - s[k].h + 1);
        ll sj = next_long(rnd, 0, N - s[k].w + 1);
        solution.emplace_back(si, sj);
    }

    vector<vector<short>> f2(N, vector<short>(N, 0));
    calc_field_status(N, M, s, solution, f2);

    double currentScore = calc_prob_score(N, M, s, field, prob, solution);
    bool updated;
    do {
        updated = false;
        // 1-opt
        for (ll k = 0; k < M; k++) {
            for (ll i = 0; i < N - s[k].h + 1; i++) {
                for (ll j = 0; j < N - s[k].w + 1; j++) {
                    vector<P> newSolution = solution;
                    newSolution[k] = P(i, j);
                    double newScore = update_prob_score(N, M, s, field, prob, newSolution, solution, f2, currentScore);
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
                    double newScore = update_prob_score(N, M, s, field, prob, newSolution, solution, f2, currentScore);
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

    currentScore = calc_score(N, M, s, field, prob, solution);
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
    vector<vector<short>> field(N, vector<short>(N, -1));

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
        if (mx_ent < 0.01) {
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

void cont_hc(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<short>> field(N, vector<short>(N, -1));

    vector<P> solution;
    vector<vector<short>> f2(N, vector<short>(N, 0));
    for (ll k = 0; k < M; k++) {
        ll si = next_long(rnd, 0, N - s[k].h + 1);
        ll sj = next_long(rnd, 0, N - s[k].w + 1);
        solution.emplace_back(si, sj);
    }
    calc_field_status(N, M, s, solution, f2);

    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    calc_init_prob(N, init_prob);
    ll remaining = calc_remaining(M, s);
    ll total = remaining;

    ll updated_since = 1;
    while (remaining > 0) {
        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            calc_prob_each(N, field, init_prob, k, s, prob_each);
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        calc_prob(N, M, field, prob_each, prob);
        color_probability(N, prob);

        double currentScore = calc_prob_score(N, M, s, field, prob, solution);

        bool updated;
        do {
            updated = false;
            // 1-opt
            for (ll cnt = 0; cnt < 1000; cnt++) {
                ll k = next_long(rnd, 0, M);
                ll i = next_long(rnd, 0, N - s[k].h + 1);
                ll j = next_long(rnd, 0, N - s[k].w + 1);
                vector<P> newSolution = solution;
                newSolution[k] = P(i, j);
                double newScore = update_prob_score(N, M, s, field, prob, newSolution, solution, f2, currentScore);
                double diff = newScore - currentScore;
                double t = 0.1 / updated_since;
                double p;
                if (diff < 0) {
                    p = 1;
                } else if (updated_since <= 1) {
                    p = 0;
                } else {
                    p = exp(-diff / t);
                }
                if (next_long(rnd, 0, 100) < p * 100) {
                    solution = newSolution;
                    currentScore = newScore;
                    calc_field_status(N, M, s, solution, f2);
                    updated = true;

                    updated_since = 1;
                }
            }
            // 2-opt
            for (ll cnt = 0; cnt < 100; cnt++) {
                ll k, l;
                do {
                    k = next_long(rnd, 0, M);
                    l = next_long(rnd, 0, M);
                } while (k == l);
                ll i = next_long(rnd, 0, N - s[k].h + 1);
                ll j = next_long(rnd, 0, N - s[k].w + 1);
                ll ni = next_long(rnd, 0, N - s[l].h + 1);
                ll nj = next_long(rnd, 0, N - s[l].w + 1);
                vector<P> newSolution = solution;
                newSolution[k] = P(i, j);
                newSolution[l] = P(ni, nj);
                double newScore = update_prob_score(N, M, s, field, prob, newSolution, solution, f2, currentScore);
                double diff = newScore - currentScore;
                double t = 0.1 / updated_since;
                double p;
                if (diff < 0) {
                    p = 1;
                } else if (updated_since <= 1) {
                    p = 0;
                } else {
                    p = exp(-diff / t);
                }
                if (next_long(rnd, 0, 100) < p * 100) {
                    solution = newSolution;
                    currentScore = newScore;
                    calc_field_status(N, M, s, solution, f2);
                    updated = true;

                    updated_since = 1;
                }
            }
        } while (updated);

        print_solution(N, M, s, solution);

        double mx_ent = calc_ent(N, prob);
        double ratio = (double) remaining / total;

        if (mx_ent < 0.01 || ratio < 0.3) {
            double score = calc_score(N, M, s, field, prob, solution);
            if (score < EPS) {
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
                if (res == 1) {
                    return;
                }
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

        updated_since++;
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

const ll MAX_BEAM = 20;
const ll MAX_DEPTH = 1;

void cont_beam(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<short>> field(N, vector<short>(N, -1));
    vector<vector<short>> f2(N, vector<short>(N, 0));

    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    calc_init_prob(N, init_prob);
    ll remaining = calc_remaining(M, s);
    ll total = remaining;

    map<vector<P>, double> solutions;
    priority_queue<pair<double, vector<P>>> Q;
    for (ll l = 0; l < MAX_BEAM; l++) {
        vector<P> solution;
        for (ll k = 0; k < M; k++) {
            ll si = next_long(rnd, 0, N - s[k].h + 1);
            ll sj = next_long(rnd, 0, N - s[k].w + 1);
            solution.emplace_back(si, sj);
        }
        double score = calc_prob_score(N, M, s, field, init_prob, solution);
        solutions[solution] = score;
        Q.push(make_pair(score, solution));
    }

    set<vector<P>> tried;

    while (remaining > 0) {
        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            calc_prob_each(N, field, init_prob, k, s, prob_each);
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        calc_prob(N, M, field, prob_each, prob);
        color_probability(N, prob);

        double current_best_score = calc_prob_score(N, M, s, field, prob, Q.top().second);
        double prev_best_score = current_best_score;
        vector<P> current_best_solution = Q.top().second;
        for (ll d = 0; d < MAX_DEPTH; d++) {
            map<vector<P>, double> next_best_solutions;
            priority_queue<pair<double, vector<P>>> nextQ;
            while (!Q.empty()) {
                auto p = Q.top();
                Q.pop();
                vector<P> solution = p.second;
                solutions.erase(solution);

                calc_field_status(N, M, s, solution, f2);
                double current_score = calc_prob_score(N, M, s, field, prob, solution);
                prev_best_score = min(prev_best_score, current_score);

                // 1-opt
                for (ll cnt = 0; cnt < 100; cnt++) {
                    ll k = next_long(rnd, 0, M);
                    ll i = next_long(rnd, 0, N - s[k].h + 1);
                    ll j = next_long(rnd, 0, N - s[k].w + 1);
                    vector<P> newSolution = solution;
                    newSolution[k] = P(i, j);

                    if (next_best_solutions.find(newSolution) != next_best_solutions.end()) {
                        continue;
                    }

                    double score = update_prob_score(N, M, s, field, prob, newSolution, solution, f2, current_score);
                    if (next_best_solutions.size() > MAX_BEAM) {
                        auto p = nextQ.top();
                        if (p.first < score) {
                            continue;
                        }
                        nextQ.pop();
                        next_best_solutions.erase(p.second);
                    }

                    next_best_solutions[newSolution] = score;
                    nextQ.push(make_pair(score, newSolution));

                    if (score < current_best_score) {
                        current_best_score = score;
                        current_best_solution = newSolution;
                    }
                }

                // 2-opt
                for (ll cnt = 0; cnt < 20; cnt++) {
                    ll k, l;
                    do {
                        k = next_long(rnd, 0, M);
                        l = next_long(rnd, 0, M);
                    } while (k == l);
                    ll i = next_long(rnd, 0, N - s[k].h + 1);
                    ll j = next_long(rnd, 0, N - s[k].w + 1);
                    ll ni = next_long(rnd, 0, N - s[l].h + 1);
                    ll nj = next_long(rnd, 0, N - s[l].w + 1);
                    vector<P> newSolution = solution;
                    newSolution[k] = P(i, j);
                    newSolution[l] = P(ni, nj);

                    if (next_best_solutions.find(newSolution) != next_best_solutions.end()) {
                        continue;
                    }

                    double score = update_prob_score(N, M, s, field, prob, newSolution, solution, f2, current_score);
                    if (next_best_solutions.size() > MAX_BEAM) {
                        auto p = nextQ.top();
                        if (p.first < score) {
                            continue;
                        }
                        nextQ.pop();
                        next_best_solutions.erase(p.second);
                    }

                    next_best_solutions[newSolution] = score;
                    nextQ.push(make_pair(score, newSolution));

                    if (score < current_best_score) {
                        current_best_score = score;
                        current_best_solution = newSolution;
                    }
                }

                // 3-opt
                if (M >= 3) {
                    for (ll cnt = 0; cnt < 10; cnt++) {
                        ll k, l, m;
                        do {
                            k = next_long(rnd, 0, M);
                            l = next_long(rnd, 0, M);
                            m = next_long(rnd, 0, M);
                        } while (k == l || l == m || m == k);
                        ll i = next_long(rnd, 0, N - s[k].h + 1);
                        ll j = next_long(rnd, 0, N - s[k].w + 1);
                        ll ni = next_long(rnd, 0, N - s[l].h + 1);
                        ll nj = next_long(rnd, 0, N - s[l].w + 1);
                        ll oi = next_long(rnd, 0, N - s[m].h + 1);
                        ll oj = next_long(rnd, 0, N - s[m].w + 1);
                        vector<P> newSolution = solution;
                        newSolution[k] = P(i, j);
                        newSolution[l] = P(ni, nj);
                        newSolution[m] = P(oi, oj);

                        if (next_best_solutions.find(newSolution) != next_best_solutions.end()) {
                            continue;
                        }

                        double score = update_prob_score(N, M, s, field, prob, newSolution, solution, f2,
                                                         current_score);
                        if (next_best_solutions.size() > MAX_BEAM) {
                            auto p = nextQ.top();
                            if (p.first < score) {
                                continue;
                            }
                            nextQ.pop();
                            next_best_solutions.erase(p.second);
                        }

                        next_best_solutions[newSolution] = score;
                        nextQ.push(make_pair(score, newSolution));

                        if (score < current_best_score) {
                            current_best_score = score;
                            current_best_solution = newSolution;
                        }
                    }
                }
            }
            Q = nextQ;
            solutions = next_best_solutions;
        }

        print_solution(N, M, s, current_best_solution);

        double mx_ent = calc_ent(N, prob);
        double ratio = (double) remaining / total;

        if (mx_ent < 0.05 || ratio < 0.4 || current_best_score - prev_best_score < EPS && M <= 5) {
            if (tried.find(current_best_solution) == tried.end()) {
                double score = calc_score(N, M, s, field, prob, current_best_solution);
                if (score < EPS) {
                    calc_field_status(N, M, s, current_best_solution, f2);
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
                    if (res == 1) {
                        return;
                    } else {
                        tried.insert(current_best_solution);
                    }
                }
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

        vector<vector<P>> solution_list;
        for (auto p: solutions) {
            solution_list.push_back(p.first);
        }
        ll v;
        if (remaining >= total * 0.1) {
            v = sense_high_ent_cell(N, prob, field);
        } else {
            vector<vector<P>> solution_list;
            for (auto p: solutions) {
                solution_list.push_back(p.first);
            }
            v = sense_high_prod_cell(N, prob, field, s, solution_list);
        }
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

const ll GEN = 20;
const ll MAX_N = 200;
const double REMAIN_PROB = 0.2;
const double MUTATION_PROB = 0.1;
const double NORMAL_PROB = 1 - REMAIN_PROB - MUTATION_PROB;

ll roulette(const vector<double> &p, mt19937 &rnd) {
    ll k = 0;
    double pp = next_prob(rnd);
    double sum = 0.0;
    for (ll m = 0; m < MAX_N; m++) {
        sum += p[m];
        if (sum >= pp) {
            k = m;
            break;
        }
    }
    return k;
}

void ga(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<short>> field(N, vector<short>(N, -1));
    vector<vector<short>> f2(N, vector<short>(N, 0));

    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    calc_init_prob(N, init_prob);
    ll remaining = calc_remaining(M, s);
    ll total = remaining;

    vector<vector<P>> solutions;
    for (ll l = 0; l < MAX_N; l++) {
        vector<P> solution;
        for (ll k = 0; k < M; k++) {
            ll si = next_long(rnd, 0, N - s[k].h + 1);
            ll sj = next_long(rnd, 0, N - s[k].w + 1);
            solution.emplace_back(si, sj);
        }
        solutions.push_back(solution);
    }

    set<vector<P>> tried;

    while (remaining > 0) {
        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            calc_prob_each(N, field, init_prob, k, s, prob_each);
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        calc_prob(N, M, field, prob_each, prob);
        color_probability(N, prob);

        double current_best_score = calc_prob_score(N, M, s, field, prob, solutions[0]);
        double prev_best_score = current_best_score;
        for (ll l = 0; l < MAX_N; l++) {
            current_best_score = min(current_best_score, calc_prob_score2(N, M, s, field, prob, solutions[l]));
        }
        vector<P> current_best_solution = solutions[0];

        for (ll g = 0; g < GEN; g++) {
            vector<double> scores;
            double mx_score = -1e18;
            for (ll l = 0; l < MAX_N; l++) {
                scores.push_back(calc_prob_score2(N, M, s, field, prob, solutions[l]));
                mx_score = max(mx_score, scores[l]);
            }
            vector<double> p(MAX_N, 0.0);
            double sum = 0.0;
            for (ll l = 0; l < MAX_N; l++) {
                p[l] = - (scores[l] - mx_score) + 1.0;
                sum += p[l];
            }
            for (ll l = 0; l < MAX_N; l++) {
                p[l] /= sum;
            }

            vector<vector<P>> next_solutions;
            for (ll l = 0; l < MAX_N; l++) {
                vector<P> newSolution;
                double op = next_prob(rnd);
                if (op < REMAIN_PROB) {
                    ll k = roulette(p, rnd);
                    newSolution = solutions[k];
                } else if (op < REMAIN_PROB + MUTATION_PROB) {
                    ll k = roulette(p, rnd);
                    ll i = next_long(rnd, 0, M);
                    ll si = next_long(rnd, 0, N - s[i].h + 1);
                    ll sj = next_long(rnd, 0, N - s[i].w + 1);

                    newSolution = solutions[k];
                    newSolution[i] = P(si, sj);
                } else {
                    ll k = roulette(p, rnd);
                    ll l = roulette(p, rnd);
                    ll i = next_long(rnd, 1, M);
                    newSolution = solutions[k];
                    for (ll j = i; j < M; j++) {
                        newSolution[j] = solutions[l][j];
                    }
                }
                next_solutions.push_back(newSolution);

                double score = calc_prob_score2(N, M, s, field, prob, newSolution);
                if (score < current_best_score) {
                    current_best_score = score;
                    current_best_solution = newSolution;
                }
            }

            solutions = next_solutions;
        }

        print_solution(N, M, s, current_best_solution);

        double mx_ent = calc_ent(N, prob);
        double ratio = (double) remaining / total;

        if (mx_ent < 0.05 || ratio < 0.4 || current_best_score - prev_best_score < EPS && M <= 5) {
            if (tried.find(current_best_solution) == tried.end()) {
                double score = calc_score(N, M, s, field, prob, current_best_solution);
                if (score < EPS) {
                    calc_field_status(N, M, s, current_best_solution, f2);
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
                    if (res == 1) {
                        return;
                    } else {
                        tried.insert(current_best_solution);
                    }
                }
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

        ll v;
        if (remaining >= total * 0.1) {
            v = sense_high_ent_cell(N, prob, field);
        } else {
            v = sense_high_prod_cell(N, prob, field, s, solutions);
        }
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

    cont_beam(N, M, e, s, rnd);

    return 0;
}
