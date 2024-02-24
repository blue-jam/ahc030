#include <bits/stdc++.h>
#include <random>

static const double EPS = 1e-6;
using namespace std;

using ll = long long;

int di[] = {-1, 0, 1, 0};
int dj[] = {0, 1, 0, -1};

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
    vector<vector<bool>> is_edge;

    stamp(short h, short w, const vector<P> &ps) : h(h), w(w), ps(ps), is_edge(ps.size(), vector<bool>(4)) {
        vector<vector<bool>> used(h, vector<bool>(w, false));
        for (ll i = 0; i < ps.size(); i++) {
            used[ps[i].i][ps[i].j] = true;
        }

        for (ll i = 0; i < ps.size(); i++) {
            ll ci = ps[i].i;
            ll cj = ps[i].j;
            for (ll d = 0; d < 4; d++) {
                ll ni = ci + di[d];
                ll nj = cj + dj[d];
                if (ni < 0 || ni >= h || nj < 0 || nj >= w) {
                    is_edge[i][d] = true;
                } else {
                    is_edge[i][d] = !used[ni][nj];
                }
            }
        }
    }

    ll size() const {
        return ps.size();
    }
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

ll simulate_sense(
        const ll &k,
        const ll &v,
        const double &e,
        mt19937 &rnd) {
    double mu = (k - v) * e + v * (1 - e);
    double sigma = sqrt(2 * e * (1 - e));
    normal_distribution<double> dist(mu, sigma);

    ll res = (ll)round(dist(rnd));
    return res < 0 ? 0 : res;
}

vector<vector<double>> init_k_prob(
        const ll &M,
        const ll &k,
        const double &e,
        mt19937 &rnd) {
    vector<vector<double>> res(M + 1, vector<double>(3 * M, 0));
    for (ll v = 0; v <= M; v++) {
        for (ll t = 0; t < 1000; t++) {
            ll x = simulate_sense(k, v, e, rnd);
            res[v][x] += 1;
        }
    }

    for (ll x = 0; x < 3 * M; x++) {
        double sum = 0;
        for (ll v = 0; v <= M; v++) {
            sum += res[v][x];
        }
        if (sum <= EPS) continue;
        for (ll v = 0; v <= M; v++) {
            res[v][x] /= sum;
        }
    }

    return res;
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

void calc_init_prob(const ll &N, vector<vector<double>> &init_prob, vector<vector<short>> &field) {
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
            if (v / a < 1.5) {
                for (ll i = 0; si + i < min(si + SN, N); i++) {
                    for (ll j = 0; sj + j < min(sj + SN, N); j++) {
                        init_prob[si + i][sj + j] = min((double) v / a + PP, 0.99);
                    }
                }
            } else {
                for (ll i = si; i < min(si + SN, N); i++) {
                    for (ll j = sj; j < min(sj + SN, N); j++) {
                        cout << "q 1 " << i << " " << j << endl;
                        flush(cout);
                        ll v;
                        cin >> v;
                        field[i][j] = v;
                        if (field[i][j] == 0) {
                            init_prob[i][j] = 0.0;
                        } else {
                            init_prob[i][j] = 1.0;
                        }
                    }
                }
            }
        }
    }
}


void update_init_prob(const ll &N, const vector<vector<short>> &field, vector<vector<double>> &init_prob) {
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] == 0) {
                init_prob[i][j] = 0;
            } else if (field[i][j] > 0) {
                init_prob[i][j] = 1;
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
                    vector<stamp> &s, vector<vector<vector<double>>> &prob_each, const ll &remaining) {
    ll cnt = 0;
    for (ll si = 0; si + s[k].h <= N; si++) {
        for (ll sj = 0; sj + s[k].w <= N; sj++) {
            bool ok = true;
            ll newCells = 0;
            for (ll l = 0; l < s[k].size(); l++) {
                ll i = s[k].ps[l].i;
                ll j = s[k].ps[l].j;
                if (field[si + i][sj + j] == 0) {
                    ok = false;
                }
                if (field[si + i][sj + j] < 0) {
                    newCells += 1;
                }
            }
            if (ok && newCells <= remaining) {
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

ll sense_high_used_cell(
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

ll sense_adjacent_cell(
        const ll &N,
        const vector<vector<double>> &prob,
        vector<vector<short>> &field,
        const vector<stamp> &s,
        const vector<vector<P>> solutions
) {
    vector<vector<short>> adj(N, vector<short>(N, 0));
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] <= 0) continue;
            for (ll d = 0; d < 4; d++) {
                ll ni = i + di[d];
                ll nj = j + dj[d];
                if (ni < 0 || ni >= N || nj < 0 || nj >= N) continue;
                adj[ni][nj] += 1;
            }
        }
    }

    ll gi = -1, gj = -1;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] < 0 && (gi < 0 || adj[i][j] > adj[gi][gj])) {
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

ll sense_high_prob_cell(
        const ll &N,
        const vector<vector<double>> &prob,
        vector<vector<short>> &field
) {
    ll gi = -1, gj = -1;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] < 0 && (gi < 0 || prob[i][j] > prob[gi][gj])) {
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
    if (expect == 0 && actual > 0) {
        result += 10;
    } else if (diff < 0){
        result += abs(diff);
    } else {
        result += abs(diff) * 1.5;
    }
    return result;
}

double calc_cell_score(const ll e, const double v, const double p) {
    double result = 0.0;
    if (e >= 0) {
        result += calc_diff_score(e, v);
    }
    result -= v * p;
    return result;
}

double calc_prob_score(const ll &N, const ll &M, vector<stamp> &s, const vector<vector<short>> &field, const vector<vector<double>> &prob, const vector<P> &solution, const ll &remaining) {
    vector<vector<short>> f2(N, vector<short>(N, 0));
    double score = 0.0;
    for (ll k = 0; k < M; k++) {
        ll si = solution[k].i;
        ll sj = solution[k].j;
        for (ll l = 0; l < s[k].size(); l++) {
            ll i = s[k].ps[l].i;
            ll j = s[k].ps[l].j;
            f2[si + i][sj + j] += 1;
        }
    }
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            score += calc_cell_score(field[i][j], f2[i][j], prob[i][j]);
        }
    }
    ll newCells = 0;
    for (ll k = 0; k < M; k++) {
        ll si = solution[k].i;
        ll sj = solution[k].j;
        for (ll l = 0; l < s[k].size(); l++) {
            ll i = si + s[k].ps[l].i;
            ll j = sj + s[k].ps[l].j;
            if (field[i][j] < 0) {
                newCells += 1;
            }
            if (field[i][j] == 0) continue;
            for (ll d = 0; d < 4; d++) {
                if (!s[k].is_edge[l][d]) {
                    continue;
                }
                ll ni = i + di[d];
                ll nj = j + dj[d];
                if (ni < 0 || ni >= N || nj < 0 || nj >= N) {
                    score -= 0.5;
                    continue;
                }
                if (field[ni][nj] == 0) {
                    score -= 0.5;
                }
                if (field[i][j] > 0 && field[ni][nj] > 0 && field[i][j] != field[ni][nj]) {
                    score -= 0.5;
                }
            }
        }
    }
    if (newCells > remaining) {
        score += 100;
    }
    return score;
}

void print_solution(const ll &N, const ll &M, const vector<stamp> &s, const vector<P> &solution) {
    for (ll k = 0; k < M; k++) {
        for (ll l = 0; l < s[k].size(); l++) {
            ll i = s[k].ps[l].i;
            ll j = s[k].ps[l].j;
            cout << "#c " << solution[k].i + i << " " << solution[k].j + j << " yellow" << endl;
        }
    }
}

const ll MAX_BEAM = 20;
const ll MAX_DEPTH = 1;

void cont_beam(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    vector<vector<short>> field(N, vector<short>(N, -1));
    vector<vector<short>> f2(N, vector<short>(N, 0));

    vector<vector<double>> init_prob(N, vector<double>(N, 1.0));

    calc_init_prob(N, init_prob, field);
    ll remaining = calc_remaining(M, s);
    ll total = remaining;

    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (field[i][j] > 0) {
                remaining -= field[i][j];
            }
        }
    }

    map<vector<P>, double> solutions;
    priority_queue<pair<double, vector<P>>> Q;
    for (ll l = 0; l < MAX_BEAM; l++) {
        vector<P> solution;
        for (ll k = 0; k < M; k++) {
            ll si = next_long(rnd, 0, N - s[k].h + 1);
            ll sj = next_long(rnd, 0, N - s[k].w + 1);
            solution.emplace_back(si, sj);
        }
        double score = calc_prob_score(N, M, s, field, init_prob, solution, remaining);
        solutions[solution] = score;
        Q.push(make_pair(score, solution));
    }

    set<vector<P>> tried;

    while (remaining > 0) {
        update_init_prob(N, field, init_prob);

        vector<vector<vector<double>>> prob_each(M, vector<vector<double>>(N, vector<double>(N, 0)));

        for (ll k = 0; k < M; k++) {
            calc_prob_each(N, field, init_prob, k, s, prob_each, remaining);
        }

        vector<vector<double>> prob(N, vector<double>(N, 0));
        calc_prob(N, M, field, prob_each, prob);
        color_probability(N, prob);

        double current_best_score = calc_prob_score(N, M, s, field, prob, Q.top().second, remaining);
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
                double current_score = calc_prob_score(N, M, s, field, prob, solution, remaining);
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

                    double score = calc_prob_score(N, M, s, field, prob, newSolution, remaining);
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

                    double score = calc_prob_score(N, M, s, field, prob, newSolution, remaining);
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

                        double score = calc_prob_score(N, M, s, field, prob, newSolution, remaining);
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
        if (M < 18) {
            if (remaining >= total * 0.1) {
                v = sense_high_prob_cell(N, prob, field);
            } else {
                vector<vector<P>> solution_list;
                for (auto p: solutions) {
                    solution_list.push_back(p.first);
                }
                v = sense_high_used_cell(N, prob, field, s, solution_list);
            }
        } else {
            if (remaining >= total * 0.3) {
                v = sense_high_prob_cell(N, prob, field);
            } else {
                vector<vector<P>> solution_list;
                for (auto p: solutions) {
                    solution_list.push_back(p.first);
                }
                v = sense_adjacent_cell(N, prob, field, s, solution_list);
            }
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

void dig_prob(const ll &N, const ll &M, const double &e, vector<stamp> &s, mt19937 &rnd) {
    const ll SN = min(sqrt(N), 3.0);
    for (ll si = 0; si < N; si += SN) {
        for (ll sj = 0; sj < N; sj += SN) {
            const ll h = min(SN, N - si);
            const ll w = min(SN, N - sj);
            const ll a = h * w;
            vector<P> ps;
            for (ll i = 0; i < h; i++) {
                for (ll j = 0; j < w; j++) {
                    ps.emplace_back(si + i, sj + j);
                }
            }
            cout << "q " << ps.size();
            for (auto p: ps) {
                cout << " " << p.i << " " << p.j;
            }
        }
    }
}

// ガウスの誤差関数の近似式
double error_function(double x) {
    double t = 1.0 / (1.0 + 0.5 * std::abs(x));
    double ans = 1 - t * std::exp(-x * x - 1.26551223 +
                                  t * (1.00002368 +
                                       t * (0.37409196 +
                                            t * (0.09678418 +
                                                 t * (-0.18628806 +
                                                      t * (0.27886807 +
                                                           t * (-1.13520398 +
                                                                t * (1.48851587 +
                                                                     t * (-0.82215223 +
                                                                          t * (0.17087277))))))))));
    return x >= 0 ? ans : -ans;
}

// 正規分布の累積密度関数 (CDF) の近似式
double normal_cdf(double x, double mean, double stddev) {
    return 0.5 * (1 + error_function((x - mean) / (stddev * std::sqrt(2))));
}

double check_prob(const ll &v, const ll &k, const double &e, const ll &x) {
    double mean = (k - v) * e + (1 - e) * v;
    double stddev = std::sqrt(k * e * (1 - e));
    if (x < EPS) {
        return normal_cdf(x + 0.5, mean, stddev);
    } else {
        return normal_cdf(x + 0.5, mean, stddev) - normal_cdf(x - 0.5, mean, stddev);
    }
}

vector<pair<vector<P>, double>> build_candidates(const ll &N, const ll &M, const double &e, const vector<stamp> &s, const vector<pair<vector<P>, ll>> &queries) {
    vector<pair<vector<P>, double>> candidates;

    double total = 0.0;
    for (ll si = 0; si <= N - s[0].h; si++) {
        for (ll sj = 0; sj <= N - s[0].w; sj++) {
            for (ll ti = 0; ti <= N - s[1].h; ti++) {
                for (ll tj = 0; tj < N - s[1].w; tj++) {
                    vector<P> solution;
                    solution.emplace_back(si, sj);
                    solution.emplace_back(ti, tj);
                    vector<vector<short>> f2(N, vector<short>(N, 0));
                    calc_field_status(N, M, s, solution, f2);

                    double score = 0.0;
                    for (ll l = 0; l < queries.size(); l++) {
                        ll v = 0;
                        for (auto p: queries[l].first) {
                            v += f2[p.i][p.j];
                        }
                        ll k = queries[l].first.size();
                        ll x = queries[l].second;
                        double p = check_prob(v, k, e, x);

                        score += log2(p);
                    }

                    score = pow(2.0, score);
                    candidates.emplace_back(solution, score);
                    total += score;
                }
            }
        }
    }

    for (auto &c: candidates) {
        c.second /= total;
    }

    return candidates;
}

double calc_mutual_information_cost(const ll &N, const double &e, const vector<stamp> &s, const vector<vector<short>> &q, const vector<pair<vector<P>, double>> &candidates) {
    double mutual_information = 0.0;
    ll cnt = 0;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (q[i][j] == 1) {
                cnt += 1;
            }
        }
    }

    ll cells = 0;
    for (ll k = 0; k < s.size(); k++) {
        cells += s[k].size();
    }

    vector<double> py(cells * 2, 0);
    for (auto c: candidates) {
        double px = c.second;
        if (px < EPS) continue;
        vector<vector<short>> f2(N, vector<short>(N, 0));
        calc_field_status(N, 2, s, c.first, f2);
        ll v = 0;
        for (ll i = 0; i < N; i++) {
            for (ll j = 0; j < N; j++) {
                if (q[i][j] == 1) {
                    v += f2[i][j];
                }
            }
        }
        for (ll y = 0; y < py.size(); y++) {
            double pxy = px * check_prob(v, cnt, e, y);
            py[y] += pxy;
        }
    }

    for (auto c: candidates) {
        double px = c.second;
        if (px < EPS) continue;
        vector<vector<short>> f2(N, vector<short>(N, 0));
        calc_field_status(N, 2, s, c.first, f2);
        ll v = 0;
        for (ll i = 0; i < N; i++) {
            for (ll j = 0; j < N; j++) {
                if (q[i][j] == 1) {
                    v += f2[i][j];
                }
            }
        }
        for (ll y = 0; y < py.size(); y++) {
            double pxy = px * check_prob(v, cnt, e, y);
            if (pxy > EPS) {
                mutual_information += pxy * log2(pxy / px / py[y]);
            }
        }
    }
    return mutual_information * std::sqrt(cnt);
}

vector<P> calc_query_cells(const ll &N, const double &e, const vector<stamp> &s, const vector<pair<vector<P>, ll>> &queries, mt19937 &rnd) {
    vector<vector<short>> q(N, vector<short>(N, 0));

    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            q[i][j] = next_long(rnd, 0, 2);
        }
    }

    auto candidates = build_candidates(N, 2, e, s, queries);
    cerr << "candidates: " << candidates.size() << endl;

    bool updated;
    do {
        updated = false;
        double cost = calc_mutual_information_cost(N, e, s, q, candidates);

        for (ll i = 0; i < N; i++) {
            for (ll j = 0; j < N; j++) {
                ll old = q[i][j];
                q[i][j] = 1 - q[i][j];
                double new_cost = calc_mutual_information_cost(N, e, s, q, candidates);
                if (new_cost > cost) {
                    cost = new_cost;
                    updated = true;
                } else {
                    q[i][j] = old;
                }
            }
        }
        cerr << "cost: " << cost << endl;
    } while (updated);

    vector<P> res;
    for (ll i = 0; i < N; i++) {
        for (ll j = 0; j < N; j++) {
            if (q[i][j] == 1) {
                res.emplace_back(i, j);
            }
        }
    }

    return res;
}

pair<vector<P>, double> infer(const ll &N, const ll &M, const double &e, const vector<stamp> &s, const vector<pair<vector<P>, ll>> &queries) {
    vector<pair<vector<P>, double>> candidates = build_candidates(N, M, e, s, queries);

    double best_score = 0.0;
    vector<P> best_solution;
    for (auto c: candidates) {
        if (c.second > best_score) {
            best_score = c.second;
            best_solution = c.first;
        }
    }

    return make_pair(best_solution, best_score);
}

void bayes(const ll &N, const ll &M, const double &e, const vector<stamp> &s, mt19937 &rnd) {
    vector<pair<vector<P>, ll>> queries;
    for (;;) {
        vector<P> cells = calc_query_cells(N, e, s, queries, rnd);

        cout << "q " << cells.size();
        for (auto p: cells) {
            cout << " " << p.i << " " << p.j;
        }
        cout << endl;
        flush(cout);
        ll v;
        cin >> v;

        queries.emplace_back(cells, v);

        pair<vector<P>, double> best = infer(N, M, e, s, queries);
        vector<P> solution = best.first;
        cout << "# " << best.second << endl;
        if (best.second >= 0.98) {
            vector<vector<short>> f2(N, vector<short>(N, 0));
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
            for (auto p: result) {
                cout << " " << p.i << " " << p.j;
            }
            cout << endl;
            flush(cout);
            cin >> v;
            if (v == 1) {
                return;
            }
        }
    }
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

    if (M == 2) {
        bayes(N, M, e, s, rnd);
    } else {
        cont_beam(N, M, e, s, rnd);
    }

    return 0;
}
