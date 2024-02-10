#include <bits/stdc++.h>

using namespace std;

using ll = long long;

struct P {
    ll i, j;
    P(ll i, ll j) : i(i), j(j) {}
};

void naive_solver(ll N, ll M, double e, vector<vector<P>> p) {
    ll sum = 0;
    for (ll i = 0; i < M; i++) {
        sum += p[i].size();
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
    ll N, M;
    double e;
    cin >> N >> M >> e;
    vector<vector<P>> p(M);
    for (ll i = 0; i < M; i++) {
        ll d;
        cin >> d;
        for (ll j = 0; j < d; j++) {
            ll a, b;
            cin >> a >> b;
            p[i].emplace_back(a, b);
        }
    }
    naive_solver(N, M, e, p);

    return 0;
}
