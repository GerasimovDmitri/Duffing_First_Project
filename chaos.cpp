#include <cmath>
#include <cstdio>

const double DELTA = 0.3;
const double ALPHA = -1.0;
const double BETA = 1.0;
const double GAMMA = 0.5;
const double OMEGA = 1.2;

const double DT = 0.001;
const int    N_STEPS = 200000;

const double X0 = 1.0;
const double V0 = 0.0;
const double D0 = 1e-8;

struct State {
    double x, v;

    State(double x = 0, double v = 0) : x(x), v(v) {}
};

State f(const State& s, double t) {
    double dx = s.v;
    double dv = -DELTA * s.v
                - ALPHA * s.x
                - BETA * s.x * s.x * s.x
                + GAMMA * std::cos(OMEGA * t);

    return State(dx, dv);
}

State rk4_step(const State& s, double t, double dt) {
    State k1 = f(s, t);
    State k2 = f(State(s.x + 0.5 * dt * k1.x, s.v + 0.5 * dt * k1.v), t + 0.5 * dt);
    State k3 = f(State(s.x + 0.5 * dt * k2.x, s.v + 0.5 * dt * k2.v), t + 0.5 * dt);
    State k4 = f(State(s.x + dt * k3.x, s.v + dt * k3.v), t + dt);

    State next;
    next.x = s.x + (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
    next.v = s.v + (dt / 6.0) * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);

    return next;
}

double benettin(const State& s0, double d0, int n_steps) {
    State s = s0;

    State delta(1.0, 0.0);

    State sp(s.x + d0 * delta.x, s.v + d0 * delta.v);

    double sum = 0.0;
    double t = 0.0;

    for (int i = 1; i <= n_steps; ++i) {
        s  = rk4_step(s,  t, DT);
        sp = rk4_step(sp, t, DT);

        double dx = sp.x - s.x;
        double dv = sp.v - s.v;

        double d = std::sqrt(dx * dx + dv * dv);

        if (d == 0.0) {
            dx = delta.x * d0;
            dv = delta.v * d0;
            d  = d0;
        }

        sum += std::log(d / d0);

        sp.x = s.x + (dx / d) * d0;
        sp.v = s.v + (dv / d) * d0;

        t += DT;
    }

    return sum / (n_steps * DT);
}

int main() {
    State s0(X0, V0);

    double lambda = benettin(s0, D0, N_STEPS);

    std::printf("%.6f\n", lambda);

    return 0;
}
