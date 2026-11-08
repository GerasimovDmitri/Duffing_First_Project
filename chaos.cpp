#include <cmath>
#include <cstdio>

const double DELTA = 0.3;
const double ALPHA = -1.0;
const double BETA  =  1.0;
const double OMEGA =  1.2;

const double DT = 0.001;

const int N_TRANSIENT = 100000;
const int N_STEPS = 1000000;

const double X0 = 1.0;
const double V0 = 0.0;
const double D0 = 1e-8;

struct State {
    double x, v;
    State(double x = 0, double v = 0) : x(x), v(v) {}
};

State f(const State& s, double t, double gamma) {
    double dx = s.v;
    double dv = -DELTA * s.v - ALPHA * s.x - BETA * s.x * s.x * s.x + gamma * std::cos(OMEGA * t);
    return State(dx, dv);
}

State rk4_step(const State& s, double t, double dt, double gamma) {
    State k1 = f(s, t, gamma);
    State k2 = f(State(s.x + 0.5 * dt * k1.x, s.v + 0.5 * dt * k1.v), t + 0.5 * dt, gamma);
    State k3 = f(State(s.x + 0.5 * dt * k2.x, s.v + 0.5 * dt * k2.v), t + 0.5 * dt, gamma);
    State k4 = f(State(s.x + dt * k3.x, s.v + dt * k3.v), t + dt, gamma);

    State next;
    next.x = s.x + (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
    next.v = s.v + (dt / 6.0) * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
    return next;
}

double benettin(const State& s0, double d0, double gamma, int n_transient, int n_steps) {
    State s = s0;
    State delta(1.0, 0.0);
    State sp(s.x + d0 * delta.x, s.v + d0 * delta.v);

    double sum = 0.0;
    double t = 0.0;
    const int total_steps = n_transient + n_steps;

    for (int i = 1; i <= total_steps; ++i) {
        s  = rk4_step(s,  t, DT, gamma);
        sp = rk4_step(sp, t, DT, gamma);

        double dx = sp.x - s.x;
        double dv = sp.v - s.v;
        double d = std::sqrt(dx * dx + dv * dv);

        if (d == 0.0) {
            dx = delta.x * d0;
            dv = delta.v * d0;
            d  = d0;
        }

        if (i > n_transient) {
            sum += std::log(d / d0);
        }

        sp.x = s.x + (dx / d) * d0;
        sp.v = s.v + (dv / d) * d0;
        t += DT;
    }

    return sum / (n_steps * DT);
}

int main() {
    struct Case {
        double gamma;
        const char* description;
    };
    
    Case cases[] = {
        {0.20, "Period-1 (regular)"},
        {0.28, "Period-2 (regular)"},
        {0.32, "Onset of chaos"},
        {0.50, "Classic chaotic attractor"},
        {0.65, "Period-3 window (regular)"}
    };

    for (const Case& c : cases) {
        State s0(X0, V0);
        double lambda = benettin(s0, D0, c.gamma, N_TRANSIENT, N_STEPS);

        const char* verdict = (lambda > 1e-3) ? "chaos: lambda > 0" : "no chaos: lambda <= 0";

        std::printf("gamma = %.2f  lambda = %+.6f  ->  %s  [%s]\n",
                    c.gamma, lambda, verdict, c.description);
    }

    return 0;
}
