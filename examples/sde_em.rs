use nalgebra::Vector1;
use ndarray::Array;
use ode_solvers::sde_em::SdeEm;
use ode_solvers::sde_shared::System;
use plotly::{Plot, Scatter};
use rand::thread_rng;
use rand_distr::Distribution;
use rand_distr::Normal;

type State = Vector1<f64>;
type Time = f64;

#[derive(Clone, Debug)]
struct Model {
    ke: f64,
    ke_iov: f64,
}
impl System<State> for Model {
    fn deterministic(&self, _t: Time, y: &State, dy: &mut State) {
        dy[0] = -self.ke * y[0];
    }
    fn stochastic(&self, _t: Time, _y: &State, l: &mut State) {
        let wiener_process = Normal::new(0.0, self.ke_iov).unwrap();
        let dw = wiener_process.sample(&mut thread_rng());
        l[0] = 1.0 * dw;
    }
}
fn main() {
    let y0 = State::new(1.0);
    let system = Model {
        ke: 1.0,
        ke_iov: 1.0,
    };
    let mut out: Vec<f64> = vec![];
    for _ in 0..1000 {
        let mut stepper = SdeEm::new(system.clone(), 0.0, y0, 1.0, 0.001);
        let res = stepper.integrate();
        let x = stepper.x_out().to_vec();
        let y = stepper.y_out();
        let yout: Vec<f64> = y.into_iter().map(|x| x.data.0[0][0]).collect();
        out.push(*yout.last().unwrap());

        // let mut plot = Plot::new();
        // let trace = Scatter::new(x, yout);
        // plot.add_trace(trace);
        // plot.show();
    }
    dbg!((-1.0_f64).exp()).abs();
    dbg!(Array::from_vec(out).mean());
}
