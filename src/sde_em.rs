use crate::{
    dop_shared::{IntegrationError, Stats},
    sde_shared::*,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OVector, Scalar};
use num_traits::Zero;
// use rand::thread_rng;
// use rand_distr::{Distribution, Normal};
use simba::scalar::{ClosedAdd, ClosedMul, ClosedNeg, ClosedSub, SubsetOf};
pub struct SdeEm<V, F>
where
    F: System<V>,
{
    f: F,
    x: f64,
    y: V,
    x_end: f64,
    step_size: f64,
    x_out: Vec<f64>,
    y_out: Vec<V>,
    stats: Stats,
    // wiener_process: Normal<f64>,
}

impl<T, D: Dim + nalgebra::DimName, F> SdeEm<OVector<T, D>, F>
where
    f64: From<T>,
    T: Copy + SubsetOf<f64> + Scalar + ClosedAdd + ClosedMul + ClosedSub + ClosedNeg + Zero,
    F: System<OVector<T, D>>,
    OVector<T, D>: std::ops::Mul<f64, Output = OVector<T, D>>,
    DefaultAllocator: Allocator<T, D>,
{
    /// Default initializer for the structure
    ///
    /// # Arguments
    ///
    /// * `f`           - Structure implementing the System<V> trait
    /// * `x`           - Initial value of the independent variable (usually time)
    /// * `y`           - Initial value of the dependent variable(s)
    /// * `x_end`       - Final value of the independent variable
    /// * `step_size`   - Step size used in the method
    ///
    pub fn new(f: F, x: f64, y: OVector<T, D>, x_end: f64, step_size: f64) -> Self {
        Self {
            f,
            x,
            y,
            x_end,
            step_size,
            x_out: Vec::new(),
            y_out: Vec::new(),
            stats: Stats::new(),
            // wiener_process: Normal::new(0.0, 0.1).unwrap(),
        }
    }

    pub fn integrate(&mut self) -> Result<Stats, IntegrationError> {
        // Save initial values
        self.x_out.push(self.x);
        self.y_out.push(self.y.clone());

        let num_steps = ((self.x_end - self.x) / self.step_size).ceil() as usize;
        for _ in 0..num_steps {
            let (x_new, y_new) = self.step();

            self.x_out.push(x_new);
            self.y_out.push(y_new.clone());

            self.x = x_new;
            self.y = y_new;

            self.stats.num_eval += 1;
            self.stats.accepted_steps += 1;
        }
        Ok(self.stats)
    }

    fn step(&self) -> (f64, OVector<T, D>) {
        let (rows, cols) = self.y.shape_generic();
        let mut k = vec![OVector::zeros_generic(rows, cols); 12];
        // let dw = self
        //     .wiener_process
        //     .sample_iter(&mut thread_rng())
        //     .take(self.y.nrows())
        //     .collect::<Vec<_>>();
        // set dw a nalgebra OVector from vec<f64>
        let x_new = self.x + self.step_size;
        self.f.deterministic(x_new, &self.y, &mut k[0]);
        self.f.stochastic(x_new, &self.y, &mut k[1]);
        let y_new = &self.y
            + k[0].clone() * self.step_size
            + (k[1].clone() * (self.step_size.sqrt()/* * dw*/));
        (x_new, y_new)
    }
    /// Getter for the independent variable's output.
    pub fn x_out(&self) -> &Vec<f64> {
        &self.x_out
    }

    /// Getter for the dependent variables' output.
    pub fn y_out(&self) -> &Vec<OVector<T, D>> {
        &self.y_out
    }
}

// use nalgebra::{Dim, DimName, OVector};
