pub trait System<V> {
    fn deterministic(&self, x: f64, y: &V, dy: &mut V);
    fn stochastic(&self, x: f64, y: &V, dy: &mut V);
    fn solout(&mut self, _x: f64, _y: &V, _dy: &V) -> bool {
        false
    }
}
