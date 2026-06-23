use std::any::Any;

pub mod cpu;
pub mod gpu;

pub trait NeuralNetwork: Any {
    fn train(&mut self, raw_images: &[u8], label_data: &[f32], batch_size: usize);
    fn test(&mut self, raw_image: &[u8], label: u32) -> bool;

    fn as_any(&self) -> &dyn Any;
}
