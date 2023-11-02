use std::cmp::min;
use std::num::NonZeroU32;

use candle_core::{
    display::set_print_options_short, safetensors, shape::Dims, DType, Device, IndexOp, Module,
    Result, Tensor, D,
};
use candle_nn::{Activation, Linear, Optimizer, VarBuilder, VarMap};

use winit::dpi::{LogicalSize, PhysicalSize, Size};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn display_img(img_rgb: &Tensor) -> anyhow::Result<()> {
    let img_rgb = (img_rgb * 255.)?.to_dtype(DType::U8)?;
    let (img_height, img_width, img_channels) = img_rgb.shape().dims3()?;

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(img_height as u32, img_width as u32))
        .build(&event_loop)?;
    let context = unsafe { softbuffer::Context::new(&window) }.unwrap();
    let mut surface = unsafe { softbuffer::Surface::new(&context, &window) }.unwrap();

    event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => {
            // Application update code.

            // Queue a RedrawRequested event.
            //
            // You only need to call this if you've determined that you need to redraw, in
            // applications which do not always need to. Applications that redraw continuously
            // can just render here instead.
            // window.request_redraw();
        }
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            window_id,
        } if window_id == window.id() => {
            let (window_width, window_height) = {
                let size = window.inner_size();
                (size.width, size.height)
            };

            surface
                .resize(
                    NonZeroU32::new(window_width).unwrap(),
                    NonZeroU32::new(window_height).unwrap(),
                )
                .unwrap();

            let mut buffer = surface.buffer_mut().unwrap();

            for row in 0..(img_height as usize) {
                for col in 0..(img_width as usize) {
                    let pixel = img_rgb
                        .i((row as usize, col as usize, ..))
                        .unwrap()
                        .to_vec1::<u8>()
                        .unwrap();

                    let [r, g, b] = pixel[..] else {
                        panic!("pixel didn't have 3 components")
                    };

                    buffer[row * (window_width as usize) + col] =
                        (b as u32) | ((g as u32) << 8) | ((r as u32) << 16);
                }
            }

            buffer.present().unwrap();
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            window_id,
        } if window_id == window.id() => {
            elwt.exit();
        }
        _ => {}
    })?;

    Ok(())
}

pub fn main() -> Result<()> {
    set_print_options_short();

    let dev = Device::cuda_if_available(0)?;
    // let dev = Device::Cpu;
    let tiny_nerf_data = safetensors::load("data/tiny_nerf_data.safetensors", &dev)?;

    let tn_images = &tiny_nerf_data["images"];
    let tn_poses = &tiny_nerf_data["poses"];
    let tn_focal = &tiny_nerf_data["focal"];

    let (img_count, height, width, img_channels) = tn_images.shape().dims4()?;
    let focal_dist = tn_focal.to_scalar()?;

    let test_pose = tn_poses.i(101)?.to_dtype(DType::F32)?;
    let test_img = tn_images.i(101)?.to_dtype(DType::F32)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let adamw_params = candle_nn::ParamsAdamW {
        lr: 5e-4,
        ..Default::default()
    };
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), adamw_params)?;

    let model = TinyNerf::new(8, 64, vs)?;

    let (rays_o, rays_d) = get_rays(height, width, focal_dist, &test_pose, &dev)?;

    for step in 0..10 {
        let (rgb, depth, acc) = render_rays(&model, &rays_o, &rays_d, 2., 6., 64, true, &dev)?;
        let loss = (rgb - &test_img)?.sqr()?.mean_all()?;
        opt.backward_step(&loss)?;
        println!("step {} / 10 done", step + 1);
    }

    let (rgb, depth, acc) = render_rays(&model, &rays_o, &rays_d, 2., 6., 64, true, &dev)?;
    // display_img(&rgb).unwrap();

    Ok(())
}

struct TinyNerf {
    hidden_layers: Vec<Linear>,
    output_layer: Linear,
    activation: Activation,
}

impl TinyNerf {
    pub fn new(depth: usize, hidden_layer_width: usize, vb: VarBuilder) -> Result<Self> {
        let input_size = 3 + 3 * POSITION_EMBED_LEVELS * 2;

        let mut hidden_layers = vec![];
        let mut prev_layer_size = input_size;

        for layer_idx in 0..depth {
            let vb = vb.pp(format!("hidden_layer_{layer_idx}"));
            hidden_layers.push(candle_nn::linear(prev_layer_size, hidden_layer_width, vb)?);

            if layer_idx % 4 == 0 && layer_idx != 0 {
                // every 4th layer, we concat the input
                prev_layer_size = hidden_layer_width + input_size;
            } else {
                prev_layer_size = hidden_layer_width;
            }
        }

        let output_layer = {
            let vb = vb.pp(format!("output_layer"));
            candle_nn::linear(prev_layer_size, 4, vb)?
        };

        Ok(Self {
            hidden_layers,
            output_layer,
            activation: Activation::Relu,
        })
    }
}

impl Module for TinyNerf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut current = xs.clone();

        for (layer_idx, (layer, activation)) in self
            .hidden_layers
            .iter()
            .map(|hl| (hl, Some(&self.activation)))
            .chain([(&self.output_layer, None)])
            .enumerate()
        {
            if layer_idx % 4 == 1 && layer_idx != 1 {
                current = Tensor::cat(&[&current, xs], D::Minus1)?;
            }

            // println!("layer_idx = {layer_idx} layer = {layer:?} activation = {activation:?} current shape = {:?}", current.shape());

            current = layer.forward(&current.contiguous()?)?;

            if let Some(activation) = activation {
                current = activation.forward(&current)?;
            }
        }

        Ok(current)
    }
}

const POSITION_EMBED_LEVELS: usize = 6;

fn embed_position(positions: &Tensor) -> Result<Tensor> {
    let mut output = vec![positions.clone()];
    for level_idx in 0..POSITION_EMBED_LEVELS {
        // this is equivalent to 2 ** (positions * level_idx)
        // b/c we don't have a variable base exponent in candle 😭
        let x = (positions * ((level_idx as f64) * f64::ln(2.)))?;
        let x = x.exp()?;
        output.push(x.sin()?);
        output.push(x.cos()?);
    }

    let output = Tensor::cat(&output, D::Minus1)?;
    Ok(output)
}

fn get_rays(
    height: usize,
    width: usize,
    focal_dist: f64,
    pose: &Tensor,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let x = Tensor::arange(0, width as i64, dev)?.to_dtype(DType::F32)?;
    let y = Tensor::arange(0, height as i64, dev)?.to_dtype(DType::F32)?;
    let mut xy = Tensor::meshgrid(&[&x, &y], true)?;
    let y = xy.pop().unwrap();
    let x = xy.pop().unwrap();

    // compute the direction of each ray based on the pixel position
    let dirs = Tensor::stack(
        &[
            // x component of direction
            ((x - (width as f64 * 0.5))? / focal_dist)?,
            // y component of direction
            ((y - (height as f64 * 0.5))? / focal_dist)?.neg()?,
            // z-component of direction (-1 for all rays)
            Tensor::ones((height, width), DType::F32, dev)?.neg()?,
        ],
        D::Minus1,
    )?;

    let dirs = dirs.unsqueeze(D::Minus2)?;

    // get top left 3x3 for the pose matrix so we don't add translation,
    // then expand it so that the shapes match
    let pose_local = pose.i((..3, ..3))?.to_dtype(DType::F32)?;
    // translate ray directions to world coordinates
    let rays_d = dirs.broadcast_mul(&pose_local)?;

    let rays_d = rays_d.sum(D::Minus1)?;
    let rays_d = rays_d.squeeze(D::Minus1)?;

    // ray origin is the same for every ray; it's the origin translated by the
    // translation vector, so we get it by just picking the translation out of
    // the matrix and then expanding it
    let rays_o = pose
        .i((..3, 3))?
        .expand(rays_d.shape())?
        .to_dtype(rays_d.dtype())?;

    Ok((rays_o, rays_d))
}

pub fn linspace(start: f64, stop: f64, steps: usize, device: &Device) -> Result<Tensor> {
    if steps < 1 {
        candle_core::bail!("cannot use linspace with steps {steps} <= 1")
    }
    let delta = (stop - start) / (steps - 1) as f64;
    let vs = (0..steps)
        .map(|step| start + step as f64 * delta)
        .collect::<Vec<_>>();
    Tensor::from_vec(vs, steps, device)
}

fn render_rays(
    network_fn: &dyn candle_nn::Module,
    rays_o: &Tensor,
    rays_d: &Tensor,
    near: f64,
    far: f64,
    n_samples: usize,
    rand: bool,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    // The batchify function is not directly translated here for simplicity. Instead,
    // consider running the network in chunks outside of this function if you have a large number of rays.

    // compute 3D query points
    let mut z_vals = linspace(near, far, n_samples, device)?;
    if rand {
        let shape = rays_o.shape().clone();
        let mut shape = shape.into_dims();
        *shape.last_mut().unwrap() = n_samples;

        let rand = Tensor::rand(0.0, (far - near) / (n_samples as f64), shape, device)?;
        z_vals = z_vals.broadcast_add(&rand)?;
    }

    let z_vals = z_vals.to_dtype(DType::F32)?;

    let points = (rays_o.unsqueeze(D::Minus2)?.broadcast_add(
        &rays_d
            .unsqueeze(D::Minus2)?
            .broadcast_mul(&z_vals.unsqueeze(D::Minus1)?)?,
    ))?;

    // run the network
    let (height, width, depths, _) = points.shape().dims4()?;
    let points_flat = points.reshape(((), 3))?;

    // we run the network on chunks of the input at a time b/c we can't train on
    // 640,000 samples at once (we get out-of-memory errors)
    let mut raw_parts = vec![];
    let sample_count = height * width * depths;
    const CHUNK_SIZE: usize = 1024 * 64;
    for chunk_idx in 0..=(sample_count / CHUNK_SIZE) {
        let start = chunk_idx * CHUNK_SIZE;
        let end = min(start + CHUNK_SIZE, sample_count);

        if end > start {
            let chunk = points_flat.i(start..end)?;
            let embedded_chunk = embed_position(&chunk)?;

            let raw_part = network_fn.forward(&embedded_chunk)?;
            raw_parts.push(raw_part);
        }
    }

    let raw = Tensor::cat(&raw_parts, 0)?;
    let raw = raw.reshape((height, width, n_samples, 4))?;

    // compute opacities and colors
    let sigma_a = Activation::Relu.forward(&raw.i((.., .., .., 3))?)?;
    let rgb = Activation::Sigmoid.forward(&raw.i((.., .., .., ..3))?)?;

    // do volume rendering
    let dists = (z_vals.i((.., .., 1..))? - z_vals.i((.., .., ..n_samples - 1))?)?;
    let dists = Tensor::cat(
        &[
            &dists,
            &Tensor::new(1e10f32, device)?.expand(z_vals.i((.., .., ..1))?.shape())?,
        ],
        D::Minus1,
    )?;

    let alpha = sigma_a.neg()?.mul(&dists)?.exp()?;
    let alpha = (1. - alpha)?;
    let alpha_e = ((1. - &alpha)? + 1e-10)?;
    let weights = alpha.broadcast_mul(&cum_prod(&alpha_e, D::Minus1, true)?)?;

    let rgb_map = weights
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&rgb)?
        .sum(D::Minus2)?;
    let depth_map = weights.mul(&z_vals)?.sum(D::Minus1)?;
    let acc_map = weights.sum(D::Minus1)?;

    Ok((rgb_map, depth_map, acc_map))
}

fn cum_prod<D: Dims>(x: &Tensor, axis: D, exclusive: bool) -> Result<Tensor> {
    let mut elements = vec![];
    let dims = axis.to_indexes(x.shape(), "cum_prod")?;
    let dim = *dims.first().unwrap();
    let len = x.shape().dims()[dim];

    let mut prev = if exclusive {
        x.ones_like()?.narrow(dim, 0, 1)?
    } else {
        x.narrow(dim, 0, 1)?
    };

    elements.push(prev.clone());

    let range = if exclusive { 0..(len - 1) } else { 1..len };

    for i in range {
        let element = x.narrow(dim, i, 1)?;
        let element = (&prev * element)?;
        elements.push(element.clone());
        prev = element;
    }

    Tensor::cat(&elements, dim)
}

#[test]
fn cum_prod_test() -> Result<()> {
    let dev = Device::Cpu;

    let t = Tensor::new(&[2, 3, 4i64], &dev)?;
    let c = cum_prod(&t, 0, false)?;
    assert_eq!(c.to_vec1::<i64>()?, [2, 6, 24]);

    let c = cum_prod(&t, 0, true)?;
    assert_eq!(c.to_vec1::<i64>()?, [1, 2, 6]);

    let t = Tensor::new(&[[2, 3, 4i64], [4, 5, 6i64]], &dev)?;
    let c = cum_prod(&t, 0, false)?;
    assert_eq!(c.to_vec2::<i64>()?, [[2, 3, 4], [8, 15, 24]]);
    let c = cum_prod(&t, 1, false)?;
    assert_eq!(c.to_vec2::<i64>()?, [[2, 6, 24], [4, 20, 120]]);

    Ok(())
}
