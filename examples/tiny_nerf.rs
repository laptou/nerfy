use std::cmp::min;
use std::collections::HashMap;
use std::num::NonZeroU32;

use tch::nn::{Linear, Module, OptimizerConfig, Path, VarStore};
use tch::{display::set_print_options_short, Device, IndexOp, Kind, Result, Tensor};

use winit::dpi::{LogicalSize, PhysicalSize, Size};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn display_img(img_rgb: &Tensor) -> anyhow::Result<()> {
    let img_rgb = (img_rgb * 255.).to_kind(Kind::Int8);
    let (img_height, img_width, img_channels) = img_rgb.size3()?;

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

            let img_rgb: Vec<Vec<Vec<u8>>> = (&img_rgb).try_into().unwrap();

            for row in 0..(img_height as usize) {
                for col in 0..(img_width as usize) {
                    // let pixels: Vec<Vec<f64>> = img_rgb.into();
                    // let pixel = img_rgb
                    //     .i((row as usize, col as usize, ..))
                    //     .unwrap()
                    //     .to_vec1::<u8>()
                    //     .unwrap();

                    let pixel = &img_rgb[row][col];
                    let &[r, g, b] = &pixel[..] else { unreachable!()};

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

    let dev = Device::Cpu;
    println!("dev = {dev:?}");

    let tiny_nerf_data: HashMap<String, Tensor> =
        HashMap::from_iter(Tensor::read_safetensors("data/tiny_nerf_data.safetensors")?);

    let tn_images = tiny_nerf_data["images"].to_device(dev);
    let tn_poses = tiny_nerf_data["poses"].to_device(dev);
    let tn_focal = &tiny_nerf_data["focal"];

    let (img_count, height, width, img_channels) = tn_images.size4()?;
    let focal_dist = tn_focal.double_value(&[]);

    let test_pose = tn_poses.i(101).to_kind(Kind::Double);
    let test_img = tn_images.i(101).to_kind(Kind::Double);

    let mut vs = VarStore::new(dev);

    let mut opt = tch::nn::AdamW::default().build(&vs, 5e-4)?;

    let model = TinyNerf::new(8, 256, vs.root())?;
    vs.set_kind(Kind::Double);

    let (rays_o, rays_d) = get_rays(height, width, focal_dist, &test_pose, dev);

    for step in 0..10 {
        let (rgb, depth, acc) = render_rays(&model, &rays_o, &rays_d, 2., 6., 64, true, dev);
        let loss = (rgb - &test_img).square().mean(None);
        opt.backward_step(&loss);
        println!("step {} / 10 done", step + 1);
    }

    let (rgb, depth, acc) = render_rays(&model, &rays_o, &rays_d, 2., 6., 64, true, dev);
    display_img(&rgb).unwrap();

    Ok(())
}

#[derive(Debug)]
struct TinyNerf {
    hidden_layers: Vec<Linear>,
    output_layer: Linear,
}

impl TinyNerf {
    pub fn new(depth: usize, hidden_layer_width: usize, vs: Path) -> Result<Self> {
        let input_size = 3 + 3 * POSITION_EMBED_LEVELS * 2;

        let mut hidden_layers = vec![];
        let mut prev_layer_size = input_size;

        for layer_idx in 0..depth {
            let vs = &vs / format!("hidden_layer_{layer_idx}");

            hidden_layers.push(tch::nn::linear(
                vs,
                prev_layer_size as i64,
                hidden_layer_width as i64,
                tch::nn::LinearConfig::default(),
            ));

            if layer_idx % 4 == 0 && layer_idx != 0 {
                // every 4th layer, we concat the input
                prev_layer_size = hidden_layer_width + input_size;
            } else {
                prev_layer_size = hidden_layer_width;
            }
        }

        let output_layer = tch::nn::linear(
            &vs / "output_layer",
            prev_layer_size as i64,
            4,
            tch::nn::LinearConfig::default(),
        );

        Ok(Self {
            hidden_layers,
            output_layer,
        })
    }
}

impl Module for TinyNerf {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut current = xs.shallow_clone();

        for (layer_idx, (layer, activation)) in self
            .hidden_layers
            .iter()
            .map(|hl| (hl, Some(())))
            .chain([(&self.output_layer, None)])
            .enumerate()
        {
            if layer_idx % 4 == 1 && layer_idx != 1 {
                current = Tensor::cat(&[&current, xs], -1);
            }

            // println!(
            //     "layer_idx = {layer_idx} layer = {layer:?} current shape = {:?} ({:?})",
            //     current.size(),
            //     current.kind()
            // );

            current = layer.forward(&current);

            if let Some(activation) = activation {
                current = current.relu();
            }
        }

        current
    }
}

const POSITION_EMBED_LEVELS: usize = 6;

fn embed_position(positions: &Tensor) -> Tensor {
    let mut output = vec![positions.shallow_clone()];
    for level_idx in 0..POSITION_EMBED_LEVELS {
        // this is equivalent to 2 ** (positions * level_idx)
        // b/c we don't have a variable base exponent in candle 😭
        let x = (positions * ((level_idx as f64) * f64::ln(2.)));
        let x = x.exp();
        output.push(x.sin());
        output.push(x.cos());
    }

    let output = Tensor::cat(&output, -1);
    output
}

fn get_rays(
    height: i64,
    width: i64,
    focal_dist: f64,
    pose: &Tensor,
    dev: Device,
) -> (Tensor, Tensor) {
    let x = Tensor::arange(width, (Kind::Double, dev));
    let y = Tensor::arange(height, (Kind::Double, dev));
    let mut xy = Tensor::meshgrid_indexing(&[&x, &y], "xy");
    let y = xy.pop().unwrap();
    let x = xy.pop().unwrap();

    // compute the direction of each ray based on the pixel position
    let dirs = Tensor::stack(
        &[
            // x component of direction
            ((x - (width as f64 * 0.5)) / focal_dist),
            // y component of direction
            -((y - (height as f64 * 0.5)) / focal_dist),
            // z-component of direction (-1 for all rays)
            -Tensor::ones([height, width], (Kind::Double, dev)),
        ],
        -1,
    );

    let dirs = dirs.unsqueeze(-2);

    // get top left 3x3 for the pose matrix so we don't add translation,
    // then expand it so that the shapes match
    let pose_local = pose.i((..3, ..3)).to_kind(Kind::Double);
    // translate ray directions to world coordinates
    let rays_d = dirs * pose_local;

    let rays_d = rays_d.sum_dim_intlist(-1, false, None);
    let rays_d = rays_d.squeeze_dim(-1);

    // ray origin is the same for every ray; it's the origin translated by the
    // translation vector, so we get it by just picking the translation out of
    // the matrix and then expanding it
    let rays_o = pose
        .i((..3, 3))
        .expand(rays_d.size(), false)
        .to_kind(rays_d.kind());

    (rays_o, rays_d)
}

fn render_rays(
    network_fn: &dyn tch::nn::Module,
    rays_o: &Tensor,
    rays_d: &Tensor,
    near: f64,
    far: f64,
    n_samples: i64,
    rand: bool,
    device: Device,
) -> (Tensor, Tensor, Tensor) {
    // The batchify function is not directly translated here for simplicity. Instead,
    // consider running the network in chunks outside of this function if you have a large number of rays.

    // compute 3D query points
    let mut z_vals = Tensor::linspace(near, far, n_samples, (Kind::Double, device));
    if rand {
        let mut shape = rays_o.size().clone();
        shape.pop();
        shape.push(n_samples);

        let rand =
            Tensor::rand(shape, (Kind::Double, device)) * ((far - near) / (n_samples as f64));
        z_vals = z_vals + rand;
    }

    let points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1);

    // run the network
    let (height, width, depths, _) = points.size4().unwrap();
    let points_flat = points.reshape([-1, 3]);

    // we run the network on chunks of the input at a time b/c we can't train on
    // 640,000 samples at once (we get out-of-memory errors)
    let mut raw_parts = vec![];
    let sample_count = height * width * depths;
    const CHUNK_SIZE: i64 = 1024 * 64;
    for chunk_idx in 0..=(sample_count / CHUNK_SIZE) {
        let start = chunk_idx * CHUNK_SIZE;
        let end = min(start + CHUNK_SIZE, sample_count);

        if end > start {
            let chunk = points_flat.i(start..end);
            let embedded_chunk = embed_position(&chunk);

            let raw_part = network_fn.forward(&embedded_chunk);
            raw_parts.push(raw_part);
        }
    }

    let raw = Tensor::cat(&raw_parts, 0);
    let raw = raw.reshape([height, width, n_samples as i64, 4]);

    // compute opacities and colors
    let sigma_a = raw.i((.., .., .., 3)).relu();
    let rgb = raw.i((.., .., .., ..3)).sigmoid();

    // do volume rendering
    let dists = z_vals.i((.., .., 1..)) - z_vals.i((.., .., ..n_samples - 1));
    let dists = Tensor::cat(
        &[
            &dists,
            &Tensor::from(1e10).expand(z_vals.i((.., .., ..1)).size(), false),
        ],
        -1,
    );

    let alpha = (-sigma_a * dists).exp();
    let alpha = (1.0f64 - &alpha);
    let alpha_e = ((1.0f64 - &alpha) + 1e-10);
    let weights = alpha * alpha_e.cumprod(-1, None);

    let rgb_map = (weights.unsqueeze(-1) * rgb).sum_dim_intlist(-2, false, None);
    let depth_map = (&weights * z_vals).sum_dim_intlist(-1, false, None);
    let acc_map = weights.sum_dim_intlist(-1, false, None);

    (rgb_map, depth_map, acc_map)
}
