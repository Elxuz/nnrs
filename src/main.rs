use std::{fs::File, io::Write, path::PathBuf};

use clap::Parser;

use crate::{
    data::NeuralNetworkData,
    mlp::{NeuralNetwork, cpu::NeuralNetworkCpu, gpu::NeuralNetworkGpu},
};

mod data;
mod mlp;

#[derive(clap_derive::Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap_derive::Subcommand, Debug)]
enum Commands {
    Train {
        output_file: PathBuf,
        #[arg(short, default_value_t = 1)]
        epochs: usize,
        #[arg(short, default_value_t = 1)]
        batch_size: usize,
        #[arg(short, value_enum, default_value_t = CalcType::Cpu)]
        calc_type: CalcType,
    },
    Test {
        input_file: PathBuf,
        #[arg(short, value_enum, default_value_t = CalcType::Cpu)]
        calc_type: CalcType,
    },
}

#[derive(clap_derive::ValueEnum, Debug, Clone, PartialEq)]
enum CalcType {
    Cpu,
    Gpu,
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Train {
            output_file: out,
            epochs,
            batch_size,
            calc_type,
        } => train(out, epochs, batch_size, calc_type),
        Commands::Test {
            input_file: input,
            calc_type,
        } => test(input, calc_type),
    }
}

fn train(output_file: PathBuf, epochs: usize, batch_size: usize, calc_type: CalcType) {
    let pixel_data = include_bytes!("../data/train-images.idx3-ubyte");
    let label_data = include_bytes!("../data/train-labels.idx1-ubyte");

    let learning_rate = 0.005;

    let mut a: Box<dyn NeuralNetwork> = match calc_type {
        CalcType::Cpu => Box::new(NeuralNetworkCpu::new()),
        CalcType::Gpu => Box::new(pollster::block_on(NeuralNetworkGpu::new(
            vec![50],
            batch_size,
        ))),
    };

    for epoch in 1..=epochs {
        for i in 0..60000 / batch_size {
            if i % (600 / batch_size) == 0 {
                print!("\rCurrently at epoch {epoch} {}%", i * batch_size / 600);
                std::io::stdout().flush().unwrap();
            }
            let (buf, label) = get_image(pixel_data, label_data, batch_size * i, batch_size);

            let mut targets = Vec::new();

            for target in label {
                for i in 0..10 {
                    if i == *target {
                        targets.push(1.0);
                    } else {
                        targets.push(0.0);
                    }
                }
            }

            a.train(buf, &targets, batch_size, learning_rate);
        }

        let leftover = 60000 % batch_size;
        if leftover == 0 {
            continue;
        }
        let mut targets = Vec::new();
        let (buf, label) = get_image(pixel_data, label_data, 60000 - leftover - 1, leftover);
        for elem_label in 0..leftover {
            let mut data = vec![0f32; 10];
            data[label[elem_label] as usize] = 1.0;
            targets.append(&mut data);
        }
        a.train(buf, &targets, leftover, learning_rate);
    }

    let save = match calc_type {
        CalcType::Cpu => {
            NeuralNetworkData::from_nn_cpu(a.as_any().downcast_ref::<NeuralNetworkCpu>().unwrap())
        }
        CalcType::Gpu => {
            NeuralNetworkData::from_nn_gpu(a.as_any().downcast_ref::<NeuralNetworkGpu>().unwrap())
        }
    };
    let mut output_file = File::create(std::env::current_dir().unwrap().join(output_file)).unwrap();
    let _ = serde_json::to_writer(&mut output_file, &save);

    println!("\r\x1b[2Kfinished.");
}

fn test(input_file: PathBuf, _calc_type: CalcType) {
    let input_file = File::open(std::env::current_dir().unwrap().join(input_file)).unwrap();
    let data: NeuralNetworkData = serde_json::from_reader(input_file).unwrap();

    let mut a = data.to_nn_cpu();

    let pixel_data = include_bytes!("../data/t10k-images.idx3-ubyte");
    let label_data = include_bytes!("../data/t10k-labels.idx1-ubyte");

    let mut success = 0;

    for i in 0..10000 {
        if i % 100 == 0 {
            print!("\rCurrently at {}%", i / 100);
            std::io::stdout().flush().unwrap();
        }

        let (buf, label) = get_image(pixel_data, label_data, i, 1);
        let res = a.calculate(buf, 1);

        let mut max_idx = 0;
        let mut cur_idx = 0;
        let mut max = 0.;

        for row in res.row_iter() {
            for elem in row.iter() {
                if *elem > max {
                    max = *elem;
                    max_idx = cur_idx;
                }
                cur_idx += 1;
            }
        }

        if max_idx == label[0] {
            success += 1;
        }
    }

    println!(
        "\rpassed {success}/10000 tests ({}%)",
        success as f32 / 10000. * 100.
    );
}

fn get_image<'a, 'b>(
    pixel_data: &'a [u8],
    label_data: &'b [u8],
    idx: usize,
    amount: usize,
) -> (&'a [u8], &'b [u8]) {
    if idx >= pixel_data.len() {
        panic!()
    }

    const IMAGE_SIZE: usize = 28 * 28;

    // skip file header
    let buf = &pixel_data[16 + IMAGE_SIZE * idx..16 + IMAGE_SIZE * (idx + amount)];

    let label = &label_data[8 + idx..8 + idx + amount];

    (buf, label)
}
