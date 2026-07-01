use std::{
    fs::File,
    io::{Read, Seek, Write},
};

use crate::mlp::{MLPDescriptor, MultiLayerPerceptron};
mod mlp;

fn main() {
    const BATCH_SIZE: u32 = 1024;
    let mut mlp = MultiLayerPerceptron::new(&MLPDescriptor {
        learning_rate: 0.001,
        input_size: 784,
        output_size: 10,
        max_batch_size: BATCH_SIZE,
        h_layer_sizes: vec![1024, 1024, 1024],
    });

    let mut data_file =
        File::open("./data/train-images.idx3-ubyte").expect("Failed to open training data");
    let mut label_file =
        File::open("./data/train-labels.idx1-ubyte").expect("Failed to open training data labels");

    let mult_train = BATCH_SIZE / 600 + 1;

    let mult_test = BATCH_SIZE / 100 + 1;

    for epoch in 0..5 {
        for i in 0..(60000 / BATCH_SIZE) {
            if i % (600 * mult_train / BATCH_SIZE) == 0 {
                print!(
                    "\repoch {epoch}: {:.0}% done",
                    i as f32 / (600. / BATCH_SIZE as f32)
                );
                std::io::stdout().flush().unwrap();
            }
            let (data, labels) = load_data(
                &mut data_file,
                &mut label_file,
                BATCH_SIZE as usize * i as usize,
                BATCH_SIZE as usize,
            );

            let data = convert_data(data);
            let labels = convert_labels(labels);

            mlp.load_data(&data, &labels, BATCH_SIZE);
            mlp.train();
        }

        let leftover = 60000 % BATCH_SIZE;
        let (data, labels) = load_data(
            &mut data_file,
            &mut label_file,
            60000 - leftover as usize - 1,
            leftover as usize,
        );

        let data = convert_data(data);
        let labels = convert_labels(labels);

        mlp.load_data(&data, &labels, leftover);
        mlp.train();
    }
    println!("\r\x1b[2Ktraining done...");

    let mut data_file =
        File::open("./data/t10k-images.idx3-ubyte").expect("Failed to open testing data");
    let mut label_file =
        File::open("./data/t10k-labels.idx1-ubyte").expect("Failed to open testing data labels");

    let mut correct = 0;
    for i in 0..(10000 / BATCH_SIZE) {
        if i % (100 * mult_test / BATCH_SIZE) == 0 {
            print!("\r{}% done", i as f32 / (100. / BATCH_SIZE as f32));
            std::io::stdout().flush().unwrap();
        }

        let (data, labels) = load_data(
            &mut data_file,
            &mut label_file,
            BATCH_SIZE as usize * i as usize,
            BATCH_SIZE as usize,
        );

        let data = convert_data(data);
        let labels = convert_labels(labels);

        mlp.load_data(&data, &labels, BATCH_SIZE);
        correct += mlp.test();
    }

    let leftover = 10000 % BATCH_SIZE;

    let (data, labels) = load_data(
        &mut data_file,
        &mut label_file,
        10000 - leftover as usize - 1,
        leftover as usize,
    );

    let data = convert_data(data);
    let labels = convert_labels(labels);

    mlp.load_data(&data, &labels, leftover);
    correct += mlp.test();

    println!("\rcorrect: {}/10000 ({}%)", correct, correct as f32 / 100.);
}

fn load_data(
    data_file: &mut File,
    label_file: &mut File,
    offset: usize,
    amount: usize,
) -> (Vec<u8>, Vec<u8>) {
    const IMAGE_SIZE: usize = 28 * 28;
    // go to offset
    let _ = data_file.seek(std::io::SeekFrom::Start((16 + IMAGE_SIZE * offset) as u64));
    let _ = label_file.seek(std::io::SeekFrom::Start((8 + offset) as u64));

    let mut data_buf = vec![0u8; IMAGE_SIZE * amount];
    let mut label_buf = vec![0u8; amount];

    let _ = data_file.read_exact(&mut data_buf);
    let _ = label_file.read_exact(&mut label_buf);

    (data_buf, label_buf)
}

fn convert_data(data: Vec<u8>) -> Vec<f32> {
    data.iter().map(|val| *val as f32 / 255.).collect()
}

fn convert_labels(labels: Vec<u8>) -> Vec<f32> {
    let mut res = Vec::with_capacity(labels.len() * 10);

    for label in labels {
        for i in 0..10 {
            if i == label {
                res.push(1_f32);
            } else {
                res.push(0_f32);
            }
        }
    }

    res
}
