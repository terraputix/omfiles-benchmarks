use clap::{App, Arg};
use omfiles::io::reader::OmFileReader;
use std::ops::Range;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let matches = App::new("OM File Rust Benchmark")
        .arg(
            Arg::with_name("FILE")
                .required(true)
                .help("Path to OM file"),
        )
        .arg(
            Arg::with_name("X")
                .required(true)
                .help("X dimension to read (optional, defaults to max)"),
        )
        .arg(
            Arg::with_name("Y")
                .required(true)
                .help("Y dimension to read (optional, defaults to max)"),
        )
        .arg(
            Arg::with_name("T")
                .required(true)
                .help("T dimension to read (optional, defaults to max)"),
        )
        .arg(
            Arg::with_name("ITERATIONS")
                .required(true)
                .help("Number of iterations to run (optional, defaults to 1)"),
        )
        .get_matches();

    // Get file path and arguments
    let file_path = matches.value_of("FILE").unwrap();
    let x = matches.value_of("X").unwrap().parse::<u64>().unwrap();
    let y = matches.value_of("Y").unwrap().parse::<u64>().unwrap();
    let t = matches.value_of("T").unwrap().parse::<u64>().unwrap();
    let iterations = matches
        .value_of("ITERATIONS")
        .unwrap()
        .parse::<u64>()
        .unwrap();

    // Open the OM file
    let om_file = OmFileReader::from_file(file_path)?;
    let dims = om_file.get_dimensions();

    // Compute max offsets for each dimension
    let x_max = if dims[0] > x { dims[0] - x } else { 0 };
    let y_max = if dims[1] > y { dims[1] - y } else { 0 };
    let t_max = if dims[2] > t { dims[2] - t } else { 0 };

    // Prepare all read selections
    let mut read_selections: Vec<Vec<Range<u64>>> = Vec::new();
    for i in 0..iterations {
        let x_start = if x_max == 0 { 0 } else { i % x_max };
        let y_start = if y_max == 0 { 0 } else { i % y_max };
        let t_start = if t_max == 0 { 0 } else { i % t_max };
        read_selections.push(vec![
            x_start..(x_start + x),
            y_start..(y_start + y),
            t_start..(t_start + t),
        ]);
    }

    let mut _data_len = 0;
    for ranges in read_selections {
        let start = std::time::Instant::now();
        // Read the data
        let data = om_file.read::<f32>(&ranges, None, None)?;
        let elapsed = start.elapsed();
        // Access data to ensure it's fully read
        _data_len = data.len();
        println!("{:.6}", elapsed.as_secs_f64());
    }

    Ok(())
}
