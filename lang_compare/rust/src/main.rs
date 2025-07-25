use clap::{App, Arg};
use omfiles_rs::io::reader::OmFileReader;
use std::ops::Range;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let matches = App::new("OM File Rust Benchmark")
        .arg(
            Arg::with_name("FILE")
                .required(true)
                .help("Path to OM file"),
        )
        .arg(Arg::with_name("T").help("T dimension to read"))
        .arg(Arg::with_name("Y").help("Y dimension to read"))
        .arg(Arg::with_name("X").help("X dimension to read"))
        .arg(Arg::with_name("ITERATIONS").help("Number of iterations"))
        .get_matches();

    // Get file path
    let file_path = matches.value_of("FILE").unwrap();

    // Open the OM file
    let om_file = OmFileReader::from_file(file_path)?;
    let dims = om_file.get_dimensions();

    // Default to full dimensions if not specified
    let mut t = dims[0];
    let mut y = dims[1];
    let mut x = dims[2];
    let mut iterations = 1;

    // Parse dimensions if provided
    if let Some(t_str) = matches.value_of("T") {
        t = t_str.parse::<u64>().unwrap_or(dims[0]);
    }
    if let Some(y_str) = matches.value_of("Y") {
        y = y_str.parse::<u64>().unwrap_or(dims[1]);
    }
    if let Some(x_str) = matches.value_of("X") {
        x = x_str.parse::<u64>().unwrap_or(dims[2]);
    }
    if let Some(iter_str) = matches.value_of("ITERATIONS") {
        iterations = iter_str.parse::<usize>().unwrap_or(1);
    }

    // Create read selection
    let read_range = vec![
        Range { start: 0, end: t },
        Range { start: 0, end: y },
        Range { start: 0, end: x },
    ];

    let mut _data_len = 0;
    // Run iterations
    for _ in 0..iterations {
        let start = std::time::Instant::now();

        // Read the data
        let _data = om_file.read::<f32>(&read_range, None, None)?;

        // Access data to ensure it's fully read
        _data_len = _data.len();

        let elapsed = start.elapsed();
        println!("{:.6}", elapsed.as_secs_f64());
    }

    // File is automatically closed when om_file goes out of scope
    Ok(())
}
