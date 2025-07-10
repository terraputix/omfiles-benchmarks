#!/usr/bin/env node
// om-typescript-bm.js - TypeScript benchmark for reading OM files

const {
  OmFileReader,
  FileBackendNode,
  OmDataType,
} = require("@openmeteo/file-reader");
const fs = require("fs");

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);

  // Need at least one argument (file path)
  if (args.length < 1) {
    console.error(
      "Usage: node om-typescript-bm.js <file_path> [t y x] [iterations]",
    );
    process.exit(1);
  }

  const filePath = args[0];

  // We load the file into a buffer, because we want to benchmark
  // the decoding performance, not how fast the file system is.
  const buffer = fs.readFileSync(filePath);
  // Create reader instance
  const backend = new FileBackendNode(buffer);
  const reader = await OmFileReader.create(backend);

  // Default values
  let t = reader.getDimensions()[0];
  let y = reader.getDimensions()[1];
  let x = reader.getDimensions()[2];
  let iterations = 1;

  // Parse dimensions if provided (3 args for dimensions)
  if (args.length >= 4) {
    t = parseInt(args[1]);
    y = parseInt(args[2]);
    x = parseInt(args[3]);

    // Check if iterations parameter is provided
    if (args.length >= 5) {
      iterations = parseInt(args[4]);
    }
  } else if (args.length >= 2) {
    // If only one extra arg, it's the iterations
    iterations = parseInt(args[1]);
  }

  try {
    const ranges = [
      { start: 0, end: t },
      { start: 0, end: y },
      { start: 0, end: x },
    ];

    for (let i = 0; i < iterations; i++) {
      // Read the specified slice
      const data = await reader.read(OmDataType.FloatArray, ranges);
      // Just accessing data to ensure it's fully read
      const totalElements = data.length;
      console.log(`Total elements: ${totalElements}`);
    }

    // Close the file
    reader.dispose();
  } catch (error) {
    console.error("Error reading file:", error);
    process.exit(1);
  }
}

// Execute the main function
main().catch((error) => {
  console.error("Unhandled error:", error);
  process.exit(1);
});
