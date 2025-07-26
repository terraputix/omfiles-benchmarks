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
      "Usage: node om-typescript-bm.js <file_path> [X] [Y] [T] [ITERATIONS]",
    );
    process.exit(1);
  }

  const filePath = args[0];
  const x = parseInt(args[1]);
  const y = parseInt(args[2]);
  const t = parseInt(args[3]);
  const iterations = parseInt(args[4]);

  // JS has one additional argument compared to other languages
  let preloadFile = false;
  if (args.length > 5) {
    preloadFile = args[5] === "true";
  }

  let backend;
  if (preloadFile) {
    // We load the file into a buffer, because we want to benchmark
    // the decoding performance, not how fast the file system is.
    const buffer = fs.readFileSync(filePath);
    // Create reader instance
    backend = new FileBackendNode(buffer);
  } else {
    // Open the OM file
    backend = new FileBackendNode(filePath);
  }

  const reader = await OmFileReader.create(backend);
  const dims = reader.getDimensions();

  const x_max = dims[0] - x;
  const y_max = dims[1] - y;
  const t_max = dims[2] - t;

  // Create read selections
  const readSelections = [];
  for (let i = 0; i < iterations; i++) {
    readSelections.push([
      {
        start: x_max === 0 ? 0 : i % x_max,
        end: x_max === 0 ? x : (i % x_max) + x,
      },
      {
        start: y_max === 0 ? 0 : i % y_max,
        end: y_max === 0 ? y : (i % y_max) + y,
      },
      {
        start: t_max === 0 ? 0 : i % t_max,
        end: t_max === 0 ? t : (i % t_max) + t,
      },
    ]);
  }

  let dataLen = 0;
  try {
    for (const ranges of readSelections) {
      const startTime = process.hrtime();

      // Read the specified slice
      const data = await reader.read(OmDataType.FloatArray, ranges);
      const elapsedHr = process.hrtime(startTime);
      const elapsed = elapsedHr[0] + elapsedHr[1] / 1e9;

      // Access data to ensure it's fully read
      dataLen = data.length;

      console.log(elapsed.toFixed(6));
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
