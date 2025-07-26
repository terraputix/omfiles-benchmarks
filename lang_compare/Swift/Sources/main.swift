import ArgumentParser
import Foundation
import OmFileFormat

// MARK: - Command Line Parsing with ArgumentParser

@main
struct OmBenchmark: AsyncParsableCommand {
    @Argument(help: "Path to OM file")
    var filePath: String

    @Argument(help: "X dimension to read (optional, defaults to max)")
    var x: UInt64

    @Argument(help: "Y dimension to read (optional, defaults to max)")
    var y: UInt64

    @Argument(help: "T dimension to read (optional, defaults to max)")
    var t: UInt64

    @Argument(help: "Number of iterations to run (optional, defaults to 1)")
    var iterations: UInt64

    // MARK: - Run Benchmark
    func run() async throws {
        // Open the OM file
        let reader = try await OmFileReader(file: filePath).asArray(of: Float.self)!
        let dims = reader.getDimensions()

        // Compute max offsets for each dimension
        let xMax = dims[0] > x ? dims[0] - x : 0
        let yMax = dims[1] > y ? dims[1] - y : 0
        let tMax = dims[2] > t ? dims[2] - t : 0

        // Prepare all read selections
        var readSelections: [[Range<UInt64>]] = []
        for i in 0..<iterations {
            let xStart = xMax == 0 ? 0 : i % xMax
            let yStart = yMax == 0 ? 0 : i % yMax
            let tStart = tMax == 0 ? 0 : i % tMax
            readSelections.append([
                xStart..<(xStart + x),
                yStart..<(yStart + y),
                tStart..<(tStart + t),
            ])
        }

        var dataLen = 0
        for ranges in readSelections {
            let startTime = Date()
            // Read the data
            let data: [Float] = try await reader.read(range: ranges)
            let elapsed = Date().timeIntervalSince(startTime)
            // Access data to ensure it's fully read
            dataLen = data.count

            print(String(format: "%.6f", elapsed))
        }
    }
}
