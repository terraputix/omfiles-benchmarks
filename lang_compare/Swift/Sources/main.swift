import ArgumentParser
import Foundation
import OmFileFormat

// MARK: - Command Line Parsing with ArgumentParser

@main
struct OmBenchmark: AsyncParsableCommand {
    @Argument(help: "Path to OM file")
    var filePath: String

    @Argument(help: "T dimension to read (optional, defaults to max)")
    var t: UInt64?

    @Argument(help: "Y dimension to read (optional, defaults to max)")
    var y: UInt64?

    @Argument(help: "X dimension to read (optional, defaults to max)")
    var x: UInt64?

    @Argument(help: "Number of iterations to run")
    var iterations: Int = 1

    // MARK: - Run Benchmark
    func run() async throws {
        // Open the OM file
        let reader = try await OmFileReader(file: filePath).asArray(of: Float.self)!
        let dims = reader.getDimensions()

        // Use provided dimensions or defaults
        let tValue = t ?? dims[0]
        let yValue = y ?? dims[1]
        let xValue = x ?? dims[2]

        // Create read selection
        let ranges = [
            0..<tValue,
            0..<yValue,
            0..<xValue,
        ]

        // Run iterations
        var dataLen = 0

        for _ in 0..<iterations {
            let startTime = Date()

            // Read the data
            let data: [Float] = try await reader.read(range: ranges)

            // Access data to ensure it's fully read
            dataLen = data.count

            let elapsed = Date().timeIntervalSince(startTime)
            print(String(format: "%.6f", elapsed))
        }
    }
}
