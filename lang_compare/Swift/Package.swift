// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "om-swift-bm",
    products: [
        .executable(name: "om-swift-bm", targets: ["om-swift-bm"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/open-meteo/om-file-format.git",
            revision: "2dd19d4dfc473c35fd278cfa5f5728105a34806c"
        ),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.2.0"),
    ],
    targets: [
        .executableTarget(
            name: "om-swift-bm",
            dependencies: [
                .product(name: "OmFileFormat", package: "om-file-format"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ])
    ]
)
