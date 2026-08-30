#!/usr/bin/env swift
// Reproducible macOS Vision OCR scaffold for the handwritten Kullu wordlists.
// The checked transcription/audit, rather than this OCR output, is authoritative.

import AppKit
import Foundation
import Vision

var arguments = Array(CommandLine.arguments.dropFirst())
var outputPath: String? = nil
if arguments.count >= 2, arguments[0] == "--output" {
    outputPath = arguments[1]
    arguments.removeFirst(2)
}

guard !arguments.isEmpty else {
    fputs("usage: vision_ocr.swift [--output FILE] IMAGE...\n", stderr)
    exit(2)
}

var lines: [String] = []
func emit(_ line: String) {
    if outputPath == nil {
        print(line)
    } else {
        lines.append(line)
    }
}

for path in arguments {
    guard let image = NSImage(contentsOfFile: path) else {
        fputs("cannot open \(path)\n", stderr)
        exit(1)
    }
    var rect = NSRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
        fputs("cannot rasterize \(path)\n", stderr)
        exit(1)
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    request.minimumTextHeight = 0.006
    try VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([request])
    emit("@@\t\(path)")
    let observations = (request.results ?? []).sorted {
        if abs($0.boundingBox.midY - $1.boundingBox.midY) > 0.005 {
            return $0.boundingBox.midY > $1.boundingBox.midY
        }
        return $0.boundingBox.minX < $1.boundingBox.minX
    }
    for observation in observations {
        let box = observation.boundingBox
        let candidates = observation.topCandidates(3).map(\.string).joined(separator: "\t")
        emit(String(format: "%.6f\t%.6f\t%.6f\t%.6f\t%@", box.minX, box.minY, box.width, box.height, candidates))
    }
}

if let outputPath {
    try (lines.joined(separator: "\n") + "\n").write(
        toFile: outputPath, atomically: true, encoding: .utf8
    )
}
