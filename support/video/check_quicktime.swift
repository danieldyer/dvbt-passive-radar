// Validate exact-frame forward/backward seeking with the macOS playback framework.
// Build: swiftc -module-cache-path build/cache/swift support/video/check_quicktime.swift -o build/check_quicktime
// Run:   build/check_quicktime VIDEO.mov
import AVFoundation
import Foundation

guard CommandLine.arguments.count == 2 else {
    fatalError("Usage: check_quicktime VIDEO.mov")
}
let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
let duration = try await asset.load(.duration)
guard let track = try await asset.loadTracks(withMediaType: .video).first else {
    fatalError("No video track")
}
let frameDuration = try await track.load(.minFrameDuration)
let frameCount = Int((duration.seconds / frameDuration.seconds).rounded())
guard frameCount > 0 else { fatalError("Empty video") }
let generator = AVAssetImageGenerator(asset: asset)
generator.requestedTimeToleranceBefore = .zero
generator.requestedTimeToleranceAfter = .zero
generator.appliesPreferredTrackTransform = true
// Include alternating distant seeks and successive frames in both directions.
let distant = [0.8, 0.1, 0.95, 0.2, 0.7, 0.3, 0.9, 0.05, 0.6, 0.4]
let length = min(12, frameCount)
let middle = min(frameCount / 2, frameCount - length)
let frames = distant.map { min(frameCount - 1, Int(Double(frameCount) * $0)) }
    + Array(middle..<(middle + length)) + Array((middle..<(middle + length)).reversed())
var results = [[String: Any]]()
for frame in frames {
    let requested = CMTimeMultiply(frameDuration, multiplier: Int32(frame))
    let started = ProcessInfo.processInfo.systemUptime
    let (image, actual) = try await generator.image(at: requested)
    let elapsed = ProcessInfo.processInfo.systemUptime - started
    guard abs(actual.seconds - requested.seconds) < 1e-6 else {
        fatalError("Seek returned the wrong frame")
    }
    results.append(["frame": frame, "requested_s": requested.seconds,
                    "actual_s": actual.seconds, "elapsed_ms": elapsed * 1000,
                    "width": image.width, "height": image.height])
}
let output: [String: Any] = ["framework": "AVFoundation", "duration_s": duration.seconds,
                            "frame_duration_s": frameDuration.seconds, "seeks": results]
let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
