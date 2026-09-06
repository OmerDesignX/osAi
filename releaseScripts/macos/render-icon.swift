import AppKit
import Foundation

guard CommandLine.arguments.count == 4 else {
  fputs("Usage: swift render-icon.swift <source.png> <destination.png> <scale>\n", stderr)
  exit(2)
}

let sourcePath = CommandLine.arguments[1]
let destinationPath = CommandLine.arguments[2]
guard let scale = Double(CommandLine.arguments[3]), scale > 0, scale <= 1 else {
  fputs("Scale must be greater than 0 and no more than 1\n", stderr)
  exit(2)
}

guard let source = NSImage(contentsOfFile: sourcePath) else {
  fputs("Could not read the source icon\n", stderr)
  exit(1)
}

guard let bitmap = NSBitmapImageRep(
  bitmapDataPlanes: nil,
  pixelsWide: 1024,
  pixelsHigh: 1024,
  bitsPerSample: 8,
  samplesPerPixel: 4,
  hasAlpha: true,
  isPlanar: false,
  colorSpaceName: .deviceRGB,
  bytesPerRow: 0,
  bitsPerPixel: 0
) else {
  fputs("Could not allocate the macOS icon canvas\n", stderr)
  exit(1)
}

guard let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
  fputs("Could not create the macOS icon graphics context\n", stderr)
  exit(1)
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.imageInterpolation = .high
NSColor.clear.setFill()
NSRect(x: 0, y: 0, width: 1024, height: 1024).fill()
let artworkSize = 1024 * scale
let inset = (1024 - artworkSize) / 2
source.draw(
  in: NSRect(x: inset, y: inset, width: artworkSize, height: artworkSize),
  from: NSRect(origin: .zero, size: source.size),
  operation: .copy,
  fraction: 1
)
context.flushGraphics()
NSGraphicsContext.restoreGraphicsState()

guard let png = bitmap.representation(using: .png, properties: [:]) else {
  fputs("Could not render the macOS icon\n", stderr)
  exit(1)
}

try png.write(to: URL(fileURLWithPath: destinationPath), options: .atomic)
