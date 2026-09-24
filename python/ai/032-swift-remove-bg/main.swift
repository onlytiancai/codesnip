import Foundation
import AppKit
import ImageIO
import Vision
import CoreVideo
import CoreImage

// MARK: - Image Loading

func loadImage(at url: URL) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(
        url as CFURL,
        nil
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法创建 CGImageSource: \(url.path)"
            ]
        )
    }

    guard let image = CGImageSourceCreateImageAtIndex(
        source,
        0,
        nil
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 2,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法从图片创建 CGImage: \(url.path)"
            ]
        )
    }

    return image
}

// MARK: - Person Segmentation

func generatePersonMask(
    from image: CGImage
) throws -> CVPixelBuffer {

    print("开始执行 Vision 人像分割...")

    let request = VNGeneratePersonSegmentationRequest()

    // 最高质量模式
    request.qualityLevel = .accurate

    // 输出 8-bit 单通道 Mask
    request.outputPixelFormat =
        kCVPixelFormatType_OneComponent8

    let handler = VNImageRequestHandler(
        cgImage: image,
        options: [:]
    )

    try handler.perform([request])

    guard let observation = request.results?.first else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 3,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Vision 没有返回人物 Mask"
            ]
        )
    }

    let mask = observation.pixelBuffer

    print("Vision 人像分割完成")
    print(
        "Mask 尺寸: \(CVPixelBufferGetWidth(mask)) x " +
        "\(CVPixelBufferGetHeight(mask))"
    )

    return mask
}

// MARK: - Save PixelBuffer as PNG

func saveMaskAsPNG(
    _ pixelBuffer: CVPixelBuffer,
    to url: URL
) throws {

    print("开始保存 Mask: \(url.path)")

    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

    let context = CIContext()

    guard let cgImage = context.createCGImage(
        ciImage,
        from: ciImage.extent
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 4,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法从 Mask 创建 CGImage"
            ]
        )
    }

    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        "public.png" as CFString,
        1,
        nil
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 5,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法创建 PNG 输出"
            ]
        )
    }

    CGImageDestinationAddImage(
        destination,
        cgImage,
        nil
    )

    guard CGImageDestinationFinalize(destination) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 6,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "PNG 保存失败"
            ]
        )
    }

    print("Mask 保存成功")
}

// MARK: - Main

func main() throws {

    let arguments = CommandLine.arguments

    guard arguments.count >= 2 else {
        print("用法:")
        print("  BackgroundRemover <input-image>")
        print("")
        print("例如:")
        print("  BackgroundRemover test.jpeg")
        return
    }

    let inputPath = arguments[1]

    let inputURL = URL(
        fileURLWithPath: inputPath
    )

    // 输出到输入图片所在目录
    let outputURL = inputURL
        .deletingPathExtension()
        .appendingPathExtension("mask.png")

    // 1. 读取原图
    let image = try loadImage(
        at: inputURL
    )

    print("")
    print("图片读取成功")
    print("文件: \(inputURL.path)")
    print(
        "尺寸: \(image.width) x \(image.height)"
    )
    print(
        "bitsPerComponent: \(image.bitsPerComponent)"
    )
    print(
        "bitsPerPixel: \(image.bitsPerPixel)"
    )
    print(
        "alphaInfo: \(image.alphaInfo)"
    )
    print("")

    // 2. Vision 生成人像 Mask
    let mask = try generatePersonMask(
        from: image
    )

    print("")

    // 3. 保存 Mask
    try saveMaskAsPNG(
        mask,
        to: outputURL
    )

    print("")
    print("完成")
    print("输出: \(outputURL.path)")
}

do {
    try main()
} catch {

    fputs(
        """
        错误:
        \(error.localizedDescription)

        """,
        stderr
    )

    exit(1)
}
