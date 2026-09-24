import Foundation
import AppKit
import ImageIO

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

func main() throws {
    let arguments = CommandLine.arguments

    guard arguments.count >= 2 else {
        print("用法:")
        print("  BackgroundRemover <input-image>")
        print("")
        print("例如:")
        print("  BackgroundRemover input.jpg")
        return
    }

    let inputPath = arguments[1]
    let inputURL = URL(fileURLWithPath: inputPath)

    let image = try loadImage(at: inputURL)

    print("图片读取成功")
    print("文件: \(inputURL.path)")
    print("尺寸: \(image.width) x \(image.height)")
    print("bitsPerComponent: \(image.bitsPerComponent)")
    print("bitsPerPixel: \(image.bitsPerPixel)")
    print("alphaInfo: \(image.alphaInfo)")
}

do {
    try main()
} catch {
    fputs(
        "错误: \(error.localizedDescription)\n",
        stderr
    )
    exit(1)
}
