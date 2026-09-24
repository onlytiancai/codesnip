import Foundation
import AppKit
import ImageIO
import Vision
import CoreVideo
import CoreImage


func generateForegroundInstanceMask(
    from image: CGImage
) throws -> VNInstanceMaskObservation {

    print("开始执行 Vision Foreground Instance 分割...")

    let request =
        VNGenerateForegroundInstanceMaskRequest()

    let handler = VNImageRequestHandler(
        cgImage: image,
        options: [:]
    )

    try handler.perform([
        request
    ])

    guard let observation =
        request.results?.first else {

        throw NSError(
            domain: "BackgroundRemover",
            code: 10,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Vision 没有返回 Foreground Instance Mask"
            ]
        )
    }

    print("Foreground Instance 分割完成")

    print(
        "Instance 数量: " +
        "\(observation.allInstances.count)"
    )

    return observation
}

func printInstances(
    _ observation: VNInstanceMaskObservation
) {

    print("")
    print("检测到的 Foreground Instances:")

    for instance in observation.allInstances {
        print(
            "Instance: \(instance)"
        )
    }

    print("")
}

func refineMask(
    _ maskPixelBuffer: CVPixelBuffer,
    blurRadius: Double = 0.8
) -> CIImage {

    let maskImage = CIImage(
        cvPixelBuffer: maskPixelBuffer
    )

    // --------------------------------------------
    // 第一步：轻微高斯模糊
    //
    // 目的：
    // 让 Mask 边缘产生自然的 alpha 过渡。
    //
    // radius 不要太大。
    // 0.5 ~ 1.5 通常比较保守。
    // --------------------------------------------

    let blurredMask = maskImage
        .applyingFilter(
            "CIGaussianBlur",
            parameters: [
                kCIInputRadiusKey: blurRadius
            ]
        )

    // --------------------------------------------
    // GaussianBlur 会扩大 image extent。
    //
    // 所以把它裁回原 Mask 范围。
    // --------------------------------------------

    let refinedMask = blurredMask.cropped(
        to: maskImage.extent
    )

    return refinedMask
}


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

    request.qualityLevel = .accurate

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

    print(
        "Mask 尺寸: " +
        "\(CVPixelBufferGetWidth(mask)) x " +
        "\(CVPixelBufferGetHeight(mask))"
    )

    return mask
}

// MARK: - Create Transparent Image

func createTransparentImage(
    image: CGImage,
    maskPixelBuffer: CVPixelBuffer
) throws -> CGImage {

    print("开始生成透明背景图片...")

    let inputImage = CIImage(
        cgImage: image
    )

    let originalExtent = inputImage.extent

    // Vision 输出的 Mask

    let maskImage = refineMask(
        maskPixelBuffer,
        blurRadius: 0.8
    )
    print(
        "原图尺寸: " +
        "\(originalExtent.width) x " +
        "\(originalExtent.height)"
    )

    print(
        "Mask 尺寸: " +
        "\(maskImage.extent.width) x " +
        "\(maskImage.extent.height)"
    )

    // ------------------------------------------------
    // 1. 把 Mask 缩放到和原图完全一致
    // ------------------------------------------------

    let scaleX =
        originalExtent.width /
        maskImage.extent.width

    let scaleY =
        originalExtent.height /
        maskImage.extent.height

    var resizedMask = maskImage.transformed(
        by: CGAffineTransform(
            scaleX: scaleX,
            y: scaleY
        )
    )

    // 确保 Mask 的 origin 和原图一致
    resizedMask = resizedMask
        .transformed(
            by: CGAffineTransform(
                translationX:
                    originalExtent.minX -
                    resizedMask.extent.minX,

                y:
                    originalExtent.minY -
                    resizedMask.extent.minY
            )
        )

    // ------------------------------------------------
    // 2. 创建完全透明的背景
    // ------------------------------------------------

    let transparentBackground =
        CIImage(
            color: CIColor(
                red: 0,
                green: 0,
                blue: 0,
                alpha: 0
            )
        )
        .cropped(
            to: originalExtent
        )

    // ------------------------------------------------
    // 3. CIBlendWithMask
    // ------------------------------------------------

    guard let filter = CIFilter(
        name: "CIBlendWithMask"
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 4,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法创建 CIBlendWithMask"
            ]
        )
    }

    filter.setValue(
        inputImage,
        forKey: kCIInputImageKey
    )

    filter.setValue(
        transparentBackground,
        forKey: kCIInputBackgroundImageKey
    )

    filter.setValue(
        resizedMask,
        forKey: kCIInputMaskImageKey
    )

    // ------------------------------------------------
    // 4. 获取最终 CIImage
    // ------------------------------------------------

    guard let outputImage =
        filter.outputImage else {

        throw NSError(
            domain: "BackgroundRemover",
            code: 5,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "CIBlendWithMask 没有产生输出"
            ]
        )
    }

    // ------------------------------------------------
    // 5. CIImage → CGImage
    // ------------------------------------------------

    let context = CIContext()

    guard let outputCGImage =
        context.createCGImage(
            outputImage,
            from: originalExtent
        )
    else {

        throw NSError(
            domain: "BackgroundRemover",
            code: 6,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法将 CIImage 转换为 CGImage"
            ]
        )
    }

    print("透明背景图片生成完成")

    return outputCGImage
}


// MARK: - Save PNG

func savePNG(
    _ image: CGImage,
    to url: URL
) throws {

    print("保存 PNG:")
    print(url.path)

    guard let destination =
        CGImageDestinationCreateWithURL(
            url as CFURL,
            "public.png" as CFString,
            1,
            nil
        )
    else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 6,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法创建 PNG 输出"
            ]
        )
    }

    CGImageDestinationAddImage(
        destination,
        image,
        nil
    )

    guard CGImageDestinationFinalize(
        destination
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 7,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "PNG 保存失败"
            ]
        )
    }

    print("PNG 保存成功")
}


func generateInstanceMask(
    observation: VNInstanceMaskObservation,
    image: CGImage,
    instances: IndexSet
) throws -> CVPixelBuffer {

    print("开始生成 Instance Mask...")

    // 创建 Vision Image Request Handler
    let handler = VNImageRequestHandler(
        cgImage: image,
        options: [:]
    )

    // 注意：
    // generateScaledMaskForImage 的 from:
    // 要传 VNImageRequestHandler，而不是 CGImage
    let mask = try observation.generateScaledMaskForImage(
        forInstances: instances,
        from: handler
    )

    print(
        "Instance Mask 尺寸: " +
        "\(CVPixelBufferGetWidth(mask)) x " +
        "\(CVPixelBufferGetHeight(mask))"
    )

    return mask
}



func saveMaskAsPNG(
    _ pixelBuffer: CVPixelBuffer,
    to url: URL
) throws {

    print("保存 Mask:")
    print(url.path)

    // CVPixelBuffer → CIImage
    let ciImage = CIImage(
        cvPixelBuffer: pixelBuffer
    )

    // CIImage → CGImage
    let context = CIContext()

    guard let cgImage = context.createCGImage(
        ciImage,
        from: ciImage.extent
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 20,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "无法从 CVPixelBuffer 创建 CGImage"
            ]
        )
    }

    // CGImage → PNG
    guard let destination =
        CGImageDestinationCreateWithURL(
            url as CFURL,
            "public.png" as CFString,
            1,
            nil
        )
    else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 21,
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

    guard CGImageDestinationFinalize(
        destination
    ) else {
        throw NSError(
            domain: "BackgroundRemover",
            code: 22,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Mask PNG 保存失败"
            ]
        )
    }

    print("Mask PNG 保存成功")
}



// MARK: - Main

func main() throws {

    let arguments = CommandLine.arguments

    guard arguments.count >= 2 else {
        print("用法:")
        print("  BackgroundRemover <input-image>")
        return
    }

    let inputPath = arguments[1]

    let inputURL = URL(
        fileURLWithPath: inputPath
    )

    // 输出文件名
    let outputURL = inputURL
        .deletingPathExtension()
        .appendingPathExtension(
            "transparent.png"
        )

    // 1. 读取原图

    let image = try loadImage(
        at: inputURL
    )

    print("")
    print("图片读取成功")
    print(
        "尺寸: \(image.width) x \(image.height)"
    )
    print("")

    let observation =
        try generateForegroundInstanceMask(
            from: image
        )

    printInstances(observation)    

    let indexSet = observation.allInstances

    let foregroundMask =
        try generateInstanceMask(
            observation: observation,
            image: image,
            instances: indexSet
        )

    let maskURL = inputURL
        .deletingPathExtension()
        .appendingPathExtension(
            "foreground-mask.png"
        )

    try saveMaskAsPNG(
        foregroundMask,
        to: maskURL
    )

    // 2. Vision 人像分割

    // let mask = try generatePersonMask(
    //     from: image
    // )

    // print("")

    // 3. Mask → Alpha → 透明图片

    let transparentImage =
        try createTransparentImage(
            image: image,
            maskPixelBuffer: foregroundMask
        )

    print("")

    // 4. 保存 PNG

    try savePNG(
        transparentImage,
        to: outputURL
    )

    print("")
    print("================================")
    print("处理完成")
    print("输出:")
    print(outputURL.path)
    print("================================")
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