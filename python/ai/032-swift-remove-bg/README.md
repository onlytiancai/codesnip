编译

    swiftc \
    -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
    -framework AppKit \
    -framework ImageIO \
    -framework Vision \
    -framework CoreImage \
    main.swift \
    -o BackgroundRemover

运行

    ./BackgroundRemover test.jpeg