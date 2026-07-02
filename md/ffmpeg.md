查看 mp4 信息

    ffprobe  ~/Documents/链式法则.mp4
    ffprobe -v quiet -print_format json -show_format -show_streams ~/Documents/链式法则.mp4

保持高宽比压缩

    # 宽度固定为 720，高度按比例自动计算
    ffmpeg -i input.mp4 -vf "scale=720:-2" output.mp4

    # 高度固定为 480，宽度按比例自动计算
    ffmpeg -i input.mp4 -vf "scale=-2:480" output.mp4

    # 限制码率 + 保持 720p
    ffmpeg -i input.mp4 -vf "scale=720:-2" -crf 23 -c:a aac -b:a 128k output.mp4