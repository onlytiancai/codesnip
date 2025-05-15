背景信息
- ffmpeg 路径是 D:\haohu\soft\ffmpeg\bin\ffmpeg.exe
- 原始的 mp3 文件在 "D:\BaiduYunDownload\NCE2-英音-(MP3+LRC)\" 路径下，文件名如如 "01－A Private Conversation.mp3","02－Breakfast or Lunch.mp3"
- 有个csv文件， 路径是 D:\haohu\github\codesnip\python\lrc-split\output.csv，有 4列，分别是File,Start,End,Sentence，示例如下

File,Start,End,Sentence
01－A Private Conversation.lrc,00:12.81,00:20.63,Why did the writer complain to the people behind him?
01－A Private Conversation.lrc,00:20.63,00:24.14,Last week I went to the theatre.
02－Breakfast or Lunch.lrc,00:13.38,00:19.69,Why was the writer's aunt surprised?
02－Breakfast or Lunch.lrc,00:19.69,00:21.83,It was Sunday.

需求：要写一个 python 代码，读取 csv 的每一行，

- 比如第一行的 File 是 "01－A Private Conversation.lrc", 那么对应的 mp3 名称和 File 名称相同，只是后缀不同，比如是该行对应的 mp3 路径是 "D:\BaiduYunDownload\NCE2-英音-(MP3+LRC)\01－A Private Conversation.mp3"
- 读取 Start 和 End 列，它们分别表示要截取的mp3的开始和结尾
- 根据以上信息，使用 ffmpeg 截取 mp3 片段，保存到 "D:\BaiduYunDownload\splited_mp3" 目录下，文件名用该行所在的csv的行号，如 1.mp3, 2.mp3等

