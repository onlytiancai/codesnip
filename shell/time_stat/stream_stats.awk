BEGIN {
  FS = ","                     # CSV 字段分隔符
  prev_window = ""             # 上一个时间窗口
  last_output_window = ""      # 上一个已输出的时间窗口（用于检测间隙）
  total_count = 0              # 当前窗口内的总记录数
  line_count = 0               # 已输出的行计数器（用于每20行输出表头）

  # 命令行参数说明：
  #   -v show_header=1  启用表头输出（每20行），show_header=0 禁用（默认）
  #   -v granularity=N 设置时间窗口粒度（分钟），可选值：5, 10, 15, 20, 30, 60
  if (show_header == "") show_header = 0
  if (granularity == "") granularity = 5
  granularity += 0
  # 校验粒度值，只允许指定的几个值
  if (granularity != 5 && granularity != 10 && granularity != 15 && \
      granularity != 20 && granularity != 30 && granularity != 60) {
    granularity = 5
  }
}

# 打印列头（时间窗口 + 计数列 + 百分比列）
function print_header() {
  print "Time\t<1s\t1-2s\t2-3s\t3-4s\t4-5s\t5s+\t<1s\t1-2s\t2-3s\t3-4s\t4-5s\t5s+"
}

# 主处理逻辑：逐行读取 CSV 数据
{
  # 去除字段中的引号
  gsub(/"/, "", $1)
  gsub(/"/, "", $2)

  ts = $1    # 时间戳字段
  rt = $2    # 响应时间字段

  # 从 ISO 时间戳中提取小时和分钟（格式如 2024-01-01T01:55:12Z）
  hour = substr(ts, 12, 2)
  min = substr(ts, 15, 2)

  # 根据粒度计算所属的时间窗口
  window_min = int(min / granularity) * granularity
  window = sprintf("%s:%02d", hour, window_min)

  # 时间窗口发生变化时，输出上一个窗口的统计结果
  if (window != prev_window && prev_window != "") {
    # 填充相邻窗口之间的数据间隙（输出全0的行）
    fill_gaps(prev_window, window)
    # 输出上一个窗口的统计数据
    output_stats(prev_window)
    # 重置计数数组，准备统计下一个窗口
    delete count
    total_count = 0
  }
  prev_window = window

  # 仅处理包含 'T' 的时间戳行（即有效的时间格式）
  if (ts ~ /T/) {
    # 处理响应时间格式，支持 "500ms"、"1.2s" 等格式
    # 先处理 ms 格式（毫秒），转换为秒
    if (rt ~ /ms$/) {
      gsub(/ms$/, "", rt)
      val = (rt + 0) / 1000
    } else {
      # 处理 s 格式（秒），去除末尾的 s
      gsub(/s$/, "", rt)
      val = rt + 0
    }

    # 将响应时间分配到对应的桶
    # 桶 0: <1s, 桶 1: 1-2s, 桶 2: 2-3s, 桶 3: 3-4s, 桶 4: 4-5s, 桶 5: 5s+
    if (val < 1)        bucket = 0
    else if (val < 2)  bucket = 1
    else if (val < 3)  bucket = 2
    else if (val < 4)  bucket = 3
    else if (val < 5)  bucket = 4
    else               bucket = 5

    count[bucket]++    # 对应桶的计数 +1
    total_count++      # 总计数 +1
  }
}

# 文件结束时，输出最后一个窗口的统计数据
END {
  if (total_count > 0) {
    fill_gaps(prev_window, "")
    output_stats(prev_window)
  }
}

# 填充两个时间窗口之间的间隙，用0填充缺失的时间点
# from_window: 当前待输出的窗口
# to_window:   下一个待输出的窗口（END 块中为空字符串）
function fill_gaps(from_window, to_window) {
  # 首次输出时，只需记录窗口位置
  if (last_output_window == "") {
    last_output_window = from_window
    return
  }

  # 将时间窗口转换为分钟数，便于计算间隔
  split(from_window, from_parts, ":")
  split(last_output_window, last_parts, ":")

  from_mins = from_parts[1] * 60 + from_parts[2]
  last_mins = last_parts[1] * 60 + last_parts[2]

  gap = from_mins - last_mins
  # 如果间隔大于一个粒度单位，则存在需要填充的间隙
  if (gap > granularity) {
    # 计算需要填充的步数（减去1因为最后一个是 from_window 本身）
    steps = int(gap / granularity) - 1
    for (i = 1; i <= steps; i++) {
      fill_mins = last_mins + i * granularity
      fill_window = sprintf("%02d:%02d", int(fill_mins / 60), fill_mins % 60)
      output_zeros(fill_window)
    }
  }
  last_output_window = from_window
}

# 输出全0的行（用于填充间隙）
function output_zeros(w) {
  line_count++
  # 每20行输出一次表头（仅当启用 show_header 时）
  if (show_header && line_count % 20 == 1) print_header()
  printf "%s\t0\t0\t0\t0\t0\t0\t0.0%%\t0.0%%\t0.0%%\t0.0%%\t0.0%%\t0.0%%\n", w
}

# 计算并输出指定时间窗口的统计数据
# w: 时间窗口字符串（如 "02:00"）
function output_stats(w) {
  # 计算各桶的百分比
  p0 = total_count > 0 ? count[0] / total_count * 100 : 0
  p1 = total_count > 0 ? count[1] / total_count * 100 : 0
  p2 = total_count > 0 ? count[2] / total_count * 100 : 0
  p3 = total_count > 0 ? count[3] / total_count * 100 : 0
  p4 = total_count > 0 ? count[4] / total_count * 100 : 0
  p5 = total_count > 0 ? count[5] / total_count * 100 : 0

  line_count++
  # 每20行输出一次表头（仅当启用 show_header 时）
  if (show_header && line_count % 20 == 1) print_header()
  # 格式：时间\t计数1\t计数2\t...\t计数6\t百分比1\t百分比2\t...\t百分比6
  printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f%%\t%.1f%%\t%.1f%%\t%.1f%%\t%.1f%%\t%.1f%%\n", \
      w, count[0]+0, count[1]+0, count[2]+0, count[3]+0, count[4]+0, count[5]+0, p0, p1, p2, p3, p4, p5
}
