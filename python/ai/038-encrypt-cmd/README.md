# myenc

带密码的文本文件加密 / 解密 / 编辑命令行工具。

- 磁盘上永远是密文，编辑时用临时明文，保存后自动重新加密。
- **输入一次密码，一段时间内不用再输**：密码由一个内存守护进程持有，空闲超时后自动擦除退出，密钥/密码**从不落盘**。
- 纯 Python + uv，Linux / macOS 行为一致。

## 安装

```bash
uv tool install myenc          # 从 PyPI 安装
uv tool install --from . myenc  # 从本地目录安装（开发用）
```

开发模式：

```bash
uv sync
uv run myenc --help
```

## 命令

```
myenc [--ttl 15m] [--no-daemon] [-v] COMMAND ...

  enc     FILE...            加密（默认输出 FILE.enc）
  cat     FILE.enc...        解密并输出到 stdout
  dec     FILE.enc...        解密到文件
  edit    FILE.enc           解密 → 编辑器 → 重新加密
  vi      FILE.enc           同 edit，但固定用 vi
  ls      [DIR]              列出密文文件
  info    FILE.enc           查看文件头信息（不需要密码）
  rm      FILE.enc...        删除密文
  passwd  FILE.enc           修改密码
  daemon  status|stop        查看 / 停止解锁缓存
  lock                      立刻擦除缓存的密码
  version                   版本信息
```

### 例子

```bash
myenc enc readme.txt                 # -> readme.txt.enc，提示两次密码
myenc cat readme.txt.enc             # 解密并打印（缓存命中时不再问密码）
myenc vi readme.txt.enc              # 用 vi 打开，保存后自动重新加密
myenc info readme.txt.enc            # 免密码查看元信息
myenc ls -l .                        # 列出密文并显示 KDF 参数
myenc dec readme.txt.enc             # 解密回 readme.txt
myenc passwd readme.txt.enc          # 换密码
myenc rm readme.txt.enc
```

管道友好：`myenc cat a.enc | grep -i secret`。

### 全局选项

| 选项 | 说明 |
|---|---|
| `--ttl 15m` | 解锁缓存的有效期（滑动过期，每次用到都会重置）。也可用 `MYENC_TTL=15m` |
| `--no-daemon` | 不使用缓存，每次都输入密码（加密在本进程内完成） |
| `--socket PATH` | 守护进程 socket 路径（也可用 `MYENC_SOCKET`） |
| `-v` | 详细输出 |

编辑器按 `$MYENC_EDITOR` → `$VISUAL` → `$EDITOR` → `vi` 顺序选择，支持 `EDITOR="code -w"` 这种带参数的写法。

## 免密缓存是怎么工作的

1. 第一次 `myenc cat` 会提示密码，密码被交给后台守护进程 `myencd`（Unix socket，进程 detached）。
2. 守护进程把密码只存在内存里，**不写磁盘**；socket 目录 0700、socket 文件 0600，并校验对端 uid。
3. 之后 15 分钟内的任何 `myenc` 调用都直接复用它，不再问密码；空闲超过 TTL 后守护进程擦除内存并退出，下次使用重新输入。
4. 守护进程最多同时持有 8 个密码：用 B 密码解锁不会让 A 密码的文件失效。文件用别的密码时，`myenc` 会自动再问一次。

想立刻失效：`myenc lock`（擦除密码但保留进程）或 `myenc daemon stop`（彻底停止）。

## 文件格式

密文文件 = 64 字节文件头 + 密文。

```
 0  6  magic            "MYENC1"
 6  1  format_version   1
 7  1  kdf_id           1 = Argon2id
 8  1  argon2_version   19
 9  3  reserved
12  4  time_cost        Argon2 时间开销
16  4  memory_cost_kib  Argon2 内存开销（KiB）
20  4  parallelism
24 16  salt             每个文件随机
40 12  nonce            每个文件随机
52  4  reserved
56  8  plaintext_len    明文长度
64  N  ciphertext       明文 + 16 字节 GCM tag
```

- 密钥派生：Argon2id（默认 `t=3, m=64MiB, p=1`，参数逐文件保存，将来改默认值不影响旧文件）。
- 加密：AES-256-GCM，**64 字节文件头作为 AAD**，所以 salt、nonce、KDF 参数、明文长度任一被改动都会解密失败。
- 每次加密都用新的 salt 和 nonce，同一密码两次加密结果完全不同。

## 安全模型

- 密码只从终端读取（不回显）。**不提供** `--password` 之类的命令行或环境变量传参方式（`ps` / `/proc` 可见）。
- 密码与密钥从不落盘；守护进程退出时尽力擦除内存中的缓冲区。
- `edit` 会在临时目录（0700）里写一份明文给编辑器，编辑结束立即删除。极高敏感场景可把编辑器指向内存盘。
- 写入一律是「同目录临时文件 → fsync → 校验能解密 → 原子 rename → fsync 目录」，中途被杀不会留下半个文件，原密文始终可用。
- 并发编辑用锁文件保护（锁放在状态目录，因为 rename 会换 inode，锁目标文件本身无效）。
- `enc --in-place` **不留明文备份**（备份会毁掉这个工具的意义）；`edit --backup` 才会留一份旧密文的 `.bak`。

退出码：`0` 成功、`2` 用法错误、`3` 密码错误或文件损坏、`4` 格式损坏、`5` 守护进程不可用、`6` 文件被其它进程占用、`7` 编辑器非零退出。

## 开发

```bash
uv sync
uv run pytest -q
```

Linux 验证（容器里跑一遍测试）：

```bash
docker run --rm --add-host=host.docker.internal:host-gateway -v "$PWD":/w -w /w \
  -e HTTP_PROXY=http://host.docker.internal:10808 -e HTTPS_PROXY=http://host.docker.internal:10808 \
  python:3.12-slim sh -c 'pip install -q -e . pytest && python -m pytest -q'
```

测试全程使用临时的 socket 和状态目录（`MYENC_SOCKET` / `MYENC_STATE_DIR`），不会碰到你自己的守护进程。
