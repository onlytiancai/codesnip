### [信息安全基础](http://baike.baidu.com/view/17249.htm)

信息安全目标

1. 真实性：对信息的来源进行判断，能对伪造来源的信息予以鉴别, 就是身份认证。
1. 保密性：保证机密信息不被窃听，盗取，或窃听者不能了解信息的真实含义。
1. 完整性：保证数据的一致性，防止数据被非法用户篡改或部分丢失。
1. 可用性：保证合法用户对信息和资源的使用不会被不正当地拒绝。
1. 不可抵赖性：建立有效的责任机制，防止用户否认其行为。

常见攻击手段

1. 破坏信息的完整性，篡改信息
1. 拒绝服务
1. 窃听,拦截信息
1. 假冒
1. 抵赖
1. 重放
1. 猜测预测
1. 拖库, 信息泄露

### 密码学基础 

**HASH**

1. 介绍
    1. 无论多长多随意的信息，最后都转换成一个固定长度的散列值；
    1. 对于大量不同的信息，最后出来的散列值呈平均分布；
    1. 对于特定的一个信息，最后出来的散列值都是相同的。
    1. 不可逆（用一个固定长度的数值，怎么可能表示任意长度的信息呢
    1. 难碰撞
    1. 可代表, 出示一个散列值，就可以证明你持有某个有效信息，比如密码
1. 用途：
    1. 签名认证，证明某段信息没有被修改；
    1. 密码验证，证明你确实知道某个密码；
    1. 数据去重
1. 常见算法：
    1. checksum: 就是简单的总和校验码，一般在通信中用于保证数据的完整性和准确性,比如tcp协议。
    1. crc32：32bit,性能好，碰撞率高，一般用于图片去重。
    1. md5: 128bit,一般用户密码加密，文件校验，对于密码加密已经不安全，因为已经找到了碰撞条件
    1. sha1: 基本同md5, 已经找到了碰撞条件, 但用于文件校验还是没问题的
    1. sha256: 相对安全，可以用于密码加密
    1. Bloom Filter: 多个hash函数组合进行去重，一般用于大数据量的去重，比如搜索引擎网页收录。
       如果它说一个项目不在一个集合里，那肯定不在，如果说在，那有很小的可能不在。


**关于密码HASH的误解**


1. MD5被破解了。
    1. MD5被破解只是可以说被碰撞，在保护信息完整性和正确性方面弱了一些，但对密码验证毫无影响
1. 已经有很大的MD5密码（碰撞）库，有7.8万亿个密码呢！
    1. 不加盐的弱密码Hash确实容易被破解，另外就是对于碰撞库来说，MD5和其它HASH的安全性是一样的。
    1. www.cmd5.com
1. MD5加盐不安全。
    1. 加盐对破解某个用户的密码不能提高难度，对破解整个库会提高难度。
1. 加固定盐。
    1. 盐不是key，盐丢了，破解整个库就可以只用一个对撞字典就行了。 
    1. 密码串+配置串+盐。配置串都一样，存文件里，这样多因素(数据库，代码，配置串)认证，增加难度。
1. MD5、SHA1、……这些公开的算法都是不安全的，就算加了盐也不安全，真正安全的只能是自己写一个算法，然后再加盐。
    1. 用户都把库拖走了，你的算法难道还能保住？
1. 用bcrypt吧，可以随意调节运算需要的时间，比MD5慢千万倍，每秒钟只能算出3个密码……
    1. 对于用字典攻击，会降低打表的速度(沉没成本)，不会降低破解的难度(边际成本)。

你的服务器里面的一切东西，无论是数据库和代码都是随时可能被公开的。只有当它们都被公开了你仍然是安全的情况下，那才是真正的安全。

**当前的密码破解方法**
1. 彩虹表破解
    1. 支持主流哈希算法
    1. 排列组合地毯式全面破解
1. 分布式查询库
    1. 数据量数万亿条
    1. 分布式快速查询
1. 传统CPU破解
    1. 支持哈希类型最广
    1. 纯文本密码字典数十亿条
1. GPU高速破解
    1. 高速破解，速度是CPU破解的数十上百倍
    1. 支持各类规则破解

**可被破解的HASH算法**

    MD5    MySQL
    md5($pass.$salt)  MySQL4.1/MySQL5
    md5($salt.$pass)  MD5(Wordpress)
    md5(md5($pass))  MD5(phpBB3)
    md5(md5(md5($pass)))  MD5(Unix)
    md5(md5($pass).$salt)  SHA-1(Base64)
    md5(md5($salt).$pass)  SSHA-1(Base64)
    md5($salt.md5($pass))  SHA-1(Django)
    md5($salt.$pass.$salt)  MD4
    md5(md5($salt).md5($pass))  NTLM
    md5(md5($pass).md5($salt)) Domain Cached Credentials
    md5($salt.md5($salt.$pass)) MD5(Chap)
    md5($salt.md5($pass.$salt)) MSSQL
    SHA1  SHA256
    sha1($pass.$salt)  MD5(APR)
    sha1($salt.$pass)  SHA512
    sha1(sha1($pass))  SHA-512(Unix)


**随机数**

1. 介绍
    1. 指定一个范围和种子，随机的生成一个数字 
1. 用途：
    1. 防猜测预测，让黑客猜测不到信息地址或加密因子。
    1. 防止重放，每次请求里的随机数不一致，用户重放请求时随机数已被使用而拒绝请求。
    1. Hash里当作salt，让相同的明文加盐后生成不同的hash值，防止被人用字典攻击破解密码。
    1. 加密算法中当作iv(初始化向量)，让相同的明文块生成不同的密文，增加破解难度。
    1. 从集合里随机抽取数据，保证一段时间内唯一，比如tcp的seq。
    1. 动态口令，和时间，种子相关的随机数。
1. 常用算法：
    1. 线性同余: 最常用的伪随机数生成算法，如果知道种子有可能被预测到。
    1. GUID: 全球唯一字符串，很难被猜测到。

**对称加密**

1. 介绍：加密和解密需要使用相同的密钥，有流加密和块加密之分，一般可以进行大数据量的加密。
1. 用途：
    1. 防止信息泄露
    1. 防止信息拦截
1. 常见算法：
    1. DES: 64bit密钥, 破解难度较低
    1. 3DES: 三重DES，128bit密钥，破解难度较高
    1. RC2: DES的建议替代算法, 密钥长度可变,1-128bit, 速度较快
    1. RC4: 强度高，速度快, 不安全
    1. AES: 广泛使用的加密算法，速度快，安全级别高，已经成为美国加密标准, 目前 AES 标准的一个实现是 Rijndael 算法

**非对称加密**

1. 介绍：
    1. 加密和解密使用不同的密钥，一般只能加密很少量的数据，而且性能较差。
    1. 公钥可以公开，公钥加密的数据私钥可以解密，反之也是。
    1. 私钥需要秘密保管，私钥签名的数据，公钥可以验证签名。
    1. 非对称加密算法的保密性比较好，它消除了最终用户交换密钥的需要。
1. 用途：
    1. 验证身份，数字签名, 可以解决否认、伪造、篡改及冒充等问题
    1. 数据加密, 防止信息泄露和拦截
1. 常见算法：
    1. RSA: 基于大数运算和数学原理，可以让加密和解密使用不同的密钥。
    1. DSA: 数据签名算法，

### 身份认证方案

**[HTTP基本认证](http://zh.wikipedia.org/wiki/HTTP%E5%9F%BA%E6%9C%AC%E8%AE%A4%E8%AF%81)**

1. 介绍：用户名追加一个冒号然后串接上口令，并将得出的结果字符串再用Base64算法编码。
1. 优点：
    1. 浏览器支持广泛。
1. 缺点：
    1. 不能防止信息泄露，base64只是编码，不是加密。
    1. 不能防窃听
    1. 不能防重放 
    1. 不能防拖库
1. 使用场景：
    1. 在可信网络环境中可使用基本认证。
    1. 使用HTTPS做传输层。

**[HMAC](http://www.cnblogs.com/songhan/archive/2012/07/29/2613898.html)**

1. 介绍：
    1. 用哈希算法，以一个密钥和一个消息为输入，生成一个消息摘要作为输出。
    1. 消息认证码是基于密钥和消息摘要【hash】所获得的一个值，可用于数据源发认证和完整性校验。
1. 原理：
    1. ![图](http://pic002.cnblogs.com/images/2012/427162/2012072911095143.png)
    1. client要给server发送message，先用key和message加起来，然后哈希得出一个MAC
    1. 然后用户把明文message和MAC发给server
    1. server知道key，用同样的算法得到MAC，看和client请求的MAC是否一致
    1. 如果MAC一致，说明message是拥有key的人发送的，而且message没有被篡改
1. 优点：
    1. 实现了身份认证，实现了不可抵赖性
    1. 保证了数据完整性，达到了防篡改的效果
    1. HMAC与一般的加密重要的区别在于它具有“瞬时”性，即认证只在当时有效
1. 缺点：
    1. message是明文，不能防窃听
    1. 不能防重放
1. 应用：
    1. 挑战/响应（Challenge/Response）身份认证,如SIP，HTTP
    1. Cookie签名

**[HTTP摘要认证(Digest access authentication, rfc2069)](http://pic002.cnblogs.com/images/2012/427162/2012072911095143.png)**

1. 介绍
    1. 它在密码发出前，先对其应用哈希函数，这相对于HTTP基本认证发送明文而言，更安全。
    1. ![图](http://upload.wikimedia.org/math/f/6/1/f61f1d5cfada956cb3f0ec17c0a6cbe0.png)
1. 原理 
    1. client请求认证页面, 不提供用户名和密码
    1. server返回401应答
        1. realm：认证域, 明文，
        1. nonce: 随机数, 明文，只使用一次
    1. client再次发起请求
        1. 对用户名、认证域(realm)以及密码的合并值计算 MD5 哈希值，结果称为 HA1。
        1. 对HTTP方法以及URI的摘要的合并值计算 MD5 哈希值，例如，"GET" 和 "/dir/index.html"，结果称为 HA2。
        1. 对 HA1、服务器密码随机数(nonce)、请求计数(nc,防止重放)、客户端密码随机数(cnonce)、 HA2 的合并值计算 MD5得到response 值以及cnonce。
    1. server收到应答，因为服务器拥有与客户端同样的信息，因此服务器可以进行同样的计算，以验证客户端提交的 response 值的正确性。
1. 优点
    1. 密码明文不需要传输，所以明文不会被泄露,这样server可以不存明文密码，而是只存HA1。
    1. 可以客户端随机数cnonce，够防止选择明文攻击(劫持到密文后猜测加密算法及明文)。
    1. nonce允许包含时间戳, 过期后就失效，防止重放攻击。
    1. 服务器也可以维护一个最近发出的nonce的列表以防止nonce重用。
    1. 防监听，防重放, 防抵赖，身份认证
1. 缺点
    1. RFC 2617 中的许多安全选项都是可选的, 某些时候会降级为RFC 2616。
    1. 容易受到中间人攻击, 摘要访问认证没有提供任何机制帮助客户端验证服务器的身份。
    1. 使用HTTPS加密同时使用这些弱明文协议解决了许多摘要访问认证试图要防止的许多威胁。
    1. 使用md5是使用到了md5的不可逆性，但md5现在有可以攻击的方式，如穷举攻击(密码比较简单时)，字典攻击，
    1. 如何面对冲突攻击(不同明文哈希后相同)（rfc2617）。
    1. 不能保护弱密码
1. 其它说明
    1. 可以允许每一个nonce只使用一次，但这样就会迫使客户端在发送每个请求的时候重复认证过程
    1. nonce在生成后立刻过期是不行的，因为客户端将没有任何机会来使用这个nonce。 
    1. 客户端多次请求可以重用nonce,但得提供新的cnonce。在后续的请求中，nc比前一次要大。


[https/tls](http://baike.baidu.com/link?url=anOyJqzJMkRs9vEAhaAJF24WcdwY_uBLNrAGjxlZw6ywkx8ppnpWFe31gaRNcBga) 

1. 介绍：它是一个安全传输协议，但也可以进行身份认证。
    1. 加密传输数据: 服务端和客户端之间的所有通讯，都是加密的。
    1. 用于身份验证: 保证服务器就是他声称的服务器。
    1. 维护数据的完整性，确保数据在传输过程中不被改变。
    1. RC4, X509
1. 握手机制-简化版
    1. client要访问一个server, 知道server的域名domain
    2. client向server发起请求
        2.1. ssl版本号
        2.2. 加密算法类型
        2.3. 随机数
    3. server给client返回应答
        3.1 ssl版本号
        3.2 加密算法类型
        3.3 随机数
        3.4 自己的证书(公钥)
        3.5 随机数签名。 
    4. client验证服务端返回的应答
        4.1 证书是否过期
        4.2 发型证书的CA是否可靠（会和本地的可信任CA列表对比）
        4.3 server的公钥能否解开server返回的随机数签名 # 确认server有该证书的私钥
        4.4 server的证书授权的域名是否是server的域名
    5. client随机产生一个用于对称加密密钥，然后用server的公钥加密，发给Server
    6. server用自己三私钥解密出对称加密密钥。
    7. 后续的通信都用对称加密密钥进行加解密。
1. 优点：
    1. 防窃听
    1. 防重放
    1. 防中间人攻击，
    1. 保证数据完整性性
    1. 防止会话劫持
1. 缺点
    1. 不能防止信息泄露，拖库，只是保证传输层安全
    1. 一般不能用于客户端身份验证，需要配合http基本认证
    1. 建立连接速度慢

oauth

1. 介绍：OAuth允许用户提供一个令牌，而不是用户名和密码来访问他们存放在特定服务提供者的数据。每一个令牌授权一个特定的网站（例如，视频编辑网站)在特定的时段（例如，接下来的2小时内）内访问特定的资源（例如仅仅是某一相册中的视频）。
1. 原理：
    1. 用户访问客户端的网站，想操作自己存放在服务提供方的资源。
    1. 客户端向服务提供方请求一个临时令牌。
    1. 服务提供方验证客户端的身份后，授予一个临时令牌。
    1. 客户端获得临时令牌后，将用户引导至服务提供方的授权页面请求用户授权。在这个过程中将临时令牌和客户端的回调连接发送给服务提供方。
    1. 用户在服务提供方的网页上输入用户名和密码，然后授权该客户端访问所请求的资源。
    1. 授权成功后，服务提供方引导用户返回客户端的网页。
    1. 客户端根据临时令牌从服务提供方那里获取访问令牌 。
    1. 服务提供方根据临时令牌和用户的授权情况授予客户端访问令牌。
    1. 客户端使用获取的访问令牌访问存放在服务提供方上的受保护的资源。

**双因素认证，动态口令**

1. 介绍：
    1.简单来说，双因素身份认证就是通过你所知道再加上你所能拥有的这二个要素组合到一起才能发挥作用的身份认证系统, 如ATM。
    1. 目前主流的双因素认证系统是基于时间同步型，
    1. 市场占有率高的有DKEY双因素认证系统、RSA双因素认证系统等
    1. 主流的有硬件令牌、手机短信密码、USB KEY、混合型令牌（USBKEY+动态口令）, 密保卡，手机令牌
1. 优点
    1. 密码丢失后，黑客不能登录你的账户
1. 缺点
    1. 使用不方便

**用密钥加密用户密码**

1. 介绍：
    1. 本机生成一个密钥key存磁盘上，对称加密密钥。
    1. 创建用户时，用户提供password, 然后数据库里保存db_password = encrypt(key, hash(password))
    1. 这样黑客把数据库拖走后，因为没有key解开用db_password，所以用户密码还是安全的。
    1. 用户登录时提供密码password, 哈希后是hash(password), 然后uncrypt(key, db_password)，
        1. 两者比较，一致就是认证通过
        1. 不一致就是终止认证
1. 优点：
    1. 防止拖库
1. 缺点
    1. key丢了就完蛋了，谁也登录不上了。

**[Secure Remote Password protocol](http://srp.stanford.edu/design.html)**

1. 介绍
    1. 一个认证和密钥交换系统，它用来在不可靠的网络中保护口令和交换密钥。
    1. 通过消除了在网络上发送明文口令的需要，并且通过安全的密钥交换机制来使用加密，改进了安全性。
    1. 服务器不保存密码或密码的散列值, 防止字典攻击. 而只是保存验证因子(verifier).
    1. 客户端和服务器可以各自计算出一个会话秘钥(session key), 其值相同. 防止窃听和会话劫持.
    1. 好多游戏服务端用SRP认证，比如魔兽世界。

1. 优点
    1. 防窃听
    1. 防暴力破解，字典攻击, 弱口令也不容易被破解
    1. 即使口令数据库被公之于众，攻击者仍然需要一个庞大的字典去搜索来获得口令。
    1. 速度快，不需要证书和第三方认证机构
1. 缺点
    1. 浏览器不支持，得自己实现

原理

    N     一个安全的大质数, 比如N=2q+1,q 是一个素数
    g     一个以N为模的生成元，对任何X，有0 < X < N，存在一个值x，使得g^x % N == X。
    k     k = H(N,G) 在 SRP6 中 k = 3
    s     User’s Salt
    I     用户名
    p     明文密码
    H()   单向 hash 函数
    ^     求幂运算
    u     随机数
    a,b   保密的临时数字
    A,B   公开的临时数字
    x     私有密匙（从 p 和 s 计算得来）
    v     密码验证数字

    N和g的值必须由双方讨论来达成一致。它们可以被提前设置好，或者主机把它们发送给客户端。

    服务器存储如下信息
    x = H(s, p)               (s is chosen randomly)
    v = g^x                   (computes password verifier)

    服务器的数据库保存 {I, s, v} 整个验证流程如下:

    User -> Host:  I, A = g^a                  (标识自己是谁, a是随机数)
    Host -> User:  s, B = kv + g^b             (把salt发送给user, b是随机数)

            Both:  u = H(A, B)

            User:  x = H(s, p)                 (用户输入密码)
            User:  S = (B - kg^x) ^ (a + ux)   (计算会话密钥)
            User:  K = H(S)

            Host:  S = (Av^u) ^ b              (计算会话密钥)
            Host:  K = H(S)

    这样双方都有一个会话密钥S, 后续的消息传输可以用S做加解密，从而保证安全。
    为了完成认证过程，双方还得向对方证明自己拥有正确的S，
    S不能让第三方知道，所以不能直接传输给对方做比较，一个可能的办法是:

    User -> Host:  M = H(H(N) xor H(g), H(I), s, A, B, K)
    Host -> User:  H(A, M, K)

    双方需要做如下保障
        1. 如果客户端收到B == 0 (mod N) 或u == 0, 客户端停止认证。
        2. 如果服务器发现 A == 0 (mod N)则停止认证。
        3. 用户必须得证明自己拥有正确的K，否则服务器就会终止认证。

### 参考链接

部分内容是直接引用过来的

1. [HTTPS的七个误解（译文）](http://www.ruanyifeng.com/blog/2011/02/seven_myths_about_https.html)
1. [谈谈近期的安全事件](http://hi.baidu.com/ncaoz/item/3877495aefd2e43595eb0589)
1. [互联网系统密码安全](http://2013.cert.org.cn/download.php?name=%E4%BA%92%E8%81%94%E7%BD%91%E5%AF%86%E7%A0%81%E5%AE%89%E5%85%A8.pdf)
1. [没知识真可怕——应用密码学的笑话之MD5+Salt不安](http://blog.sina.com.cn/s/blog_77e8d1350100wfc7.html)