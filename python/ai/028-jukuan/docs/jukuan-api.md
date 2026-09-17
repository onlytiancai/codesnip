### 指数相关
#### 获取所有指数数据

获取平台支持的所有指数数据

**调用方法**

```
get_all_securities(types=['index'], date=None)
```

这里请在使用时注意防止未来函数。

**返回**  
- display_name # 中文名称  
- name # 缩写简称  
- start_date # 上市日期  
- end_date # 退市日期，如果没有退市则为2200-01-01  
- type # 类型，index(指数)

pandas.DataFrame, 比如:`get_all_securities(['index'])[:2]`返回:

||display_name|name|start_date|end_date|type|
|---|---|---|---|---|---|
|000001.XSHG|上证指数|SZZS|1991-07-15|2200-01-01|index|
|000002.XSHG|A股指数|AGZS|1992-02-21|2200-01-01|index|

#### 获取单支指数数据

获取单支指数的信息.

**调用方法**

```python
get_security_info(code)
```

**参数**

- code: 指数代码

**返回值**

- 一个对象, 有如下属性:

- display_name # 中文名称
- name # 缩写简称
- start_date # 上市日期, [datetime.date] 类型
- end_date # 退市日期，[datetime.date] 类型, 如果没有退市则为2200-01-01
- type # 类型，index(指数)

#### 获取指数成份股

获取一个指数给定日期在平台可交易的成分股列表，我们支持近600种股票指数数据，包括指数的行情数据以及成分股数据。为了避免未来函数，我们支持获取历史任意时刻的指数成分股信息。

**调用方法**

```
get_index_stocks(index_symbol, date=None)
```

**参数**  
- index_symbol, 指数代码  
- date: 查询日期, 一个字符串(格式类似’2015-10-15’)或者[datetime.date]/[datetime.datetime]对象, 可以是None, 使用默认日期. 这个默认日期在回测和研究模块上有点差别:

1. 回测模块: 默认值会随着回测日期变化而变化, 等于context.current_dt
2. 研究模块: 默认是今天

**返回**  
- 返回股票代码的list

**示例**

```
# 获取所有沪深300的股票
  codes= get_index_stocks('000300.XSHG')
  print(codes)
```

#### 历史行情数据

获取指数历史交易数据，**可以通过参数设置获取日k线、分钟k线数据**。获取数据的基本属性如下：

- open 时间段开始时价格
- close 时间段结束时价格
- low 最低价
- high 最高价
- volume 成交的指数数量
- money 成交的金额
- factor 前复权因子, 我们提供的价格都是前复权后的, 但是利用这个值可以算出原始价格, 方法是价格除以factor, 比如: `close/factor`
- high_limit 涨停价
- low_limit 跌停价
- avg 这段时间的平均价, 等于`money/volume`
- pre_close 前一个单位时间结束时的价格, 按天则是前一天的收盘价, 按分钟这是前一分钟的结束价格
- paused 布尔值, 这只指数是否停牌, 停牌时open/close/low/high/pre_close依然有值,都等于停牌前的收盘价, volume=money=0

**调用方法**

```
get_price(security, start_date='2015-01-01', end_date='2015-12-31', frequency='daily', fields=None, skip_paused=False, fq='pre')
```

**注：设定不同的unit参数，获取日K线或分钟k线，详情见参数。** 这里请在使用时注意防止未来函数.


**关于停牌**: 因为此API可以获取多只股票的数据, 可能有的股票停牌有的没有, 为了保持时间轴的一致, 我们默认没有跳过停牌的日期, 停牌时使用停牌前的数据填充(请看[SecurityUnitData]的paused属性). 如想跳过, 请使用 skip_paused=True 参数, 同时只取一只股票的信息

**参数**

- security: 一支指数代码或者一个指数代码的list  
- start_date: 字符串或者[datetime.datetime]/[datetime.date]对象, 开始时间, 默认是’2015-01-01’. 注意:

- 当取分钟数据时, 时间可以精确到分钟, 比如: 传入 `datetime.datetime(2015, 1, 1, 10, 0, 0)` 或者 `'2015-01-01 10:00:00'`.
- 当取分钟数据时, 如果只传入日期, 则日内时间是当日的 00:00:00.
- 当取天数据时, 传入的日内时间会被忽略

- end_date: 格式同上, 结束时间, 默认是’2015-12-31’, 包含此日期. **注意: 当取分钟数据时, 如果 end_date 只有日期, 则日内时间等同于 00:00:00, 所以返回的数据是不包括 end_date 这一天的**.  
- frequency: 单位时间长度, 几天或者几分钟, 现在支持’Xd’,’Xm’, ‘daily’(等同于’1d’), ‘minute’(等同于’1m’), X是一个正整数, 分别表示X天和X分钟(不论是按天还是按分钟回测都能拿到这两种单位的数据), 注意, 当X > 1时, field只支持[‘open’, ‘close’, ‘high’, ‘low’, ‘volume’, ‘money’]这几个标准字段. 默认值是daily  
- fields: 字符串list, 选择要获取的行情数据字段, 默认是None(表示[‘open’, ‘close’, ‘high’, ‘low’, ‘volume’, ‘money’]这几个标准字段), 支持[属性](https://www.joinquant.com/help/data/index#%E5%8E%86%E5%8F%B2%E8%A1%8C%E6%83%85%E6%95%B0%E6%8D%AE)里面的所有基本属性.  
- skip_paused: 是否跳过不交易日期(包括停牌, 未上市或者退市后的日期). 如果不跳过, 停牌时会使用停牌前的数据填充(具体请看[SecurityUnitData]的paused属性), 上市前或者退市后数据都为 nan, , 但要注意:

- 默认为 False

- 当 skip_paused 是 True 时, 只能取一只股票的信息  
- fq: 复权选项:

- `'pre'`: 前复权(根据’use_real_price’选项不同含义会有所不同, 参见[set_option](https://www.joinquant.com/api#set_option)), 默认是前复权
- `None`: 不复权, 返回实际价格
- `'post'`: 后复权

**返回**

- **请注意, 为了方便比较一只指数的多个属性, 同时也满足对比多只指数的一个属性的需求, 我们在security参数是一只指数和多只指数时返回的结构完全不一样**
- 如果是一支指数, 则返回[pandas.DataFrame]对象, 行索引是[datetime.datetime]对象, 列索引是行情字段名字, 比如’open’/’close’. 比如: `get_price('000300.XSHG')[:2]`返回:

||open|close|high|low|volume|money|
|---|---|---|---|---|---|---|
|2015-01-05 00:00:00|3566.09|3641.54|3669.04|3551.51|451198098.0|519849817448.0|
|2015-01-06 00:00:00|3608.43|3641.06|3683.23|3587.23|420962185.0|498529588258.0|

- 如果是多支指数, 则返回[pandas.Panel]对象, 里面是很多[pandas.DataFrame]对象, 索引是行情字段(open/close/…), 每个[pandas.DataFrame]的行索引是[datetime.datetime]对象, 列索引是指数代号. 比如`get_price(['000300.XSHG', '000001.XSHG'])['open'][:2]`返回:

||000300.XSHG|000001.XSHG|
|---|---|---|
|2015-01-05 00:00:00|3566.09|3258.63|
|2015-01-06 00:00:00|3608.43|3330.80|

**示例**

```python
# 获取一支指数
df = get_price('000001.XSHG') # 获取000001.XSHG的2015年的按天数据
df = get_price('000001.XSHG', start_date='2015-01-01', end_date='2015-02-01', frequency='minute', fields=['open', 'close']) # 获得000001.XSHG的2015年02月的分钟数据, 只获取open+close字段
df = get_price('000001.XSHG', start_date='2015-12-01 14:00:00', end_date='2015-12-02 12:00:00', frequency='1m') # 获得000001.XSHG的2015年12月1号14:00-2015年12月2日12:00的分钟数据

# 获取多只指数
panel =  get_price(get_index_stocks('000903.XSHG')) # 获取中证100的所有成分股的2015年的天数据, 返回一个[pandas.Panel]
df_open = panel['open']  # 获取开盘价的[pandas.DataFrame],  行索引是[datetime.datetime]对象, 列索引是指数代号
df_volume = panel['volume']  # 获取交易量的[pandas.DataFrame]

print(df_open['000001.XSHE']) # 获取平安银行的2015年每天的开盘价数据
```

### 场内基金相关

#### 获取所有基金数据

获取平台支持的所有基金数据

**调用方法**

```
get_all_securities(['fund'])
```

这里请在使用时注意防止未来函数。

**返回**  
- display_name # 中文名称  
- name # 缩写简称  
- start_date # 上市日期  
- end_date # 退市日期，如果没有退市则为2200-01-01  
- type # 类型， etf(ETF基金)，fja（分级A），fjb（分级B）,fjm(分级母基金），lof(场内交易开发基金），mmf(场内交易货币基金） [pandas.DataFrame], 比如:`get_all_securities(['fund'])[:2]`返回:

|display_name|name|start_date|end_date|type|
|---|---|---|---|---|
|150008.XSHE|国投瑞银沪深300指数分级-A|RHXK|2009-11-19|2200-01-01|
|150009.XSHE|国投瑞银沪深300指数分级-B|RHYJ|2009-11-19|2200-01-01|

#### 获取单支基金数据

获取单支基金的信息.

**调用方法**

```
get_security_info(code)
```

#### 场内基金份额数据

```python
from jqdata import *
finance.run_query(query(finance.FUND_SHARE_DAILY).filter(finance.FUND_SHARE_DAILY.date=='2019-05-23').limit(n))
```

描述：记录每日场内基金份额数据

**参数：**

- **query(finance.FUND_SHARE_DAILY)**：表示从finance.FUND_SHARE_DAILY这张表中查询每日场内基金份额数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：query简易教程
- **finance.FUND_SHARE_DAILY**：收录了每日场内基金份额数据，表结构和字段信息如下：

**字段设计**：

|名称|类型|描述|
|---|---|---|
|code|varchar(12)|基金代码|
|name|varchar(50）|基金简称|
|exchange_code|varchar(12)|交易市场编码，XSHG-上海证券交易所；XSHE-深圳证券交易所|
|date|date|日期|
|shares|bigint|基金份额（份）|

- **filter(finance.FUND_SHARE_DAILY.date==date)**：指定筛选条件，通过finance.FUND_SHARE_DAILY.date==date可以指定你想要查询的日期；除此之外，还可以对表中其他字段指定筛选条件；多个筛选条件用英文逗号分隔。
- **limit(n)**：限制返回的数据条数，n指定返回条数。

**返回结果：**

- 返回一个 dataframe，每一行对应数据表中的一条数据， 列索引是您所查询的字段名称

**注意：**

1. **为了防止返回数据量过大, 我们每次最多返回4000行**
2. 不能进行连表查询，即同时查询多张表的数据

**示例：**

```python
#查询2019-05-23的场内基金份额数据。
from jqdata import *
df=finance.run_query(query(finance.FUND_SHARE_DAILY).filter(finance.FUND_SHARE_DAILY.date=='2019-05-23').limit(10))
df

    id  code    name    exchange_code   date    shares
0    960881  150008.XSHE 瑞和小康    XSHE    2019-05-23  17749200
1    960882  150009.XSHE 瑞和远见    XSHE    2019-05-23  17749200
2    960883  150012.XSHE 中证100A  XSHE    2019-05-23  36139500
```

#### 获取etf跟踪指数信息

```
from jqdata import *
finance.run_query(query(finance.FUND_INVEST_TARGET).filter(finance.FUND_INVEST_TARGET.code== '510190.XSHG'))
```

- **query(finance.FUND_INVEST_TARGET)**：表示从finance.FUND_INVEST_TARGET这张表中查询etf跟踪指数信息，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：query简易教程
- **finance.FUND_INVEST_TARGET**：收录了ETF基金跟踪的指数信息，表结构和字段信息如下：

**字段设计**：

|名称|类型|描述|
|---|---|---|
|code|varchar(12)|基金代码|
|name|varchar(50）|基金简称|
|pub_date|DATE|公告日期|
|start_date|DATE|生效日期|
|end_date|DATE|失效日期(未失效则为空)|
|traced_index_name|varchar(100)|跟踪指数名称|
|traced_index_code|varchar(12)|跟踪指数代码(不支持的指数填充为空|

- **filter(finance.FUND_INVEST_TARGET.code==code)**：指定筛选条件，通过finance.FUND_INVEST_TARGET.code==code可以指定你想要查询的基金标的；除此之外，还可以对表中其他字段指定筛选条件；多个筛选条件用英文逗号分隔。
- **limit(n)**：限制返回的数据条数，n指定返回条数。

**返回结果：**

- 返回一个 dataframe，每一行对应数据表中的一条数据， 列索引是您所查询的字段名称

**注意：**

1. **为了防止返回数据量过大, 我们每次最多返回4000行**
2. 不能进行连表查询，即同时查询多张表的数据

**示例：**

```python
# 查询510190追踪的指数信息
from jqdata import *
q = query(finance.FUND_INVEST_TARGET).filter(finance.FUND_INVEST_TARGET.code== '510190.XSHG')
df = finance.run_query(q)
print(df)
>>>

    id         code   name    pub_date  start_date    end_date traced_index_name traced_index_code
0  930  510190.XSHG  上证50基  2010-10-20  2010-10-25  2023-02-23      上证龙头(000065)       000065.XSHG
1  742  510190.XSHG  上证50基  2023-02-23  2023-02-23        None      上证50(000016)       000016.XSHG
```

### 场外基金相关

#### 获取公募基金主体信息

```python
from jqdata import finance
finance.run_query(query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.main_code==main_code).limit(n))
```

描述：记录不同基金的主体信息

**参数：**

- **query(finance.FUND_MAIN_INFO)**：表示从finance.FUND_MAIN_INFO这张表中查询公募基金主体信息数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：sqlalchemy.orm.query.Query对象
- **finance.FUND_MAIN_INFO**：收录了公募基金主体信息数据，表结构和字段信息如下：

**字段设计**

|**字段**|**名称**|**类型**|
|---|---|---|
|main_code|基金主体代码|varchar(12)|
|name|基金名称|varchar(100)|
|advisor|基金管理人|varchar(100)|
|trustee|基金托管人|varchar(100)|
|operate_mode_id|基金运作方式编码|int|
|operate_mode|基金运作方式|varchar(32)|
|underlying_asset_type_id|投资标的类型编码|int|
|underlying_asset_type|投资标的类型|varchar(32)|
|start_date|成立日期|date|
|end_date|结束日期|date|
|invest_style_id|投资风格编码|int|
|invest_style|投资风格|varchar(32)|
|statistics_main_code|基金统计主代码(仅多份额基金存在此字段)|varchar(32)|

基金运作方式编码

|编码|基金运作方式|
|---|---|
|401001|开放式基金|
|401002|封闭式基金|
|401003|QDII|
|401004|FOF|
|401005|ETF|
|401006|LOF|
|401007|MOM|
|401008|基础设施基金|

基金类别编码

|编码|基金类别|
|---|---|
|402001|股票型|
|402002|货币型|
|402003|债券型|
|402004|混合型|
|402005|基金型|
|402006|贵金属|
|402007|封闭式|

- **filter(finance.FUND_MAIN_INFO.main_code==main_code)**：指定筛选条件，通过finance.FUND_MAIN_INFO.main_code==main_code可以指定你想要查询的基金主体代码；除此之外，还可以对表中其他字段指定筛选条件；多个筛选条件用英文逗号分隔。
- **limit(n)**：限制返回的数据条数，n指定返回条数。

**返回结果：**

- 返回一个 dataframe，每一行对应数据表中的一条数据， 列索引是您所查询的字段名称

**注意：**

1. **为了防止返回数据量过大, 我们每次最多返回4000行**
2. 不能进行连表查询，即同时查询多张表的数据

**示例：**

```python
#查询华夏成长公募基金主体信息数据,传入的基金代码无须添加后缀
from jqdata import finance
q=query(finance.FUND_MAIN_INFO).filter(finance.FUND_MAIN_INFO.main_code=='000001')
df=finance.run_query(q)
print(df)

  id main_code  name     advisor       trustee  operate_mode_id operate_mode  \
0   1    000001  华夏成长  华夏基金管理有限公司  中国建设银行股份有限公司           401001        开放式基金   

   underlying_asset_type_id underlying_asset_type  start_date end_date  
0                    402004                   混合型  2001-12-18     None  
```

#### 获取公募基金净值信息

```python
from jqdata import finance
finance.run_query(query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.code==code).limit(n))
```

描述：记录公募基金的净值数据

**参数：**

- **query(finance.FUND_NET_VALUE)**：表示从finance.FUND_NET_VALUE这张表中查询基金净值数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：[sqlalchemy.orm.query.Query对象](http://docs.sqlalchemy.org/en/rel_1_0/orm/query.html)
- **finance.FUND_NET_VALUE**：收录了基金净值数据，表结构和字段信息如下：

**字段设计**

|**字段**|**名称**|**类型**|**注释**|
|---|---|---|---|
|code|基金代码|varchar(12)||
|day|交易日|date||
|net_value|单位净值|decimal(20,6)|基金单位净值=（基金资产总值－基金负债）÷ 基金总份额|
|sum_value|累计净值|decimal(20,6)|累计单位净值＝单位净值＋成立以来每份累计分红派息的金额|
|factor|复权因子|decimal(20,6)|交易日最近一次分红拆分送股的复权因子|
|acc_factor|累计复权因子|decimal(20,6)|复权因子的累乘|
|refactor_net_value|累计复权净值|decimal(20,6)|单位净值*累计复权因子|

- **filter(finance.FUND_NET_VALUE.code==code)**：指定筛选条件，通过finance.FUND_NET_VALUE.code==code可以指定你想要查询的基金代码；除此之外，还可以对表中其他字段指定筛选条件；多个筛选条件用英文逗号分隔。
- **limit(n)**：限制返回的数据条数，n指定返回条数。

**返回结果：**

- 返回一个 dataframe，每一行对应数据表中的一条数据， 列索引是您所查询的字段名称

**注意：**

1. **为了防止返回数据量过大, 我们每次最多返回4000行**
2. 不能进行连表查询，即同时查询多张表的数据

**示例：**

```python
#查询华夏成长证券投资基金("000001")基金净值数据，传入的基金代码无需添加后缀
from jqdata import finance
q=query(finance.FUND_NET_VALUE).filter(finance.FUND_NET_VALUE.code=="000001").order_by(finance.FUND_NET_VALUE.day.desc()).limit(10)
df=finance.run_query(q)
print(df)

id    code         ...         acc_factor  refactor_net_value
0  20578245  000001         ...           5.580782            6.216991
1  20569568  000001         ...           5.580782            6.177926
```

#### 获取基金持股信息

```python
from jqdata import finance
finance.run_query(query(finance.FUND_PORTFOLIO_STOCK).filter(finance.FUND_PORTFOLIO_STOCK.code==code).limit(n))
```

描述：统计基金季度报表、半年度报表和年度报表披露的股票持仓数据  
数据更新时间：报告披露当天将数据入库，但整体的检查确认工作一般要在报告披露期后一周的时间内完成，即数据在披露一周后可以查询到更新的数据。  
**参数：**

- **query(finance.FUND_PORTFOLIO_STOCK)**：表示从finance.FUND_PORTFOLIO_STOCK这张表中查询基金持仓股票组合数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：sqlalchemy.orm.query.Query对象
- **finance.FUND_PORTFOLIO_STOCK**：收录了基金持仓股票组合数据，表结构和字段信息如下：

**字段设计**

|字段名称|中文名称|字段类型|
|---|---|---|
|code|基金代码|varchar(12)|
|period_start|开始日期|date|
|period_end|报告期|date|
|pub_date|公告日期|date|
|report_type_id|报告类型编码|int|
|report_type|报告类型|varchar(32)|
|rank|持仓排名|int|
|symbol|股票代码|varchar(32)|
|name|股票名称|varchar(100)|
|shares|持有股票股数|decimal(20,4)|
|market_cap|持有股票的市值|decimal(20,4)|
|proportion|占净值比例|decimal(10,4)|

- **filter(finance.FUND_PORTFOLIO_STOCK.code==code)**：指定筛选条件，通过finance.FUND_PORTFOLIO_STOCK.code==code可以指定你想要查询的基金代码；除此之外，还可以对表中其他字段指定筛选条件；多个筛选条件用英文逗号分隔。
- **limit(n)**：限制返回的数据条数，n指定返回条数。

**返回结果：**

- 返回一个 dataframe，每一行对应数据表中的一条数据， 列索引是您所查询的字段名称

#### 获取基金持有的债券信息

```python
from jqdata import finance
finance.run_query(query(finance.FUND_PORTFOLIO_BOND).filter(finance.FUND_PORTFOLIO_BOND.code==code).limit(n))
```

描述：记录公募基金按季度公布的债券组合，为债券投资者提供一些参照

**参数：**

- **query(finance.FUND_PORTFOLIO_BOND)**：表示从finance.FUND_PORTFOLIO_BOND这张表中查询基金持仓债券组合数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：sqlalchemy.orm.query.Query对象
- **finance.FUND_PORTFOLIO_BOND**：收录了基金持仓债券组合数据，表结构和字段信息如下：

**字段设计**

|字段名称|中文名称|字段类型|
|---|---|---|
|code|基金代码|varchar(12)|
|period_start|开始日期|date|
|period_end|报告期|date|
|pub_date|公告日期|date|
|report_type_id|报告类型编码|int|
|report_type|报告类型|varchar(32)|
|rank|持仓排名|int|
|symbol|债券代码|varchar(32)|
|name|债券名称|varchar(100)|
|shares|持有债券数量|decimal(20,4)|
|market_cap|持有债券的市值|decimal(20,4)|
|proportion|占净值比例|decimal(10,4)|

- **filter(finance.FUND_PORTFOLIO_BOND.code==code)**：指定筛选条件，通过finance.FUND_PORTFOLIO_BOND.code==code可以指定你想要查询的基金代码；除此之外，还可以对表中其他字段指定筛选条件；多个筛选条件用英文逗号分隔。
- **limit(n)**：限制返回的数据条数，n指定返回条数。

**返回结果：**

- 返回一个 dataframe，每一行对应数据表中的一条数据， 列索引是您所查询的字段名称

#### 获取基金资产组合概况

```python
from jqdata import finance
finance.run_query(query(finance.FUND_PORTFOLIO).filter(finance.FUND_PORTFOLIO.code==code).limit(n))
```

描述：基金资产组合概况

**参数：**

- **query(finance.FUND_PORTFOLIO)**：表示从finance.FUND_PORTFOLIO这张表中查询基金资产组合概况数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：[sqlalchemy.orm.query.Query对象](http://docs.sqlalchemy.org/en/rel_1_0/orm/query.html)
- **finance.FUND_PORTFOLIO**：收录了基金资产组合概况数据，表结构和字段信息如下：

**字段设计**

|字段名称|中文名称|字段类型|
|---|---|---|
|code|基金代码|varchar(12)|
|name|基金名称|varchar(80)|
|period_start|开始日期|date|
|period_end|报告期|date|
|pub_date|公告日期|date|
|report_type_id|报告类型编码|int|
|report_type|报告类型|varchar(32)|
|equity_value|权益类投资金额|decimal(20,4)|
|equity_rate|权益类投资占比|decimal(10,4)|
|stock_value|股票投资金额|decimal(20,4)|
|stock_rate|股票投资占比|decimal(10,4)|
|fixed_income_value|固定收益投资金额|decimal(20,4)|
|fixed_income_rate|固定收益投资占比|decimal(10,4)|
|precious_metal_value|贵金属投资金额|decimal(20,4)|
|precious_metal_rate|贵金属投资占比|decimal(10,4)|
|derivative_value|金融衍生品投资金额|decimal(20,4)|
|derivative_rate|金融衍生品投资占比|decimal(10,4)|
|buying_back_value|买入返售金融资产金额|decimal(20,4)|
|buying_back_rate|买入返售金融资产占比|decimal(10,4)|
|deposit_value|银行存款和结算备付金合计|decimal(20,4)|
|deposit_rate|银行存款和结算备付金合计占比|decimal(10,4)|
|others_value|其他资产|decimal(20,4)|
|others_rate|其他资产占比|decimal(10,4)|
|total_asset|总资产合计|decimal(20,4)|
|CDR_value|存托凭证(QDII专用)|decimal(20,4)|
|CDR_rate|存托凭证占总值比例(QDII专用)|decimal(20,4)|
|fund_value|基金投资(QDII专用)|decimal(20,4)|
|fund_rate|基金投资占总值比例(QDII专用)|decimal(20,4)|
|mm_inst_value|货币市场工具(QDII专用)||
|mm_inst_rate|货币市场工具占总值比例(QDII专用)|decimal(20,4)|
|REIT_value|房地产信托|decimal(20,4)|
|REIT_rate|房地产信托占总值比例|decimal(20,4)|
|preferred_value|优先股|decimal(20,4)|
|preferred_rate|优先股占总值比例|decimal(20,4)|

#### 获取基金财务指标信息

```python
from jqdata import finance
finance.run_query(query(finance.FUND_FIN_INDICATOR).filter(finance.FUND_FIN_INDICATOR.code==code).limit(n))
```

描述：基金财务指标表

**参数：**

- **query(finance.FUND_FIN_INDICATOR)**：表示从finance.FUND_FIN_INDICATOR这张表中查询基金财务指标数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：[sqlalchemy.orm.query.Query对象](http://docs.sqlalchemy.org/en/rel_1_0/orm/query.html)
- **finance.FUND_FIN_INDICATOR**：收录了基金财务指标数据，表结构和字段信息如下：

**字段设计**

|**字段**|**名称**|**类型**|
|---|---|---|
|code|基金代码|varchar(12)|
|name|基金名称|varchar(80)|
|period_start|开始日期|date|
|period_end|结束日期|date|
|pub_date|公告日期|date|
|report_type_id|报告类型编码|int|
|report_type|报告类型|varchar(32)|
|profit|本期利润||
|adjust_profit|本期利润扣减本期公允价值变动损益后的净额||
|avg_profit|加权平均份额本期利润||
|avg_roe|加权平均净值利润率||
|profit_available|期末可供分配利润||
|profit_avaialbe_per_share|期末可供分配份额利润||
|total_tna|期末基金资产净值||
|nav|期末基金份额净值||
|adjust_nav|期末还原后基金份额累计净值||
|nav_growth|本期净值增长率||
|acc_nav_growth|累计净值增长率||
|adjust_nav_growth|扣除配售新股基金净值增长率||
|total_asset|期末基金资产总值|

#### 获取基金分红拆分合并信息

```python
from jqdata import finance
finance.run_query(query(finance.FUND_DIVIDEND).filter(finance.FUND_DIVIDEND.code==code).limit(n))
```

描述：记录基金分红、拆分和合并的方案

**参数：**

- **query(finance.FUND_DIVIDEND)**：表示从finance.FUND_DIVIDEND这张表中查询基金分红拆分合并数据，还可以指定所要查询的字段名，格式如下：query(库名.表名.字段名1，库名.表名.字段名2），多个字段用逗号分隔进行提取；query函数的更多用法详见：[sqlalchemy.orm.query.Query对象](http://docs.sqlalchemy.org/en/rel_1_0/orm/query.html)
- **finance.FUND_DIVIDEND**：收录了基金分红拆分合并数据，表结构和字段信息如下：

**字段设计**

|**字段**|**名称**|**类型**|
|---|---|---|
|code|基金代码|varchar(12)|
|name|基金名称|varchar(80)|
|pub_date|公布日期|date|
|event_id|事项类别|int|
|event|事项名称|varchar(100)|
|distribution_date|分配收益日|date|
|process_id|方案进度编码|int|
|process|方案进度|varchar(100)|
|proportion|派现比例|decimal(20,8)|
|split_ratio|分拆（合并、赠送）比例|decimal(20,8)|
|record_date|权益登记日|date|
|ex_date|除息日|date|
|fund_paid_date|基金红利派发日|date|
|redeem_date|再投资赎回起始日|date|
|dividend_implement_date|分红实施公告日|dated|
|dividend_cancel_date|取消分红公告日|date|
|otc_ex_date|场外除息日|date|
|pay_date|红利派发日|date|
|new_share_code|新增份额基金代码|varchar(10)|
|new_share_name|新增份额基金名称|varchar(100)|

事项类别编码 404

|**编码**|**名称**|
|---|---|
|404001|基金分红|
|404002|基金分拆|
|404003|基金合并|
|404004|基金赠送|
|404005|分级基金折算|

基金分红拆分合并进度编码 405

|**编码**|**名称**|
|---|---|
|405001|分红预案|
|405002|实施方案|
|405003|取消折算|