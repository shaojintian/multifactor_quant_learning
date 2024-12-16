hft_c++_system
这是一个高频交易系统的C++代码库，包含了各种与交易相关的类和函数。以下是该代码库的主要组成部分：

文件结构
include/：包含头文件，定义了各种枚举类型、映射表、函数和类。
src/：包含源文件，实现了头文件中定义的类和函数。
CMakeLists.txt：CMake构建脚本，用于编译和构建项目。
README.md：项目的说明文档。

主要功能
定义了交易所类型、连接状态、报价模式、订单方向、订单类型、订单有效期、流动性、货币类型等枚举类型。
提供了将ISO 8601格式的字符串转换为Poco::DateTime对象的函数。
提供了从字符串获取报价模式的函数。
`定义了多个与交易相关的类，包括Timestamped（时间戳类）、MarketQuote（市场报价类）、Quote（报价类）、TwoSidedQuote（双向报价类）、FairValue（公平价值类）、QuotingParameters（报价参数类）、NewOrder（新订单类）、ReplaceOrder（替换订单类）、CancelOrder（取消订单类）、Trade（交易类）和Skew（偏斜类）。`