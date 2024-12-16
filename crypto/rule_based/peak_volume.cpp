#include <iostream>
#include <vector>
#include <numeric>
#include <thread>
#include <chrono>
#include <algorithm>

using namespace std;




// 设置时间窗口大小和成交额倍数阈值
const int time_window = 100;  // 100个bar
const double threshold = 100.0;  // 成交额是均值的100倍

// 策略函数
bool trade(auto& records) {
    if (records.size() < time_window + 1) {
        return false;  // 如果数据不足101个bar，无法进行策略计算
    }

    // 提取过去100个bar的成交量数据（不包含最新的成交量）
    vector<double> volumes;
    for (int i = records.size() - time_window - 1; i < records.size() - 1; ++i) {
        volumes.push_back(records[i].Volume);
    }

    // 计算过去100个bar的成交量均值
    double mean_vol = accumulate(volumes.begin(), volumes.end(), 0.0) / volumes.size();

    // 当前的成交量（最新3分钟成交量）
    double current_vol = records.back().Volume;

    // 检查当前成交额是否是均值的100倍以上
    if (current_vol > mean_vol * threshold) {
        return true;  // 符合轮动条件
    } else {
        return false;  // 不符合轮动条件
    }
}

// 执行策略
void execute_trade() {
    // 获取k线数据
    auto records = exchange.GetRecords();

    // 判断是否符合轮动条件
    if (trade(records)) {
        // 执行买入操作（这个买入操作需要根据你的API来实现）
        exchange.Buy(-1, 100);
        cout << "符合轮动条件，执行策略" << endl;
        // 在这里执行买入操作，比如调用API执行订单
    } else {
        cout << "不符合轮动条件" << endl;
    }
}

int main() {
    while (true) {
        execute_trade();
        // 每3分钟检查一次
        this_thread::sleep_for(chrono::minutes(3));  // 每3分钟检查一次
    }

    return 0;
}
