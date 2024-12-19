//dawd
#[derive(Debug)]
struct Config {
    gamma: f64,         // 风险厌恶系数
    k: f64,            // 订单到达率参数
    sigma: f64,        // 波动率
    initial_price: f64, // 初始价格
    position_limit: f64,// 持仓限制
    terminal_time: f64, // 终止时间(秒)
}

fn main(){
    let config = Config {
        gamma: 0.1,
        k: 0.5,
        sigma: 0.2,
        initial_price: 100.0,
        position_limit: 1000.0,
        terminal_time: 100.0,
    };
    println!("{:#?}",config);
}