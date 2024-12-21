use std::time::SystemTime;
use plotters::prelude::*;
use rand::Rng;
// 本地模拟as mm
// 策略配置
#[derive(Clone)]
struct Config {
    gamma: f64,         // 风险厌恶系数
    k: f64,            // 订单到达率参数
    sigma: f64,        // 波动率
    initial_price: f64, // 初始价格
    position_limit: f64,// 持仓限制
    terminal_time: f64, // 终止时间(秒)
    A: f64, // 交易概率系数:P=A⋅e −λ
    dt: f64, // 时间步长
}

// 市场状态
struct MarketState {
    current_price: f64,
    position: f64,
    start_time: SystemTime,
    cash: f64,
}

// 报价
struct Quote {
    bid: f64,
    ask: f64,
    timestamp: SystemTime,
}

// 主策略结构
struct AvellanedaStoikov {
    config: Config,
    state: MarketState,
}

impl AvellanedaStoikov {
    fn new(config: Config) -> Self {
        let state = MarketState {
            current_price: config.initial_price,
            position: 0.0,
            start_time: SystemTime::now(),
        };
        
        AvellanedaStoikov {
            config,
            state,
        }
    }

    // 计算预留价格
    fn calculate_reservation_price(&self) -> f64 {
        let elapsed = self.state.start_time.elapsed().unwrap().as_secs_f64();
        let time_remaining = self.config.terminal_time - elapsed;
        
        if time_remaining <= 0.0 {
            return self.state.current_price;
        }

        self.state.current_price - 
            self.state.position * 
            self.config.gamma * 
            self.config.sigma.powi(2) * 
            time_remaining
    }

    // 计算最优价差
    fn calculate_optimal_spread(&self) -> f64 {
        let elapsed = self.state.start_time.elapsed().unwrap().as_secs_f64();
        let time_remaining = self.config.terminal_time - elapsed;
        
        if time_remaining <= 0.0 {
            return 0.0;
        }

        self.config.gamma * self.config.sigma.powi(2) * time_remaining +
            (2.0 / self.config.gamma) * (1.0 + self.config.gamma / self.config.k).ln()
    }

    // 获取报价
    fn get_quotes(&self) -> Quote {
        let r = self.calculate_reservation_price();
        let s = self.calculate_optimal_spread();
        let half_spread = s / 2.0;

        Quote {
            bid: r - half_spread,
            ask: r + half_spread,
            timestamp: SystemTime::now(),
        }
    }

    // 更新市场状态
    fn update_state(&mut self, price: f64, new_position: f64) {
        // 记录交易历史
        


        // 更新当前状态
        self.state.position = new_position;
        self.state.current_price = price; // 更新当前价格
    }

    // 处理成交
    // ... existing code ...

    // 处理成交!!!!!!
    fn handle_trade(&mut self, price: f64, size: f64, is_buy: bool) {
        let position_delta = if is_buy { size } else { -size };
        let new_position = self.state.position + position_delta;
        
        // 检查持仓限制
        if new_position.abs() > self.config.position_limit {
            println!("订单超出持仓限制，未执行: 价格 {:.2}, 数量 {:.2}, 当前持仓: {:.2}", price, size, self.state.position);
            return; // 不执行交易
        }
        self.update_state(price, new_position);
        // 模拟挂单交易
        let order_type = if is_buy { "买入" } else { "卖出" };
        println!("执行交易{} 订单: 价格 {:.2}, 数量 {:.2}, 新持仓: {:.2}", order_type, price, size, new_position);
    }

    // 分别计算交易概率
    fn calculate_trade_probability(&self,quotes:&Quote) -> (f64,f64){
        let delta_a = quotes.ask - self.state.current_price;
        let lambda_a = self.config.A * (-self.config.k *delta_a ).exp(); // Rust中使用exp函数计算e的幂
        let prob_ask = lambda_a;

        let delta_b = self.state.current_price - quotes.bid;
        let lambda_b = self.config.A * (-self.config.k * delta_b).exp();
        let prob_bid = lambda_b;
        // 使用泊松分布公式 P = A * e^(-λ)
        //λ = k * detla
        //let probability = self.condig.A * (-lambda).exp();
        //println!("quotes.bid{:.2},quotes.ask{:.2},lambda_a{:.3},prob_bid{:.3},prob_ask{:.3}",quotes.bid,quotes.ask,lambda_a,prob_bid,prob_ask);
        (prob_bid,prob_ask)
        
    }

    // 风险度量
    fn calculate_risk_metrics(&self) -> (f64, f64) {
        let position_risk = self.state.position.abs() * 
            self.config.sigma * 
            self.state.current_price;
            
        let spread_risk = self.calculate_optimal_spread() * 
            self.state.current_price;
            
        (position_risk, spread_risk)
    }

    // 执行交易如果符合概率
    fn execute_trade_if_probable(&mut self, quotes: &Quote,trade_size: f64) {
        let new_price = self.state.current_price;
        // 计算交易概率
        let (prob_bid,prob_ask) = self.calculate_trade_probability(quotes);
        let mut rng = rand::thread_rng();
        let random_value_bid: f64 = rng.gen::<f64>(); // 生成一个 [0.0, 1.0) 的随机数
        let random_value_ask: f64 = rng.gen::<f64>(); // 生成一个 [0.0, 1.0) 的随机数

        // 根据随机数和交易概率决定是否行交易
        if random_value_bid < prob_bid && random_value_ask > prob_ask{
            // 执行买入
            self.handle_trade(new_price, trade_size, true); // 执行买入
            println!("Executed Buy: Price: {:.2}, Size: {:.2}", new_price, trade_size);
        } else if random_value_bid > prob_bid && random_value_ask < prob_bid{
            // 执行卖出
            self.handle_trade(new_price, trade_size, false); // 执行卖出
            println!("Executed Sell: Price: {:.2}, Size: {:.2}", new_price, trade_size);
        } else if random_value_bid > prob_bid && random_value_ask > prob_bid{
            //什么都不做
            println!("No trade executed: Price: {:.2}, Size: {:.2} ,--prob_bid:{:.2}%----,prob_ask:{:.2}%, random_value_bid:{:.2}%,random_value_ask:{:.2}%", new_price, trade_size,prob_bid*100.0,prob_ask*100.0,random_value_bid*100.0,random_value_ask *100.0);
        }else{
            //同时买卖
            self.handle_trade(new_price, trade_size, true); // 执行买入
            self.handle_trade(new_price, trade_size, false); // 执行卖出
            println!("Executed Buy and Sell: Price: {:.2}, Size: {:.2}", new_price, trade_size);
        }
        
    }
}

// 模拟市场环境
struct MarketSimulator {
    current_price: f64,
    volatility: f64,
}

// struct SimulatorConfig {
//     initial_price: f64,
//     volatility:f64
// }

impl MarketSimulator {
    fn new(initial_price: f64, volatility: f64) -> Self {
        MarketSimulator {
            current_price: initial_price,
            volatility,
        }
    }

    // 模拟价格更新
    fn update_price(&mut self) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let random_walk = rng.gen::<f64>() * 2.0 - 1.0;
        self.current_price *= 1.0 + self.volatility * random_walk;
        self.current_price
    }
}

fn plot_results(prices: &[f64], bids: &[f64], asks: &[f64], positions: &[f64], pnl:[f64] filename: &str) {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Market Making Strategy Results", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..prices.len() as u32, 
            *prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()..*prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // 绘制价格、买���、卖出和持仓
    chart.draw_series(LineSeries::new(prices.iter().enumerate().map(|(x, &y)| (x as u32, y)), &RED)).unwrap()
        .label("Price")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &RED));
    
    chart.draw_series(LineSeries::new(bids.iter().enumerate().map(|(x, &y)| (x as u32, y)), &BLUE)).unwrap()
        .label("Bid")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &BLUE));
    
    chart.draw_series(LineSeries::new(asks.iter().enumerate().map(|(x, &y)| (x as u32, y)), &GREEN)).unwrap()
        .label("Ask")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &GREEN));
    
    chart.draw_series(LineSeries::new(positions.iter().enumerate().map(|(x, &y)| (x as u32, y)), &BLACK)).unwrap()
        .label("Position")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &BLACK));

    chart.draw_series(LineSeries::new(positions.iter().enumerate().map(|(x, &y)| (x as u32, y)), &BLACK)).unwrap()
        .label("PNL")
        .legend(|(x, y)| PathElement::new(vec![(x, y)], &YELLOW));

}

// 计算 PnL 的辅助函数
fn calculate_pnl(&self, current_price: f64) -> f64 {
    let pnl = self.cash + current_price * self.state.position;
    pnl
        
}

pub fn main() {
    // 策略配置
    let config = Config {
        gamma: 0.1,           // 风险厌恶系数
        k: 1.5,              // 订单到达率参数
        sigma: 0.2,          // 年化波动率
        initial_price: 100.0, // 初始价格
        position_limit: 100.0,// 持仓限制
        terminal_time: 3600.0,// 1小时
        A:200.0,
        dt:1.0/3600.0 //config.dt = 1.0 / config.terminal_time;
    };

    // 初始化策略
    let mut strategy = AvellanedaStoikov::new(config.clone());
    let initial_price = 100.0;
    let simulator_volatility = 0.001;
    let mut simulator = MarketSimulator::new(initial_price, simulator_volatility);

    // 数据收集
    let mut prices = Vec::new();
    let mut bids = Vec::new();
    let mut asks = Vec::new();
    let mut positions = Vec::new();
    let mut pnl = Vec::new();

    // 模拟交易循环1h
    let simulation_time = config.terminal_time as usize;
    for _ in 0..simulation_time {  // 每秒一次报价
        // 更新市场价格
        let new_price = simulator.update_price();
        strategy.update_state(new_price, strategy.state.position);

        // 获取新的报价
        let quotes = strategy.get_quotes();

        //执行概率交易
        

        // 收集数据
        prices.push(new_price);
        bids.push(quotes.bid);
        asks.push(quotes.ask);
        positions.push(strategy.state.position);
        pnl.push(calculate_pnl(&pnl, new_price));
        
        // 计算风险指标
        let (position_risk, _spread_risk) = strategy.calculate_risk_metrics();

        // 执行交易
        let trade_size = new_price * 0.5 / 100.0; //[0.1%,0.5%]
        strategy.execute_trade_if_probable(&quotes, trade_size);
        
        // 打印状态
        println!("Price: {:.3}, Bid: {:.3}, Ask: {:.3}, Position: {:.3}, Risk: {:.3}", 
            new_price,
            quotes.bid,
            quotes.ask,
            strategy.state.position,
            position_risk
        );

        // 模拟等待1秒
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    //
    plot_results(&prices, &bids, &asks, &positions, &pnl,"market_making_results.png");
}