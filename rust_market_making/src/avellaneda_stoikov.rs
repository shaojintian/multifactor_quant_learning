use std::time::SystemTime;

// 策略配置
struct Config {
    gamma: f64,         // 风险厌恶系数
    k: f64,            // 订单到达率参数
    sigma: f64,        // 波动率
    initial_price: f64, // 初始价格
    position_limit: f64,// 持仓限制
    terminal_time: f64, // 终止时间(秒)
}

// 市场状态
struct MarketState {
    current_price: f64,
    position: f64,
    start_time: SystemTime,
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
    fn update_state(&mut self, new_price: f64, new_position: f64) {
        self.state.current_price = new_price;
        self.state.position = new_position;
    }

    // 处理成交
    fn handle_trade(&mut self, price: f64, size: f64, is_buy: bool) {
        let position_delta = if is_buy { size } else { -size };
        let new_position = self.state.position + position_delta;
        
        // 检查持仓限制
        if new_position.abs() <= self.config.position_limit {
            self.update_state(price, new_position);
        }
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
}

// 模拟市场环境
struct MarketSimulator {
    current_price: f64,
    volatility: f64,
}

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

fn main() {
    // 策略配置
    let config = Config {
        gamma: 0.1,           // 风险厌恶系数
        k: 1.5,              // 订单到达率参数
        sigma: 0.2,          // 年化波动率
        initial_price: 100.0, // 初始价格
        position_limit: 100.0,// 持仓限制
        terminal_time: 3600.0,// 1小时
    };

    // 初始化策略
    let mut strategy = AvellanedaStoikov::new(config);
    let mut simulator = MarketSimulator::new(100.0, 0.001);

    // 模拟交易循环
    for _ in 0..3600 {  // 每秒一次报价
        // 更新市场价格
        let new_price = simulator.update_price();
        strategy.update_state(new_price, strategy.state.position);

        // 获取新的报价
        let quotes = strategy.get_quotes();
        
        // 计算风险指标
        let (position_risk, spread_risk) = strategy.calculate_risk_metrics();

        // 打印状态
        println!("Price: {:.2}, Bid: {:.2}, Ask: {:.2}, Position: {:.2}, Risk: {:.2}", 
            new_price,
            quotes.bid,
            quotes.ask,
            strategy.state.position,
            position_risk
        );

        // 模拟等待1秒
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}