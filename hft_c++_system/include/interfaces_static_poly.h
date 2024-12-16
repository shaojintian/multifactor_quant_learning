#include <iostream>
#include <vector>
#include <optional>
#include "models.h"
#include "Poco/BasicEvent.h"

using json = nlohmann::json;

namespace Interfaces
{
  // 定义接口
  class IMarketDataGateway {
  public:
    virtual Poco::BasicEvent<json> market_quote = 0;
  };

  class IExchangeDetailsGateway {
  public:
    virtual std::string pair = 0;
    virtual double maker_fee = 0;
    virtual double taker_fee = 0;
    virtual double min_tick_increment = 0;
    virtual double min_size_increment = 0;
    virtual double face_value = 0;
    virtual unsigned int max_leverage = 0;
  };

  class IOrderEntryGateway {
  public:
    virtual std::string generate_client_id() = 0;
    virtual void batch_send_order(std::vector<Models::NewOrder> &orders) = 0;
    virtual void batch_cancel_order(std::vector<Models::CancelOrder> &cancels) = 0;
    virtual void batch_replace_order(std::vector<Models::ReplaceOrder> &replaces) = 0;
    virtual unsigned int cancel_all() = 0;
    virtual std::optional<json> open_orders() = 0;
    virtual Poco::BasicEvent<Models::Trade> trade = 0;
    virtual Poco::BasicEvent<long> n_orders = 0;
  };

  class IPositionGateway {
  public:
    virtual std::optional<json> get_latest_position() = 0;
    virtual std::optional<json> get_latest_margin() = 0;
  };

  class IRateLimitMonitor {
  public:
    virtual void update_rate_limit(int limit, int remain, Poco::DateTime next_reset) = 0;
    virtual bool is_rate_limited() = 0;
  };

  // 使用CRTP实现的基类模板
  template <typename Derived>
  class MarketDataGatewayBase : public IMarketDataGateway {
  public:
    Poco::BasicEvent<json> market_quote;
  };

  template <typename Derived>
  class ExchangeDetailsGatewayBase : public IExchangeDetailsGateway {
  public:
    std::string pair;
    double maker_fee;
    double taker_fee;
    double min_tick_increment;
    double min_size_increment;
    double face_value;
    unsigned int max_leverage;

    ExchangeDetailsGatewayBase(const Derived &details) {
      pair = details.pair;
      maker_fee = details.maker_fee;
      taker_fee = details.taker_fee;
      min_tick_increment = details.min_tick_increment;
      min_size_increment = details.min_size_increment;
      face_value = details.face_value;
      max_leverage = details.max_leverage;
    }
  };

  template <typename Derived>
  class OrderEntryGatewayBase : public IOrderEntryGateway {
  public:
    std::string generate_client_id() override {
      return static_cast<Derived*>(this)->generate_client_id();
    }

    void batch_send_order(std::vector<Models::NewOrder> &orders) override {
      static_cast<Derived*>(this)->batch_send_order(orders);
    }

    void batch_cancel_order(std::vector<Models::CancelOrder> &cancels) override {
      static_cast<Derived*>(this)->batch_cancel_order(cancels);
    }

    void batch_replace_order(std::vector<Models::ReplaceOrder> &replaces) override {
      static_cast<Derived*>(this)->batch_replace_order(replaces);
    }

    unsigned int cancel_all() override {
      return static_cast<Derived*>(this)->cancel_all();
    }

    std::optional<json> open_orders() override {
      return static_cast<Derived*>(this)->open_orders();
    }

    Poco::BasicEvent<Models::Trade> trade;
    Poco::BasicEvent<long> n_orders;
  };

  template <typename Derived>
  class PositionGatewayBase : public IPositionGateway {
  public:
    std::optional<json> get_latest_position() override {
      return static_cast<Derived*>(this)->get_latest_position();
    }

    std::optional<json> get_latest_margin() override {
      return static_cast<Derived*>(this)->get_latest_margin();
    }
  };

  template <typename Derived>
  class RateLimitMonitorBase : public IRateLimitMonitor {
  public:
    void update_rate_limit(int limit, int remain, Poco::DateTime next_reset) override {
      static_cast<Derived*>(this)->update_rate_limit(limit, remain, next_reset);
    }

    bool is_rate_limited() override {
      return static_cast<Derived*>(this)->is_rate_limited();
    }
  };
}