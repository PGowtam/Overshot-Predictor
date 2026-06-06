#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <map>
#include <deque>

extern "C" {

    struct Trade {
        int order_id;
        int direction;
        double original_limit;
        double limit_price;
        double tp_price;
        double sl_price;
        double brick_size;
        int delay_mins;
        int state; // 0=WAITING_FOR_TOUCH, 1=STABILIZING, 2=ACTIVE
        int64_t created_t_msc;
        int64_t touch_t_msc;
        int64_t activation_t_msc;
        bool filled;
        int64_t fill_t_msc;
        int64_t exit_t_msc;
        double pnl_R;
        int result_code; // 0=Win, 1=Loss, 2=Invalidated_Missed, 3=Invalidated_Stabilization, 4=Expired/Timeout
    };

    struct Brick {
        double open;
        double close;
        double high;
        double low;
        int uptrend;
        int64_t time_msc;
    };

    class RenkoBuilder {
    public:
        double brick_size;
        double current_price;
        std::vector<Brick> bricks;

        RenkoBuilder(double day_open, double b_size) {
            brick_size = b_size;
            current_price = day_open;
        }

        std::vector<Brick> update_tick(double price, int64_t t_msc) {
            std::vector<Brick> new_bricks;
            while (true) {
                if (price >= current_price + brick_size) {
                    Brick b;
                    b.open = current_price;
                    b.close = current_price + brick_size;
                    b.high = b.close;
                    b.low = b.open;
                    b.uptrend = 1;
                    b.time_msc = t_msc;
                    bricks.push_back(b);
                    new_bricks.push_back(b);
                    current_price += brick_size;
                } else if (price <= current_price - brick_size) {
                    Brick b;
                    b.open = current_price;
                    b.close = current_price - brick_size;
                    b.high = b.open;
                    b.low = b.close;
                    b.uptrend = 0;
                    b.time_msc = t_msc;
                    bricks.push_back(b);
                    new_bricks.push_back(b);
                    current_price -= brick_size;
                } else {
                    break;
                }
            }
            return new_bricks;
        }
    };

    double find_optimal_anchor(const double* lb_bids, const int64_t* lb_times, int lb_count, double day_open, double brick_size) {
        if (lb_count < 100) return day_open;
        
        double best_anchor = day_open;
        double best_pnl = -1e9;

        // Optimization: limit the grid to +/- 50 steps
        for (int i = -50; i <= 50; ++i) {
            double anchor = day_open + (i * 0.1 * brick_size);
            RenkoBuilder rb(anchor, brick_size);
            for (int j = 0; j < lb_count; ++j) {
                rb.update_tick(lb_bids[j], lb_times[j]);
            }
            
            double pnl = 0;
            for (size_t b = 1; b < rb.bricks.size(); ++b) {
                if (rb.bricks[b].uptrend == rb.bricks[b-1].uptrend) {
                    pnl += 1.0; 
                } else {
                    pnl -= 2.0; 
                }
            }
            if (pnl > best_pnl) {
                best_pnl = pnl;
                best_anchor = anchor;
            }
        }
        return best_anchor;
    }

    Trade* run_backtest_cpp(
        const double* bids, 
        const double* asks, 
        const int64_t* times_msc, 
        int num_ticks,
        double k_multiplier,
        const int* delays,
        int num_delays,
        int* out_num_trades
    ) {
        std::vector<Trade> all_orders;
        std::vector<Trade> pending_orders;
        std::vector<Trade> active_trades;
        int order_counter = 0;

        int64_t ORDER_EXPIRY_MS = 12LL * 3600LL * 1000LL;
        int64_t TRADE_TIMEOUT_MS = 24LL * 3600LL * 1000LL;

        if (num_ticks == 0) {
            *out_num_trades = 0;
            return nullptr;
        }

        std::vector<int> day_start_indices;
        day_start_indices.push_back(0);
        
        // Simple day boundaries using GMT epoch
        int64_t current_day = times_msc[0] / 86400000LL;
        for (int i = 1; i < num_ticks; ++i) {
            int64_t d = times_msc[i] / 86400000LL;
            if (d > current_day) {
                day_start_indices.push_back(i);
                current_day = d;
            }
        }
        day_start_indices.push_back(num_ticks);

        for (size_t day_idx = 0; day_idx < day_start_indices.size() - 1; ++day_idx) {
            int day_start = day_start_indices[day_idx];
            int day_end = day_start_indices[day_idx+1];
            int day_count = day_end - day_start;
            
            if (day_count < 100) continue;

            int lb_start_day_idx = (day_idx >= 7) ? (day_idx - 7) : 0;
            int lb_start = day_start_indices[lb_start_day_idx];
            int lb_count = day_start - lb_start;

            double day_open = bids[day_start];
            double brick_size = day_open * k_multiplier;

            double best_anchor = find_optimal_anchor(bids + lb_start, times_msc + lb_start, lb_count, day_open, brick_size);
            
            RenkoBuilder renko(best_anchor, brick_size);

            for (int i = lb_start; i < day_start; ++i) {
                renko.update_tick(bids[i], times_msc[i]);
            }

            for (int i = day_start; i < day_end; ++i) {
                double bid = bids[i];
                double ask = asks[i];
                int64_t t_msc = times_msc[i];

                std::vector<Trade> still_pending;
                for (auto& order : pending_orders) {
                    if ((t_msc - order.created_t_msc) > ORDER_EXPIRY_MS) {
                        order.result_code = 4; 
                        all_orders.push_back(order);
                        continue;
                    }

                    if (order.state == 0) { 
                        bool missed = false;
                        if (order.direction == 1 && bid >= order.tp_price) missed = true;
                        else if (order.direction == -1 && ask <= order.tp_price) missed = true;

                        if (missed) {
                            order.result_code = 2; 
                            all_orders.push_back(order);
                            continue;
                        }

                        bool touched = false;
                        if (order.direction == 1 && ask <= order.original_limit) touched = true;
                        else if (order.direction == -1 && bid >= order.original_limit) touched = true;

                        if (touched) {
                            order.touch_t_msc = t_msc;
                            order.activation_t_msc = t_msc + (order.delay_mins * 60LL * 1000LL);
                            order.state = 1; 
                        }
                        still_pending.push_back(order);
                    } 
                    else if (order.state == 1) { 
                        bool invalidated = false;
                        if (order.direction == 1) {
                            if (bid <= order.sl_price || bid >= order.tp_price) invalidated = true;
                        } else {
                            if (ask >= order.sl_price || ask <= order.tp_price) invalidated = true;
                        }

                        if (invalidated) {
                            order.result_code = 3; 
                            all_orders.push_back(order);
                            continue;
                        }

                        if (t_msc >= order.activation_t_msc) {
                            order.state = 2; 
                            if (order.direction == 1 && ask <= order.original_limit) {
                                order.limit_price = ask;
                                order.filled = true;
                                order.fill_t_msc = t_msc;
                                active_trades.push_back(order);
                                continue;
                            } else if (order.direction == -1 && bid >= order.original_limit) {
                                order.limit_price = bid;
                                order.filled = true;
                                order.fill_t_msc = t_msc;
                                active_trades.push_back(order);
                                continue;
                            }
                        }
                        still_pending.push_back(order);
                    }
                    else if (order.state == 2) { 
                        bool filled = false;
                        if (order.direction == 1 && ask <= order.limit_price) filled = true;
                        else if (order.direction == -1 && bid >= order.limit_price) filled = true;

                        if (filled) {
                            order.filled = true;
                            order.fill_t_msc = t_msc;
                            active_trades.push_back(order);
                        } else {
                            still_pending.push_back(order);
                        }
                    }
                }
                pending_orders = still_pending;

                std::vector<Trade> still_active;
                for (auto& trade : active_trades) {
                    if ((t_msc - trade.fill_t_msc) > TRADE_TIMEOUT_MS) {
                        trade.result_code = 4; 
                        all_orders.push_back(trade);
                        continue;
                    }

                    bool resolved = false;
                    
                    // Breakeven Logic (0.3125 * brick_size)
                    bool be_triggered = false;
                    if (trade.direction == 1) {
                        if (bid >= trade.original_limit + (0.3125 * trade.brick_size)) be_triggered = true;
                    } else {
                        if (ask <= trade.original_limit - (0.3125 * trade.brick_size)) be_triggered = true;
                    }
                    
                    if (be_triggered) {
                        if (trade.direction == 1) {
                            trade.sl_price = std::max(trade.sl_price, trade.limit_price);
                        } else {
                            trade.sl_price = std::min(trade.sl_price, trade.limit_price);
                        }
                    }

                    if (trade.direction == 1) {
                        if (bid >= trade.tp_price) {
                            trade.result_code = 0; 
                            trade.pnl_R = (trade.tp_price - trade.limit_price) / trade.brick_size;
                            resolved = true;
                        } else if (bid <= trade.sl_price) {
                            trade.result_code = 1; 
                            trade.pnl_R = (trade.sl_price - trade.limit_price) / trade.brick_size;
                            resolved = true;
                        }
                    } else {
                        if (ask <= trade.tp_price) {
                            trade.result_code = 0; 
                            trade.pnl_R = (trade.limit_price - trade.tp_price) / trade.brick_size;
                            resolved = true;
                        } else if (ask >= trade.sl_price) {
                            trade.result_code = 1; 
                            trade.pnl_R = (trade.limit_price - trade.sl_price) / trade.brick_size;
                            resolved = true;
                        }
                    }

                    if (resolved) {
                        trade.exit_t_msc = t_msc;
                        all_orders.push_back(trade);
                    } else {
                        still_active.push_back(trade);
                    }
                }
                active_trades = still_active;

                std::vector<Brick> new_bricks = renko.update_tick(bid, t_msc);
                for (const auto& brick : new_bricks) {
                    int direction = (brick.uptrend == 1) ? 1 : -1;
                    double limit = brick.open;
                    double tp = (direction == 1) ? limit + 2 * brick_size : limit - 2 * brick_size;
                    double sl = (direction == 1) ? limit - 1 * brick_size : limit + 1 * brick_size;

                    for (int d = 0; d < num_delays; ++d) {
                        Trade o;
                        o.order_id = ++order_counter;
                        o.direction = direction;
                        o.original_limit = limit;
                        o.limit_price = limit;
                        o.tp_price = tp;
                        o.sl_price = sl;
                        o.brick_size = brick_size;
                        o.delay_mins = delays[d];
                        o.state = 0;
                        o.created_t_msc = t_msc;
                        o.touch_t_msc = 0;
                        o.activation_t_msc = 0;
                        o.filled = false;
                        o.fill_t_msc = 0;
                        o.exit_t_msc = 0;
                        o.pnl_R = 0;
                        o.result_code = -1;
                        pending_orders.push_back(o);
                    }
                }
            } 
        } 

        for (auto& t : active_trades) { t.result_code = 4; all_orders.push_back(t); }
        for (auto& o : pending_orders) { o.result_code = 4; all_orders.push_back(o); }

        *out_num_trades = all_orders.size();
        Trade* out_array = new Trade[*out_num_trades];
        std::copy(all_orders.begin(), all_orders.end(), out_array);
        return out_array;
    }

    void free_trades(Trade* ptr) {
        delete[] ptr;
    }

} // extern "C"
