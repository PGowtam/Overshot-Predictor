#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdint.h>

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
        int state; // 0=DELAYING, 2=ACTIVE
        int64_t created_t_msc;
        int64_t activation_t_msc;
        bool filled;
        int64_t fill_t_msc;
        int64_t exit_t_msc;
        double pnl_R;
        int result_code; // 0=Win, 1=Loss, 2=Invalidated_Missed, 3=Invalidated_Dodged, 4=Expired/Timeout
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
        int uptrend;
        std::vector<Brick> bricks;

        RenkoBuilder(double day_open, double b_size) {
            brick_size = b_size;
            current_price = day_open;
            uptrend = 0;
        }

        std::vector<Brick> update_tick(double price, int64_t t_msc) {
            std::vector<Brick> new_bricks;
            while (true) {
                if (uptrend == 0) {
                    if (price >= current_price + brick_size) {
                        Brick b; b.open = current_price; b.close = current_price + brick_size; b.high = b.close; b.low = b.open; b.uptrend = 1; b.time_msc = t_msc;
                        bricks.push_back(b); new_bricks.push_back(b); current_price += brick_size; uptrend = 1;
                    } else if (price <= current_price - brick_size) {
                        Brick b; b.open = current_price; b.close = current_price - brick_size; b.high = b.open; b.low = b.close; b.uptrend = 0; b.time_msc = t_msc;
                        bricks.push_back(b); new_bricks.push_back(b); current_price -= brick_size; uptrend = -1;
                    } else { break; }
                } else if (uptrend == 1) {
                    if (price >= current_price + brick_size) {
                        Brick b; b.open = current_price; b.close = current_price + brick_size; b.high = b.close; b.low = b.open; b.uptrend = 1; b.time_msc = t_msc;
                        bricks.push_back(b); new_bricks.push_back(b); current_price += brick_size;
                    } else if (price <= current_price - 2 * brick_size) {
                        current_price -= 2 * brick_size; uptrend = -1;
                        Brick b; b.open = current_price + brick_size; b.close = current_price; b.high = current_price + brick_size; b.low = current_price; b.uptrend = 0; b.time_msc = t_msc;
                        bricks.push_back(b); new_bricks.push_back(b);
                    } else { break; }
                } else if (uptrend == -1) {
                    if (price <= current_price - brick_size) {
                        Brick b; b.open = current_price; b.close = current_price - brick_size; b.high = b.open; b.low = b.close; b.uptrend = 0; b.time_msc = t_msc;
                        bricks.push_back(b); new_bricks.push_back(b); current_price -= brick_size;
                    } else if (price >= current_price + 2 * brick_size) {
                        current_price += 2 * brick_size; uptrend = 1;
                        Brick b; b.open = current_price - brick_size; b.close = current_price; b.high = current_price; b.low = current_price - brick_size; b.uptrend = 1; b.time_msc = t_msc;
                        bricks.push_back(b); new_bricks.push_back(b);
                    } else { break; }
                }
            }
            return new_bricks;
        }
    };

    double find_optimal_anchor(const double* lb_bids, const int64_t* lb_times, int lb_count, double day_open, double brick_size) {
        if (lb_count < 100) return day_open;
        double best_anchor = day_open;
        double best_pnl = -1e9;
        for (int i = -50; i <= 50; ++i) {
            double anchor = day_open + (i * 0.1 * brick_size);
            RenkoBuilder rb(anchor, brick_size);
            for (int j = 0; j < lb_count; ++j) { rb.update_tick(lb_bids[j], lb_times[j]); }
            double pnl = 0;
            for (size_t b = 1; b < rb.bricks.size(); ++b) {
                if (rb.bricks[b].uptrend == rb.bricks[b-1].uptrend) pnl += 1.0; else pnl -= 2.0; 
            }
            if (pnl > best_pnl) { best_pnl = pnl; best_anchor = anchor; }
        }
        return best_anchor;
    }

    Trade* run_backtest_reversal(
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

                // 1. Process Pending Orders (DELAYING)
                std::vector<Trade> still_pending;
                for (auto& order : pending_orders) {
                    if ((t_msc - order.created_t_msc) > ORDER_EXPIRY_MS) {
                        order.result_code = 4; // Expired
                        all_orders.push_back(order);
                        continue;
                    }

                    if (order.state == 0) { // DELAYING
                        bool dodged = false;
                        bool missed = false;
                        
                        if (order.direction == 1) { // LONG
                            if (bid <= order.sl_price) dodged = true; // hit SL during wait
                            else if (bid >= order.tp_price) missed = true; // hit TP during wait
                        } else { // SHORT
                            if (ask >= order.sl_price) dodged = true;
                            else if (ask <= order.tp_price) missed = true;
                        }

                        if (dodged) {
                            order.result_code = 3; // Invalidated Dodged (saved from loss!)
                            all_orders.push_back(order);
                            continue;
                        }
                        if (missed) {
                            order.result_code = 2; // Invalidated Missed
                            all_orders.push_back(order);
                            continue;
                        }

                        if (t_msc >= order.activation_t_msc) {
                            order.state = 2; // ACTIVE
                            // Limit/Market Entry Check
                            if (order.direction == 1) { // LONG
                                if (ask <= order.original_limit) { // Better or equal price
                                    order.limit_price = ask;
                                    order.filled = true;
                                    order.fill_t_msc = t_msc;
                                } else { // Worse price, place LIMIT order
                                    order.limit_price = order.original_limit;
                                    order.filled = false;
                                }
                            } else { // SHORT
                                if (bid >= order.original_limit) { // Better or equal price
                                    order.limit_price = bid;
                                    order.filled = true;
                                    order.fill_t_msc = t_msc;
                                } else { // Worse price, place LIMIT order
                                    order.limit_price = order.original_limit;
                                    order.filled = false;
                                }
                            }
                            active_trades.push_back(order);
                        } else {
                            still_pending.push_back(order);
                        }
                    }
                }
                pending_orders = still_pending;

                // 2. Check Active Trades
                std::vector<Trade> still_active;
                for (auto& trade : active_trades) {
                    if ((t_msc - trade.created_t_msc) > TRADE_TIMEOUT_MS) {
                        trade.result_code = 4; 
                        all_orders.push_back(trade);
                        continue;
                    }

                    if (!trade.filled) {
                        bool filled = false;
                        if (trade.direction == 1 && ask <= trade.limit_price) filled = true;
                        else if (trade.direction == -1 && bid >= trade.limit_price) filled = true;
                        
                        if (filled) {
                            trade.filled = true;
                            trade.fill_t_msc = t_msc;
                        }
                        still_active.push_back(trade);
                        continue;
                    }

                    bool resolved = false;
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

                // 3. New Bricks -> New Orders
                std::vector<Brick> new_bricks = renko.update_tick(bid, t_msc);
                for (const auto& brick : new_bricks) {
                    // Bet on Full Reversal
                    int direction = (brick.uptrend == 1) ? -1 : 1;
                    double entry = brick.close;
                    double tp = (direction == 1) ? entry + 3 * brick_size : entry - 3 * brick_size;
                    double sl = (direction == 1) ? entry - 1 * brick_size : entry + 1 * brick_size;

                    for (int d = 0; d < num_delays; ++d) {
                        Trade o;
                        o.order_id = ++order_counter;
                        o.direction = direction;
                        o.original_limit = entry;
                        o.limit_price = entry;
                        o.tp_price = tp;
                        o.sl_price = sl;
                        o.brick_size = brick_size;
                        o.delay_mins = delays[d];
                        o.state = (delays[d] == 0) ? 2 : 0; 
                        o.created_t_msc = t_msc;
                        o.activation_t_msc = t_msc + (delays[d] * 60LL * 1000LL);
                        o.filled = (delays[d] == 0);
                        o.fill_t_msc = (delays[d] == 0) ? t_msc : 0;
                        o.exit_t_msc = 0;
                        o.pnl_R = 0;
                        o.result_code = -1;
                        if (delays[d] == 0) active_trades.push_back(o);
                        else pending_orders.push_back(o);
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
