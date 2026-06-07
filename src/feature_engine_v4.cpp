#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include <deque>

extern "C" {

    struct FeatureRow {
        int brick_id;
        int64_t timestamp;
        int direction;
        double entry_price;
        double brick_size;
        double time_sin;
        double time_cos;
        
        double ema_50_5m_dist;
        double ema_200_5m_dist;
        double atr_14_5m;
        double return_12_5m;
        
        double ema_50_15m_dist;
        double ema_200_15m_dist;
        double atr_14_15m;
        double return_4_15m;
        
        int label_t1; // 1:1 Cont
        int label_t2; // 1:2 Pullback Cont (-2 if not triggered)
        int label_t3; // 1:2 Reversal
        int label_t4; // 1:3 Double Reversal
    };

    struct Brick {
        double open;
        double close;
        double high;
        double low;
        int uptrend;
        int64_t time_msc;
    };

    struct TimeCandle {
        double open;
        double high;
        double low;
        double close;
        int64_t start_t_msc;
        bool closed;
    };

    class CandleTracker {
    public:
        int64_t period_ms;
        TimeCandle current_candle;
        double ema_50;
        double ema_200;
        std::deque<double> tr_history;
        std::deque<double> close_history;
        double atr_14;
        double prev_close;
        bool initialized;

        CandleTracker(int64_t p_ms) : period_ms(p_ms), ema_50(0), ema_200(0), atr_14(0), prev_close(0), initialized(false) {
            current_candle.closed = true;
            current_candle.start_t_msc = 0;
        }

        void update(double price, int64_t t_msc) {
            int64_t period_start = (t_msc / period_ms) * period_ms;
            
            if (current_candle.closed || period_start > current_candle.start_t_msc) {
                if (!current_candle.closed) {
                    double close_price = current_candle.close;
                    
                    close_history.push_back(close_price);
                    if (close_history.size() > 20) close_history.pop_front();
                    
                    if (!initialized) {
                        ema_50 = close_price;
                        ema_200 = close_price;
                        prev_close = close_price;
                        atr_14 = 0;
                        initialized = true;
                    } else {
                        double k_50 = 2.0 / (50.0 + 1.0);
                        ema_50 = (close_price - ema_50) * k_50 + ema_50;
                        
                        double k_200 = 2.0 / (200.0 + 1.0);
                        ema_200 = (close_price - ema_200) * k_200 + ema_200;
                        
                        double tr = std::max({
                            current_candle.high - current_candle.low,
                            std::abs(current_candle.high - prev_close),
                            std::abs(current_candle.low - prev_close)
                        });
                        
                        tr_history.push_back(tr);
                        if (tr_history.size() > 14) tr_history.pop_front();
                        
                        double tr_sum = 0;
                        for (double v : tr_history) tr_sum += v;
                        atr_14 = tr_sum / tr_history.size();
                        
                        prev_close = close_price;
                    }
                }
                
                current_candle.open = price;
                current_candle.high = price;
                current_candle.low = price;
                current_candle.close = price;
                current_candle.start_t_msc = period_start;
                current_candle.closed = false;
            } else {
                current_candle.high = std::max(current_candle.high, price);
                current_candle.low = std::min(current_candle.low, price);
                current_candle.close = price;
            }
        }
        
        double get_momentum(int periods, double current_price, double brick_size) {
            if (close_history.size() < periods) return 0;
            double old_price = close_history[close_history.size() - periods];
            return (current_price - old_price) / brick_size;
        }
    };

    class RenkoBuilderV4 {
    public:
        double brick_size;
        double current_price;
        int uptrend;
        std::vector<Brick> bricks;

        RenkoBuilderV4(double day_open, double b_size) {
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

    void resolve_all_trades(int start_idx, int num_ticks, const double* bids, const double* asks, const int64_t* times, int uptrend, double close_price, double brick_size, FeatureRow& row) {
        int64_t start_time = times[start_idx];
        
        row.label_t1 = -1;
        row.label_t2 = -2; 
        row.label_t3 = -1;
        row.label_t4 = -1;

        bool t1_done = false;
        bool t2_triggered = false, t2_done = false;
        bool t3_done = false;
        bool t4_done = false;

        if (uptrend == 1) { 
            double t1_entry = asks[start_idx];
            double t1_tp = t1_entry + 1 * brick_size;
            double t1_sl = t1_entry - 1 * brick_size;
            
            double t2_entry = close_price - brick_size;
            double t2_tp = t2_entry + 2 * brick_size;
            double t2_sl = t2_entry - 1 * brick_size;
            
            double t3_entry = bids[start_idx];
            double t3_tp = t3_entry - 2 * brick_size;
            double t3_sl = t3_entry + 1 * brick_size;
            
            double t4_entry = bids[start_idx];
            double t4_tp = t4_entry - 3 * brick_size;
            double t4_sl = t4_entry + 1 * brick_size;

            for (int i = start_idx; i < num_ticks; ++i) {
                if (t1_done && t2_done && t3_done && t4_done) break;
                if (times[i] - start_time > 86400000LL) break; 
                
                double bid = bids[i];
                double ask = asks[i];
                
                if (!t1_done) {
                    if (bid >= t1_tp) { row.label_t1 = 1; t1_done = true; }
                    else if (bid <= t1_sl) { row.label_t1 = 0; t1_done = true; }
                }
                if (!t2_done) {
                    if (!t2_triggered) {
                        if (ask <= t2_entry) { t2_triggered = true; row.label_t2 = -1; }
                    }
                    if (t2_triggered) {
                        if (bid >= t2_tp) { row.label_t2 = 1; t2_done = true; }
                        else if (bid <= t2_sl) { row.label_t2 = 0; t2_done = true; }
                    }
                }
                if (!t3_done) {
                    if (ask <= t3_tp) { row.label_t3 = 1; t3_done = true; }
                    else if (ask >= t3_sl) { row.label_t3 = 0; t3_done = true; }
                }
                if (!t4_done) {
                    if (ask <= t4_tp) { row.label_t4 = 1; t4_done = true; }
                    else if (ask >= t4_sl) { row.label_t4 = 0; t4_done = true; }
                }
            }
        } else { 
            double t1_entry = bids[start_idx];
            double t1_tp = t1_entry - 1 * brick_size;
            double t1_sl = t1_entry + 1 * brick_size;
            
            double t2_entry = close_price + brick_size;
            double t2_tp = t2_entry - 2 * brick_size;
            double t2_sl = t2_entry + 1 * brick_size;
            
            double t3_entry = asks[start_idx];
            double t3_tp = t3_entry + 2 * brick_size;
            double t3_sl = t3_entry - 1 * brick_size;
            
            double t4_entry = asks[start_idx];
            double t4_tp = t4_entry + 3 * brick_size;
            double t4_sl = t4_entry - 1 * brick_size;

            for (int i = start_idx; i < num_ticks; ++i) {
                if (t1_done && t2_done && t3_done && t4_done) break;
                if (times[i] - start_time > 86400000LL) break; 
                
                double bid = bids[i];
                double ask = asks[i];
                
                if (!t1_done) {
                    if (ask <= t1_tp) { row.label_t1 = 1; t1_done = true; }
                    else if (ask >= t1_sl) { row.label_t1 = 0; t1_done = true; }
                }
                if (!t2_done) {
                    if (!t2_triggered) {
                        if (bid >= t2_entry) { t2_triggered = true; row.label_t2 = -1; }
                    }
                    if (t2_triggered) {
                        if (ask <= t2_tp) { row.label_t2 = 1; t2_done = true; }
                        else if (ask >= t2_sl) { row.label_t2 = 0; t2_done = true; }
                    }
                }
                if (!t3_done) {
                    if (bid >= t3_tp) { row.label_t3 = 1; t3_done = true; }
                    else if (bid <= t3_sl) { row.label_t3 = 0; t3_done = true; }
                }
                if (!t4_done) {
                    if (bid >= t4_tp) { row.label_t4 = 1; t4_done = true; }
                    else if (bid <= t4_sl) { row.label_t4 = 0; t4_done = true; }
                }
            }
        }
    }

    double find_optimal_anchor(const double* lb_bids, const int64_t* lb_times, int lb_count, double day_open, double brick_size) {
        if (lb_count < 100) return day_open;
        double best_anchor = day_open;
        double best_pnl = -1e9;
        for (int i = -50; i <= 50; ++i) {
            double anchor = day_open + (i * 0.1 * brick_size);
            RenkoBuilderV4 rb(anchor, brick_size);
            for (int j = 0; j < lb_count; ++j) { rb.update_tick(lb_bids[j], lb_times[j]); }
            double pnl = 0;
            for (size_t b = 1; b < rb.bricks.size(); ++b) {
                if (rb.bricks[b].uptrend == rb.bricks[b-1].uptrend) pnl += 1.0; else pnl -= 2.0; 
            }
            if (pnl > best_pnl) { best_pnl = pnl; best_anchor = anchor; }
        }
        return best_anchor;
    }

    FeatureRow* generate_hybrid_features(
        const double* bids, 
        const double* asks, 
        const int64_t* times_msc, 
        int num_ticks,
        double k_multiplier,
        int* out_num_rows
    ) {
        std::vector<FeatureRow> all_features;
        int brick_counter = 0;

        if (num_ticks == 0) {
            *out_num_rows = 0;
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
            RenkoBuilderV4 renko(best_anchor, brick_size);
            
            CandleTracker tracker_5m(5LL * 60LL * 1000LL);
            CandleTracker tracker_15m(15LL * 60LL * 1000LL);

            // Warmup
            for (int i = lb_start; i < day_start; ++i) {
                double mid = (bids[i] + asks[i]) / 2.0;
                tracker_5m.update(mid, times_msc[i]);
                tracker_15m.update(mid, times_msc[i]);
                
                renko.update_tick(bids[i], times_msc[i]);
            }

            for (int i = day_start; i < day_end; ++i) {
                double bid = bids[i];
                double ask = asks[i];
                int64_t t_msc = times_msc[i];
                double mid = (bid + ask) / 2.0;

                tracker_5m.update(mid, t_msc);
                tracker_15m.update(mid, t_msc);

                auto new_bricks = renko.update_tick(bid, t_msc);
                for (const auto& brick : new_bricks) {
                    brick_counter++;

                    int uptrend = brick.uptrend;
                    double close_price = brick.close;

                    FeatureRow row;
                    row.brick_id = brick_counter;
                    row.timestamp = t_msc;
                    row.direction = uptrend;
                    row.entry_price = close_price;
                    row.brick_size = brick_size;
                    
                    double day_fraction = (t_msc % 86400000LL) / 86400000.0;
                    row.time_sin = std::sin(day_fraction * 2.0 * M_PI);
                    row.time_cos = std::cos(day_fraction * 2.0 * M_PI);
                    
                    row.ema_50_5m_dist = (tracker_5m.initialized && tracker_5m.ema_50 > 0) ? (mid - tracker_5m.ema_50) / brick_size : 0;
                    row.ema_200_5m_dist = (tracker_5m.initialized && tracker_5m.ema_200 > 0) ? (mid - tracker_5m.ema_200) / brick_size : 0;
                    row.atr_14_5m = tracker_5m.atr_14 / brick_size;
                    row.return_12_5m = tracker_5m.get_momentum(12, mid, brick_size);
                    
                    row.ema_50_15m_dist = (tracker_15m.initialized && tracker_15m.ema_50 > 0) ? (mid - tracker_15m.ema_50) / brick_size : 0;
                    row.ema_200_15m_dist = (tracker_15m.initialized && tracker_15m.ema_200 > 0) ? (mid - tracker_15m.ema_200) / brick_size : 0;
                    row.atr_14_15m = tracker_15m.atr_14 / brick_size;
                    row.return_4_15m = tracker_15m.get_momentum(4, mid, brick_size);
                    
                    resolve_all_trades(i, num_ticks, bids, asks, times_msc, uptrend, close_price, brick_size, row);

                    all_features.push_back(row);
                }
            } 
        } 

        *out_num_rows = all_features.size();
        FeatureRow* out_array = new FeatureRow[*out_num_rows];
        std::copy(all_features.begin(), all_features.end(), out_array);
        return out_array;
    }

    void free_features(FeatureRow* ptr) {
        delete[] ptr;
    }
} // extern "C"
