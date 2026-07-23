#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdint.h>

extern "C" {

    struct LabelRow {
        int brick_id;
        int64_t timestamp;
        int direction;
        double close_price;
        double open_price;
        double brick_size;
        
        int t1_win;
        double t1_y_mag;
        int t2_win;
        double t2_y_mag;
        int t2_filled;
        int t3_win;
        double t3_y_mag;
        int t4_win;
        double t4_y_mag;
        int t4_filled;
        
        int label; 
        int exclude_flag;
        int64_t brick_duration_seconds;
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

    void resolve_all_trades(int start_idx, int num_ticks, const double* bids, const double* asks, const int64_t* times, 
                            int uptrend, double close_price, double open_price, double brick_size, LabelRow& row) {
        
        row.t1_win = 0; row.t1_y_mag = 0.0;
        row.t2_win = 0; row.t2_y_mag = 0.0; row.t2_filled = 0;
        row.t3_win = 0; row.t3_y_mag = 0.0;
        row.t4_win = 0; row.t4_y_mag = 0.0; row.t4_filled = 0;
        row.label = 0;
        row.exclude_flag = 0;
        
        bool t1_done = false;
        bool t2_done = false;
        bool t3_done = false;
        bool t4_done = false;

        double t1_entry = (uptrend == 1) ? asks[start_idx] : bids[start_idx];
        double t1_tp = (uptrend == 1) ? close_price + brick_size : close_price - brick_size;
        double t1_sl = (uptrend == 1) ? close_price - brick_size : close_price + brick_size;

        double t2_entry = open_price;
        double t2_tp = (uptrend == 1) ? open_price + 2 * brick_size : open_price - 2 * brick_size;
        double t2_sl = (uptrend == 1) ? open_price - brick_size : open_price + brick_size;
        
        double t3_entry = (uptrend == 1) ? bids[start_idx] : asks[start_idx];
        double t3_tp = (uptrend == 1) ? close_price - 2 * brick_size : close_price + 2 * brick_size;
        double t3_sl = (uptrend == 1) ? close_price + brick_size : close_price - brick_size;

        double t4_entry = (uptrend == 1) ? bids[start_idx] : asks[start_idx];
        double t4_tp = (uptrend == 1) ? close_price - 3 * brick_size : close_price + 3 * brick_size;
        double t4_sl = (uptrend == 1) ? close_price + brick_size : close_price - brick_size;

        for (int i = start_idx; i < num_ticks; ++i) {
            if (t1_done && t2_done && t3_done && t4_done) break;
            
            double bid = bids[i];
            double ask = asks[i];
            
            if (uptrend == 1) { 
                if (!t1_done) {
                    if (bid >= t1_tp) { row.t1_win = 1; row.t1_y_mag = std::abs(bid - t1_entry)/brick_size; t1_done = true; }
                    else if (bid <= t1_sl) { row.t1_win = 0; row.t1_y_mag = std::abs(bid - t1_entry)/brick_size; t1_done = true; }
                }
                if (!t2_done) {
                    if (row.t2_filled == 0) {
                        if (ask <= t2_entry) { row.t2_filled = 1; }
                        else if (bid <= t2_sl) { t2_done = true; } 
                    }
                    if (row.t2_filled == 1 && !t2_done) {
                        if (bid >= t2_tp) { row.t2_win = 1; row.t2_y_mag = std::abs(bid - t2_entry)/brick_size; t2_done = true; }
                        else if (bid <= t2_sl) { row.t2_win = 0; row.t2_y_mag = std::abs(bid - t2_entry)/brick_size; t2_done = true; }
                    }
                }
                if (!t3_done) {
                    if (ask <= t3_tp) { row.t3_win = 1; row.t3_y_mag = std::abs(t3_entry - ask)/brick_size; t3_done = true; }
                    else if (ask >= t3_sl) { row.t3_win = 0; row.t3_y_mag = std::abs(t3_entry - ask)/brick_size; t3_done = true; }
                }
                if (!t4_done) {
                    if (ask <= t4_tp) { row.t4_win = 1; row.t4_y_mag = std::abs(t4_entry - ask)/brick_size; t4_done = true; row.t4_filled = 1; }
                    else if (ask >= t4_sl) { row.t4_win = 0; row.t4_y_mag = std::abs(t4_entry - ask)/brick_size; t4_done = true; row.t4_filled = 1; }
                }
            } else { 
                if (!t1_done) {
                    if (ask <= t1_tp) { row.t1_win = 1; row.t1_y_mag = std::abs(t1_entry - ask)/brick_size; t1_done = true; }
                    else if (ask >= t1_sl) { row.t1_win = 0; row.t1_y_mag = std::abs(t1_entry - ask)/brick_size; t1_done = true; }
                }
                if (!t2_done) {
                    if (row.t2_filled == 0) {
                        if (bid >= t2_entry) { row.t2_filled = 1; }
                        else if (ask >= t2_sl) { t2_done = true; }
                    }
                    if (row.t2_filled == 1 && !t2_done) {
                        if (ask <= t2_tp) { row.t2_win = 1; row.t2_y_mag = std::abs(t2_entry - ask)/brick_size; t2_done = true; }
                        else if (ask >= t2_sl) { row.t2_win = 0; row.t2_y_mag = std::abs(t2_entry - ask)/brick_size; t2_done = true; }
                    }
                }
                if (!t3_done) {
                    if (bid >= t3_tp) { row.t3_win = 1; row.t3_y_mag = std::abs(bid - t3_entry)/brick_size; t3_done = true; }
                    else if (bid <= t3_sl) { row.t3_win = 0; row.t3_y_mag = std::abs(bid - t3_entry)/brick_size; t3_done = true; }
                }
                if (!t4_done) {
                    if (bid >= t4_tp) { row.t4_win = 1; row.t4_y_mag = std::abs(bid - t4_entry)/brick_size; t4_done = true; row.t4_filled = 1; }
                    else if (bid <= t4_sl) { row.t4_win = 0; row.t4_y_mag = std::abs(bid - t4_entry)/brick_size; t4_done = true; row.t4_filled = 1; }
                }
            }
        }

        if (!t1_done || !t2_done || !t3_done || !t4_done) {
            row.exclude_flag = 1;
        }

        if (row.t2_win && row.t2_filled) {
            row.label = 2;
        } else if (row.t4_win && row.t4_filled) {
            row.label = 4;
        } else if (row.t1_win) {
            row.label = 1;
        } else if (row.t3_win) {
            row.label = 3;
        } else {
            row.label = 0;
        }
    }

    LabelRow* generate_labels(
        const double* bids, 
        const double* asks, 
        const int64_t* times_msc, 
        int num_ticks,
        double k_multiplier,
        int* out_num_rows
    ) {
        std::vector<LabelRow> all_labels;
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
            RenkoBuilder renko(best_anchor, brick_size);
            
            // Warmup
            for (int i = lb_start; i < day_start; ++i) {
                renko.update_tick(bids[i], times_msc[i]);
            }

            int64_t prev_brick_time = (day_start > 0) ? times_msc[day_start - 1] : times_msc[0];

            for (int i = day_start; i < day_end; ++i) {
                double bid = bids[i];
                double ask = asks[i];
                int64_t t_msc = times_msc[i];

                auto new_bricks = renko.update_tick(bid, t_msc);
                for (const auto& brick : new_bricks) {
                    brick_counter++;

                    LabelRow row;
                    row.brick_id = brick_counter;
                    row.timestamp = t_msc;
                    row.direction = brick.uptrend;
                    row.close_price = brick.close;
                    row.open_price = brick.open;
                    row.brick_size = brick_size;
                    row.brick_duration_seconds = (t_msc - prev_brick_time) / 1000LL;
                    
                    resolve_all_trades(i, num_ticks, bids, asks, times_msc, brick.uptrend, brick.close, brick.open, brick_size, row);

                    all_labels.push_back(row);
                    prev_brick_time = t_msc;
                }
            } 
        } 

        *out_num_rows = all_labels.size();
        LabelRow* out_array = new LabelRow[*out_num_rows];
        std::copy(all_labels.begin(), all_labels.end(), out_array);
        return out_array;
    }

    void free_labels(LabelRow* ptr) {
        delete[] ptr;
    }
} 
