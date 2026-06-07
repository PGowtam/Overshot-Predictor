#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdint.h>

extern "C" {

    struct FeatureLabelRow {
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

        float ancs_fine[60];
        float ancs_coarse[30];
        float candle_features[15];
        float momentum[19];
        float history[150];
    };

    struct Tick {
        double bid;
        double ask;
        int64_t time_msc;
    };

    struct BrickEvent {
        double open;
        double close;
        double high;
        double low;
        int uptrend;
        int64_t time_msc;
        std::vector<Tick> intra_brick_ticks;
    };

    class RenkoBuilder {
    public:
        double brick_size;
        double current_price;
        int uptrend;
        std::vector<Tick> current_brick_ticks;

        RenkoBuilder(double day_open, double b_size) {
            brick_size = b_size;
            current_price = day_open;
            uptrend = 0;
        }

        std::vector<BrickEvent> update_tick(double bid, double ask, int64_t t_msc) {
            std::vector<BrickEvent> new_bricks;
            Tick t = {bid, ask, t_msc};
            current_brick_ticks.push_back(t);
            double price = bid;

            while (true) {
                if (uptrend == 0) {
                    if (price >= current_price + brick_size) {
                        BrickEvent b; b.open = current_price; b.close = current_price + brick_size; b.high = b.close; b.low = b.open; b.uptrend = 1; b.time_msc = t_msc;
                        b.intra_brick_ticks = current_brick_ticks; 
                        new_bricks.push_back(b); current_price += brick_size; uptrend = 1;
                        current_brick_ticks.clear();
                        current_brick_ticks.push_back(t);
                    } else if (price <= current_price - brick_size) {
                        BrickEvent b; b.open = current_price; b.close = current_price - brick_size; b.high = b.open; b.low = b.close; b.uptrend = 0; b.time_msc = t_msc;
                        b.intra_brick_ticks = current_brick_ticks;
                        new_bricks.push_back(b); current_price -= brick_size; uptrend = -1;
                        current_brick_ticks.clear();
                        current_brick_ticks.push_back(t);
                    } else { break; }
                } else if (uptrend == 1) {
                    if (price >= current_price + brick_size) {
                        BrickEvent b; b.open = current_price; b.close = current_price + brick_size; b.high = b.close; b.low = b.open; b.uptrend = 1; b.time_msc = t_msc;
                        b.intra_brick_ticks = current_brick_ticks;
                        new_bricks.push_back(b); current_price += brick_size;
                        current_brick_ticks.clear();
                        current_brick_ticks.push_back(t);
                    } else if (price <= current_price - 2 * brick_size) {
                        current_price -= 2 * brick_size; uptrend = -1;
                        BrickEvent b; b.open = current_price + brick_size; b.close = current_price; b.high = current_price + brick_size; b.low = current_price; b.uptrend = 0; b.time_msc = t_msc;
                        b.intra_brick_ticks = current_brick_ticks;
                        new_bricks.push_back(b);
                        current_brick_ticks.clear();
                        current_brick_ticks.push_back(t);
                    } else { break; }
                } else if (uptrend == -1) {
                    if (price <= current_price - brick_size) {
                        BrickEvent b; b.open = current_price; b.close = current_price - brick_size; b.high = b.open; b.low = b.close; b.uptrend = 0; b.time_msc = t_msc;
                        b.intra_brick_ticks = current_brick_ticks;
                        new_bricks.push_back(b); current_price -= brick_size;
                        current_brick_ticks.clear();
                        current_brick_ticks.push_back(t);
                    } else if (price >= current_price + 2 * brick_size) {
                        current_price += 2 * brick_size; uptrend = 1;
                        BrickEvent b; b.open = current_price - brick_size; b.close = current_price; b.high = current_price; b.low = current_price - brick_size; b.uptrend = 1; b.time_msc = t_msc;
                        b.intra_brick_ticks = current_brick_ticks;
                        new_bricks.push_back(b);
                        current_brick_ticks.clear();
                        current_brick_ticks.push_back(t);
                    } else { break; }
                }
            }
            return new_bricks;
        }
    };

    void compute_ancs(const std::vector<Tick>& ticks, int n_segments, double brick_open_price, double brick_size, float* out) {
        int N = ticks.size();
        int seg_size = std::max(1, N / n_segments);
        if (N == 0) {
            for (int i=0; i<n_segments*6; ++i) out[i] = 0.0f;
            return;
        }
        int64_t brick_start_time = ticks[0].time_msc;
        int64_t brick_end_time = ticks.back().time_msc;
        double brick_duration = std::max(1.0, (double)(brick_end_time - brick_start_time));

        for (int i = 0; i < n_segments; ++i) {
            int start = i * seg_size;
            int end = (i < n_segments - 1) ? std::min((i + 1) * seg_size, N) : N;
            
            if (start >= N || end <= start) {
                for (int j=0; j<6; ++j) out[i*6 + j] = 0.0f;
                continue;
            }

            double seg_open = ticks[start].bid;
            double seg_close = ticks[end - 1].bid;
            double seg_high = seg_open;
            double seg_low = seg_open;
            for (int k = start; k < end; ++k) {
                if (ticks[k].bid > seg_high) seg_high = ticks[k].bid;
                if (ticks[k].bid < seg_low) seg_low = ticks[k].bid;
            }

            out[i*6 + 0] = (float)((seg_open - brick_open_price) / brick_size);
            out[i*6 + 1] = (float)((seg_high - brick_open_price) / brick_size);
            out[i*6 + 2] = (float)((seg_low - brick_open_price) / brick_size);
            out[i*6 + 3] = (float)((seg_close - brick_open_price) / brick_size);
            
            double duration_frac = (ticks[end - 1].time_msc - brick_start_time) / brick_duration;
            double tick_frac = (double)(end - start) / N;
            out[i*6 + 4] = (float)duration_frac;
            out[i*6 + 5] = (float)tick_frac;
        }
    }

    void compute_candle_features(const std::vector<Tick>& ticks, double brick_size, int direction, float* out) {
        if (ticks.empty()) {
            for(int i=0; i<15; ++i) out[i] = 0.0f;
            return;
        }
        double O = ticks[0].bid;
        double C = ticks.back().bid;
        double H = O, L = O;
        for (const auto& t : ticks) {
            if (t.bid > H) H = t.bid;
            if (t.bid < L) L = t.bid;
        }
        double body = std::abs(C - O);
        double max_OC = std::max(O, C);
        double min_OC = std::min(O, C);
        double upper_wick = H - max_OC;
        double lower_wick = min_OC - L;
        double full_range = H - L;

        out[0] = (float)(body / brick_size);
        out[1] = (float)(upper_wick / brick_size);
        out[2] = (float)(lower_wick / brick_size);
        out[3] = (float)(full_range / brick_size);
        out[4] = (float)((C - O) / brick_size);
        out[5] = (float)(upper_wick / (full_range + 1e-8));
        out[6] = (float)(lower_wick / (full_range + 1e-8));
        out[7] = (float)(body / (full_range + 1e-8));
        out[8] = (float)((C - L) / (full_range + 1e-8));
        out[9] = (float)((O - L) / (full_range + 1e-8));
        out[10] = (C > (H + L) / 2.0) ? 1.0f : 0.0f;
        out[11] = (direction == 1) ? (float)((H - C) / brick_size) : (float)((C - L) / brick_size);
        out[12] = (float)std::min(full_range / brick_size, 3.0);
        out[13] = (float)(std::abs(O - L) / brick_size);
        out[14] = (float)(std::abs(H - O) / brick_size);
    }

    void compute_momentum_features(const std::vector<Tick>& ticks, double brick_size, float* out) {
        int N = ticks.size();
        if (N == 0) {
            for(int i=0; i<19; ++i) out[i] = 0.0f;
            return;
        }
        
        auto phase_stats = [&](int start, int end, int out_idx) {
            if (end - start < 2) {
                out[out_idx] = 0; out[out_idx+1] = 0; out[out_idx+2] = 0; out[out_idx+3] = 0;
                return;
            }
            double p0 = ticks[start].bid;
            double p1 = ticks[end-1].bid;
            double H = p0, L = p0;
            double sum = 0, sq_sum = 0;
            for (int k = start; k < end; ++k) {
                double p = ticks[k].bid;
                if (p > H) H = p;
                if (p < L) L = p;
                sum += p;
                sq_sum += p * p;
            }
            double mean = sum / (end - start);
            double var = (sq_sum / (end - start)) - (mean * mean);
            double std_dev = (var > 0) ? std::sqrt(var) : 0;

            out[out_idx] = (float)((p1 - p0) / brick_size);
            out[out_idx+1] = (float)((H - p0) / brick_size); 
            out[out_idx+2] = (float)((L - p0) / brick_size); 
            out[out_idx+3] = (float)(std_dev / brick_size);
        };

        phase_stats(0, N/3, 0);
        phase_stats(N/3, 2*N/3, 4);
        phase_stats(2*N/3, N, 8);

        double early_move = std::abs((double)out[0]);
        double late_move = std::abs((double)out[8]);
        out[12] = (float)(late_move - early_move); // acceleration

        double velocity_ratio = 1.0;
        if (N > 10) {
            int early_count = N/3;
            int late_count = N - 2*N/3;
            double dt_early = (early_count > 1) ? (double)(ticks[early_count-1].time_msc - ticks[0].time_msc) / (early_count - 1) : 1000.0;
            double dt_late = (late_count > 1) ? (double)(ticks.back().time_msc - ticks[2*N/3].time_msc) / (late_count - 1) : 1000.0;
            velocity_ratio = dt_early / (dt_late + 1e-3);
        }
        out[13] = (float)velocity_ratio;

        double spread_open = 0;
        int n_open = std::max(1, N/10);
        for(int k=0; k<n_open; ++k) spread_open += (ticks[k].ask - ticks[k].bid);
        spread_open /= n_open;

        double spread_close = 0;
        int start_close = std::max(0, 9*N/10);
        int n_close = N - start_close;
        for(int k=start_close; k<N; ++k) spread_close += (ticks[k].ask - ticks[k].bid);
        if (n_close > 0) spread_close /= n_close;

        out[14] = (float)((spread_close - spread_open) / (spread_open + 1e-8));

        int dir_changes = 0;
        for (int i = 1; i < N - 1; ++i) {
            double d1 = ticks[i].bid - ticks[i-1].bid;
            double d2 = ticks[i+1].bid - ticks[i].bid;
            if (d1 * d2 < 0) dir_changes++;
        }
        out[15] = (float)((double)dir_changes / std::max(1, N));

        double duration_ms = ticks.back().time_msc - ticks[0].time_msc;
        out[16] = (float)std::log1p(duration_ms / 1000.0);
        out[17] = (float)(spread_open / brick_size);
        out[18] = (float)(spread_close / brick_size);
    }

    double find_optimal_anchor(const double* lb_bids, const int64_t* lb_times, int lb_count, double day_open, double brick_size) {
        if (lb_count < 100) return day_open;
        double best_anchor = day_open;
        double best_pnl = -1e9;
        for (int i = -50; i <= 50; ++i) {
            double anchor = day_open + (i * 0.1 * brick_size);
            RenkoBuilder rb(anchor, brick_size);
            for (int j = 0; j < lb_count; ++j) { rb.update_tick(lb_bids[j], lb_bids[j], lb_times[j]); }
            double pnl = 0;
            std::vector<int> up_arr;
            for(int k=0; k<rb.current_brick_ticks.size(); ++k) {} // dummy
            // Actually, we need to collect the bricks to evaluate PnL
            // But doing so inside find_optimal_anchor by instantiating another builder is fine
        }
        return best_anchor;
    }

    // A better anchor finder matching exactly the previous logic
    double find_optimal_anchor_v2(const double* lb_bids, const double* lb_asks, const int64_t* lb_times, int lb_count, double day_open, double brick_size) {
        if (lb_count < 100) return day_open;
        double best_anchor = day_open;
        double best_pnl = -1e9;
        for (int i = -50; i <= 50; ++i) {
            double anchor = day_open + (i * 0.1 * brick_size);
            RenkoBuilder rb(anchor, brick_size);
            double pnl = 0;
            int last_up = 0;
            bool first = true;
            for (int j = 0; j < lb_count; ++j) { 
                auto bricks = rb.update_tick(lb_bids[j], lb_asks[j], lb_times[j]); 
                for(const auto& b : bricks) {
                    if (first) { last_up = b.uptrend; first = false; continue; }
                    if (b.uptrend == last_up) pnl += 1.0; else pnl -= 2.0;
                    last_up = b.uptrend;
                }
            }
            if (pnl > best_pnl) { best_pnl = pnl; best_anchor = anchor; }
        }
        return best_anchor;
    }

    void resolve_all_trades(int start_idx, int num_ticks, const double* bids, const double* asks, const int64_t* times, 
                            int uptrend, double close_price, double open_price, double brick_size, FeatureLabelRow& row) {
        
        row.t1_win = 0; row.t1_y_mag = 0.0;
        row.t2_win = 0; row.t2_y_mag = 0.0; row.t2_filled = 0;
        row.t3_win = 0; row.t3_y_mag = 0.0;
        row.t4_win = 0; row.t4_y_mag = 0.0; row.t4_filled = 0;
        row.label = 0;
        row.exclude_flag = 0;
        
        bool t1_done = false, t2_done = false, t3_done = false, t4_done = false;

        double t1_entry = (uptrend == 1) ? asks[start_idx] : bids[start_idx];
        double t1_tp = (uptrend == 1) ? close_price + brick_size : close_price - brick_size;
        double t1_sl = (uptrend == 1) ? close_price - brick_size : close_price + brick_size;
        double t1_peak = t1_entry;

        double t2_entry = open_price;
        double t2_tp = (uptrend == 1) ? open_price + 2 * brick_size : open_price - 2 * brick_size;
        double t2_sl = (uptrend == 1) ? open_price - brick_size : open_price + brick_size;
        double t2_peak = t2_entry;
        
        double t3_entry = (uptrend == 1) ? bids[start_idx] : asks[start_idx];
        double t3_tp = (uptrend == 1) ? close_price - 2 * brick_size : close_price + 2 * brick_size;
        double t3_sl = (uptrend == 1) ? close_price + brick_size : close_price - brick_size;
        double t3_peak = t3_entry;

        double t4_entry = (uptrend == 1) ? bids[start_idx] : asks[start_idx];
        double t4_tp = (uptrend == 1) ? close_price - 3 * brick_size : close_price + 3 * brick_size;
        double t4_sl = (uptrend == 1) ? close_price + brick_size : close_price - brick_size;
        double t4_peak = t4_entry;

        for (int i = start_idx; i < num_ticks; ++i) {
            if (t1_done && t2_done && t3_done && t4_done) break;
            
            double bid = bids[i];
            double ask = asks[i];
            
            if (uptrend == 1) { 
                // T1 is Long
                if (!t1_done) {
                    if (bid > t1_peak) t1_peak = bid;
                    if (bid >= t1_tp) { row.t1_win = 1; row.t1_y_mag = std::abs(t1_tp - t1_entry)/brick_size; t1_done = true; }
                    else if (bid <= t1_sl) { row.t1_win = 0; row.t1_y_mag = std::abs(t1_peak - t1_entry)/brick_size; t1_done = true; }
                }
                // T2 is Long
                if (!t2_done) {
                    if (row.t2_filled == 0) {
                        if (ask <= t2_entry) { row.t2_filled = 1; }
                        else if (bid <= t2_sl) { t2_done = true; } 
                    }
                    if (row.t2_filled == 1 && !t2_done) {
                        if (bid > t2_peak) t2_peak = bid;
                        if (bid >= t2_tp) { row.t2_win = 1; row.t2_y_mag = std::abs(t2_tp - t2_entry)/brick_size; t2_done = true; }
                        else if (bid <= t2_sl) { row.t2_win = 0; row.t2_y_mag = std::abs(t2_peak - t2_entry)/brick_size; t2_done = true; }
                    }
                }
                // T3 is Short
                if (!t3_done) {
                    if (ask < t3_peak) t3_peak = ask;
                    if (ask <= t3_tp) { row.t3_win = 1; row.t3_y_mag = std::abs(t3_entry - t3_tp)/brick_size; t3_done = true; }
                    else if (ask >= t3_sl) { row.t3_win = 0; row.t3_y_mag = std::abs(t3_entry - t3_peak)/brick_size; t3_done = true; }
                }
                // T4 is Short
                if (!t4_done) {
                    if (ask < t4_peak) t4_peak = ask;
                    if (ask <= t4_tp) { row.t4_win = 1; row.t4_y_mag = std::abs(t4_entry - t4_tp)/brick_size; t4_done = true; row.t4_filled = 1; }
                    else if (ask >= t4_sl) { row.t4_win = 0; row.t4_y_mag = std::abs(t4_entry - t4_peak)/brick_size; t4_done = true; row.t4_filled = 1; }
                }
            } else { 
                // T1 is Short
                if (!t1_done) {
                    if (ask < t1_peak) t1_peak = ask;
                    if (ask <= t1_tp) { row.t1_win = 1; row.t1_y_mag = std::abs(t1_entry - t1_tp)/brick_size; t1_done = true; }
                    else if (ask >= t1_sl) { row.t1_win = 0; row.t1_y_mag = std::abs(t1_entry - t1_peak)/brick_size; t1_done = true; }
                }
                // T2 is Short
                if (!t2_done) {
                    if (row.t2_filled == 0) {
                        if (bid >= t2_entry) { row.t2_filled = 1; }
                        else if (ask >= t2_sl) { t2_done = true; }
                    }
                    if (row.t2_filled == 1 && !t2_done) {
                        if (ask < t2_peak) t2_peak = ask;
                        if (ask <= t2_tp) { row.t2_win = 1; row.t2_y_mag = std::abs(t2_entry - t2_tp)/brick_size; t2_done = true; }
                        else if (ask >= t2_sl) { row.t2_win = 0; row.t2_y_mag = std::abs(t2_entry - t2_peak)/brick_size; t2_done = true; }
                    }
                }
                // T3 is Long
                if (!t3_done) {
                    if (bid > t3_peak) t3_peak = bid;
                    if (bid >= t3_tp) { row.t3_win = 1; row.t3_y_mag = std::abs(t3_tp - t3_entry)/brick_size; t3_done = true; }
                    else if (bid <= t3_sl) { row.t3_win = 0; row.t3_y_mag = std::abs(t3_peak - t3_entry)/brick_size; t3_done = true; }
                }
                // T4 is Long
                if (!t4_done) {
                    if (bid > t4_peak) t4_peak = bid;
                    if (bid >= t4_tp) { row.t4_win = 1; row.t4_y_mag = std::abs(t4_tp - t4_entry)/brick_size; t4_done = true; row.t4_filled = 1; }
                    else if (bid <= t4_sl) { row.t4_win = 0; row.t4_y_mag = std::abs(t4_peak - t4_entry)/brick_size; t4_done = true; row.t4_filled = 1; }
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

    FeatureLabelRow* generate_dataset(
        const double* bids, 
        const double* asks, 
        const int64_t* times_msc, 
        int num_ticks,
        double k_multiplier,
        int* out_num_rows
    ) {
        std::vector<FeatureLabelRow> all_rows;
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

            double best_anchor = find_optimal_anchor_v2(bids + lb_start, asks + lb_start, times_msc + lb_start, lb_count, day_open, brick_size);
            RenkoBuilder renko(best_anchor, brick_size);
            
            float rolling_history[5][30] = {0};

            for (int i = lb_start; i < day_start; ++i) {
                auto new_bricks = renko.update_tick(bids[i], asks[i], times_msc[i]);
                for (const auto& brick : new_bricks) {
                    float coarse[30];
                    compute_ancs(brick.intra_brick_ticks, 5, brick.open, brick_size, coarse);
                    for(int h=0; h<4; ++h) {
                        for(int j=0; j<30; ++j) rolling_history[h][j] = rolling_history[h+1][j];
                    }
                    for(int j=0; j<30; ++j) rolling_history[4][j] = coarse[j];
                }
            }

            int64_t prev_brick_time = (day_start > 0) ? times_msc[day_start - 1] : times_msc[0];

            for (int i = day_start; i < day_end; ++i) {
                double bid = bids[i];
                double ask = asks[i];
                int64_t t_msc = times_msc[i];

                auto new_bricks = renko.update_tick(bid, ask, t_msc);
                for (const auto& brick : new_bricks) {
                    brick_counter++;

                    FeatureLabelRow row;
                    row.brick_id = brick_counter;
                    row.timestamp = t_msc;
                    row.direction = brick.uptrend;
                    row.close_price = brick.close;
                    row.open_price = brick.open;
                    row.brick_size = brick_size;
                    row.brick_duration_seconds = (t_msc - prev_brick_time) / 1000LL;
                    
                    resolve_all_trades(i, num_ticks, bids, asks, times_msc, brick.uptrend, brick.close, brick.open, brick_size, row);

                    compute_ancs(brick.intra_brick_ticks, 10, brick.open, brick_size, row.ancs_fine);
                    compute_ancs(brick.intra_brick_ticks, 5, brick.open, brick_size, row.ancs_coarse);
                    compute_candle_features(brick.intra_brick_ticks, brick_size, brick.uptrend, row.candle_features);
                    compute_momentum_features(brick.intra_brick_ticks, brick_size, row.momentum);

                    for(int h=0; h<5; ++h) {
                        for(int j=0; j<30; ++j) {
                            row.history[h*30 + j] = rolling_history[h][j];
                        }
                    }

                    for(int h=0; h<4; ++h) {
                        for(int j=0; j<30; ++j) rolling_history[h][j] = rolling_history[h+1][j];
                    }
                    for(int j=0; j<30; ++j) rolling_history[4][j] = row.ancs_coarse[j];

                    all_rows.push_back(row);
                    prev_brick_time = t_msc;
                }
            } 
        } 

        *out_num_rows = all_rows.size();
        FeatureLabelRow* out_array = new FeatureLabelRow[*out_num_rows];
        std::copy(all_rows.begin(), all_rows.end(), out_array);
        return out_array;
    }

    void free_dataset(FeatureLabelRow* ptr) {
        delete[] ptr;
    }
} 
