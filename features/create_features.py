# features/feature_engineering.py
import pandas as pd
import numpy as np
from cycle_theory.fourier.fft_analysis import fft_analysis
from cycle_theory.peak_trough.detect_cycles import detect_cycles

def create_features(date, detrended_prices, detrended_weekly_prices, detrended_monthly_prices):
    feature = {}

    # detect_cycles()　戻り値
        # peaks,  # インデックス
        # troughs,  # インデックス
        # avg_peak_cycle,
        # median_peak_cycle,
        # mean_absolute_error_peak,
        # mode_peak_cycle,
        # avg_trough_cycle,
        # median_trough_cycle,
        # mean_absolute_error_trough, # 誤差率
        # mode_trough_cycle,

    # fft_analysis()　戻り値
        # fft_period, # 
        # fft_amplitude, #
        # dominant_periods # 支配的な周期の配列
    

    # 日足の特徴量
    _, daily_troughs, _, _, _, _, avg_trough_cycle, median_trough_cycle, _, mode_trough_cycle = detect_cycles(detrended_prices)
    if len(daily_troughs) > 1:
        feature['daily_avg_trough_cycle'] = avg_trough_cycle
        feature['daily_med_trough_cycle'] = median_trough_cycle
        feature['daily_mode_trough_cycle'] = mode_trough_cycle
    else:
        feature['daily_avg_trough_cycle'] = np.nan
        feature['daily_med_trough_cycle'] = np.nan
        feature['daily_mode_trough_cycle'] = np.nan

    _, _, dominant_periods = fft_analysis(detrended_prices.values)
    feature['daily_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan

    # 週足の特徴量
    _, weekly_troughs, _, _, _, _, avg_trough_cycle, median_trough_cycle, _, mode_trough_cycle = detect_cycles(detrended_weekly_prices)
    if len(weekly_troughs) > 1:
        feature['weekly_avg_trough_cycle'] = avg_trough_cycle
        feature['weekly_med_trough_cycle'] = median_trough_cycle
        feature['weekly_mode_trough_cycle'] = mode_trough_cycle
    else:
        feature['weekly_avg_trough_cycle'] = np.nan
        feature['weekly_med_trough_cycle'] = np.nan
        feature['weekly_mode_trough_cycle'] = np.nan

    _, _, dominant_periods = fft_analysis(detrended_weekly_prices.values)
    feature['weekly_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan

    # 月足の特徴量
    _, monthly_troughs, _, _, _, _, avg_trough_cycle, median_trough_cycle, _, mode_trough_cycle = detect_cycles(detrended_monthly_prices)
    if len(monthly_troughs) > 1:
        feature['monthly_avg_trough_cycle'] = avg_trough_cycle
        feature['monthly_med_trough_cycle'] = median_trough_cycle
        feature['monthly_mode_trough_cycle'] = mode_trough_cycle
    else:
        feature['monthly_avg_trough_cycle'] = np.nan
        feature['monthly_med_trough_cycle'] = np.nan
        feature['monthly_mode_trough_cycle'] = np.nan

    _, _, dominant_periods = fft_analysis(detrended_monthly_prices.values)
    feature['monthly_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan

    return feature