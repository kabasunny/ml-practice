# features/feature_engineering.py

import pandas as pd
import numpy as np
from cycle_theory.fourier.fft_analysis import fft_analysis
from cycle_theory.peak_trough.detect_cycles import detect_cycles

def create_features(date, detrended_prices, detrended_weekly_prices, detrended_monthly_prices):
    feature = {}

    # 日足の特徴量
    past_daily_data = detrended_prices[:date][-60:]
    print(f"past_daily_data: {past_daily_data}")
    daily_peaks, daily_troughs, _, _, _, _, _, _, _, _ = detect_cycles(past_daily_data)
    if len(daily_troughs) > 1:
        daily_trough_cycles = np.diff(daily_troughs)
        feature['daily_avg_trough_cycle'] = np.mean(daily_trough_cycles)
        feature['daily_med_trough_cycle'] = np.median(daily_trough_cycles)
        feature['daily_mode_trough_cycle'] = pd.Series(daily_trough_cycles).mode()[0]
    else:
        feature['daily_avg_trough_cycle'] = np.nan
        feature['daily_med_trough_cycle'] = np.nan
        feature['daily_mode_trough_cycle'] = np.nan

    fft_period, fft_amplitude, dominant_periods = fft_analysis(past_daily_data.values)
    print("a")
    feature['daily_fft_max_amp'] = fft_amplitude.max()
    feature['daily_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan

    # 週足の特徴量
    past_weekly_data = detrended_weekly_prices[:date][-26:]
    print(f"past_weekly_data: {past_weekly_data}")
    weekly_peaks, weekly_troughs, _, _, _, _, _, _, _, _ = detect_cycles(past_weekly_data)
    if len(weekly_troughs) > 1:
        weekly_trough_cycles = np.diff(weekly_troughs)
        feature['weekly_avg_trough_cycle'] = np.mean(weekly_trough_cycles)
        feature['weekly_med_trough_cycle'] = np.median(weekly_trough_cycles)
        feature['weekly_mode_trough_cycle'] = pd.Series(weekly_trough_cycles).mode()[0]
    else:
        feature['weekly_avg_trough_cycle'] = np.nan
        feature['weekly_med_trough_cycle'] = np.nan
        feature['weekly_mode_trough_cycle'] = np.nan

    fft_period, fft_amplitude, dominant_periods = fft_analysis(past_weekly_data.values)
    print(f"5, fft_amplitude: {fft_amplitude}")
    feature['weekly_fft_max_amp'] = fft_amplitude.max()
    feature['weekly_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan

    # 月足の特徴量
    past_monthly_data = detrended_monthly_prices[:date][-12:]
    monthly_peaks, monthly_troughs, _, _, _, _, _, _, _, _ = detect_cycles(past_monthly_data)
    if len(monthly_troughs) > 1:
        monthly_trough_cycles = np.diff(monthly_troughs)
        feature['monthly_avg_trough_cycle'] = np.mean(monthly_trough_cycles)
        feature['monthly_med_trough_cycle'] = np.median(monthly_trough_cycles)
        feature['monthly_mode_trough_cycle'] = pd.Series(monthly_trough_cycles).mode()[0]
    else:
        feature['monthly_avg_trough_cycle'] = np.nan
        feature['monthly_med_trough_cycle'] = np.nan
        feature['monthly_mode_trough_cycle'] = np.nan

    fft_period, fft_amplitude, dominant_periods = fft_analysis(past_monthly_data.values)
    print("c")
    feature['monthly_fft_max_amp'] = fft_amplitude.max()
    feature['monthly_fft_dominant_period'] = dominant_periods[0] if len(dominant_periods) > 0 else np.nan

    return feature