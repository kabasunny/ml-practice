import pandas as pd
import numpy as np
from features.cycle_theory.fourier.fft_analysis import fft_analysis
from features.cycle_theory.peak_trough.detect_cycles import detect_cycles
from data_processing.detrend_prices import detrend_prices


def create_features_for_dates(dates, daily_prices, weekly_prices, monthly_prices, remove_trend):
    features = []
    feature_dates = []
    for date in dates:
        detrended_daily_prices = detrend_prices(daily_prices, remove_trend)
        detrended_weekly_prices = detrend_prices(weekly_prices, remove_trend)
        detrended_monthly_prices = detrend_prices(monthly_prices, remove_trend)

        recent_prices = {
            "daily": detrended_daily_prices.loc[:date].tail(90),
            "weekly": detrended_weekly_prices.loc[:date].tail(60),
            "monthly": detrended_monthly_prices.loc[:date].tail(36),
        }
        feature = create_individual_features(recent_prices)
        features.append(feature)
        feature_dates.append(date)
    return features, feature_dates


def create_individual_features(recent_prices):
    feature = {}
    for freq, prices in recent_prices.items():
        prefix = freq
        feature.update(process_frequency_features(prices, prefix))
    return feature


def process_frequency_features(prices, prefix):
    feature = {}
    if len(prices) > 0:
        # サイクル検出
        (
            _,
            troughs,
            _,
            _,
            _,
            _,
            avg_trough_cycle,
            median_trough_cycle,
            _,
            mode_trough_cycle,
        ) = detect_cycles(prices)
        if len(troughs) > 1:
            feature[f"{prefix}_avg_trough_cycle"] = avg_trough_cycle
            feature[f"{prefix}_med_trough_cycle"] = median_trough_cycle
            feature[f"{prefix}_mode_trough_cycle"] = mode_trough_cycle
        else:
            feature[f"{prefix}_avg_trough_cycle"] = np.nan
            feature[f"{prefix}_med_trough_cycle"] = np.nan
            feature[f"{prefix}_mode_trough_cycle"] = np.nan

        # FFT解析
        _, _, dominant_periods = fft_analysis(prices.values)
        feature[f"{prefix}_fft_dominant_period"] = (
            dominant_periods[0] if len(dominant_periods) > 0 else np.nan
        )
    else:
        feature[f"{prefix}_avg_trough_cycle"] = np.nan
        feature[f"{prefix}_med_trough_cycle"] = np.nan
        feature[f"{prefix}_mode_trough_cycle"] = np.nan
        feature[f"{prefix}_fft_dominant_period"] = np.nan
    return feature