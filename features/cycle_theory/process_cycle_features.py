import numpy as np
from features.cycle_theory.fourier.fft_analysis import fft_analysis
from features.cycle_theory.peak_trough.detect_cycles import detect_cycles


def process_cycle_features(prices, prefix):
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
            feature[f"{prefix}_avg_cycle"] = avg_trough_cycle
            # feature[f"{prefix}_med_cycle"] = median_trough_cycle
            # feature[f"{prefix}_mode_cycle"] = mode_trough_cycle
        else:
            feature[f"{prefix}_avg_cycle"] = np.nan
            # feature[f"{prefix}_med_cycle"] = np.nan
            # feature[f"{prefix}_mode_cycle"] = np.nan

        # FFT解析
        _, _, dominant_periods = fft_analysis(prices.values)
        feature[f"{prefix}_fft_dom_pri"] = (
            dominant_periods[0] if len(dominant_periods) > 0 else np.nan
        )
    else:
        feature[f"{prefix}_avg_cycle"] = np.nan
        # feature[f"{prefix}_med_cycle"] = np.nan
        # feature[f"{prefix}_mode_cycle"] = np.nan
        feature[f"{prefix}_fft_dom_pri"] = np.nan
    return feature
