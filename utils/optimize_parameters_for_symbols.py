from setting_stop.optimize_parameters import optimize_parameters


def optimize_parameters_for_symbols(symbol_signals, symbol_data_dict):
    optimal_params = []
    least_optimal_params = []
    rejected_params = []

    for symbol, signals in symbol_signals.items():
        daily_data = symbol_data_dict[symbol]
        for signal_date in signals:
            best_result, worst_result, _ = optimize_parameters(daily_data, signal_date)
            if not any(
                param["stop_loss_percentage"] == best_result["stop_loss_percentage"]
                and param["trailing_stop_trigger"]
                == best_result["trailing_stop_trigger"]
                and param["trailing_stop_update"] == best_result["trailing_stop_update"]
                for param in optimal_params
            ):
                optimal_params.append(best_result)
            else:
                rejected_params.append(best_result)
            if not any(
                param["stop_loss_percentage"] == worst_result["stop_loss_percentage"]
                and param["trailing_stop_trigger"]
                == worst_result["trailing_stop_trigger"]
                and param["trailing_stop_update"]
                == worst_result["trailing_stop_update"]
                for param in least_optimal_params
            ):
                least_optimal_params.append(worst_result)
            else:
                rejected_params.append(worst_result)

    return symbol_signals, optimal_params, least_optimal_params, rejected_params
