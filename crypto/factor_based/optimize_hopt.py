import optuna

import pandas as pd


def sortino_ratio(returns: pd.Series, rf: float = 0.01) -> float:
    downside = returns[returns < rf]
    downside_std = downside.std()
    if downside_std == 0:
        return 0
    return (returns.mean() - rf) / downside_std




def optimize_with_optuna(factor_func):
    def wrapper(df: pd.DataFrame, n_trials: int = 30):
        def objective(trial):
            params = {
                "atr_period": trial.suggest_int('atr_period', 5, 30),
                "vol_multiplier": trial.suggest_float('vol_multiplier', 0.3, 2.0),
                "body_ratio_max": trial.suggest_float('body_ratio_max', 0.1, 1.0),
                "min_lower_shadow_ratio": trial.suggest_float('min_lower_shadow_ratio', 0.1, 0.7),
                "window":trial.suggest_int('window', 5, 20*24)
            }

            factor = factor_func(df, **params)
            position = factor

            if "target_returns" not in df.columns:
                raise ValueError("df must contain 'target_returns'")

            strategy_ret = position * df["target_returns"]
            return sortino_ratio(strategy_ret)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print("Best parameters:", best_params)

        return factor_func(df, **best_params)

    return wrapper


