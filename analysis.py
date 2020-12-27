"""
Модуль анализа моделди на предмет ошибок
"""
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from utils import get_metrics

PATHDATA = "data/"
PATHMODELS = "models/"
PATHIMAGE = "images/"


def distance_id(x_train, x_current, metric="euclidean", weigths=None, n=20, fillna=-1):
    """Вычислить N ближайших соседей
    """
    x_train = np.nan_to_num(x_train, nan=fillna)
    x_current = np.nan_to_num(x_current, nan=fillna)
    scaler = StandardScaler()
    if weigths is not None:
        x_transformed = scaler.fit_transform(x_train) / weigths
        x_transformed_cur = scaler.transform(x_current) / weigths
    else:
        x_transformed = scaler.fit_transform(x_train)
        x_transformed_cur = scaler.transform(x_current)

    distances = pairwise_distances(x_transformed, x_transformed_cur, metric=metric).reshape(-1)
    if metric == "euclidean":
        indexes = np.argsort(distances)
    if metric == "cosine":
        indexes = np.argsort(-distances)

    distances = distances[indexes]
    return distances[:n], indexes[:n]


###################
# КЛАСС АНАЛИТИКИ
###################

class AnalysisResult:
    """
    Аналитика.
    """

    def __init__(self):
        """Fake constructor
        """
        self.booster_dict = {}
        self.shaps_all = {}
        self.shaps_feature_importance = {}
        self.feature_names = {}
        self.explainer = {}
        self.train_data = {}
        self.valid_data = {}
        self.shaps_valid_data = {}
        self.metrics = {}
        self.model_errors = {}
        self.ids_shaps = {}
        self.ids_pred = {}

    def get_booster(self, model_id):
        """Получить бустер по ID

        Arguments:
            model_id {str} -- уникальное имя модели
        """
        if model_id not in self.booster_dict:
            self.booster_dict[model_id] = lgb.Booster(model_file=PATHMODELS + model_id)
            self.booster_dict[model_id].params["objective"] = "binary"
            self.feature_names[model_id] = self.booster_dict[model_id].feature_name()
            self.explainer[model_id] = shap.TreeExplainer(self.booster_dict[model_id])
            self.ids_shaps[model_id] = {}
            self.ids_pred[model_id] = {}


    def read_model_data_set(self, model_id, type_set="valid"):
        """Считать данные, на которых модель обучалась и валидировалась
        """
        ParserError = pd.errors.ParserError
        if type_set == "valid":
            if model_id not in self.valid_data.keys():
                try:
                    self.valid_data[model_id] = pd.read_csv(PATHDATA + model_id[:-6] + "_valid.csv")
                except FileNotFoundError:
                    print("Файл с данными отсутствует!")
                except ParserError:
                    print("Данные не могут быть загружены!")

        if type_set == "train":
            if model_id not in self.train_data.keys():
                try:
                    self.train_data[model_id] = pd.read_csv(PATHDATA + model_id[:-6] + "_train.csv")
                except FileNotFoundError:
                    print("Файл с данными отсутствует!")
                except ParserError:
                    print("Данные не могут быть загружены!")
    # pylint: disable=line-too-long
    def get_all_shaps(self, model_id, param_dict):
        """Вычислить shap values
        """
        sample = param_dict["sample"]
        recalc_shap = param_dict["recalc_shap"]
        save_image_shap_all = param_dict["save_image_shap_all"]

        try:

            self.get_booster(model_id)
            self.read_model_data_set(model_id, "valid")
            feats = self.feature_names[model_id]
            x_val = self.valid_data[model_id][feats].sample(frac=sample)
            if (model_id not in self.shaps_all) or recalc_shap:
                self.shaps_valid_data[model_id] = self.valid_data[model_id][feats].sample(frac=sample)
                self.shaps_all[model_id] = self.explainer[model_id].shap_values(self.shaps_valid_data[model_id])[1]
                self.shaps_feature_importance[model_id] = np.array(x_val.columns[np.argsort(-np.abs(self.shaps_all[model_id]).mean(0))])
                print(self.shaps_feature_importance[model_id])
                print("shaps вычислены!")

            fig = plt.figure(figsize=(40, 40), tight_layout=True)
            shap.summary_plot(self.shaps_all[model_id], self.shaps_valid_data[model_id], max_display=len(x_val.columns), show=False)

            if save_image_shap_all:
                print("prepare to save")
                fig.savefig(PATHIMAGE + model_id + "_shap.png", bbox_inches="tight")
                print("shap-plot saved!")
            return fig

        except FileNotFoundError:
            print("Неверный путь")

    # pylint: disable=line-too-long
    def validate_model(self, model_id, target="price", error_ratio=0.75):
        """
        Валидация модели.
        Сбор ошибок
        """

        if model_id not in self.metrics:
            self.get_booster(model_id)
            self.read_model_data_set(model_id, type_set="valid")
            feats = self.feature_names[model_id]
            x_val, y_val = self.valid_data[model_id][feats], self.valid_data[model_id][target]
            y_pred, mse, mape = get_metrics(self.booster_dict[model_id], x_val, y_val)
            self.metrics[model_id] = {
                "mse": mse,
                "mape": mape,
                "len_valid": len(x_val),
                "mean_price": np.mean(y_val),
                "mean_pred" : np.mean(y_pred),
                "y_val": y_val,
                "y_pred": y_pred
            }

            self.valid_data[model_id]["y_pred"] = y_pred
            self.valid_data[model_id]["true_target"] = y_val

            self.valid_data[model_id]["y_pred_true_target_ape1"] = np.abs(y_pred - y_val) / (y_val + 1e-6)
            self.valid_data[model_id]["y_pred_true_target_ape2"] = np.abs(y_pred - y_val) / (y_pred + 1e-6)

            self.model_errors[model_id] = self.valid_data[model_id][(self.valid_data[model_id]["y_pred_true_target_ape1"] > error_ratio) | (self.valid_data[model_id]["y_pred_true_target_ape2"] > error_ratio)]

    def histogramm_feature(self, model_id, id_df, feature_name, nan_like=-1, target="price", id_col="id", n_closest=1000):
        """
        Гистограмма по ближайшим значениям к прогнозной и истинной ценам
        """
        self.read_model_data_set(model_id, type_set="train")
        x_train_temp = self.train_data[model_id].copy()
        x_train_temp.fillna(nan_like, inplace=True)
        id_df.fillna(nan_like, inplace=True)

        current_id = id_df[id_col].values[0]
        current_value = id_df[feature_name].values[0]

        self.current_shaps(model_id, id_df)
        current_shaps = self.ids_shaps[model_id][current_id]

        current_pred = id_df["y_pred"].values[0]
        current_target = id_df[target].values[0]

        x_train_temp["delta true target"] = np.abs(x_train_temp[target] - current_target)
        x_train_temp["delta pred target"] = np.abs(x_train_temp[target] - current_pred)

        x_feat_close_true = x_train_temp.sort_values("delta true target")[feature_name].values[:n_closest]
        x_feat_close_pred = x_train_temp.sort_values("delta pred target")[feature_name].values[:n_closest]


        fig_close_true = plt.figure()
        plt.hist(x_feat_close_true, bins=30, label=f"Распределение {feature_name} для близких к истинном значению")
        plt.plot(current_value, 0, "o", c="r", label=f"Значение {feature_name} для {current_id}", linewidth=2, markersize=8)
        plt.legend(loc='best', prop={'size': 7})
        plt.grid()

        fig_close_pred = plt.figure()
        plt.hist(x_feat_close_pred, bins=30, label=f"Распределение {feature_name} для близких к прогнозному значению")
        plt.plot(current_value, 0, "o", c="r", label=f"Значение {feature_name} для {current_id}", linewidth=2, markersize=8)
        plt.legend(loc='best', prop={'size': 7})
        plt.grid()

        fig_force_plot = shap.force_plot(self.explainer[model_id].expected_value[1], current_shaps, id_df[self.feature_names[model_id]], matplotlib=True)

        return fig_close_true, fig_close_pred, fig_force_plot

    def histogramm(self, y_val, y_pred):
        """
        Гистограмма
        """
        fig_ = plt.figure()
        plt.hist(y_val, bins=20, label=f"Распределение истинной цены")
        plt.hist(y_pred, bins=20, label=f"Распределение прогнозной цены", alpha=0.8)
        plt.legend(loc='best', prop={'size': 10})
        plt.grid()

        return fig_

    def table_score(self, model_id, id_df):
        """
        """
        self.booster_dict[model_id]
        current_id = id_df["id"].values[0]

        current_pred_ = id_df["y_pred"].values[0]
        true_pred = id_df["true_target"].values[0]

        table = f"""|id| Оценка стоимости моделью | Стоимость |
        |-|-:|-:|
        |{current_id} |{round(current_pred_, 4)}|{round(true_pred, 4)}|
        """
        return table

    def current_shaps(self, model_id, id_df, id_col="id"):

        id_ = id_df[id_col].values[0]

        if id_ not in self.ids_shaps[model_id].keys():
            self.ids_shaps[model_id][id_] = self.explainer[model_id].shap_values(
                id_df[self.feature_names[model_id]]
            )[1]

    # pylint: disable=too-many-arguments
    def closest_ids(self, model_id, id_df, param_dict_shap, n_top_feats=10, num_closest=20, method="Текущее shap-value"):
        """show closest
        """
        self.read_model_data_set(model_id, type_set="train")
        if method == "current shap":
            _, _, shap_feats_warm_sorted, shap_feats_cold_sorted = self.get_sorted_features(
                model_id, id_df, n=n_top_feats
                )
            feats_for_res = list(shap_feats_cold_sorted[:n_top_feats]) + list(shap_feats_warm_sorted[:n_top_feats])
            # weigths = np.array(list(shap_values_cold_sorted[:n_top_feats]) + list(shap_values_warm_sorted[:n_top_feats]))
        else:
            self.get_all_shaps(model_id, param_dict_shap)
            feats_for_res = np.array(self.shaps_feature_importance[model_id])[:n_top_feats]
            print(feats_for_res)
            # weigths = None

        x_train_all = np.array(self.train_data[model_id][feats_for_res].values).astype("float32")
        current_msis = np.array(id_df[feats_for_res].values).astype("float32")
        _, i_all = distance_id(x_train_all, current_msis, "euclidean", None, num_closest)

        df_closest = self.train_data[model_id][
            list(feats_for_res) + ["price"] + ["id"]
        ].loc[i_all]
        current_df = id_df[list(feats_for_res) + ["price"] + ["id"]]
        return df_closest, current_df

    def __sorted_shaps_current_msis__(self, model_id, shap_values_tmp_):
        """
        Вычисление теплых и холодных признаков
        """
        shap_feats_warm = np.array(self.feature_names[model_id])[shap_values_tmp_ > 0]
        shap_values_res = np.abs(shap_values_tmp_[shap_values_tmp_ > 0])
        idx = np.argsort(-shap_values_res)
        shap_feats_warm_sorted = shap_feats_warm[idx]
        shap_values_warm_sorted = np.abs(shap_values_res[idx])

        shap_feats_cold = np.array(self.feature_names[model_id])[shap_values_tmp_ < 0]
        shap_values_res = np.abs(shap_values_tmp_[shap_values_tmp_ < 0])
        idx = np.argsort(-shap_values_res)
        shap_feats_cold_sorted = shap_feats_cold[idx]
        shap_values_cold_sorted = np.abs(shap_values_res[idx])
        return (
            shap_values_warm_sorted,
            shap_values_cold_sorted,
            shap_feats_warm_sorted,
            shap_feats_cold_sorted,
        )

    def get_sorted_features(self, model_id, id_df, id_col="id", n=10):

        current_id = id_df[id_col].values[0]

        self.current_shaps(model_id, id_df)
        current_shaps = self.ids_shaps[model_id][current_id]

        shap_values_warm_sorted, shap_values_cold_sorted, shap_feats_warm_sorted, shap_feats_cold_sorted = self.__sorted_shaps_current_msis__(
            model_id, current_shaps[0]
        )
        return (
            shap_values_warm_sorted[:n],
            shap_values_cold_sorted[:n],
            shap_feats_warm_sorted[:n],
            shap_feats_cold_sorted[:n],
        )


if __name__ == "__main__":
    pass
