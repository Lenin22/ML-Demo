"""
Инструмент аналитика
"""
import streamlit as st
from analysis import AnalysisResult


#########################################
# Функции - реакции виджетов
#########################################
# pylint: disable=no-value-for-parameter
def on_click_execute(model_id, pars_dict_1):
    """Click button
    """
    if st.sidebar.button("Выполнить"):
        if RESEARCH_DICT["shaps_all"]:
            fig_shap = analys_service.get_all_shaps(model_id, pars_dict_1)

            if pars_dict_1["show_shap_all"]:
                st.markdown("### Наиболее значимые признаки (SHAP-VALUES ГРАФИК):")
                st.pyplot(fig_shap)
        if RESEARCH_DICT["valid"]:
            analys_service.validate_model(model_id)
            st.markdown("### Немного метрик:")
            metrics_ = analys_service.metrics[model_id]
            table = f"""|Имя|Значение|
        |-:|-:|
        |Имя модели |{model_id}|
        |Длина выборки для валидации |{round(metrics_["len_valid"], 4)}|
        |Средняя цена выборки для валидации |{round(metrics_["mean_price"], 4)}|
        |Средняя прогнозная цена выборки для валидации |{round(metrics_["mean_pred"], 4)}|
        |MSE выборки для валидации|{round(metrics_["mse"], 4)}|
        |MAPE выборки для валидации|{round(metrics_["mape"], 4)}|
        """
            st.markdown(table)
            fig_hist = analys_service.histogramm(metrics_["y_val"], metrics_["y_pred"])
            st.pyplot(fig_hist)

# pylint: disable=no-value-for-parameter
def choose_research_model(research):
    """
    Выбор исследования
    """
    pars_dict_1 = {
        "sample": 0.5,
        "recalc_shap": False,
        "save_image_shap_all": False,
        "show_shap_all": False
        }
    # pylint: disable=no-value-for-parameter
    if research == RESEARCHES[0]:
        RESEARCH_DICT["shaps_all"] = True
        str1 = "Доля выборки для расчета shap-values"
        pars_dict_1 = {
            "sample": st.sidebar.slider(label=str1, min_value=0.0, max_value=0.999, value=0.5),
            "recalc_shap": st.sidebar.checkbox(label="Пересчитать shap-values для модели"),
            "save_image_shap_all": st.sidebar.checkbox("Сохранить график shap-values"),
            "show_shap_all": st.sidebar.checkbox("Показать график shap-values"),
        }

    if research == RESEARCHES[1]:
        RESEARCH_DICT["valid"] = True
    return pars_dict_1

# pylint: disable=no-value-for-parameter
def choose_research_id(model_id, research, id_df, param_dict_shap):
    """
    Выбор исследования
    """
    if id_df is None:
        return

    if research == RESEARCHES_ABON[0]:
        show_histogramms(model_id, id_df)

    if research == RESEARCHES_ABON[1]:
        method = st.sidebar.radio("Значимые признаки:", ["Текущее shap-value", "Все shap-values"])
        num_neib = st.sidebar.slider("Число ближайших соседей", min_value=1, max_value=40, value=20)
        # pylint: disable=line-too-long
        n_top_feats = st.sidebar.slider("Число значимых признаков", min_value=1, max_value=40, value=10)

        if st.sidebar.button("Вычислить расстояние"):
            show_closest(model_id, id_df, param_dict_shap, n_top_feats, num_neib, method)

# pylint: disable=no-value-for-parameter
def show_list_id(model_id):
    """show_list_id
    """
    if model_id not in analys_service.model_errors:
        st.markdown("## Проведите валидацию модели")
        return None

    fail_df_ = analys_service.model_errors[model_id]
    all_df_ = analys_service.valid_data[model_id]
    ids_fail = ["Выберите id"] + list(fail_df_["id"].values.astype("str"))
    ids_all = ["Выберите id"] + list(all_df_["id"].values.astype("str"))
    st.sidebar.markdown(f"Сильно ошибочных id: {len(ids_fail) - 1}")
    if st.sidebar.checkbox("Показать только ошибочные id"):
        id_ = st.sidebar.selectbox("Сильно ошибочные id:", ids_fail)
    else:
        id_ = st.sidebar.selectbox("Все id:", ids_all)

    if id_ == "Выберите id":
        return None
    return all_df_[all_df_["id"].isin([id_])]

# pylint: disable=no-value-for-parameter
def show_table(model_id, id_df):
    """show_table
    """
    table1 = analys_service.table_score(model_id, id_df)
    st.markdown("Таблица результатов")
    st.markdown(table1)

# pylint: disable=no-value-for-parameter
def show_histogramms(model_id, id_df):
    """show_histogramms
    """

    _, _, shap_feats_warm_sorted, shap_feats_cold_sorted = analys_service.get_sorted_features(
        model_id, id_df
    )
    feat_zero = "Выберите признак"
    feats_warm = [feat_zero] + list(shap_feats_warm_sorted)
    feats_cold = [feat_zero] + list(shap_feats_cold_sorted)
    warm_cold_radio = st.sidebar.radio("Тип признаков для абонента", WARM_COLD)
    if warm_cold_radio == WARM_COLD[0]:
        feature_name = st.sidebar.selectbox("", feats_warm)
    if warm_cold_radio == WARM_COLD[1]:
        feature_name = st.sidebar.selectbox("", feats_cold)
    if feature_name != feat_zero:
        fig_true, fig_pred, fig_force_plot = analys_service.histogramm_feature(
            model_id, id_df, feature_name
        )

        show_table(model_id, id_df)
        st.markdown("Гистограмма выбранного признака для ближайших к истинной цене:")
        st.pyplot(fig_true)
        st.markdown("Гистограмма выбранного признака для ближайших к прогнозной цене:")
        st.pyplot(fig_pred)
        st.markdown("Наиболее значимые признаки для id (FORCE PLOT)")
        st.pyplot(fig_force_plot)

# pylint: disable=too-many-arguments
# pylint: disable=line-too-long
def show_closest(model_id, id_df, param_dict_shap, n_top_feats=10, num_neib=20, method="current shap"):
    """show_closest
    """
    # pylint: disable=line-too-long
    df_closest, current_df = analys_service.closest_ids(model_id, id_df, param_dict_shap, n_top_feats, num_neib, method)
    show_table(model_id, id_df)
    st.write("Важнейшие признаки id:")
    st.write(current_df)

    st.write("Важнейшие признаки ближайших соседей:")
    st.write(df_closest)
    st.write(f"Средняя цена ближайших соседей: {df_closest['price'].mean()}")


#########################################
# Константы
#########################################

MODELS = ["model_1.model", "model_2.model", "no model yet"]
RESEARCHES = ["Вычислить shap-values", "Валидировать модель"]  # pylint: disable=invalid-name
RESEARCHES_ABON = ["Гистограммы (+ force plot)", "Расстояния"]  # pylint: disable=invalid-name
RESEARCH_DICT = {"shaps_all": False, "valid": False}  # pylint: disable=invalid-name
WARM_COLD = ["Повышающие", "Занижающие"]

#########################################
# Загрузка сервисов
#########################################

# pylint: disable=invalid-name
analys_service = st.cache(AnalysisResult, allow_output_mutation=True)()


#########################################
# Основная функция
#########################################
# pylint: disable=no-value-for-parameter
def main():
    """
    Функция - построение веб-формы
    """
    st.markdown("# Модуль анализа модели")

    st.sidebar.markdown("# Исследование модели")
    model_id = st.sidebar.selectbox("Выберите модель для анализа", MODELS)
    st.sidebar.markdown("## Выберите исследование модели:")
    research = st.sidebar.radio("", RESEARCHES, index=1)
    st.sidebar.markdown(f"Настройки выбранного исследования ({research}):")
    pars_dict_1 = choose_research_model(research)
    print(pars_dict_1)
    on_click_execute(model_id, pars_dict_1)

    st.sidebar.markdown("# Исследование конкретных id")

    print(analys_service.train_data.keys())

    id_df = show_list_id(model_id)
    st.sidebar.markdown("## Выберите исследование id:")
    research_abon = st.sidebar.radio("", RESEARCHES_ABON)
    choose_research_id(model_id, research_abon, id_df, pars_dict_1)


if __name__ == "__main__":
    main()
