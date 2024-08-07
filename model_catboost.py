import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def outlier_thresholds(dataframe, col_name, q1=0.15, q3=0.85):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


le = LabelEncoder()


def label_encoder(dataframe, binary_col):
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def quick_missing_imp(data, num_method="median", cat_length=20, target="price"):
    variables_with_na = [col for col in data.columns if
                         data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x,
                      axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


def amsterdam_data_prep(awd_df):
    # En önemli amenities sütunlarını ekleyelim
    awd_df = awd_df.assign(
        has_private_entrance=awd_df['amenities'].apply(lambda x: x.find('Private entrance') != -1),
        has_self_checkin=awd_df['amenities'].apply(lambda x: x.find('Self check-in') != -1),
        has_kitchen=awd_df['amenities'].apply(lambda x: x.find('Kitchen') != -1),
        has_bathtub=awd_df['amenities'].apply(lambda x: x.find('Bathtub') != -1),
        has_host_greeting=awd_df['amenities'].apply(lambda x: x.find('Host greets you') != -1),
        has_dishwasher=awd_df['amenities'].apply(lambda x: x.find('Dishwasher') != -1),
        has_longterm=awd_df['amenities'].apply(lambda x: x.find('Long term stays allowed') != -1),
        has_fireplace=awd_df['amenities'].apply(lambda x: x.find('Indoor fireplace') != -1),
        has_parking=awd_df['amenities'].apply(lambda x: x.find('Free parking on premises') != -1)
    )

    awd_df = awd_df.drop(['amenities'], axis=1)

    # use string.replace to get rid of string items in price column
    awd_df = awd_df.assign(price=awd_df['price'].str.replace(r'$', ''))
    awd_df = awd_df.assign(price=awd_df['price'].str.replace(r',', ''))

    # Set price as float type
    awd_df['price'] = awd_df['price'].astype(float)

    # Değiştirilmesi gereken ifadeler listesi
    replace_terms = ['Private half-bath', 'Half-bath', 'Shared half-bath']
    delete_term = ["baths", "shared bath", "private", "bath", "shared"]
    # İfadeleri 0.5 ile değiştir
    for term in replace_terms:
        awd_df["bathrooms_text"] = awd_df["bathrooms_text"].str.replace(term, '0.5')
        # "baths" kelimesini kaldır
    for term in delete_term:
        awd_df["bathrooms_text"] = awd_df["bathrooms_text"].str.replace(term, '')

        # Sütunu float türüne çevir
    awd_df["bathrooms_text"] = awd_df["bathrooms_text"].astype(float)
    awd_df.rename(columns={'bathrooms_text': 'bathrooms'}, inplace=True)
    awd_df.dropna(subset=["bathrooms"], inplace=True)

    # has availibity düşür
    awd_df = awd_df[awd_df['has_availability'] == 't']
    awd_df.drop("has_availability", axis=1, inplace=True)

    awd_df["bedrooms"].fillna(value=0, inplace=True)
    awd_df["beds"].fillna(awd_df['bedrooms'].astype(int), inplace=True)
    awd_df["beds"].astype(int)

    awd_df["host_is_superhost"].fillna(value="f", inplace=True)

    awd_df = quick_missing_imp(awd_df)

    # Değişknelerin tipini getirelim
    cat_cols, num_cols, cat_but_car = grab_col_names(awd_df)
    num_cols = [col for col in num_cols if col not in ['id', 'host_id', 'calendar_last_scraped']]

    binary_cols = [col for col in awd_df.columns if
                   awd_df[col].dtype not in [int, float] and awd_df[col].nunique() == 2]

    for col in binary_cols:
        awd_df = label_encoder(awd_df, col)

    ohe_cols = [col for col in cat_cols if 10 >= awd_df[col].nunique() > 2]
    awd_df = one_hot_encoder(awd_df, ohe_cols, drop_first=True)

    # Aykırı değerleri temizleyelim
    for col in num_cols:
        replace_with_thresholds(awd_df, col)

    categories_to_replace = [
        "Camper/RV", "Private room in barn", "Entire chalet", "Private room in bungalow",
        "Shared room in boat", "Cave", "Yurt", "Earthen home", "Barn", "Private room in casa particular",
        "Private room in villa", "Private room", "Room in hostel", "Entire cabin", "Tiny home",
        "Room in serviced apartment",
        "Private room in farm stay", "Entire cottage", "Private room in earthen home", "Shared room in houseboat",
        "Casa particular",
        "Private room in nature lodge", "Shared room in condo", "Shared room in home", "Shared room in hotel",
        "Shared room in rental unit", "Private room in cabin", "Private room in tiny home"]

    # Belirtilen kategorileri "Other" ile değiştiren fonksiyon
    def replace_with_other(dataframe, variable, categories):
        dataframe[variable] = dataframe[variable].apply(lambda x: 'Other' if x in categories else x)
        return dataframe

    # Sadece property_count değişkenine uygulama
    awd_df = replace_with_other(awd_df, 'property_type', categories_to_replace)


    # Önem düzeyine sahip review değişkenlerini ağırlıklandırıp genel bir score değişkeni oluşturalım
    weights = {
        'review_scores_rating': 0.20,
        'review_scores_communication': 0.20,
        'review_scores_location': 0.50,
    }

    # Bu ağırlıklarla genel skoru hesaplayın
    def calculate_weighted_score(df, weights):
        score = 0
        for col, weight in weights.items():
            if col in df.columns:
                score += df[col] * weight
        return score

        # Genel skoru dataframe'e ekleyin

    awd_df['general_score'] = calculate_weighted_score(awd_df, weights)

    awd_df = awd_df.drop(['review_scores_rating', 'review_scores_communication', 'review_scores_location'], axis=1)

    model_df = awd_df[['host_is_superhost', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
                       'price', 'minimum_nights', 'maximum_nights', 'availability_60', 'availability_90',
                       'number_of_reviews', 'review_scores_value', 'has_private_entrance', 'has_self_checkin',
                       'has_kitchen',
                       'has_bathtub', 'has_host_greeting', 'has_dishwasher', 'has_longterm', 'has_fireplace',
                       'has_parking',
                       'instant_bookable', 'reviews_per_month', 'room_type_Hotel room', 'room_type_Private room',
                       'room_type_Shared room', "general_score"]]

    # Değişkenleri standartlaştırınız.
    # Scaling
    sc = MinMaxScaler((0, 1))
    model_scaling = sc.fit_transform(model_df)
    model_df = pd.DataFrame(model_scaling, columns=model_df.columns)

    train_df = model_df[model_df['price'].notnull()]
    test_df = model_df[model_df['price'].isnull()]

    y = np.log1p(train_df['price'])
    X = train_df.drop(["price"], axis=1)

    return X, y


def catboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    cat_model = CatBoostRegressor()
    cat_rmse = np.mean(np.sqrt(-cross_val_score(cat_model, X, y, cv=5, scoring="neg_mean_squared_error")))

    # Randomized Search için parametre grid'i belirle
    random_cat_param_dist = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 50, 100]
    }

    # Randomized Search ile hiperparametre optimizasyonu
    random_cat_search = RandomizedSearchCV(estimator=cat_model,
                                           param_distributions=random_cat_param_dist,
                                           n_iter=100, cv=5,
                                           scoring='neg_mean_squared_error',
                                           n_jobs=-1, random_state=42)

    random_cat_search.fit(X_train, y_train)

    print("Best parameters found: ", random_cat_search.best_params_)
    print("Best RMSE score: ", -random_cat_search.best_score_)

    # Best parameters found:  {'learning_rate': 0.05, 'l2_leaf_reg': 1, 'iterations': 300, 'depth': 6, 'border_count': 32}
    # Best RMSE score:  0.009508592570946463

    final_model = cat_model.set_params(**random_cat_search.best_params_).fit(X, y)

    print(f"İlk RMSE: {cat_rmse}")  # 0.10038272243541443
    rmse_new_cat = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"Yeni RMSE: {rmse_new_cat}")
    # Yeni RMSE:  0.09973612230790614

    return final_model


def main():
    amsterdam_list = pd.read_csv("datasets/listings.csv")

    df = amsterdam_list.drop(
        ['listing_url', 'name', 'description', 'calendar_updated', 'latitude', 'longitude', 'last_scraped', 'scrape_id',
         'source', 'host_url', 'host_location', 'host_about', 'host_thumbnail_url', 'host_total_listings_count',
         'host_neighbourhood', 'host_verifications',
         'host_identity_verified', 'host_picture_url', 'host_has_profile_pic',
         'calculated_host_listings_count_shared_rooms', 'license', 'picture_url',
         'neighbourhood_group_cleansed', 'neighborhood_overview', 'neighbourhood', 'minimum_minimum_nights',
         'maximum_minimum_nights', 'minimum_maximum_nights',
         'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calculated_host_listings_count',
         'host_response_time', 'host_response_rate', 'host_acceptance_rate', "bathrooms",
         "calculated_host_listings_count_entire_homes", 'first_review', 'last_review',
         "calculated_host_listings_count_private_rooms", "host_listings_count", "calendar_last_scraped",
         "number_of_reviews_l30d", "number_of_reviews_ltm"], axis=1)

    x, y = amsterdam_data_prep(df)
    final_model = catboost_model(x, y)
    joblib.dump(final_model, "final_model.pkl")
    return final_model


if __name__ == "__main__":
    print("İşlem başladı")
    main()

