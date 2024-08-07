import numpy as np
import pandas as pd
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


weights = {
    'review_scores_value': 0.2,
    'review_scores_rating': 0.2,
    'review_scores_accuracy': 0.15,
    'review_scores_cleanliness': 0.15,
    'review_scores_communication': 0.15,
    'review_scores_location': 0.10,
    'review_scores_checkin': 0.05
}

def get_top_recommendations(df, weights, min_reviews=10, min_reviews_per_month=0.1, top_n=10):
    """
    Verilen dataframe üzerinde ağırlıklı genel skoru hesaplar ve en iyi önerileri getirir.

    :param df: İnceleme verilerini içeren dataframe
    :param weights: İnceleme puanları için ağırlıklar sözlüğü
    :param min_reviews: Filtreleme için minimum yorum sayısı
    :param min_reviews_per_month: Filtreleme için minimum aylık yorum sayısı
    :param top_n: Gösterilecek öneri sayısı
    :return: Ağırlıklı skora göre sıralanmış en iyi öneriler
    """
    # Kopya oluştur
    df_copy = df.copy()

    # Genel skoru hesapla
    df_copy['weighted_score'] = (
        df_copy['review_scores_value'] * weights.get('review_scores_value', 0) +
        df_copy['review_scores_rating'] * weights.get('review_scores_rating', 0) +
        df_copy['review_scores_accuracy'] * weights.get('review_scores_accuracy', 0) +
        df_copy['review_scores_cleanliness'] * weights.get('review_scores_cleanliness', 0) +
        df_copy['review_scores_communication'] * weights.get('review_scores_communication', 0) +
        df_copy['review_scores_location'] * weights.get('review_scores_location', 0) +
        df_copy['review_scores_checkin'] * weights.get('review_scores_checkin', 0)
    )

    # Filtreleme
    filtered_df = df_copy[
        (df_copy['number_of_reviews'] > min_reviews) &
        (df_copy['reviews_per_month'] > min_reviews_per_month)
    ]

    # Skora göre sıralama yap
    top_recommendations = filtered_df.sort_values(
        by=['weighted_score', 'number_of_reviews', 'reviews_per_month'],
        ascending=[False, False, False]
    ).head(top_n)

    return top_recommendations


def data_prep(df):
    # df['id'] = df['id'].astype(str)
    df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)
    df['price'] = (df.groupby(['neighbourhood_cleansed'])['price'].transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

    df = df[df['has_availability'] == 't']  # Aktif olanları gösterelim

    rev_cols = [col for col in df.columns if 'review' in col]
    for col in rev_cols:
        df.loc[:, col] = df[col].fillna(0)

    return df


def main():
    listing = pd.read_csv('datasets/listings.csv')

    listing = listing[
        ['id', 'price', 'name', 'listing_url', 'minimum_nights', 'maximum_nights','host_is_superhost',
         'neighbourhood_cleansed', 'has_availability', 'number_of_reviews',
         'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness','review_scores_value',
         'review_scores_communication', 'review_scores_location', 'review_scores_checkin', 'reviews_per_month']]

    filtered_df = data_prep(listing)

    new_df = get_top_recommendations(filtered_df,weights)
    # Skora göre sıralama yap
    top_recommendations = new_df.sort_values(by=['weighted_score', 'number_of_reviews', 'reviews_per_month'],
                                                  ascending=False).head(10)
    # En yüksek genel skorları listeleme
    top_10_scores = top_recommendations[["id",'name','listing_url', 'weighted_score']].sort_values(by='weighted_score', ascending=False).head(10)

    #print(top_10_scores)
    joblib.dump(top_10_scores, "top_10_scores.pkl")
    return top_10_scores


if __name__ == "__main__":
    print("İşlem başladı")
    main()