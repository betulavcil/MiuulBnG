import numpy as np
import pandas as pd
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

##########################################################################################################
weights = {
    'review_scores_value': 0.2,
    'review_scores_rating': 0.2,
    'review_scores_accuracy': 0.15,
    'review_scores_cleanliness': 0.15,
    'review_scores_communication': 0.15,
    'review_scores_location': 0.10,
    'review_scores_checkin': 0.05
}

def get_top_recommendations(df, weights, min_reviews=10, min_reviews_per_month=0.1, top_n=1000):
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


##########################################################################################################
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    return analyzer.polarity_scores(text)['compound']  # -1 (negatif) ile 1 (pozitif) arasında bir değer


def best_comments(df):
    df['vader_sentiment'] = df['comments'].apply(get_vader_sentiment)
    vader_sentiments = df['vader_sentiment'].tolist()
    avg_sentiment2 = df.groupby('id')['sentiment2'].mean().reset_index()

    top_comments = avg_sentiment2.sort_values(by='sentiment2', ascending=False).head(10)

##########################################################################################################
def data_prep(df):
    df['description'] = df['description'].fillna(df['name'])

    df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)
    df['price'] = (df.groupby(['neighbourhood_cleansed'])['price']
                   .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

    rev_cols = [col for col in df.columns if 'review' in col]
    for col in rev_cols:
        df.loc[:, col] = df[col].fillna(0)

    df = df.dropna()
    df = df[df['has_availability'] == 't']

    return df


def main(name):

    listing_detail = pd.read_csv('listings.csv')
    reviews_comment = pd.read_csv('reviews.csv')

    listing_detail = listing_detail.rename(columns={'id': 'listing_id'})
    df = reviews_comment.merge(listing_detail, how="left", on="listing_id")

    df = df[
        ['listing_id', 'id', 'name', 'description','picture_url', 'date', 'comments', 'listing_url','host_id', 'host_name', 'host_since', 'neighbourhood_cleansed',
         'property_type', 'room_type', 'bedrooms', 'price', 'minimum_nights', 'comments','maximum_nights','price',
         'has_availability', 'first_review', 'last_review', 'availability_30', 'availability_365',
         'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
         'review_scores_communication', 'review_scores_location', 'review_scores_checkin', 'instant_bookable',
         'reviews_per_month','review_scores_value']]

    prep_df = data_prep(df)

#######################################################################################################################
    # En çok yorum alan ve en iyi ratinge sahip listeyi getirir
    new_df = get_top_recommendations(prep_df, weights)
    # Skora göre sıralama yap
    top_recommendations = new_df.sort_values(by=['weighted_score', 'number_of_reviews', 'reviews_per_month'],ascending=False)
    # En yüksek genel skorları listeleme
    top_scores = top_recommendations[['listing_id', 'name', 'listing_url','number_of_reviews', 'weighted_score']].sort_values(by='weighted_score', ascending=False)
##################################################################################################################3
    rev_df = reviews_comment.copy()
    rev_df.dropna(inplace=True)

    name_counts = pd.DataFrame(
        prep_df['name'].value_counts())  # en çok kiralanan evlerin sıralaması

    # İki veri çerçevesini 'listing_id' sütunu üzerinden birleştirme
    merged_df = pd.merge(top_scores, rev_df, on='listing_id', how='inner').sort_values(by='weighted_score', ascending=False)

###########################################################################################################################3
###########################################################################################################################3
    # Normalizasyon işlemi
    merged_df['normalized_reviews'] = merged_df['number_of_reviews'] / merged_df['number_of_reviews'].max()
    merged_df['normalized_score'] = merged_df['weighted_score'] / merged_df['weighted_score'].max()

    # Ağırlıklı puanı hesapla (örneğin: %50 yorum sayısı + %50 skor)
    merged_df['weighted_value'] = 0.5 * merged_df['normalized_reviews'] + 0.5 * merged_df['normalized_score']

    # Sonuçları sıralayıp en yüksekten düşüğe doğru 20 sonuç al
    result_df = merged_df.sort_values(by='weighted_value', ascending=False).drop_duplicates(subset='listing_id')
    top_20_weighted = result_df[['listing_id', 'name', 'listing_url', 'number_of_reviews', 'weighted_score', 'weighted_value']].head(21)
    # top_20_comments.to_csv('datasets/Amsterdam_Datasets/Top_20_comments.csv', index=False)
    joblib.dump(top_20_weighted, "top_20_comment.pkl")
    # print(top_20_weighted)
    return top_20_weighted




if __name__ == "__main__":
    print("İşlem başladı")
    name = 'Comfortable double room'
    main(name)



