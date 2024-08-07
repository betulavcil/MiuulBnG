import numpy as np
import pandas as pd
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def preprocess_data(df):
    # Fiyat sütununu temizle ve dönüştür
    df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)

    # Tarih sütununu işleyerek yılları hesapla
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['years_as_host'] = (pd.to_datetime('today') - df['host_since']).dt.days // 365

    # 'host_is_superhost' değerini Boolean'a dönüştür
    df['host_is_superhost'] = df['host_is_superhost'].apply(lambda x: x == 't')

    # Null değerleri temizle
    df = df.fillna(0)

    return df


def get_top_superhosts(df, top_n=10):
    """
    Verilen dataframe üzerinde süperhost'ları en çok yoruma sahip ve en yüksek puanlara göre sıralar.

    :param df: Ev sahipliği ve puanları içeren dataframe
    :param top_n: Gösterilecek süperhost sayısı
    :return: En iyi süperhost'ların bilgileri
    """
    # Sadece süperhost olanları filtrele
    superhosts_df = df[df['host_is_superhost']]

    # Süperhost'ları en çok yorum sayısına göre sıralama yap
    superhosts_sorted = superhosts_df.sort_values(by=['number_of_reviews', 'review_scores_rating'], ascending=False)

    # İlk 'top_n' süperhost'u seç
    top_superhosts = superhosts_sorted.head(top_n)

    return top_superhosts[['host_url', 'host_name', 'years_as_host', 'number_of_reviews', 'review_scores_rating']]


def main():
    # Veri setinizi oku
    listing = pd.read_csv('listings.csv')

    # Gereken sütunları seç
    listing = listing[['id', 'price', 'name', 'host_id', 'host_url', 'listing_url', 'host_name', 'host_since',
                       'minimum_nights', 'maximum_nights', 'host_is_superhost', 'host_total_listings_count',
                       'neighbourhood_cleansed', 'has_availability', 'number_of_reviews', 'review_scores_rating',
                       'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_value',
                       'review_scores_communication', 'review_scores_location', 'review_scores_checkin',
                       'reviews_per_month']]

    # Veriyi ön işleme tabi tut
    processed_df = preprocess_data(listing)

    # En iyi süperhost'ları getir
    top_superhosts = get_top_superhosts(processed_df)

    print(top_superhosts)
    joblib.dump(top_superhosts, "top_superhosts.pkl")
    return top_superhosts


if __name__ == "__main__":
    print("İşlem başladı")
    main()
