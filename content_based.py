import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Benzerlik için Word2Vec yöntemi kullanıldı
def get_vector_w2v(text, model_w2v):
    words = simple_preprocess(text)
    word_vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
    if len(word_vectors) == 0:
        return np.zeros(model_w2v.vector_size)
    return np.mean(word_vectors, axis=0)


def calculate_cosine_sim_Word2Vec(dataframe):
    texts_w2v = dataframe['description'].apply(simple_preprocess).tolist()
    model_w2v = Word2Vec(sentences=texts_w2v, vector_size=100, window=5, min_count=1, workers=4)
    dataframe['vector_w2v'] = dataframe['description'].apply(lambda x: get_vector_w2v(x, model_w2v))
    vectors_w2v = np.array(dataframe['vector_w2v'].tolist())
    cosine_sim_w2v = cosine_similarity(vectors_w2v, vectors_w2v)
    return cosine_sim_w2v, model_w2v


def content_based_recommender_Word2Vec(title, price, cosine_sim_w2v, dataframe):
    if title not in dataframe['name'].values:
        return "Title not found in the dataset"

    # Kullanıcının seçtiği fiyat aralığına göre filtreleme yapın
    bins = [0, 9, 100, 200, 300, float('inf')]
    labels = ['0-9', '10-100', '100-200', '200-300', '300+']
    user_label = pd.cut([price], bins=bins, labels=labels)[0]
    dataframe['price_range'] = pd.cut(dataframe['price'], bins=bins, labels=labels)
    filtered_df = dataframe[dataframe['price_range'] == user_label]

    if filtered_df.empty:
        return "Otur evinde be yaaa!!"

    # Filtrelenmiş dataframe için indeksleme yapın
    property_index_w2v = filtered_df[filtered_df['name'] == title].index
    if len(property_index_w2v) == 0:
        return "Bu fiyata böyle bir ev yok!"
    property_index_w2v = property_index_w2v[0]

    similarity_scores_w2v = pd.DataFrame(cosine_sim_w2v[property_index_w2v], columns=["score"])
    similarity_scores_w2v = similarity_scores_w2v.sort_values(by='score', ascending=False)
    listing_indices_w2v = similarity_scores_w2v.index

    # Filtrelenmiş df'deki geçerli indekslere göre sonuçları al
    valid_indices = [idx for idx in listing_indices_w2v if idx < len(filtered_df)]
    if not valid_indices:
        return "No valid recommendations found."


    recommendations = filtered_df.iloc[valid_indices][['name', 'listing_url', 'price']].head(12)
    return recommendations



# Veri Ön Hazırlık
def data_prep(df):
    df['price'] = df['price'].replace({r'^\$': '', r',': ''}, regex=True).astype(float)
    df['price'] = (df.groupby(['neighbourhood_cleansed'])['price']
                   .transform(lambda x: x.replace(0, np.nan).fillna(x.mean()).replace(np.nan, 0)))

    df['description'] = df['description'].fillna('No description')
    df['has_availability'] = df['has_availability'].fillna('f')
    return df


def main():
    listing = pd.read_csv('datasets/listings.csv')
    df = listing.copy()
    df = df[
        ['id', 'name','listing_url','description', 'neighbourhood_cleansed', 'property_type', 'price',
         'minimum_nights', 'maximum_nights', 'has_availability', 'number_of_reviews',
         'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
         'review_scores_communication', 'review_scores_location', 'review_scores_checkin',
         'instant_bookable', 'reviews_per_month']]

    df = data_prep(df)

    user_input_name = 'Cozy apartment in Amsterdam West'
    user_input_price = 200

    cosine_sim_w2v, model_w2v = calculate_cosine_sim_Word2Vec(df)

    recommendations_w2v = content_based_recommender_Word2Vec(user_input_name, user_input_price, cosine_sim_w2v, df)

    #print('Word2Vec sonuçları: ', '\n', recommendations_w2v)
    # print('-----------------------------------------')
    # print('Score ortalaması: ', '\n', mean_score_w2v)
    # joblib.dump(recommendations_w2v, "recommendations.pkl")


if __name__ == "__main__":
    print("İşlem başladı")
    main()


