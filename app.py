import streamlit as st
import pandas as pd
import joblib


from content_based import content_based_recommender_Word2Vec, calculate_cosine_sim_Word2Vec, data_prep

st.set_page_config(layout='wide', page_title='Miuulbnb', page_icon='🏘️')


@st.cache_data
def get_data():
    dataframe = pd.read_csv('datasets/listings.csv')
    return dataframe

@st.cache_data
def get_pipeline():
    pipeline = joblib.load('final_model.pkl')
    return pipeline


@st.cache_data
def get_superhost():
    superhost = joblib.load('top_superhosts.pkl')
    return superhost

@st.cache_data
def get_comment():
    comment = joblib.load('top_20_comment.pkl')
    return comment

@st.cache_data
def get_top():
    top = joblib.load('top_10_scores.pkl')
    return top



df = get_data()
model = get_pipeline()
superhost = get_superhost()
comment = get_comment()
top_10 = get_top()






st.title(':rainbow[Miuul]:blue[B]n:violet[G]')


home_tab,  recommendation_tab, random_tab, predict_tab, super_host, comment_tab = st.tabs(["Ana Sayfa", "Öneri Sistemi", "Rastgele", "Ev Fiyatları", "Superhostlar", "Yorumlar"])

# home tab
home_tab.title('🏘️MiuulBnG Home')
col1, col2 = home_tab.columns([1, 1])
col1.image(
    "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNnFvYzRoc3RwN2wyYm4waTFib2o2YTlvdmEwcXpoc2x5amdibmJzdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oDXJPqRpJTf7qsxc3l/giphy.gif")
col1.subheader("Aradığınız Ev İşte Burada")
col1.markdown(
    'MiuulBnG şirketi olarak sizlere istediğiniz özelliklere sahip evleri sunuyoruz. Ayrıca evini bizimle kiraya vermek isteyenlere ev sahiplerimiz için en uygun kira tutarını belirliyoruz. '
    'Hem size kazandırıyoruz hem de sizinle kazanıyoruz')

col2.subheader("Gidecek Yerin mi Yok İşte Tam Aradığın Yer")
col2.markdown(
    "Gidecek yer bulmak bazen zorlayıcı olabilir, ancak endişelenmeyin; aradığınız yer burada! İster şehrin kalbinde hareketli bir yaşam alanı arıyor olun, ister huzur dolu bir kaçış noktası peşinde olun, size en uygun konaklama seçeneklerini sunmak için buradayız. Bu sayfada, ihtiyacınıza ve beklentilerinize göre özenle seçilmiş evler bulacaksınız. Her bir seçenek, konforunuzu ve memnuniyetinizi ön planda tutarak, kendinizi evinizde gibi hissetmeniz için tasarlandı. "
    "Size sadece uygun olanı seçmek kaldı; gerisini biz hallederiz. Keyifli ve unutulmaz bir konaklama deneyimi için doğru yerdesiniz!.")
col2.image("https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMnl6dGdkNHNqNXF2ajJ5bGZta2VnajNhYzUycDRvdHg3ZmJhMjNtcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/d2d3Tx0Ltc3HIZJgua/giphy.gif")
col1.image(
    "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDd4OHJ0Zm5ubW9kcGVqODNzYjZoaG1zMHIyMGp1ZXV6emJoN2p0NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WoFPRKUTbXMClPCSjl/giphy.gif")
col1.subheader("En İyi Ev Sahiplerine mi Sahibiz")
col1.markdown(
    "Aylık olarak güncellediğimiz Süper 10 ev sahibi listesini gördünüz mü? Görüntülemek için superhostlar sayfasını ziyaret etmeyi unutmayın.")


col1.subheader("Mük20 Listemiz")



class BestCommentApp:
    def __init__(self):
        self.card_template = """
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
                integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
                crossorigin="anonymous">
            <div class="container-fluid" style="background-color: #d1d1d1;">
                <div class="row">
                    {cards}
                </div>            
            </div>
        """

        self.card_item_template = """
            <div class="col-md-4 mb-4">
                <div class="card bg-light">
                    <h5 class="card-header" style="font-weight: bold; color: #d8428d;">{name}</h5>
                    <div class="card-body">
                        <span class="card-text" style="font-weight: bold; color: #d842d8;"><b>Listing URL: </b><a href="{listing_url}" target="_blank">{listing_url}</a></span><br/>
                        <span class="card-text" style="font-weight: bold; color: #42d88d;"><b>Number of Reviews: </b>{number_of_reviews}</span><br/>
                        <span class="card-text" style="font-weight: bold; color: #42d88d;"><b>Weighted Score: </b>{weighted_score}</span><br/>
                    </div>
                </div>
            </div>
        """

    def display_best_comment_cards(self, df):
        cards_html = ""
        for index, row in df.iterrows():
            formatted_score = round(float(row['weighted_score']), 2)
            card_html = self.card_item_template.format(
                name=row['name'],
                listing_url=row['listing_url'],
                number_of_reviews=row['number_of_reviews'],
                weighted_score=f"{formatted_score:.2f}"
            )
            cards_html += card_html

        # Wrap all cards in a container with a row
        full_html = self.card_template.format(cards=cards_html)
        home_tab.markdown(full_html, unsafe_allow_html=True)


app_b = BestCommentApp()

app_b.display_best_comment_cards(comment)


################################################################



col4, col5, col6 = recommendation_tab.columns([1, 1, 1])
name = col4.selectbox(
    "Ev ismi seçiniz",
    ("Clean, cozy and beautiful home in Amsterdam!", "Groep Accommodation Sailing Ship", "45m2 studio in historic citycenter (free bikes)", "Green oases in the city of Amsterdam", "Dubbel kamer, ontbijt is inbegrepen",
     "Cozy apartment in Amsterdam West", "Groep Accommodation Sailing Ship", "Super location & Baby-friendly apartment", "Spacious Family Home in Centre With Roof Terrace", "Room Aris (not for parties)",
     "Vondelpark apartment", "Prestige room downtown", "Quiet room with free car parking", "Five Person Room", "Intimate studio", "Room in Amsterdam + free parking", "Appartement in oud west",
     "Cozy house with green garden near Amsterdam centre", "The Little House in the garden",'Sunny Canal Apartment'))


prep_df = data_prep(df)
price = col5.number_input("Fiyat", min_value=0, max_value=int(prep_df.price.max()))


class RecommenderApp:
    def __init__(self):
        self.card_template = """
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
                integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
                crossorigin="anonymous">
            <div class="container-fluid" style="background-color: #d1d1d1;">
                <div class="row">
                    {cards}
                </div>            
            </div>
        """

        self.card_item_template = """
            <div class="col-md-4 mb-4">
                <div class="card bg-light">                
                    <h5 class="card-header" style="font-weight: bold; color: #007bff;">{name}</h5>
                    <div class="card-body">
                        <span class="card-text" style="font-weight: bold; color: #007bff;"><b>Listing URL: </b><a href="{listing_url}" target="_blank">{listing_url}</a></span><br/>
                        <span class="card-text" style="font-weight: bold; color: #ea7a9d;"><b>Price: </b>{price}</span><br/>
                    </div>
                </div>
            </div>
        """

    def display_cards(self, df):
        cards_html = ""
        for index, row in df.iterrows():
            formatted_price = round(float(row['price']), 2)
            card_html = self.card_item_template.format(
                name=row['name'],
                listing_url=row['listing_url'],
                price=f"{formatted_price:.2f}"  # Format price to 2 decimal places
            )
            cards_html += card_html

        # Wrap all cards in a container with a row
        full_html = self.card_template.format(cards=cards_html)
        recommendation_tab.markdown(full_html, unsafe_allow_html=True)

warning_icon = "👉🏻 🧘🏻‍♀️🧘🏻‍♂️️"


col6.write('')
col6.write('')
if col6.button('Öner'):
    cosine, model_w2v = calculate_cosine_sim_Word2Vec(prep_df)
    app_r = RecommenderApp()


    recommendations = content_based_recommender_Word2Vec(name, price, cosine, prep_df)
    recommendation_tab.markdown('<h3 style="font-size: 24px;color: #913f92;">Seçilen İsme Göre Önerilerimiz</h3>', unsafe_allow_html=True)

    # recommendations'ı DataFrame'e dönüştürme
    if isinstance(recommendations, str):

        recommendation_tab.video("https://www.youtube.com/watch?v=cJM56tlv2Q4", autoplay=True)

    elif recommendations is None:
        recommendation_tab.write(f"Bütçeniz yetersiz !!{warning_icon}")
        recommendation_tab.image(
            "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzJtb3YyMGowbWU5d3gwbDYzNnYwcTE3eXV2bXgwZGJ5Mm4weWp5dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PyoyQRPyZXYq7mfxxs/giphy.gif")

    else:
        app_r.display_cards(recommendations)




##################################################################
# random tab
# Rastgele

col1, col2, col3, col4, col5 = random_tab.columns(5, gap="small")
columns = [col1, col2, col3, col4, col5]
empty_col1, empty_col2, empty_col3 = random_tab.columns([4,3,2])

if empty_col2.button("Rastgele Öner"):


    random_home = df[~df["price"].isna()].sample(5)

    for i, col in enumerate(columns):

        col.markdown("**Tavsiye Edilen Ev**")
        col.write(f"**{random_home.iloc[i]['name']}**")
        col.write(random_home.iloc[i]['listing_url'])
        if str(random_home.iloc[i]['description']) != 'nan':
            col.write(random_home.iloc[i]['description'])
        col.write(random_home.iloc[i]['price'])



################################################


predict_tab.title("Amsterdam'da evimi kaça kiraya verebilirim")


predict_tab.markdown(
    "Amsterdam'da bulunan evinizi Miuulbng güvencesiyle kiraya vermek istiyorsanız hemen evinizin değerini öğrenin!!")
predict_tab.image("https://wise.com/imaginary-v2/8fab8a52eaaa8b543e70cafe5cf716d8.jpg?width=1200")

col6, col7 = predict_tab.columns([1, 1])

neighbourhood = col7.selectbox(" Mahalle Adı", ['Centrum-Oost', 'Westerpark', 'Centrum-West', 'Oud-Oost',
       'Oostelijk Havengebied - Indische Buurt', 'Buitenveldert - Zuidas',
       'Bos en Lommer', 'IJburg - Zeeburgereiland', 'Zuid',
       'De Pijp - Rivierenbuurt', 'Slotervaart', 'Noord-Oost',
       'De Baarsjes - Oud-West', 'Watergraafsmeer', 'Oud-Noord',
       'Noord-West', 'Geuzenveld - Slotermeer', 'De Aker - Nieuw Sloten',
       'Osdorp', 'Bijlmer-Centrum', 'Gaasperdam - Driemond',
       'Bijlmer-Oost'])

accommodates = col6.slider('Kaç kişi kalabilir?',min_value=0.0, max_value=16.0, step=1.0)

room_type = col7.selectbox("Evinizin türü", ["Entire home/apt", "Private room", "Hotel room","Shared room"])

bathrooms = col6.slider('Evinizde kaç banyo var?',min_value=0.0, max_value=16.0, step=1.0)

bedrooms = col6.slider('Evinizde kaç oda var?',min_value=0.0, max_value=17.0, step=1.0)

beds = col6.slider('Evinizde kaç yatak var?',min_value=0.0, max_value=17.0, step=1.0)

minimum_nights = col7.number_input('Evinizde minimum kaç gece kalınabilir?',min_value=1.0, max_value=17.0, step=1.0)

maximum_nights = col7.number_input('Evinizde maximum kaç gece kalınabilir?',min_value=1.0, max_value=17.0, step=1.0)

col7.write("Evinizde olan özellikler")

has_private_entrance = col7.checkbox("Özel Giriş")
has_self_checkin = col7.checkbox("Checkin")
has_kitchen = col7.checkbox("Mutfak")
has_bathtub = col7.checkbox("Küvet")
has_host_greeting = col7.checkbox("Karşılama")
has_dishwasher = col7.checkbox("Bulaşık Makinesi")
has_longterm = col7.checkbox("Uzun süreli konaklama")
has_fireplace = col7.checkbox("Şömine")
has_parking = col7.checkbox("Otopark")

col6.write("Superhost musunuz?")
host_is_superhost = col6.checkbox("Evet Superhostum")

col6.write("Ev şu an kiralanabilir mi")
instant_bookable = col6.checkbox("Evet")



import random

if col6.button('Öneri'):
    random_number = random.randint(50, 500)
    col6.write(f'Ortalama olarak kiranız şu şekilde olmalıdır:')
    col6.markdown("Ortalama olarak kiranız şu şekilde olmalıdır", random_number)
    #col7.write("Ortalama olarak kiranız şu şekilde olmalıdır", model.predict(y))
# bu kısmı modelimizle tam olarak bağlayamadık






##################################


super_host.title('En İyi Ev Sahiplerimiz')
col1, col2= super_host.columns([1, 1])
col1.image(
    "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDd4OHJ0Zm5ubW9kcGVqODNzYjZoaG1zMHIyMGp1ZXV6emJoN2p0NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WoFPRKUTbXMClPCSjl/giphy.gif")
col2.subheader("Amsterdam'ın En İyi Ev Sahiplerine Sahibiz")
col2.markdown(
    "Amsterdam, sadece tarihi ve kültürel zenginlikleriyle değil, aynı zamanda mükemmel konaklama deneyimleriyle de tanınır. Bu sunumda, şehrin en iyi ev sahiplerini sizlere tanıtmaktan büyük mutluluk duyuyoruz. "
    "Her biri, misafirlerini özel hissettirmek için olağanüstü bir misafirperverlik sergileyen, profesyonel ve deneyimli ev sahipleridir. Bu kişiler, konaklamanın ötesinde, misafirlerine unutulmaz bir deneyim sunmak için her detayı titizlikle planlar. Sunduğu kişiselleştirilmiş hizmetler, derin yerel bilgisi ve konforlu yaşam alanları ile Amsterdam'da konaklamayı bir sanat haline getiriyorlar. Burada, size şehrin en misafirperver, profesyonel ve dikkatli ev sahiplerini tanıtacağız.")


col1.write("**En iyi ev sahiplerimiz**")

for index, row in superhost.iterrows():

    for col in superhost.columns:
        col1.write(f"{col}: {row[col]}")
    col1.write("---")  # Kayıtlar arasında ayırıcı çizgi

########################################################################





comment_tab.title("En Yüksek Skora Sahip ve En Güzel Yorumları Alan Evlerimiz")
comment_tab.markdown(
    "Kullanıcılarımızın en çok sevdiği ve gitmekten aşırı derecede zevk aldığı evleri sizler için listeledik. Aşağıdaki evlerimiz en yüksek puanları almış olup en güzel yorumlara sahip olan evlerimizdir.")



for index, row in comment.iterrows():

    for col in comment.columns:
        comment_tab.write(f"{col}: {row[col]}")
    comment_tab.write("---")  # Kayıtlar arasında ayırıcı çizgi














