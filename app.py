import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq


model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))


data = pd.read_csv("dataset.csv")


client = Groq(api_key="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")


st.title(" Product Review System")


products = data['product'].unique()
selected_product = st.selectbox("Select Phone Model", products)


product_data = data[data['product'] == selected_product]

st.subheader(f"Sentiment Analysis for {selected_product}")


sentiment_counts = product_data['sentiment'].value_counts()

fig, ax = plt.subplots()
ax.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%'
)
st.pyplot(fig)


st.subheader(" Add Your Review")
review = st.text_area("Enter your review")


if st.button("Submit"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        vec = tfidf.transform([review])
        prediction = model.predict(vec)[0]

        st.success(f"Predicted Sentiment: {prediction}")

        
        new_row = pd.DataFrame({
            "product": [selected_product],
            "review": [review],
            "sentiment": [prediction]
        })

        data = pd.concat([data, new_row], ignore_index=True)

        
        data.to_csv("dataset.csv", index=False)

        st.success("Review added successfully!")

        
        st.rerun()


if st.button("Generate Summary"):
    reviews_text = " ".join(product_data['review'].tolist())

    prompt = f"""
    These are customer reviews for {selected_product}:

    {reviews_text}

    Give a short summary including:
    - Overall sentiment
    - Common pros
    - Common cons
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader(" Product Summary")
    st.write(response.choices[0].message.content)