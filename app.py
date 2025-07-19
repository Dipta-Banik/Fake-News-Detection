import streamlit as st
from verifier import verify_fact_bert
from news_utils import fetch_articles, find_most_similar, extract_news_source

st.markdown(
    """
    <div style='
        background-color: #dcdcdc; 
        padding: 8px;
        border-radius: 12px;
        text-align: center;
        font-family: Arial, Helvetica, sans-serif;
        color: #111111; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    '>
        <h1>🧠 Fake News Verifier <br>
        <span style='color:#0056b3;'>(BERT Powered)</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)

user_input = st.text_input("📰 Type or paste a news headline here...")

if user_input:
    st.info("🔎 Checking against trusted news sources...")

    articles = fetch_articles(user_input)
    if not articles:
        st.error("❌ No news articles found. Can't verify.")
    else:
        best_article, similarity = find_most_similar(user_input, articles)
        evidence = best_article["title"] + ". " + best_article["summary"]
        source = extract_news_source(best_article["link"])
        
        predicted_label, bert_score = verify_fact_bert(user_input, evidence)
        #st.write("🔍 **BERT Entailment Confidence:**", round(bert_score, 2))
        #st.write("🧠 **Model Judgment:**", predicted_label)
        
        st.markdown(f"""
🔍 **BERT Entailment Confidence:** <span style='color:{"green" if bert_score >= 0.7 else "orange" if bert_score >= 0.5 else "red"}; font-weight:bold; font-size:18px;'>{round(bert_score, 2)}</span><br>
🧠 **Model Judgment:** <span style='color:{"red" if predicted_label == "Contradiction" else "orange" if predicted_label == "Neutral" else "green"}; font-weight:bold; font-size:18px;'>{predicted_label}</span>
""", unsafe_allow_html=True)

        
        percent = int(bert_score * 100)
        st.markdown("#### 🎯 Prediction Confidence Meter"

)

        def get_confidence_level(score):
            if score >= 80:
                return "High"
            elif score >= 50:
                return "Moderate"
            else:
                return "Low"
 
        def confidence_color(score):
            if score >= 80:
                return "rgb(0, 153, 51)"   
            elif score >= 50:
                return "rgb(255, 193, 7)"     
            else:
                return "rgb(220, 53, 69)" 


        level = get_confidence_level(percent)
        color = confidence_color(percent)
        st.markdown(f"""
        <div title="Confidence: {level}" style="position: relative; height: 20px; background-color: #bfbfbf; border-radius: 8px; overflow: hidden;">
                <div style="
                    width: 0%;
                    background-color: {color};
                    height: 100%;
                    border-radius: 8px;
                    transition: width 1s ease;
                    animation: growBar 1s forwards;">
                </div>
                <div style="
                    position: absolute;
                    width: 100%;
                    text-align: center;
                    top: 0;
                    line-height: 20px;
                    font-weight: bold;
                    color: {'white' if percent > 50 else 'black'};">
                    {percent}%
                </div>
        </div>

        <style>
        @keyframes growBar {{
        to {{
            width: {percent}%;
        }}
        }}
        </style>
        """, unsafe_allow_html=True)


        if predicted_label == "Entailment":        
            if bert_score > 0.65:
                st.success("✅ Trusted sources report similar news. Likely **True**.")
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Article:** [{best_article['title']}]({best_article['link']})")
            else:
                st.info(f"🟡 The claim *seems* supported, but confidence is only **{round(bert_score, 2)}** — more reliable evidence may be needed.")
                st.markdown("🔁 **Closest Related Real News:**")
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Article:** [{best_article['title']}]({best_article['link']})")

        elif predicted_label == "Contradiction":
            if bert_score > 0.65:
                st.error("❌ This claim contradicts reliable sources. Likely **Fake**.")
                st.markdown("🔁 **Closest Related Real News:**")
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Article:** [{best_article['title']}]({best_article['link']})")
            else:
                st.info(f"🟡 The claim may be incorrect, but the model is **not confident enough** (confidence: {round(bert_score, 2)}). Additional verification is advised.")
                st.markdown("🔁 **Closest Related Real News:**")
                st.markdown(f"**Source:** {source}")
                st.markdown(f"**Article:** [{best_article['title']}]({best_article['link']})")

        else:               
            st.warning("⚠️ Insufficient or unclear evidence to verify this claim.")
            st.markdown("🔁 **Closest Related Real News:**")
            st.markdown(f"**Source:** {source}")
            st.markdown(f"**Article:** [{best_article['title']}]({best_article['link']})")
