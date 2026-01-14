!pip install -q gradio supabase transformers pandas plotly scipy

import gradio as gr
from supabase import create_client, Client
from transformers import pipeline
import pandas as pd
import plotly.express as px

# --- 1. CONFIGURATION ---
SUPABASE_URL = "https://noxsvelejtdgqmaqgtgy.supabase.co"
SUPABASE_KEY = "sb_publishable_ixONKGsSKc9dRdbBQuVmNQ_yyGO3S90"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Sentiment Analysis Model (Hugging Face)
# We use a 'distilbert' model which is fast and accurate for reviews
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- 2. LOGIC LAYER ---
def fetch_and_analyze():
    try:
        response = supabase.table("comments").select("*, videos(video_title)").execute()
    except Exception as e:
        print(f"⚠️ Database Fetch Error: {e}")
        return pd.DataFrame()

    data = response.data

    if not data:
        return pd.DataFrame() 

    processed_rows = []
    for row in data:
        raw_text = row.get('text_content') or row.get('text') or row.get('comment_text')
        
        comment_text = raw_text if raw_text else ""

        if not comment_text.strip():
            continue

        timestamp = row.get('published_at') or row.get('created_at')

        if row.get('videos'):
             video_label = row['videos'].get('video_title') or "Unknown Video"
        else:
             video_label = row.get('video_id', "Unknown Video")

        try:
            # Truncate to 512 chars for BERT
            analysis = sentiment_pipeline(comment_text[:512])[0]

            sentiment_score = analysis['score'] if analysis['label'] == 'POSITIVE' else -analysis['score']

            processed_rows.append({
                "Date": pd.to_datetime(timestamp),
                "Video": video_label,
                "Comment": comment_text,
                "Sentiment Label": analysis['label'],
                "Sentiment Score": sentiment_score
            })
        except Exception as e:
            print(f"⚠️ AI Error on comment: {e}")
            continue

    return pd.DataFrame(processed_rows)

def generate_dashboard():
    df = fetch_and_analyze()

    if df.empty:
        empty_fig = px.line(title="Waiting for Data... (Database is Empty)")
        return empty_fig, pd.DataFrame(), pd.DataFrame()

    df_daily = df.groupby([pd.Grouper(key='Date', freq='D'), 'Video'])['Sentiment Score'].mean().reset_index()

    try:
        # Group by Day and Video Title
        df_daily = df.groupby([pd.Grouper(key='Date', freq='D'), 'Video'])['Sentiment Score'].mean().reset_index()
        
        fig = px.line(df_daily, x="Date", y="Sentiment Score", color="Video",
                      title="Brand Sentiment Pulse (Over Time)",
                      labels={"Sentiment Score": "Vibe Score (-1 to +1)"},
                      template="plotly_white")
    except Exception as e:
        fig = px.line(title=f"Chart Error: {e}")

    top_pos = df.sort_values(by="Sentiment Score", ascending=False).head(2)
    top_pos_display = top_pos[['Video', 'Comment', 'Sentiment Score']]

    top_neg = df.sort_values(by="Sentiment Score", ascending=True).head(2)
    top_neg_display = top_neg[['Video', 'Comment', 'Sentiment Score']]

    return fig, top_pos_display, top_neg_display

# --- 3. UI LAYER ---
pulse_css = """
body {background-color: #f4f4f9;}
h1 {color: #2c3e50; text-align: center;}
.gradio-container {max-width: 1200px !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=pulse_css) as app:

    gr.Markdown("# 📉 The Market Pulse Engine")
    gr.Markdown("### Real-time AI analysis of the 'Digital Town Square'")

    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh Pulse Data", variant="primary")

    
    with gr.Tab("📊 Macro Strategy"):
        sentiment_chart = gr.Plot(label="Sentiment Over Time")

    
    with gr.Tab("🔍 Micro Drill-Down"):
        gr.Markdown("### 🏆 Top Brand Advocates (Most Positive)")
        pos_table = gr.Dataframe()

        gr.Markdown("### ⚠️ Critical Alerts (Most Negative)")
        neg_table = gr.Dataframe()

    refresh_btn.click(
        fn=generate_dashboard,
        inputs=None,
        outputs=[sentiment_chart, pos_table, neg_table]
    )

# Launch the App
app.launch(share=True, debug=True)
