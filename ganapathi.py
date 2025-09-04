import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import re

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Google Ads Keyword Intelligence", layout="wide")
st.title("🚀 Google Ads Keyword Intelligence Dashboard")

st.markdown("""
Upload multiple **Google Ads Search Term Reports (CSV)**.  
This tool will let you:
- See what customers are searching (exact phrases)
- Identify top-performing keywords
- List all search terms under each keyword
- Find **search gaps** (terms not covered by your paid keywords)
- View **Search Term → Keyword Mapping**
- Get **AI-powered campaign suggestions**
""")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload CSV files (Google Ads exports)", 
    accept_multiple_files=True, 
    type=["csv"]
)

if not uploaded_files:
    st.warning("⬆️ Please upload your CSV reports.")
    st.stop()

# ----------------------------
# DATA LOADING & CLEANING
# ----------------------------
all_dfs = []
for file in uploaded_files:
    try:
        df = pd.read_csv(file, skiprows=2)  # Google Ads reports often have 2 metadata rows
        df["account"] = file.name
        all_dfs.append(df)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

df = pd.concat(all_dfs, ignore_index=True)

# Standardize column names
df = df.rename(columns={
    "Search term": "search_term",
    "Keyword": "keyword",
    "Campaign": "campaign",
    "Ad group": "ad_group",
    "Impr.": "impressions",
    "Interactions": "clicks",
    "Cost": "cost",
    "Match type": "match_type"
})

# Clean numbers
for col in ["impressions", "clicks", "cost"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df = df.dropna(subset=["search_term"])

# ----------------------------
# CLEANING FUNCTION
# ----------------------------
def clean_text(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r'^[\[\]\"]+|[\[\]\"]+$', '', x)  # remove quotes/brackets
    x = x.replace("+", "").strip()
    x = re.sub(r'\s+', ' ', x)  # collapse spaces
    return x

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔍 Filters")

accounts = df["account"].unique()
account_sel = st.sidebar.selectbox("Select Account", ["All"] + list(accounts))
filtered_df = df if account_sel == "All" else df[df["account"] == account_sel]

campaigns = filtered_df["campaign"].unique()
campaign_sel = st.sidebar.selectbox("Select Campaign", ["All"] + list(campaigns))
if campaign_sel != "All":
    filtered_df = filtered_df[filtered_df["campaign"] == campaign_sel]

# ----------------------------
# KEYWORD SUMMARY
# ----------------------------
st.subheader("📊 Keyword Summary")

summary = (
    filtered_df.groupby("keyword")
    .agg(
        total_search_terms=("search_term", "nunique"),
        total_impressions=("impressions", "sum"),
        total_clicks=("clicks", "sum"),
        total_cost=("cost", "sum")
    )
    .reset_index()
    .sort_values("total_search_terms", ascending=False)
)
summary["CTR"] = (summary["total_clicks"] / summary["total_impressions"]).fillna(0)
summary["CPC"] = (summary["total_cost"] / summary["total_clicks"]).replace([float("inf")], 0)

st.dataframe(summary, use_container_width=True)

# ----------------------------
# EXACT SEARCH TERMS FOR EACH KEYWORD
# ----------------------------
st.subheader("🔍 Exact Search Terms by Keyword")

chosen_kw = st.selectbox("Pick a keyword to see its exact search terms:", summary["keyword"].dropna().tolist())

if chosen_kw:
    sub = filtered_df[filtered_df["keyword"] == chosen_kw]
    term_details = (
        sub.groupby("search_term")
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            cost=("cost", "sum")
        )
        .reset_index()
        .sort_values("clicks", ascending=False)
    )
    st.write(f"**Keyword:** `{chosen_kw}` — {len(term_details)} unique search terms")
    st.dataframe(term_details, use_container_width=True)

# ----------------------------
# SEARCH GAP ANALYSIS
# ----------------------------
st.subheader("⚠️ Search Terms Not Covered by Keywords")

filtered_df["search_term_clean"] = filtered_df["search_term"].apply(clean_text)
filtered_df["keyword_clean"] = filtered_df["keyword"].apply(clean_text)

all_keywords = set(filtered_df["keyword_clean"].unique()) - {""}
all_search_terms = set(filtered_df["search_term_clean"].unique())
uncovered_terms = all_search_terms - all_keywords

uncovered_df = filtered_df[filtered_df["search_term_clean"].isin(uncovered_terms)].copy()

if uncovered_df.empty:
    st.info("✅ All search terms are already covered by your paid keywords!")
else:
    uncovered_summary = (
        uncovered_df.groupby("search_term")
        .agg(
            total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"),
            total_cost=("cost", "sum")
        )
        .reset_index()
        .sort_values(by=["total_clicks", "total_impressions"], ascending=[False, False])
    )

    st.write("These search terms drive traffic but are **not directly mapped to your keywords**:")

    st.markdown("### 📊 Aggregated Summary (by uncovered search term)")
    st.dataframe(uncovered_summary, use_container_width=True)

    st.markdown("### 📋 Full Detailed List")
    st.dataframe(
        uncovered_df[
            ["search_term", "match_type", "campaign", "ad_group", 
             "impressions", "clicks", "cost", "keyword"]
        ].sort_values(by="clicks", ascending=False),
        use_container_width=True
    )

# ----------------------------
# SEARCH TERM → KEYWORD MAPPING
# ----------------------------
st.subheader("🔗 Search Term → Keyword Mapping")

mapping = (
    filtered_df[["search_term", "keyword", "impressions", "clicks", "cost"]]
    .sort_values(by="clicks", ascending=False)
    .reset_index(drop=True)
)

st.write("This table shows **what people typed** vs **which keyword triggered your ad**:")
st.dataframe(mapping, use_container_width=True)

# ----------------------------
# AI BUSINESS SUGGESTIONS
# ----------------------------
st.subheader("🤖 AI Campaign Insights")

if st.button("Generate AI Campaign Insights"):
    with st.spinner("⌛ Analyzing your campaign data..."):
        try:
            top_context = (
                filtered_df.groupby("search_term")
                .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"))
                .sort_values(by="clicks", ascending=False)
                .head(30)
                .reset_index()
            )

            terms_summary = "\n".join(
                f"- {row.search_term} (Clicks: {row.clicks}, Impr: {row.impressions})"
                for _, row in top_context.iterrows()
            )

            prompt = f"""
            You are a PPC strategist. Analyze the following **Google Ads search terms** from a roofing campaign:

            {terms_summary}

            Focus ONLY on insights directly related to this campaign.
            
            Provide a structured output with:
            1. Key customer needs shown by the search terms
            2. Recommended campaign optimizations (keywords to add, negatives to exclude, ad copy ideas, location targeting)
            3. Growth opportunities visible from the campaign data

            Keep it short, specific, and actionable for the advertiser.
            """

            # --- Your API Key (hard-coded here) ---
            api_key = "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"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Google Ads Dashboard"
            }

            payload = {
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                    {"role": "system", "content": "You are a PPC strategist analyzing Google Ads data."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            result = response.json()

            if "choices" in result:
                st.success(result["choices"][0]["message"]["content"])
            else:
                st.error(f"AI response error: {result}")

        except Exception as e:
            st.error(f"AI request failed: {e}")

