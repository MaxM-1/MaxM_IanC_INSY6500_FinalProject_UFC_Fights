import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, ttest_ind

st.set_page_config(
    page_title="UFC Dashboard",
    page_icon="ufc_icon.png",  # Path to your icon file
    layout="wide"
)


# Load data
@st.cache_data
def load_data():
    attrs = pd.read_csv("data/fighter_attributes.csv")
    hist = pd.read_csv("data/fighter_history.csv")
    stats = pd.read_csv("data/fighter_stats.csv")
    return attrs, hist, stats

attrs, hist, stats = load_data()

# Precompute fighter records for use in tables
fighter_records = (
    hist.groupby('fighter_id')['fight_result']
        .agg(
            wins=lambda x: (x == 'W').sum(),
            losses=lambda x: (x == 'L').sum(),
            draws=lambda x: (x == 'D').sum(),
            no_contests=lambda x: (x == 'NC').sum(),
            total_fights='count'
        )
        .reset_index()
)

fighter_records['record'] = (
    fighter_records['wins'].astype(str) + "-" +
    fighter_records['losses'].astype(str) + "-" +
    fighter_records['draws'].astype(str)
)
# Title
st.image("ufc_icon.png", width=150)
st.title("UFC Fighter Analytics")
st.markdown("Interactive exploration of UFC fighter statistics, performance, and trends.")

with st.sidebar:
    st.image("ufc_icon.png", width=120)
    st.markdown("### UFC Data Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", [
    "Data Overview",
    "Age Analysis",
    "Fighter Style Analysis",
    "Reach/Height Ratio",
    "🇷🇺 Russian Grappler Dominance"
])

# ============================================
# PAGE 1: Data Overview
# ============================================
if page == "Data Overview":
    st.header("Fighter Directory")

    # Merge attributes with records
    fighter_table = attrs.merge(
        fighter_records[
            ['fighter_id', 'wins', 'losses', 'draws', 'no_contests', 'total_fights', 'record']
        ],
        on='fighter_id',
        how='left'
    )

    # Optional: choose which columns to show first
    cols_front = [
        'name', 'record', 'total_fights', 'wins', 'losses', 'draws',
        'weight_class', 'country', 'style', 'height', 'reach'
    ]
    # Keep only columns that actually exist + all remaining ones
    cols_front = [c for c in cols_front if c in fighter_table.columns]
    remaining_cols = [c for c in fighter_table.columns if c not in cols_front]
    fighter_table = fighter_table[cols_front + remaining_cols]

    # 🔍 Search + filter controls
    col_search, col_wc = st.columns([2, 1])

    with col_search:
        name_query = st.text_input("Search fighter by name", value="")

    with col_wc:
        wc_options = ["All weight classes"] + sorted(fighter_table['weight_class'].dropna().unique().tolist())
        selected_wc = st.selectbox("Filter by weight class", wc_options)

    filtered = fighter_table.copy()

    if name_query:
        filtered = filtered[filtered['name'].str.contains(name_query, case=False, na=False)]

    if selected_wc != "All weight classes":
        filtered = filtered[filtered['weight_class'] == selected_wc]

    # Scrollable, wide fighter table
    st.subheader("All Fighters (Attributes + Record)")
    st.dataframe(
        filtered,
        use_container_width=True,
        height=400  # makes it scrollable
    )

    st.markdown("---")
    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fighters", f"{len(attrs):,}")
    col2.metric("Total Fights", f"{len(hist):,}")
    col3.metric("Countries", f"{attrs['country'].nunique()}")

    # Weight class distribution
    st.subheader("Fighters by Weight Class")
    wc_counts = attrs['weight_class'].value_counts()
    fig = px.bar(
        x=wc_counts.index,
        y=wc_counts.values, 
        labels={'x': 'Weight Class', 'y': 'Count'},
        color=wc_counts.values,
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Country distribution
    st.subheader("Top 15 Countries")
    country_counts = attrs['country'].value_counts().head(15)
    fig2 = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        labels={'x': 'Number of Fighters', 'y': 'Country'},
        color=country_counts.values,
        color_continuous_scale='reds'
    )
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 2: Age Analysis
# ============================================
elif page == "Age Analysis":
    st.header("Does Fighter Age Affect Performance?")
    
    # Prepare data
    attrs['dob'] = pd.to_datetime(attrs['dob'], errors='coerce')
    hist['event_date'] = pd.to_datetime(hist['event_date'], errors='coerce')
    
    merged = hist.merge(attrs[['fighter_id', 'dob']], on='fighter_id', how='left')
    merged['age_at_fight'] = (merged['event_date'] - merged['dob']).dt.days / 365.25
    merged = merged[(merged['age_at_fight'] >= 18) & (merged['age_at_fight'] <= 50)]
    
    # Age group selector
    age_bins = st.slider("Select Age Range", 18, 50, (18, 45))
    merged_filtered = merged[(merged['age_at_fight'] >= age_bins[0]) & 
                              (merged['age_at_fight'] <= age_bins[1])]
    
    # Calculate win rates by age group
    bins = [18, 25, 30, 35, 40, 45, 50]
    labels = ['18-24', '25-29', '30-34', '35-39', '40-44', '45+']
    merged_filtered['age_group'] = pd.cut(merged_filtered['age_at_fight'], bins=bins, labels=labels)
    
    win_rates = merged_filtered.groupby('age_group')['fight_result'].apply(
        lambda x: (x == 'W').mean()
    ).dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Win Rate by Age Group")
        fig = px.bar(x=win_rates.index.astype(str), y=win_rates.values,
                     labels={'x': 'Age Group', 'y': 'Win Rate'},
                     color=win_rates.values, color_continuous_scale='RdYlGn')
        fig.update_layout(yaxis_range=[0, 0.7])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fight Count by Age Group")
        fight_counts = merged_filtered['age_group'].value_counts().sort_index()
        fig2 = px.bar(x=fight_counts.index.astype(str), y=fight_counts.values,
                      labels={'x': 'Age Group', 'y': 'Number of Fights'},
                      color=fight_counts.values, color_continuous_scale='blues')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Key insight
    st.info("**Key Finding**: Win rate generally decreases as fighter age increases, " 
            "suggesting that younger fighters tend to have better performance outcomes.")

# ============================================
# PAGE 3: Fighter Style Analysis
# ============================================
elif page == "Fighter Style Analysis":
    st.header("Fighter Style Analysis")
    
    # Prepare data
    wins = hist[hist['fight_result'] == 'W'].copy()
    wins['finish_category'] = np.where(
        wins['fight_result_type'] == 'KO-TKO', 'KO/TKO',
        np.where(wins['fight_result_type'] == 'SUBMISSION', 'Submission', 'Decision')
    )
    
    wins_merged = wins.merge(attrs[['fighter_id', 'style']], on='fighter_id', how='left')
    
    # Filter styles
    min_wins = st.slider("Minimum wins to include style", 10, 100, 20)
    
    style_finish = wins_merged.groupby(['style', 'finish_category']).size().reset_index(name='count')
    style_totals = style_finish.groupby('style')['count'].sum()
    valid_styles = style_totals[style_totals >= min_wins].index
    style_finish = style_finish[style_finish['style'].isin(valid_styles)]
    
    # Calculate rates
    style_stats = wins_merged[wins_merged['style'].isin(valid_styles)].groupby('style').agg(
        total_wins=('fight_result', 'count'),
        ko_wins=('finish_category', lambda x: (x == 'KO/TKO').sum()),
        sub_wins=('finish_category', lambda x: (x == 'Submission').sum())
    )
    style_stats['ko_rate'] = style_stats['ko_wins'] / style_stats['total_wins']
    style_stats['sub_rate'] = style_stats['sub_wins'] / style_stats['total_wins']
    style_stats = style_stats.sort_values('ko_rate', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("KO/TKO Rate by Style")
        fig = px.bar(x=style_stats.index, y=style_stats['ko_rate'],
                     labels={'x': 'Fighting Style', 'y': 'KO/TKO Rate'},
                     color=style_stats['ko_rate'], color_continuous_scale='reds')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Submission Rate by Style")
        fig2 = px.bar(x=style_stats.sort_values('sub_rate', ascending=False).index, 
                      y=style_stats.sort_values('sub_rate', ascending=False)['sub_rate'],
                      labels={'x': 'Fighting Style', 'y': 'Submission Rate'},
                      color=style_stats.sort_values('sub_rate', ascending=False)['sub_rate'], 
                      color_continuous_scale='greens')
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Striker vs non-striker comparison
    st.subheader("Striker vs Non-Striker KO Rate")
    
    striking_keywords = ["boxing", "kickboxing", "muay thai", "striker", "taekwondo", "karate"]
    attrs['is_striker'] = attrs['style'].fillna('').str.lower().apply(
        lambda x: any(k in x for k in striking_keywords)
    )
    
    wins_striker = wins.merge(attrs[['fighter_id', 'is_striker']], on='fighter_id', how='left')
    
    striker_ko = (wins_striker[wins_striker['is_striker'] == True]['finish_category'] == 'KO/TKO').mean()
    non_striker_ko = (wins_striker[wins_striker['is_striker'] == False]['finish_category'] == 'KO/TKO').mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Striker KO Rate", f"{striker_ko:.1%}")
    col2.metric("Non-Striker KO Rate", f"{non_striker_ko:.1%}")
    
    st.success(f"Strikers have a **{(striker_ko - non_striker_ko)*100:.1f}%** higher KO rate than non-strikers")

# ============================================
# PAGE 4: Reach/Height Ratio Analysis
# ============================================
elif page == "Reach/Height Ratio":
    st.header("Reach-to-Height Ratio Analysis")
    st.markdown("**Hypothesis**: Fighters with longer reach relative to height have better records")
    
    # Calculate ratio
    attrs['reach_height_ratio'] = attrs['reach'] / attrs['height']
    
    # Calculate win records
    fighter_records = hist.groupby('fighter_id').agg(
        wins=('fight_result', lambda x: (x == 'W').sum()),
        total_fights=('fight_result', 'count')
    )
    fighter_records['win_rate'] = fighter_records['wins'] / fighter_records['total_fights']
    fighter_records = fighter_records.reset_index()
    
    # Merge
    df = attrs.merge(fighter_records, on='fighter_id', how='inner')
    
    # Filters
    min_fights = st.slider("Minimum fights", 1, 20, 3)
    df_filtered = df[(df['reach_height_ratio'].notna()) & (df['total_fights'] >= min_fights)]
    
    # Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Fighters Analyzed", f"{len(df_filtered):,}")
    col2.metric("Avg Ratio", f"{df_filtered['reach_height_ratio'].mean():.3f}")
    col3.metric("Avg Win Rate", f"{df_filtered['win_rate'].mean():.1%}")
    
    # Scatter plot
    st.subheader("Reach/Height Ratio vs Win Rate")
    fig = px.scatter(df_filtered, x='reach_height_ratio', y='win_rate',
                     color='weight_class', hover_data=['name'],
                     labels={'reach_height_ratio': 'Reach/Height Ratio', 'win_rate': 'Win Rate'},
                     opacity=0.6)
    
    # Add trend line
    z = np.polyfit(df_filtered['reach_height_ratio'], df_filtered['win_rate'], 1)
    x_line = np.linspace(df_filtered['reach_height_ratio'].min(), df_filtered['reach_height_ratio'].max(), 100)
    fig.add_trace(go.Scatter(x=x_line, y=np.poly1d(z)(x_line), mode='lines', 
                             name='Trend', line=dict(color='red', width=2)))
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation
    corr, p_val = pearsonr(df_filtered['reach_height_ratio'], df_filtered['win_rate'])
    st.write(f"**Correlation**: {corr:.4f} (p-value: {p_val:.4f})")
    
    # Quintile comparison
    st.subheader("Win Rate by Ratio Quintile")
    df_filtered['quintile'] = pd.qcut(df_filtered['reach_height_ratio'], q=5, 
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    quintile_rates = df_filtered.groupby('quintile')['win_rate'].mean()
    
    fig2 = px.bar(x=quintile_rates.index.astype(str), y=quintile_rates.values,
                  labels={'x': 'Reach/Height Quintile', 'y': 'Win Rate'},
                  color=quintile_rates.values, color_continuous_scale='RdYlGn')
    st.plotly_chart(fig2, use_container_width=True)

# ============================================
# PAGE 5: Russian Grappler Dominance
# ============================================
elif page == "🇷🇺 Russian Grappler Dominance":
    st.header("Russian Grappler Dominance Analysis")
    st.markdown("Are Russian grapplers more dominant in the UFC?")
    
    # Define grapplers
    grappling_styles = ['wrestling', 'brazilian jiu-jitsu', 'grappling', 'sambo', 'judo']
    attrs['is_grappler'] = attrs['style'].fillna('').str.lower().isin(grappling_styles)
    attrs['is_russian'] = attrs['country'].fillna('').str.lower() == 'russia'
    
    def categorize(row):
        if row['is_russian'] and row['is_grappler']:
            return 'Russian Grappler'
        elif row['is_russian']:
            return 'Russian Non-Grappler'
        elif row['is_grappler']:
            return 'Non-Russian Grappler'
        else:
            return 'Other'
    
    attrs['category'] = attrs.apply(categorize, axis=1)
    
    # Calculate records
    fighter_records = hist.groupby('fighter_id').agg(
        wins=('fight_result', lambda x: (x == 'W').sum()),
        total_fights=('fight_result', 'count')
    )
    fighter_records['win_rate'] = fighter_records['wins'] / fighter_records['total_fights']
    fighter_records = fighter_records.reset_index()
    
    df = attrs.merge(fighter_records, on='fighter_id', how='inner')
    
    min_fights = st.slider("Minimum fights", 1, 15, 3)
    df_filtered = df[df['total_fights'] >= min_fights]
    
    # Category stats
    cat_stats = df_filtered.groupby('category').agg(
        n_fighters=('fighter_id', 'count'),
        avg_win_rate=('win_rate', 'mean'),
        total_wins=('wins', 'sum')
    ).round(3)
    
    st.subheader("Fighter Categories")
    st.dataframe(cat_stats)
    
    # Bar chart
    st.subheader("Average Win Rate by Category")
    fig = px.bar(x=cat_stats.index, y=cat_stats['avg_win_rate'],
                 labels={'x': 'Category', 'y': 'Average Win Rate'},
                 color=cat_stats['avg_win_rate'], color_continuous_scale='RdYlGn')
    fig.add_hline(y=df_filtered['win_rate'].mean(), line_dash="dash", 
                  annotation_text="Overall Average")
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical test
    russian_grapplers = df_filtered[df_filtered['category'] == 'Russian Grappler']['win_rate']
    non_russian_grapplers = df_filtered[df_filtered['category'] == 'Non-Russian Grappler']['win_rate']
    
    if len(russian_grapplers) > 5 and len(non_russian_grapplers) > 5:
        t_stat, p_val = ttest_ind(russian_grapplers, non_russian_grapplers)
        
        col1, col2 = st.columns(2)
        col1.metric("Russian Grappler Win Rate", f"{russian_grapplers.mean():.1%}")
        col2.metric("Non-Russian Grappler Win Rate", f"{non_russian_grapplers.mean():.1%}")
        
        st.write(f"**T-test p-value**: {p_val:.4f}")
        
        if p_val < 0.05:
            st.success("The difference is statistically significant!")
        else:
            st.warning("The difference is not statistically significant.")
    
    # Top Russian grapplers
    st.subheader("Top Russian Grapplers")
    top_russians = df_filtered[df_filtered['category'] == 'Russian Grappler'].nlargest(10, 'wins')
    st.dataframe(top_russians[['name', 'style', 'wins', 'total_fights', 'win_rate']])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**INSY 6500 Final Project**")
st.sidebar.markdown("Max M & Ian C")
