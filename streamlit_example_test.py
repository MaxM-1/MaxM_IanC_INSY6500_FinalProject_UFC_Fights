import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, ttest_ind, norm, chi2_contingency

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


# ------------------------------------
    # AGE DISTRIBUTION (Overall)
    # ------------------------------------
    st.subheader("Age Distribution of Fighters")

    # Compute ages
    attrs['dob'] = pd.to_datetime(attrs['dob'], errors='coerce')
    today = pd.to_datetime("today")
    attrs['age_years'] = (today - attrs['dob']).dt.days / 365.25

    # Keep reasonable ages
    age_data = attrs['age_years'].dropna()
    age_data = age_data[(age_data >= 18) & (age_data <= 50)]

    mean_age = age_data.mean()
    median_age = age_data.median()

    # Skinny histogram settings
    fig_age = px.histogram(
        age_data,
        nbins=40,                   # more bins → skinnier histogram
        labels={'value': 'Age (years)', 'count': 'Fighter Count'},
        title="Overall Age Distribution",
        opacity=0.8
    )

    # Add mean and median lines
    fig_age.add_vline(x=mean_age, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_age:.1f}", annotation_position="top left")
    fig_age.add_vline(x=median_age, line_dash="dot", line_color="blue",
                      annotation_text=f"Median: {median_age:.1f}", annotation_position="top right")

    # Make bars skinny-looking
    fig_age.update_layout(
        bargap=0.08,               # spacing between bars
        width=800,                 # skinnier look
        height=350
    )

    st.plotly_chart(fig_age, use_container_width=True)

# ------------------------------------
    # Top 10 Fighting Styles
    # ------------------------------------
    st.subheader("Top 10 Fighting Styles (by Number of Fighters)")

    style_counts = (
        attrs['style']
        .fillna("Unknown")
        .value_counts()
        .head(10)
    )

    fig_styles = px.bar(
        x=style_counts.values,
        y=style_counts.index,
        orientation='h',
        labels={'x': 'Number of Fighters', 'y': 'Fighting Style'},
        title="Top 10 Fighting Styles"
    )
    fig_styles.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_styles, use_container_width=True)

# ------------------------------------
    # Number of Fights per Year
    # ------------------------------------
    st.subheader("Number of Fights per Year")

    hist['event_date'] = pd.to_datetime(hist['event_date'], errors='coerce')
    fights_per_year = (
        hist['event_date']
        .dt.year
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    fig_fights_year = px.bar(
        x=fights_per_year.index,
        y=fights_per_year.values,
        labels={'x': 'Year', 'y': 'Number of Fights'},
        title="Fights per Year"
    )
    st.plotly_chart(fig_fights_year, use_container_width=True)

# ------------------------------------
    # Top 15 Fighters (Wins)
    # ------------------------------------
    st.subheader("Top 15 Fighters (Wins)")

    # Work from wins only to get KO/Sub breakdown
    wins_only = hist[hist['fight_result'] == 'W'].copy()
    wins_only['ko_win'] = (wins_only['fight_result_type'] == 'KO-TKO').astype(int)
    wins_only['sub_win'] = (wins_only['fight_result_type'] == 'SUBMISSION').astype(int)

    finish_counts = (
        wins_only
        .groupby('fighter_id')
        .agg(
            ko_wins=('ko_win', 'sum'),
            sub_wins=('sub_win', 'sum')
        )
        .reset_index()
    )

    # Merge with precomputed fighter_records (which already has total wins)
    fighter_summary = fighter_records.merge(finish_counts, on='fighter_id', how='left')
    fighter_summary[['ko_wins', 'sub_wins']] = fighter_summary[['ko_wins', 'sub_wins']].fillna(0)

    # Attach fighter names and weight class
    fighter_summary = fighter_summary.merge(
        attrs[['fighter_id', 'name', 'weight_class']],
        on='fighter_id',
        how='left'
    )

    # Top 15 by total wins
    top_total = (
        fighter_summary
        .sort_values('wins', ascending=False)
        .head(15)
    )

    # Top 15 by KO/TKO wins
    top_ko = (
        fighter_summary
        .sort_values('ko_wins', ascending=False)
        .head(15)
    )

    # Top 15 by submission wins
    top_sub = (
        fighter_summary
        .sort_values('sub_wins', ascending=False)
        .head(15)
    )

    tab1, tab2, tab3 = st.tabs([
        "Top 15 by Total Wins",
        "Top 15 by KO/TKO Wins",
        "Top 15 by Submission Wins"
    ])

    with tab1:
        fig_top_total = px.bar(
            top_total,
            x='wins',
            y='name',
            orientation='h',
            labels={'wins': 'Total Wins', 'name': 'Fighter'},
            title="Top 15 Fighters by Total Wins",
            text='wins'
        )
        fig_top_total.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_total, use_container_width=True)

    with tab2:
        fig_top_ko = px.bar(
            top_ko,
            x='ko_wins',
            y='name',
            orientation='h',
            labels={'ko_wins': 'KO/TKO Wins', 'name': 'Fighter'},
            title="Top 15 Fighters by KO/TKO Wins",
            text='ko_wins'
        )
        fig_top_ko.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_ko, use_container_width=True)

    with tab3:
        fig_top_sub = px.bar(
            top_sub,
            x='sub_wins',
            y='name',
            orientation='h',
            labels={'sub_wins': 'Submission Wins', 'name': 'Fighter'},
            title="Top 15 Fighters by Submission Wins",
            text='sub_wins'
        )
        fig_top_sub.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_sub, use_container_width=True)


    
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

    # --- Win Probability vs. Age (by year) with regression line and p-value ---
    st.subheader("Win Probability vs. Age (by Year)")
    merged_filtered['win_binary'] = (merged_filtered['fight_result'] == 'W').astype(int)
    win_rate_by_age = merged_filtered.groupby(merged_filtered['age_at_fight'].round())['win_binary'].mean()

    # Logistic regression for win probability
    import statsmodels.api as sm
    age_vals = merged_filtered['age_at_fight'].values
    win_vals = merged_filtered['win_binary'].values
    age_grid = np.linspace(age_vals.min(), age_vals.max(), 100)
    X = sm.add_constant(age_vals)
    logit_model = sm.Logit(win_vals, X, missing='drop').fit(disp=0)
    X_pred = sm.add_constant(age_grid)
    win_pred = logit_model.predict(X_pred)

    import plotly.graph_objects as go
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=win_rate_by_age.index, y=win_rate_by_age.values,
                              mode='lines+markers', name='Actual Win Rate'))
    fig3.add_trace(go.Scatter(x=age_grid, y=win_pred, mode='lines', name='Logistic Regression',
                              line=dict(color='green', dash='dash')))
    # Get p-value for age coefficient
    pval_win = logit_model.pvalues[1] if len(logit_model.pvalues) > 1 else None
    pval_text_win = f"p-value (age): {pval_win:.4g}" if pval_win is not None else "p-value unavailable"
    fig3.update_layout(title='Win Probability vs. Age (Actual + Regression)',
                      xaxis_title='Age at Fight', yaxis_title='Win Probability', yaxis_range=[0, 1],
                      annotations=[dict(x=age_grid.mean(), y=0.98, text=pval_text_win, showarrow=False, font=dict(size=12, color='green'))])
    st.plotly_chart(fig3, use_container_width=True)

    # --- KO Risk vs. Age (by year) with regression line ---
    st.subheader("KO Risk vs. Age (by Year)")
    merged_filtered['ko_suffered'] = ((merged_filtered['fight_result'] == 'L') & (merged_filtered['fight_result_type'] == 'KO-TKO')).astype(int)
    ko_risk_by_age = merged_filtered.groupby(merged_filtered['age_at_fight'].round())['ko_suffered'].mean()

    # Logistic regression for KO risk
    ko_vals = merged_filtered['ko_suffered'].values
    logit_model_ko = sm.Logit(ko_vals, X, missing='drop').fit(disp=0)
    ko_pred = logit_model_ko.predict(X_pred)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=ko_risk_by_age.index, y=ko_risk_by_age.values,
                              mode='lines+markers', name='Actual KO Risk', line=dict(color='red')))
    fig4.add_trace(go.Scatter(x=age_grid, y=ko_pred, mode='lines', name='Logistic Regression',
                              line=dict(color='black', dash='dash')))
    # Get p-value for age coefficient (KO risk)
    pval_ko = logit_model_ko.pvalues[1] if len(logit_model_ko.pvalues) > 1 else None
    pval_text_ko = f"p-value (age): {pval_ko:.4g}" if pval_ko is not None else "p-value unavailable"
    fig4.update_layout(title='KO Risk vs. Age (Actual + Regression)',
                      xaxis_title='Age at Fight', yaxis_title='KO Risk', yaxis_range=[0, 1],
                      annotations=[dict(x=age_grid.mean(), y=0.98, text=pval_text_ko, showarrow=False, font=dict(size=12, color='red'))])
    st.plotly_chart(fig4, use_container_width=True)

# ============================================
# PAGE 3: Fighter Style Analysis
# ============================================
elif page == "Fighter Style Analysis":
    st.header("Fighter Style Analysis")
    min_wins = 5
    
    # Prepare data: only wins, and simplified finish categories
    wins = hist[hist['fight_result'] == 'W'].copy()
    wins['finish_category'] = np.where(
        wins['fight_result_type'] == 'KO-TKO', 'KO/TKO',
        np.where(wins['fight_result_type'] == 'SUBMISSION', 'Submission', 'Decision')
    )
    
    wins_merged = wins.merge(attrs[['fighter_id', 'style']], on='fighter_id', how='left')
    
    # -----------------------------
    # 1. Finish tendencies by style
    # -----------------------------
    style_finish = wins_merged.groupby(['style', 'finish_category']).size().reset_index(name='count')
    style_totals = style_finish.groupby('style')['count'].sum()
    valid_styles = style_totals[style_totals >= min_wins].index
    style_finish = style_finish[style_finish['style'].isin(valid_styles)]
    
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
        fig = px.bar(
            x=style_stats.index,
            y=style_stats['ko_rate'],
            labels={'x': 'Fighting Style', 'y': 'KO/TKO Rate'},
            color=style_stats['ko_rate'],
            color_continuous_scale='reds'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Submission Rate by Style")
        style_sub_sorted = style_stats.sort_values('sub_rate', ascending=False)
        fig2 = px.bar(
            x=style_sub_sorted.index,
            y=style_sub_sorted['sub_rate'],
            labels={'x': 'Fighting Style', 'y': 'Submission Rate'},
            color=style_sub_sorted['sub_rate'],
            color_continuous_scale='greens'
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")

    # -----------------------------
    # 1b. Finish type rates by style *category* + chi-square test
    # -----------------------------
    st.subheader("Finish Type Rates by Style Category")

    # Classify fighters into style categories
    def classify_style(style):
        if pd.isna(style):
            return "Generalist"
        s = style.lower()

        bjj_keywords = ["bjj", "brazilian jiu jitsu", "jiu-jitsu", "jiu jitsu"]
        if any(k in s for k in bjj_keywords):
            return "BJJ"

        wrestling_keywords = ["wrestling", "wrestler", "freestyle wrestling", "folkstyle"]
        if any(k in s for k in wrestling_keywords):
            return "Wrestler"

        striking_keywords = [
            "boxing", "kickboxing", "muay thai", "striker",
            "taekwondo", "karate", "savate"
        ]
        if any(k in s for k in striking_keywords):
            return "Striker"

        return "Generalist"

    attrs['style_group'] = attrs['style'].apply(classify_style)

    wins_style = wins.merge(
        attrs[['fighter_id', 'style_group']],
        on='fighter_id',
        how='left'
    )

    # Contingency table: style_group x finish_category
    finish_table = pd.crosstab(
        wins_style['style_group'],
        wins_style['finish_category']
    )

    # Convert to rates (row-wise proportions)
    finish_rates = finish_table.div(finish_table.sum(axis=1), axis=0)

    # Plot grouped bar chart of finish type rates by style category
    finish_long = (
        finish_rates
        .reset_index()
        .melt(id_vars='style_group', var_name='Finish Type', value_name='Rate')
    )

    fig_cat = px.bar(
        finish_long,
        x='style_group',
        y='Rate',
        color='Finish Type',
        barmode='group',
        labels={'style_group': 'Style Category', 'Rate': 'Proportion of Wins'},
        range_y=[0, 1]
    )
    fig_cat.update_layout(legend_title_text="Finish Type")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Chi-square test of independence: does finish type depend on style category?
    chi2, p_finish, dof, expected = chi2_contingency(finish_table)

    st.write(f"**Chi-square test p-value:** {p_finish:.4g}")
    if p_finish < 0.05:
        st.success(
            "Finish type **does** depend on style category (statistically significant). "
            "This supports the idea that different style groups tend to win in different ways "
            "(for example, strikers by KO/TKO and BJJ specialists by submission)."
        )
    else:
        st.info(
            "Finish type does **not** show a statistically significant dependence on style category "
            "at the 5% level. The observed differences in finish patterns could be due to chance."
        )

    st.markdown("---")
    
    # -----------------------------
    # 2. Strikers vs Non-Strikers (KO rate + z-test)
    # -----------------------------
    st.subheader("Strikers vs Non-Strikers: KO/TKO Finish Rate")
    
    striking_keywords = ["boxing", "kickboxing", "muay thai", "striker", "taekwondo", "karate", "savate"]
    attrs['is_striker'] = attrs['style'].fillna('').str.lower().apply(
        lambda s: any(k in s for k in striking_keywords)
    )
    
    wins_striker = wins.merge(attrs[['fighter_id', 'is_striker']], on='fighter_id', how='left')
    
    # Counts and totals
    ko_strikers = ((wins_striker['is_striker'] == True)  & (wins_striker['finish_category'] == "KO/TKO")).sum()
    ko_non      = ((wins_striker['is_striker'] == False) & (wins_striker['finish_category'] == "KO/TKO")).sum()
    total_strikers = (wins_striker['is_striker'] == True).sum()
    total_non      = (wins_striker['is_striker'] == False).sum()
    
    striker_rate = ko_strikers / total_strikers if total_strikers > 0 else np.nan
    non_striker_rate = ko_non / total_non if total_non > 0 else np.nan
    
    # Two-proportion z-test for KO rates
    if total_strikers > 0 and total_non > 0:
        p_pool = (ko_strikers + ko_non) / (total_strikers + total_non)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/total_strikers + 1/total_non))
        z_stat = (striker_rate - non_striker_rate) / se
        p_ko = 2 * (1 - norm.cdf(abs(z_stat)))
    else:
        z_stat, p_ko = np.nan, np.nan
    
    # Bar chart
    ko_df = pd.DataFrame({
        "Group": ["Striker", "Non-striker"],
        "KO Rate": [striker_rate, non_striker_rate]
    })
    fig_ko = px.bar(
        ko_df,
        x="Group",
        y="KO Rate",
        range_y=[0, 1],
        text=ko_df["KO Rate"].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"),
        labels={"KO Rate": "KO/TKO Win Rate"}
    )
    fig_ko.update_traces(textposition="outside")
    st.plotly_chart(fig_ko, use_container_width=True)
    
    st.write(
        f"**Striker KO/TKO rate:** {striker_rate:.1%} "
        f"vs **Non-striker KO/TKO rate:** {non_striker_rate:.1%}"
    )
    if not np.isnan(p_ko):
        st.write(f"**Two-proportion z-test p-value:** {p_ko:.4g}")
        if p_ko < 0.05:
            st.success(
                "This difference is statistically significant, suggesting that "
                "strikers are genuinely more likely to win by KO/TKO than non-strikers."
            )
        else:
            st.info(
                "This difference is *not* statistically significant at the 5% level, "
                "so we cannot confidently say strikers have a higher KO/TKO rate than non-strikers."
            )
    
    st.markdown("---")
    
    # -----------------------------
    # 3. Grapplers vs Non-Grapplers (Submission rate + z-test)
    # -----------------------------
    st.subheader("Grapplers vs Non-Grapplers: Submission Finish Rate")
    
    def is_grappler(style):
        if pd.isna(style):
            return False
        s = style.lower()
        grappling_keywords = [
            "bjj", "brazilian jiu jitsu", "jiu-jitsu", "jiu jitsu",
            "grappler", "wrestling", "wrestler", "sambo", "judo"
        ]
        return any(k in s for k in grappling_keywords)
    
    attrs['is_grappler'] = attrs['style'].apply(is_grappler)
    
    wins_grappler = wins.merge(attrs[['fighter_id', 'is_grappler']], on='fighter_id', how='left')
    wins_grappler['SUB_binary'] = wins_grappler['finish_category'].apply(
        lambda x: 1 if x == "Submission" else 0
    )
    
    sub_grapplers = wins_grappler[(wins_grappler['is_grappler'] == True)  & (wins_grappler['SUB_binary'] == 1)].shape[0]
    sub_non       = wins_grappler[(wins_grappler['is_grappler'] == False) & (wins_grappler['SUB_binary'] == 1)].shape[0]
    total_grapplers = (wins_grappler['is_grappler'] == True).sum()
    total_non_grap = (wins_grappler['is_grappler'] == False).sum()
    
    grappler_rate = sub_grapplers / total_grapplers if total_grapplers > 0 else np.nan
    non_grappler_rate = sub_non / total_non_grap if total_non_grap > 0 else np.nan
    
    # Two-proportion z-test for submission rates
    if total_grapplers > 0 and total_non_grap > 0:
        p_pool_sub = (sub_grapplers + sub_non) / (total_grapplers + total_non_grap)
        se_sub = np.sqrt(p_pool_sub * (1 - p_pool_sub) * (1/total_grapplers + 1/total_non_grap))
        z_sub = (grappler_rate - non_grappler_rate) / se_sub
        p_sub = 2 * (1 - norm.cdf(abs(z_sub)))
    else:
        z_sub, p_sub = np.nan, np.nan
    
    # Bar chart
    sub_df = pd.DataFrame({
        "Group": ["Grappler", "Non-grappler"],
        "Submission Rate": [grappler_rate, non_grappler_rate]
    })
    fig_sub = px.bar(
        sub_df,
        x="Group",
        y="Submission Rate",
        range_y=[0, 1],
        text=sub_df["Submission Rate"].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"),
        labels={"Submission Rate": "Submission Win Rate"}
    )
    fig_sub.update_traces(textposition="outside")
    st.plotly_chart(fig_sub, use_container_width=True)
    
    st.write(
        f"**Grappler submission rate:** {grappler_rate:.1%} "
        f"vs **Non-grappler submission rate:** {non_grappler_rate:.1%}"
    )
    if not np.isnan(p_sub):
        st.write(f"**Two-proportion z-test p-value:** {p_sub:.4g}")
        if p_sub < 0.05:
            st.success(
                "This difference is statistically significant, suggesting that grapplers "
                "are genuinely more likely to win by submission than non-grapplers."
            )
        else:
            st.info(
                "This difference is *not* statistically significant at the 5% level, "
                "so we cannot confidently say grapplers have a higher submission rate."
            )

# ============================================
# PAGE 4: Reach/Height Ratio Analysis
# ============================================
elif page == "Reach/Height Ratio":
    st.header("Reach-to-Height Ratio Analysis")
    st.markdown("**Hypothesis**: Fighters with longer reach relative to height have better records")
    min_fights = 3
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

# ------------------------------------
    # Correlation between Reach/Height Ratio and Win Rate by Fighting Style
    # ------------------------------------
    st.subheader("Correlation Between Reach/Height Ratio and Win Rate by Fighting Style")

    # Compute correlation per *raw* style (not grouped)
    corr_rows = []
    # Drop styles that are missing
    df_style = df_filtered.dropna(subset=['style']).copy()

    for style, sub in df_style.groupby('style'):
        # Require a reasonable sample size and variation
        if (
            len(sub) >= 10 and
            sub['reach_height_ratio'].nunique() > 1 and
            sub['win_rate'].nunique() > 1
        ):
            c, p = pearsonr(sub['reach_height_ratio'], sub['win_rate'])
            corr_rows.append({
                "style": style,
                "correlation": c,
                "p_value": p,
                "n_fighters": len(sub)
            })

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)

        # Sort so negative correlations at top, positives at bottom (like your example)
        corr_df = corr_df.sort_values("correlation", ascending=True)

        # Color bars: red for negative, green for positive
        colors = ["#e74c3c" if c < 0 else "#27ae60" for c in corr_df["correlation"]]

        # Horizontal bar chart
        fig_style_corr = go.Figure(
            go.Bar(
                x=corr_df["correlation"],
                y=corr_df["style"],
                orientation="h",
                marker_color=colors
            )
        )

        # Vertical line at 0
        fig_style_corr.add_vline(x=0, line_width=2, line_color="black")

        fig_style_corr.update_layout(
            title="Correlation between Reach/Height Ratio and Win Rate by Fighting Style",
            xaxis_title="Correlation (Reach/Height Ratio vs Win Rate)",
            yaxis_title="Fighting Style",
            xaxis=dict(range=[-0.35, 0.35])  # tweak this if your values are narrower/wider
        )

        st.plotly_chart(fig_style_corr, use_container_width=True)

        # Optional: show numeric table below
        st.dataframe(
            corr_df[["style", "correlation", "p_value", "n_fighters"]].style.format({
                "correlation": "{:.3f}",
                "p_value": "{:.4f}"
            })
        )
    else:
        st.info("Not enough data per fighting style to compute meaningful correlations.")

# ============================================
# PAGE 5: Russian Grappler Dominance
# ============================================
elif page == "🇷🇺 Russian Grappler Dominance":
    st.header("Russian Grappler Dominance Analysis")
    st.markdown("Are Russian grapplers more dominant in the UFC?")
    min_fights = 3
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

# ------------------------------------
    # Average Win Rate of Grapplers by Country
    # ------------------------------------
    st.subheader("Average Win Rate of Grapplers by Country")

    # Use only grapplers from df_filtered (already has win_rate, total_fights, country, is_grappler)
    grapplers_df = df_filtered[df_filtered['is_grappler'] == True].copy()
    grapplers_df = grapplers_df.dropna(subset=['country'])

    # Aggregate: average win rate and number of grappler fighters per country
    country_stats = (
        grapplers_df
        .groupby('country')
        .agg(
            avg_win_rate=('win_rate', 'mean'),
            n_fighters=('fighter_id', 'nunique')
        )
        .reset_index()
    )

    # Optional: keep countries with at least a few grapplers
    min_grapplers = 5
    country_stats = country_stats[country_stats['n_fighters'] >= min_grapplers]

    # Sort by average win rate (like your example)
    country_stats = country_stats.sort_values('avg_win_rate')

    # Overall mean win rate for all grapplers
    overall_mean = grapplers_df['win_rate'].mean()

    # Colors: highlight Russia in red, others in blue
    def color_for_country(c):
        return "#e74c3c" if c.strip().lower() == "russia" else "#5dade2"

    colors = [color_for_country(c) for c in country_stats['country']]

    # Build horizontal bar chart
    fig_country = go.Figure(
        go.Bar(
            x=country_stats['avg_win_rate'],
            y=country_stats['country'],
            orientation='h',
            marker_color=colors,
            text=country_stats['n_fighters'].apply(lambda n: f"n={n}"),
            textposition='outside'
        )
    )

    # Vertical dashed line at overall mean grappler win rate
    fig_country.add_vline(
        x=overall_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Overall Grappler Mean: {overall_mean:.1%}",
        annotation_position="top right"
    )

    fig_country.update_layout(
        title="Average Win Rate of Grapplers by Country",
        xaxis_title="Average Win Rate",
        yaxis_title="Country",
        xaxis_tickformat=".0%"  # show as percentages
    )

    st.plotly_chart(fig_country, use_container_width=True)
    
    # Top Russian grapplers
    st.subheader("Top Russian Grapplers")
    top_russians = df_filtered[df_filtered['category'] == 'Russian Grappler'].nlargest(10, 'wins')
    st.dataframe(top_russians[['name', 'style', 'wins', 'total_fights', 'win_rate']])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**INSY 6500 Final Project**")
st.sidebar.markdown("Max M & Ian C")