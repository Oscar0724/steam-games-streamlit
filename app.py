# loading file
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import random
import os
def read_single_csv(file_path):
    df_chunk = pd.read_csv(file_path, chunksize=1000)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df = pd.concat(res_chunk, ignore_index=True)
    return res_df
@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "games.csv")
    df = pd.read_csv(file_path, index_col = False)
    return df
df = load_data()

columns = df.columns.tolist()
shift_start = 7  
new_columns = columns[:shift_start] + [''] + columns[shift_start:]
df.columns = pd.Index(new_columns[:len(df.columns)])
df.drop(df.columns[7], axis=1, inplace=True)
df_copy = df.copy()
df = df.drop('Screenshots', axis = 1)
df = df.drop('Header image', axis = 1)
df = df.drop('Website', axis = 1)
df = df.drop('Support email', axis = 1)
df = df.drop('Support url', axis = 1)
df = df.drop('Metacritic url', axis = 1)
df_base = df.copy()
df['Estimated owners'] = df['Estimated owners'].astype(str)
df['owner_min'] = df['Estimated owners'].str.replace(' ', '').str.split('-', expand=True)[0].astype(int)
df['owner_max'] = df['Estimated owners'].str.replace(' ', '').str.split('-', expand=True)[1].astype(int)

st.markdown('(The first load will be slow)...loading...')
    
    
# 3.1
@st.cache_resource
def get_fig1(df):
    df['Estimated owners'] = df['Estimated owners'].astype(str)
    df['owner_min'] = df['Estimated owners'].str.replace(' ', '').str.split('-', expand=True)[0].astype(int)
    df['owner_max'] = df['Estimated owners'].str.replace(' ', '').str.split('-', expand=True)[1].astype(int)
    def convert_to_group(row):
        min_val = row['owner_min']
        max_val = row['owner_max']
        if min_val == 0 and max_val == 0:
            return '0 - 0'
        elif max_val <= 20_000:
            return '0 - 20k'
        elif max_val <= 50_000:
            return '20k - 50k'
        elif max_val <= 100_000:
            return '50k - 100k'
        elif max_val <= 1_000_000:
            return '100k - 1M'
        elif max_val > 1_000_000:
            return 'Above 1M'
        else:
            return 'Other'
    df['owner_group'] = df.apply(convert_to_group, axis=1)
    range_counts = df['owner_group'].value_counts().reset_index()
    range_counts.columns = ['Owner Range', 'Game Count']
    order = ['0 - 0', '0 - 20k', '20k - 50k', '50k - 100k', '100k - 1M', 'Above 1M']
    range_counts['sort_key'] = range_counts['Owner Range'].apply(lambda x: order.index(x) if x in order else 999)
    range_counts = range_counts.sort_values('sort_key')
    fig1, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=range_counts, x='Owner Range', y='Game Count', palette='viridis', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=2)
    ax.set_xlabel('Owner Count Range')
    ax.set_ylabel('Number of Steam Games')
    ax.set_title('Number of Games by Player Count Range')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig1

@st.cache_resource
def get_fig1_1(df):
    df = df.copy()
    df['Estimated owners'] = df['Estimated owners'].astype(str)
    df['owner_min'] = df['Estimated owners'].str.replace(' ', '').str.split('-', expand=True)[0].astype(int)
    range_counts = df.groupby(['Estimated owners', 'owner_min']).size().reset_index(name='count')
    range_counts_sorted = range_counts.sort_values(by='owner_min').reset_index(drop=True)
    range_counts_filtered = range_counts_sorted.iloc[6:].copy()
    def format_range(s):
        try:
            parts = s.replace(' ', '').split('-')
            min_val = int(parts[0])
            max_val = int(parts[1])
            def format_num(n):
                if n >= 1_000_000:
                    return f"{n // 1_000_000}M"
                else:
                    return f"{n // 1_000}k"
            return f"{format_num(min_val)} - {format_num(max_val)}"
        except:
            return s
    range_counts_filtered['owner_range_label'] = range_counts_filtered['Estimated owners'].apply(format_range)
    fig1_1, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=range_counts_filtered,
        x='owner_range_label',
        y='count',
        palette='viridis',
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=2)
    ax.set_xlabel('Owner Count Range')
    ax.set_ylabel('Number of Steam Games')
    ax.set_title('Number of Games by Player Count Range (above 500k owners games)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig1_1

#3.2

@st.cache_resource
def get_fig2(df):
    df['Estimated owners'] = df['Estimated owners'].astype(str)
    df['owner_max'] = df['Estimated owners'].str.replace(' ', '').str.split('-', expand=True)[1].astype(int)
    top6_games = df.sort_values(by='owner_max', ascending=False).head(6)
    system_cols = ['Windows', 'Mac', 'Linux']
    system_status_counts = df[system_cols].melt(var_name='System', value_name='Supported')
    system_status_summary = system_status_counts.value_counts().reset_index(name='Game Count')
    system_status_summary.sort_values(by=['System', 'Supported'], ascending=[True, False], inplace=True)
    fig2, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=system_status_summary, x='System', y='Game Count', hue='Supported', palette='Set2', hue_order=[True, False], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=2)
    plt.title('Game Compatibility by System')
    ax.set_ylabel('Number of Steam Games')
    return fig2

#3.3

@st.cache_resource
def get_fig3(df):
    dev_df = df[['Developers', 'owner_max']].copy()
    dev_df['Developers'] = dev_df['Developers'].astype(str)
    dev_df['Developers'] = dev_df['Developers'].str.replace(r'\s*\(.*?\)', '', regex=True)
    dev_df['Developers'] = dev_df['Developers'].str.replace(r'[\s,]*(inc|ltd)\.?', '', regex=True, flags=re.IGNORECASE)
    dev_df = dev_df.assign(Developer=dev_df['Developers'].str.split(',')).explode('Developer')
    dev_df['Developer'] = dev_df['Developer'].str.strip()
    dev_df = dev_df[dev_df['Developer'] != '']
    dev_grouped = dev_df.groupby('Developer')['owner_max'].sum().reset_index()
    top5_devs = dev_grouped.sort_values(by='owner_max', ascending=False).head(5)
    dev_df['dev_group'] = dev_df['Developer'].apply(lambda x: x if x in top5_devs['Developer'].values else 'Others')
    final_grouped = dev_df.groupby('dev_group')['owner_max'].sum().reset_index()
    final_grouped = final_grouped.sort_values(by='owner_max', ascending=False)
    fig3, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=final_grouped, x='dev_group', y='owner_max', palette='Set2', ax=ax)
    def format_val(val):
        if val >= 1_000_000:
            return f"{val // 1_000_000}M"
        elif val >= 1000:
            return f"{val // 1000}k"
        else:
            return str(val)
    for container in ax.containers:
        labels = [format_val(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=3)
    ax.set_xlabel('Developer')
    ax.set_ylabel('Total max owner')
    ax.set_title('Top 5 Developers and other game developers')
    plt.xticks(rotation=20)
    return fig3

@st.cache_resource
def get_fig3_1(df):
    dev_df = df[['Developers', 'owner_max']].copy()
    dev_df['Developers'] = dev_df['Developers'].astype(str)
    dev_df['Developers'] = dev_df['Developers'].str.replace(r'\s*\(.*?\)', '', regex=True)
    dev_df['Developers'] = dev_df['Developers'].str.replace(r'[\s,]*(inc|ltd)\.?', '', regex=True, flags=re.IGNORECASE)
    dev_df = dev_df.assign(Developer=dev_df['Developers'].str.split(',')).explode('Developer')
    dev_df['Developer'] = dev_df['Developer'].str.strip()
    dev_df = dev_df[dev_df['Developer'] != '']
    top10_devs = dev_df.groupby('Developer')['owner_max'].sum().reset_index()
    top10_devs = top10_devs.sort_values(by='owner_max', ascending=False).head(10)
    fig3_1, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top10_devs, x='Developer', y='owner_max', palette='viridis', ax=ax)
    def format_val(val):
        if val >= 1_000_000:
            return f"{val // 1_000_000}M"
        elif val >= 1000:
            return f"{val // 1000}k"
        else:
            return str(val)
    for container in ax.containers:
        labels = [format_val(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=3)
    ax.set_title('Top 10 Developers')
    ax.set_xlabel('Developer')
    ax.set_ylabel('Total max owner')
    plt.xticks(rotation=30)
    return fig3_1

    
@st.cache_resource
def get_fig3_2(df):
    pub_df = df[['Publishers', 'owner_max']].copy()
    pub_df['Publishers'] = pub_df['Publishers'].astype(str)
    pub_df['Publishers'] = pub_df['Publishers'].str.replace(r'\s*\(.*?\)', '', regex=True)
    pub_df['Publishers'] = pub_df['Publishers'].str.replace(
        r'[\s,]*(inc|ltd|llc|gmbh|plc|co)\.?', '', regex=True, flags=re.IGNORECASE)
    pub_df = pub_df.assign(Publisher=pub_df['Publishers'].str.split(',')).explode('Publisher')
    pub_df['Publisher'] = pub_df['Publisher'].str.strip()
    pub_df = pub_df[pub_df['Publisher'] != '']
    pub_grouped = pub_df.groupby('Publisher')['owner_max'].sum().reset_index()
    top5_pubs = pub_grouped.sort_values(by='owner_max', ascending=False).head(5)
    pub_df['pub_group'] = pub_df['Publisher'].apply(lambda x: x if x in top5_pubs['Publisher'].values else 'Others')
    final_pub_grouped = pub_df.groupby('pub_group')['owner_max'].sum().reset_index()
    final_pub_grouped = final_pub_grouped.sort_values(by='owner_max', ascending=False)
    fig3_2, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=final_pub_grouped, x='pub_group', y='owner_max', palette='Set2', ax=ax)
    def format_val(val):
        if val >= 1_000_000:
            return f"{val // 1_000_000}M"
        elif val >= 1000:
            return f"{val // 1000}k"
        else:
            return str(val)
    for container in ax.containers:
        labels = [format_val(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=3)
    ax.set_xlabel('Publisher')
    ax.set_ylabel('Total max owner')
    ax.set_title('Top 5 Publishers and Other Publishers')
    plt.xticks(rotation=10)
    return fig3_2

@st.cache_resource
def get_fig3_3(df):
    pub_df = df[['Publishers', 'owner_max']].copy()
    pub_df['Publishers'] = pub_df['Publishers'].astype(str)
    pub_df['Publishers'] = pub_df['Publishers'].str.replace(r'\s*\(.*?\)', '', regex=True)
    pub_df['Publishers'] = pub_df['Publishers'].str.replace(
        r'[\s,]*(inc|ltd|llc|gmbh|plc|co)\.?', '', regex=True, flags=re.IGNORECASE)
    pub_df = pub_df.assign(Publisher=pub_df['Publishers'].str.split(',')).explode('Publisher')
    pub_df['Publisher'] = pub_df['Publisher'].str.strip()
    pub_df = pub_df[pub_df['Publisher'] != '']
    top10_pubs = pub_df.groupby('Publisher')['owner_max'].sum().reset_index()
    top10_pubs = top10_pubs.sort_values(by='owner_max', ascending=False).head(10)
    fig3_3, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=top10_pubs, x='Publisher', y='owner_max', palette='viridis', ax=ax)

    def format_val(val):
        if val >= 1_000_000:
            return f"{val // 1_000_000}M"
        elif val >= 1000:
            return f"{val // 1000}k"
        else:
            return str(val)

    for container in ax.containers:
        labels = [format_val(bar.get_height()) for bar in container]
        ax.bar_label(container, labels=labels, padding=3)
    ax.set_title('Top 10 Publishers')
    ax.set_xlabel('Publisher')
    ax.set_ylabel('Total max owner')
    plt.xticks(rotation=20)
    return fig3_3

#3.4

@st.cache_resource
def get_fig4(df):
    def price_category(price):
        if price == 0:
            return 'Free'
        elif price < 5:
            return '$0.01 - $4.99'
        elif price < 10:
            return '$5 - $9.99'
        elif price < 15:
            return '$10 - $14.99'
        elif price < 20:
            return '$15 - $19.99'
        elif price < 30:
            return '$20 - $29.99'
        elif price < 50:
            return '$30 - $49.99'
        elif price < 70:
            return '$50 - $69.99'
        else:
            return '$70+'
    df['price_group'] = df['Price'].apply(price_category)
    price_counts = df['price_group'].value_counts().reset_index()
    price_counts.columns = ['Price Range', 'Game Count']
    order = ['Free', '$0.01 - $4.99', '$5 - $9.99', '$10 - $14.99', '$15 - $19.99',
             '$20 - $29.99', '$30 - $49.99', '$50 - $69.99', '$70+']
    price_counts['sort_key'] = price_counts['Price Range'].apply(lambda x: order.index(x))
    price_counts = price_counts.sort_values('sort_key')

    fig4, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=price_counts, x='Price Range', y='Game Count', palette='mako', ax=ax)
    for container in ax.containers:
        labels = [f'{int(bar.get_height())}' for bar in container]
        ax.bar_label(container, labels=labels, padding=3)
    ax.set_title('Distribution of Games by Price Range')
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Number of Games')
    plt.xticks(rotation=20)
    return fig4

@st.cache_resource
def get_fig4_1(df):
    def owner_group(owner):
        if owner == 0:
            return '0'
        elif owner <= 20_000:
            return '0 - 20k'
        elif owner <= 50_000:
            return '20k - 50k'
        elif owner <= 100_000:
            return '50k - 100k'
        elif owner <= 500_000:
            return '100k - 500k'
        elif owner <= 1_000_000:
            return '500k - 1M'
        elif owner <= 5_000_000:
            return '1M - 5M'
        else:
            return '5M+'
    df['owner_group'] = df['owner_max'].apply(owner_group)
    order = ['0', '0 - 20k', '20k - 50k', '50k - 100k', '100k - 500k', '500k - 1M', '1M - 5M', '5M+']
    filtered_df = df[(df['Price'].notna()) & (df['owner_max'].notna())]
    max_prices = filtered_df.groupby('owner_group')['Price'].max().reindex(order)

    fig4_1, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x='owner_group', y='Price', order=order, palette='coolwarm', showfliers=False, ax=ax)
    for i, group in enumerate(order):
        max_price = max_prices[group]
        ax.scatter(i, max_price, color='red', marker='D', s=40, label='Max' if i == 0 else "")
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Price Distribution by Owner Count Group')
    ax.set_xlabel('Owner Max Range')
    ax.set_ylabel('Price (USD)')
    return fig4_1

#3.5

@st.cache_data
def get_fig5(df):
    df = df.copy()
    df['Release month'] = pd.to_datetime(df['Release date'], errors='coerce').dt.to_period('M').astype(str)
    monthly_counts = df['Release month'].value_counts().sort_index()
    monthly_df = monthly_counts.reset_index()
    monthly_df.columns = ['Month', 'Game Count']
    fig5 = px.line(
        monthly_df,
        x='Month',
        y='Game Count',
        title='Monthly Game Release Count',
        labels={'Month': 'Release Month', 'Game Count': 'Number of Games'},
        markers=True
    )
    fig5.update_layout(xaxis_tickangle=-45)
    return fig5

#3.6

from collections import Counter
@st.cache_resource
def get_fig6(df):
    def safe_parse_language_list(s):
        if isinstance(s, list):
            return s
        if not isinstance(s, str):
            return []
        matches = re.findall(r"'(.*?)'|\"(.*?)\"", s)
        return [m[0] if m[0] else m[1] for m in matches]
    df['Supported languages'] = df['Supported languages'].apply(safe_parse_language_list)
    lang_counter = Counter()
    for langs in df['Supported languages']:
        lang_counter.update(set(langs))
    top5_langs = lang_counter.most_common(5)
    labels = ['All Games'] + [lang for lang, _ in top5_langs]
    counts = [len(df)] + [count for _, count in top5_langs]
    plot_df = pd.DataFrame({
        'Language': labels,
        'Game Count': counts
    })
    sns.set(style='whitegrid')
    fig6, ax = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(data=plot_df, x='Language', y='Game Count', palette='viridis', ax=ax)
    for bar, label in zip(barplot.patches, plot_df['Language']):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, height + height*0.01, f'{int(height)}', ha='center', va='bottom', fontsize=10, color='black')
    ax.set_title('Game Count for All Games and Top 5 Supported Languages', fontsize=14)
    ax.set_xlabel('Supported Language')
    ax.set_ylabel('Number of Games')
    return fig6

@st.cache_resource
def get_fig6_1(df):
    def parse_audio_list(s):
        if isinstance(s, list):
            return s
        if not isinstance(s, str) or s.strip() in ['', '[]']:
            return ['No Audio']
        matches = re.findall(r"'(.*?)'|\"(.*?)\"", s)
        langs = [m[0] if m[0] else m[1] for m in matches]
        return langs if langs else ['No Audio']
    df['Full audio languages'] = df['Full audio languages'].apply(parse_audio_list)
    audio_counter = Counter()
    for langs in df['Full audio languages']:
        audio_counter.update(set(langs))
    top5_audio_langs = [item for item in audio_counter.most_common() if item[0] != 'No Audio'][:5]
    labels = ['All Games', 'No Audio'] + [lang for lang, _ in top5_audio_langs]
    counts = [len(df), audio_counter['No Audio']] + [count for _, count in top5_audio_langs]
    plot_df = pd.DataFrame({
        'Language': labels,
        'Game Count': counts
    })
    sns.set(style='whitegrid')
    fig6_1, ax = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(data=plot_df, x='Language', y='Game Count', palette='viridis', ax=ax)
    for bar in barplot.patches:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, height + height * 0.01, f'{int(height)}', ha='center', va='bottom', fontsize=10)
    ax.set_title('Game Count for All Games and Top 5 Full Audio Languages (Including No Audio)', fontsize=14)
    ax.set_xlabel('Full Audio Language')
    ax.set_ylabel('Number of Games')
    return fig6_1

@st.cache_resource
def get_fig6_2(df):
    has_audio_count = df['Full audio languages'].apply(lambda x: 'No Audio' not in x).sum()
    no_audio_count = df['Full audio languages'].apply(lambda x: 'No Audio' in x).sum()
    pie_labels = ['Has Full Audio', 'No Audio']
    pie_counts = [has_audio_count, no_audio_count]
    colors = ['#66c2a5', '#fc8d62']
    fig6_2, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        pie_counts,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'edgecolor': 'black'}
    )
    plt.setp(autotexts, size=12, weight='bold', color='white')
    ax.set_title('Proportion of Games with and without Full Audio Support', fontsize=13)
    ax.axis('equal')
    return fig6_2

#3.7

@st.cache_resource
def get_fig7(df):
    df = df.copy()
    df['owner_max'] = pd.to_numeric(df['owner_max'], errors='coerce')
    df['Average playtime forever'] = pd.to_numeric(df['Average playtime forever'], errors='coerce')
    if 'owner_group' in df.columns:
        df.drop(columns=['owner_group'], inplace=True)
    def convert_to_group(max_val):
        if pd.isna(max_val):
            return 'Other'
        elif max_val == 0:
            return '0 - 0'
        elif max_val <= 20_000:
            return '0 - 20k'
        elif max_val <= 50_000:
            return '20k - 50k'
        elif max_val <= 100_000:
            return '50k - 100k'
        elif max_val <= 1_000_000:
            return '100k - 1M'
        elif max_val > 1_000_000:
            return 'Above 1M'
        else:
            return 'Other'
    df['owner_group'] = df['owner_max'].apply(convert_to_group)
    owner_order = ['0 - 0', '0 - 20k', '20k - 50k', '50k - 100k', '100k - 1M', 'Above 1M']
    df['owner_group'] = pd.Categorical(df['owner_group'], categories=owner_order, ordered=True)
    grouped = df.groupby('owner_group')['Average playtime forever'].mean().reset_index()
    grouped.columns = ['Owner Group', 'Average Playtime (Forever)']
    fig7 = px.bar(
         grouped,
         x='Owner Group',
         y='Average Playtime (Forever)',
         title='Average Playtime vs Owner Group',
         text_auto='.1f',
         color='Owner Group',
         color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig7.update_layout(
        xaxis_title='Owner Max Group',
        yaxis_title='Average Playtime (Minutes)',
        showlegend=False
    )
    return fig7

@st.cache_resource
def get_fig7_1(df):
    df = df.copy()
    df['Average playtime forever'] = pd.to_numeric(df['Average playtime forever'], errors='coerce')
    df['Average playtime two weeks'] = pd.to_numeric(df['Average playtime two weeks'], errors='coerce')
    df_plot = df.dropna(subset=['Average playtime forever', 'Average playtime two weeks'])
    fig7_1 = px.scatter(
           df_plot,
           x='Average playtime forever',
           y='Average playtime two weeks',
           title='Playtime: Forever vs Last Two Weeks',
           labels={
               'Average playtime forever': 'Average Playtime (Forever)',
               'Average playtime two weeks': 'Average Playtime (2 Weeks)'
           },
        hover_data=['name'] if 'name' in df.columns else None,
        color='Average playtime forever',
        color_continuous_scale='Viridis',
        opacity=0.7,
    )
    fig7_1.update_traces(marker=dict(size=6))
    fig7_1.update_layout(
        xaxis_title='Average Playtime (Forever, mins)',
        yaxis_title='Average Playtime (2 Weeks, mins)',
        height=600
    )
    return fig7_1

#3.8

@st.cache_resource
def get_fig8(df):
    cols = ['owner_max', 'Achievements', 'Price', 'Average playtime forever', 
            'DiscountDLC count', 'Metacritic score', 'User score', 
            'Positive', 'Negative', 'Recommendations', 'Peak CCU']
    df_corr = df[cols].copy()
    df_corr = df_corr.apply(pd.to_numeric, errors='coerce')
    df_corr_clean = df_corr.dropna()
    corr_matrix = df_corr_clean.corr(method='pearson')
    fig8 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='Reds',
        title='Correlation Heatmap between Game Metrics',
        labels=dict(color='Correlation'),
        aspect='auto'
    )
    fig8.update_layout(
        width=800,
        height=800,
        xaxis=dict(tickangle=45, tickfont=dict(size=15)),
        yaxis=dict(tickangle=45, tickfont=dict(size=15))
    )
    return fig8

#3.9

@st.cache_resource
def get_fig9(df):
    df_tags = df[['Tags', 'owner_max']].dropna(subset=['Tags']).copy()
    df_tags['Tags'] = df_tags['Tags'].str.split(',')
    df_tags = df_tags.explode('Tags')
    df_tags['Tags'] = df_tags['Tags'].str.strip()
    df_tags = df_tags[df_tags['Tags'].notna() & (df_tags['Tags'] != '')]
    tag_owner = df_tags.groupby('Tags')['owner_max'].sum().reset_index()
    top_tags = tag_owner.sort_values('owner_max', ascending=False).head(10)
    total_owners = df['owner_max'].sum()
    total_row = pd.DataFrame([{'Tags': 'All Games', 'owner_max': total_owners}])
    full_data = pd.concat([top_tags, total_row], ignore_index=True)
    fig9 = px.bar(
         full_data,
         x='Tags',
         y='owner_max',
         text_auto='.2s',
         title='Top 10 Game Tags & All Games by Total Owner Count',
         labels={'owner_max': 'Total Owners', 'Tags': 'Game Tag'},
         color='Tags',
         color_discrete_sequence=px.colors.sequential.Viridis
     ) 
    fig9.update_layout(
        xaxis_title='Tag',
        yaxis_title='Total Owners',
        showlegend=False
    )
    return fig9

#3.10

@st.cache_resource
def get_fig10(df):
    df['Required age'] = pd.to_numeric(df['Required age'], errors='coerce')
    df['Age Category'] = df['Required age'].apply(
        lambda x: 'Age 0 (No Restriction)' if x == 0 else 'Age > 0 (Restricted)'
    )
    age_group_counts = df['Age Category'].value_counts().reset_index()
    age_group_counts.columns = ['Age Category', 'Game Count']
    
    fig10 = px.pie(
          age_group_counts,
          names='Age Category',
          values='Game Count',
          title='Game Count & Required Age',
          hole=0.3,
          color_discrete_sequence=px.colors.sequential.Viridis
      )
    fig10.update_traces(textinfo='percent+label')
    return fig10

@st.cache_resource
def get_fig10_1(df):
    df['Required age'] = pd.to_numeric(df['Required age'], errors='coerce')
    age_limited_df = df[df['Required age'] > 0]
    age_counts = age_limited_df['Required age'].value_counts().reset_index()
    age_counts.columns = ['Required Age', 'Game Count']
    age_counts = age_counts.sort_values('Required Age')
    fig10_1 = px.bar(
            age_counts,
            x='Required Age',
            y='Game Count',
            title='Game Count by Required Age (Age > 0)',
            text_auto=True,
            labels={'Required Age': 'Required Age', 'Game Count': 'Number of Games'},
            color='Game Count',
            color_continuous_scale='Plasma'
    )
    fig10_1.update_layout(xaxis=dict(type='category'))
    return fig10_1
# fig1
if 'fig1' not in st.session_state:
    st.session_state['fig1'] = get_fig1(df)

# fig1_1
if 'fig1_1' not in st.session_state:
    st.session_state['fig1_1'] = get_fig1_1(df)

# fig2
if 'fig2' not in st.session_state:
    st.session_state['fig2'] = get_fig2(df)

# fig3
if 'fig3' not in st.session_state:
    st.session_state['fig3'] = get_fig3(df)

# fig3_1
if 'fig3_1' not in st.session_state:
    st.session_state['fig3_1'] = get_fig3_1(df)

# fig3_2
if 'fig3_2' not in st.session_state:
    st.session_state['fig3_2'] = get_fig3_2(df)
# fig3_3
if 'fig3_3' not in st.session_state:
    st.session_state['fig3_3'] = get_fig3_3(df)

# fig4
if 'fig4' not in st.session_state:
    st.session_state['fig4'] = get_fig4(df)

# fig4_1
if 'fig4_1' not in st.session_state:
    st.session_state['fig4_1'] = get_fig4_1(df)

# fig5
if 'fig5' not in st.session_state:
    st.session_state['fig5'] = get_fig5(df)

# fig6
if 'fig6' not in st.session_state:
    st.session_state['fig6'] = get_fig6(df)

# fig6_1
if 'fig6_1' not in st.session_state:
    st.session_state['fig6_1'] = get_fig6_1(df)

# fig6_2
if 'fig6_2' not in st.session_state:
    st.session_state['fig6_2'] = get_fig6_2(df)

# fig7
if 'fig7' not in st.session_state:
    st.session_state['fig7'] = get_fig7(df)

# fig7_1
if 'fig7_1' not in st.session_state:
    st.session_state['fig7_1'] = get_fig7_1(df)

# fig8
if 'fig8' not in st.session_state:
    st.session_state['fig8'] = get_fig8(df)

# fig9
if 'fig9' not in st.session_state:
    st.session_state['fig9'] = get_fig9(df)

# fig10
if 'fig10' not in st.session_state:
    st.session_state['fig10'] = get_fig10(df)

# fig10_1
if 'fig10_1' not in st.session_state:
    st.session_state['fig10_1'] = get_fig10_1(df)
st.markdown('finish')

#dashboard designing

st.set_page_config(page_title="Steam Games Dashboard", layout="wide")
menu = [
    "Home",
    "Steam game player statistics",
    "Steam game compatibility statistics",
    "Top 10 Steam game developers and publishers",
    "Steam game price statistics",
    "Steam game release time statistics",
    "Steam games support language and voice statistics",
    "Steam game average player playing time and player churn statistics",
    "Steam game popularity survey",
    "Statistics of the top ten popular tags in steam games",
    "Steam game required age statistics"
]
st.sidebar.title("🧭 Navigation")
selected_page = st.sidebar.radio("Go to", menu)

if selected_page == "Home":
    st.markdown("# 🎮🕹️ Steam Games Analysis 🎯👾")
    st.markdown("Welcome to Steam games analysis dashboard")
    st.markdown("## 🔍 Search for a Steam Game")
    search_query = st.text_input("Enter game name to search:", "")
    if search_query:
        filtered_df_base = df_base[df_base['Name'].str.contains(search_query, case=False, na=False)]
        if not filtered_df_base.empty:
            st.markdown(f"### 🎯 Found {len(filtered_df_base)} result(s):")
            st.dataframe(filtered_df_base)
        else:
            st.warning("No games found matching your input.")
    @st.cache_data
    def filter_valid_games(df):
        df_filtered = df[(df['Positive'] > 0) & (df['owner_max'] > 20000)].copy()
        return df_filtered[['AppID', 'Name', 'Price', 'Developers', 'Publishers']].dropna()

    def recommend_games(df_filtered, n=3):
        return df_filtered.sample(n=n)

    st.header("🎮 Daily recommended games")
    df_filtered = filter_valid_games(df)
    if 'recommend_click' not in st.session_state:
        st.session_state['recommend_click'] = 0
    if st.button("🔄 Don't like? Change it!"):
        st.session_state['recommend_click'] += 1
    recommended = recommend_games(df_filtered)
    st.subheader("Today’s recommended games")
    for i, row in recommended.iterrows():
        st.markdown(f"""
        **🎯 Game Name:** {row['Name']}  
        **🆔 App ID:** {row['AppID']}  
        **💵 Price:** {row['Price']}  
        **🛠️ Developers:** {row['Developers']}  
        **🏢 Publishers:** {row['Publishers']}  
        ---  
        """)
    st.markdown('Raw data')
    st.dataframe(df_base)
    st.write(f"data has {df.shape[0]} lines，{df.shape[1]} columns")
        
elif selected_page == "Steam game player statistics":
    st.markdown("# Steam game player statistics")
    st.pyplot(st.session_state['fig1'])
    st.pyplot(st.session_state['fig1_1'])
elif selected_page == "Steam game compatibility statistics":
    st.markdown("# Steam game player statistics")
    st.pyplot(st.session_state['fig2'])
elif selected_page == "Top 10 Steam game developers and publishers":
    st.markdown("# Top 10 Steam game developers and publishers")
    st.pyplot(st.session_state['fig3'])
    st.pyplot(st.session_state['fig3_1'])
    st.pyplot(st.session_state['fig3_2'])
    st.pyplot(st.session_state['fig3_3'])
elif selected_page == "Steam game price statistics":
    st.markdown("# Steam game price statistics")
    st.pyplot(st.session_state['fig4'])
    st.pyplot(st.session_state['fig4_1'])
elif selected_page == "Steam game release time statistics":
    st.markdown("# Steam game release time statistics")
    df['Release date'] = pd.to_datetime(df['Release date'], errors='coerce')
    df['Release month'] = df['Release date'].dt.to_period('M').astype(str)
    all_months = df['Release month'].dropna().sort_values().unique().tolist()
    default_start = all_months[0]
    default_end = all_months[-1]
    selected_range = st.select_slider(
        "Select release month range",
       options=all_months,
        value=(default_start, default_end)
    )
    start_month, end_month = selected_range
    filtered_df = df[(df['Release month'] >= start_month) & (df['Release month'] <= end_month)]
    fig5 = get_fig5(filtered_df)
    st.plotly_chart(fig5)
elif selected_page == "Steam games support language and voice statistics":
    st.markdown("# Steam games support language and voice statistics")
    st.pyplot(st.session_state['fig6'])
    st.pyplot(st.session_state['fig6_1'])
    st.pyplot(st.session_state['fig6_2'])
elif selected_page == "Steam game average player playing time and player churn statistics":
    st.markdown("# Steam game average player playing time and player churn statistics")
    st.plotly_chart(st.session_state['fig7'])
    st.plotly_chart(st.session_state['fig7_1'])
elif selected_page == "Steam game popularity survey":
    st.markdown("# team game popularity survey")
    st.plotly_chart(st.session_state['fig8'])
elif selected_page == "Statistics of the top ten popular tags in steam games":
    st.markdown("# Statistics of the top ten popular tags in steam games")
    st.plotly_chart(st.session_state['fig9'])
elif selected_page =="Steam game required age statistics":
    st.markdown("# Steam game required age statistics")
    st.plotly_chart(st.session_state['fig10'])
    st.plotly_chart(st.session_state['fig10_1'])
