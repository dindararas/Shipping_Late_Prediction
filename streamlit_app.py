# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import pickle
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(
    page_title='E-Commerce Shipping Predictive Analytics',
    page_icon='📦',
    layout='wide',
    initial_sidebar_state='expanded')

# Function to load dataset
@st.cache_data
def load_data() :
    return pd.read_csv('data\DataCoSupplyChainDataset.csv',
                       delimiter = ',', encoding='latin-1')

# Load dataset
df = load_data()

# ---------------DATA PREPROCESSING------------------------------------
# Data preprocessing
# Convert data from object to datetime
df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'])

# Drop some columns with missing values
drop_cols_isna = ['Product Description', 'Order Zipcode', 
                  'Customer Zipcode', 'Customer Lname']
df = df.drop(columns=drop_cols_isna, axis=1)

# ---------------FEATURE ENGINEERING------------------------------------

# Feature Engineering
# extract month from shipping date
df['shipping_month'] = df['shipping date (DateOrders)'].dt.month

month_map = {1 : 'Jan', 2 : 'Feb', 3 : 'Mar',
             4 : 'Apr', 5 : 'May', 6 : 'Jun',
             7 : 'Jul', 8 : 'Aug', 9 : 'Sep',
             10 : 'Oct', 11 : 'Nov', 12 : 'Dec'}

df['shipping_month'] = df['shipping_month'].replace(month_map)

# order from January to December
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 
               'Sep', 'Oct', 'Nov', 'Dec']
df['shipping_month'] = pd.Categorical(df['shipping_month'], ordered=True, 
                                      categories = month_order)

# add column "DayofWeek"
df['shipping_DayofWeek'] = df['shipping date (DateOrders)'].dt.dayofweek

# change day name
day_map = {0 : 'Monday', 1 : 'Tuesday', 2 : 'Wednesday',
           3 : 'Thursday', 4 : 'Friday', 5 : 'Saturday', 6 : 'Sunday'}
df['shipping_DayofWeek'] = df['shipping_DayofWeek'].replace(day_map)

# order from Monday to Sunday
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['shipping_DayofWeek'] = pd.Categorical(df['shipping_DayofWeek'], ordered=True, 
                                          categories = day_order)

# Add column "Hour"
df['shipping_Hour'] = df['shipping date (DateOrders)'].dt.hour

# categorize column "Hour" into morning, afternoon, and evening
def categorize_time(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

# apply function to column "Hour"
df['shipping_TimeCategory'] = df['shipping_Hour'].apply(categorize_time)

# calculate the difference in shipping (real) and shipping (expected)
df['shipping_day_deviation'] =  df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

# assign store ID based on longitude and latitude
unique_stores = df[['Latitude','Longitude']].drop_duplicates().reset_index(drop=True)

# assign ID Store
unique_stores['Store_ID'] = range(1, len(unique_stores) + 1)

# merge with the original dataframe
df = df.merge(unique_stores, on=['Latitude','Longitude'], how='left')

# --------------------------LOAD MODEL-------------------------------------

# load model
@st.cache_resource
def load_model():
    with open('model/xgb_best.pkl', 'rb') as f :
        model, feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()

# -------------TITLE & SIDEBAR-------------------------
# Dashboard title
st.title('📦 SHIPPING PREDICTIVE ANALYTICS')
st.markdown('Logistic industry plays the key role in various industries ' \
'as it is reponsible for the entire supply chain processes. ' \
'This project focuses on data analysis and predictive modeling ' \
'to assess the risk of late deliveries')

# Border line
st.markdown('---')

# sidebar for global filter and navigation
st.sidebar.header('Setting & Navigation')

page = st.sidebar.radio(
    'Choose Page : ',
    ['🚚 Overview Dashboard','🤖 Prediction Model'])

# -------------FILTER FOR DASHBOARD----------
# Filter if the page = 'Overview Dashboard'
if page == '🚚 Overview Dashboard' :
    st.sidebar.markdown('⚙️ Dashboard Filter')

    # Filter based on year
    # Make a new column "year"
    df['shipping_year'] = df['shipping date (DateOrders)'].dt.year

    # Make a list of unique year
    year_list = df['shipping_year'].unique().tolist()

    # sort year
    year_list.sort(reverse=True)

    # Add "All" as part of filter
    year_list = ['All'] + year_list

    # Year filter
    st.sidebar.subheader('📅 Year Filter')
    selected_year = st.sidebar.selectbox('Select Year', options=year_list, index = 0 ) # set default to "All Year"

    # Apply year filter
    if selected_year == 'All' :
        df_filtered = df.copy()
    else :
        df_filtered = df[df['shipping_year'] == selected_year]
    

    # Make a list of market
    market_list = df['Market'].unique().tolist()

    # Market filter
    st.sidebar.subheader('🌏 Market Filter')
    selected_market = st.sidebar.multiselect('Select Market', options=market_list, default=market_list)

    # Apply market filter
    df_filtered = df_filtered[df_filtered['Market'].isin(selected_market)]

# if there's no data anymore after filter
if page == '🚚 Overview Dashboard' and df_filtered.empty :
    st.warning('No data avaible. Please choose suitable filters')
    st.stop()

#---------PAGE 1 : OVERVIEW DASHBOARD MAIN PAGE---------
if page == '🚚 Overview Dashboard' :
    # KPI Metrics
    st.subheader('Key Performance Indicators')

    # create columns
    col1, col2, col3, col4 = st.columns([3,3,3,3])

    # calculate metrics
    late_rate = df_filtered['Late_delivery_risk'].mean() * 100
    actual_lead_time = df_filtered['Days for shipping (real)'].mean()
    expected_lead_time = df_filtered['Days for shipment (scheduled)'].mean()
    day_deviation = df_filtered['shipping_day_deviation'].mean()

    # make metrics
    with col1 :
        st.metric(label = 'Delivery Late Rate', value = f'{round(late_rate,1)}%')
    with col2 :
        st.metric(label='Actual Lead Time', value = f'{round(actual_lead_time,1)} days')
    with col3 :
        st.metric(label='Expected Lead Time', value = f'{round(expected_lead_time,1)} days')
    with col4 :
        st.metric(label='Shipping Day Deviation', value = f'{round(day_deviation,1)} days')

    st.markdown('---')

    # monthly late rate vs quantity
    st.subheader('Monthly Trend')

    df_filtered['month_num'] = df_filtered['shipping date (DateOrders)'].dt.month

    # data aggregation
    df_monthly = df_filtered.groupby('shipping_month').agg({
        'month_num' : 'unique',
        'Late_delivery_risk': 'mean',  
        'Order Item Total' : 'sum'}).reset_index()
    
    df_monthly.columns = ['Month', 'Month_number', 'Late Delivery Risk', 'Total Quantity']

    # create subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # main y-axis
    fig.add_trace(
        go.Scatter(x=df_monthly['Month'], y=df_monthly['Late Delivery Risk']*100, name='Late Rate (%)'), secondary_y=False)
    
    # secondary y-axis
    fig.add_trace(
        go.Scatter(x=df_monthly['Month'], y=df_monthly['Total Quantity'], name='Total Quantity'), secondary_y=True)
    
    # add title
    fig.update_layout(title_text = 'Late Rate and Total Quantity by Month')

    # add x-axis name
    fig.update_xaxes(title_text='Month')

    # add y-axes names
    fig.update_yaxes(title_text='Late Rate (%)', secondary_y=False)
    fig.update_yaxes(title_text='Total Quantity', secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # ---late rate by shipping mode----
    # create two columns
    st.subheader('Late Rate by Shipping Mode')
    col5, col6 = st.columns([5,5])

    # set colors so it has consistent color across different chart
    color_map = {
    "Standard Class": "#1f77b4",   
    "Second Class": "#ff7f0e",  
    "First Class": "#2ca02c",      
    "Same Day": "#d62728"}

    with col5 :
        df_shipping = df_filtered.groupby('Shipping Mode')['Late_delivery_risk'].mean().reset_index()
        df_shipping['Late Rate (%)'] = df_shipping['Late_delivery_risk'] * 100

        bar_shippig = px.bar(df_shipping, x='Shipping Mode', y = 'Late Rate (%)', 
                             title = 'Late Rate by Shipping Mode', color = 'Shipping Mode',
                             color_discrete_map=color_map)
        bar_shippig.update_xaxes(title_text='Shipping Mode')
        bar_shippig.update_yaxes(title_text='Late Rate (%)')
        st.plotly_chart(bar_shippig, use_container_width=True)

    st.markdown("---")

    # boxplot
    with col6 :
        boxplot = px.box(df_filtered, x='Shipping Mode', y='shipping_day_deviation',
                     color='Shipping Mode', title='Shipping Day Deviation by Shipping Mode',
                     color_discrete_map=color_map)

        boxplot.update_xaxes(title_text='Shipping Mode')
        bar_shippig.update_yaxes(title_text='Shipping Day Deviation (days)')
        st.plotly_chart(boxplot, use_container_width=True)


if page == '🤖 Prediction Model' :
    st.header('🤖 Prediction Model')
    st.markdown('Use this machine learning model to predict wether the order will be late or not')

    # Input form
    st.subheader('Input your data')

    col1, col2 = st.columns(2)

    with col1 :
        days_scheduled = st.number_input('Expected Shipment Days', min_value=1, max_value=30)
        shipping_month = st.selectbox('Shipping Month', df['shipping_month'].cat.categories)
        shipping_hour = st.slider('Shipping Hour', 0, 23, 0)
        shipping_mode = st.selectbox('Shipping Mode', df['Shipping Mode'].unique())
        customer_street = st.selectbox('Customer Street', df['Customer Street'].unique())
    
    with col2 :
        order_city = st.selectbox('Order City', df['Order City'].unique())
        store_id = st.selectbox('Store ID', df['Store_ID'].unique())
        payment_type = st.selectbox('Payment Type', df['Type'].unique())
        order_item = st.selectbox('Order Item ID', df['Order Item Id'].unique())
    
    # make input dataframe
    input_df = pd.DataFrame([{
        'Shipping Mode' : 'shipping_mode',
        'shipping_month' : shipping_month,
        'shipping_Hour' : shipping_hour,
        'Days for shipment (scheduled)' : days_scheduled,
        'Order City' : order_city,
        'Store_ID' : store_id,
        'Type' : payment_type,
        'Order Item Id' : order_item}])
    
    st.markdown('### Data Input')
    st.dataframe(input_df)

    # prediction
    if st.button('Prediction of Late Risk') :
        # preprocessing
        X = pd.get_dummies(input_df, drop_first=False)

        # add some columns
        for col in feature_names:
            if col not in X.columns :
                X[col] = 0
        X = X.reindex(columns=feature_names, fill_value=0)

        # prediction
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]

        # Output
        st.subheader('Prediction Result')
        st.metric('Late Probability', f"{proba*100:.1f}%")
        if pred == 1:
            st.error('🚨 WARNING : Order is predicted as **LATE**')
        else:
            st.success('✅ Order is predicted as **ON TIME**')
            