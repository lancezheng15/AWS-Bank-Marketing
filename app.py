import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import boto3
import joblib
import logging
import traceback
from io import BytesIO
from datetime import datetime

# ---- Logging ----
logging.basicConfig(
    filename="app_log.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---- Configuration ----
BUCKET = "zmd7353-cloud"  # Replace with your bucket name
MODEL_PATHS = {
    "xgboost": "models/xgboost.pkl",
    "random_forest": "models/random_forest.pkl",
    "logistic_regression": "models/logistic_regression.pkl"
}

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main content styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* Card styling */
    .styledCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Input group styling */
    .input-group {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Section headers */
    .section-header {
        color: #1652f0;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1652f0;
    }
    
    /* Button and select box alignment */
    .model-selection-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 2rem;
        max-width: 300px;
    }

    .model-selection-container .stSelectbox {
        width: 100%;
    }

    .model-selection-container .stButton {
        width: 100%;
    }

    .stSelectbox > div > div {
        height: 48px !important;
        border-radius: 6px !important;
        display: flex;
        align-items: center;
        border: none;
        background-color: rgb(242, 244, 247) !important;
        padding: 0 12px;
    }
    
    .stButton > button {
        background-color: #1652f0 !important;
        color: white;
        font-weight: 600;
        width: 100%;
        height: 48px !important;
        border-radius: 6px !important;
        justify-content: center;
        padding: 0;
        margin: 0;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #0f3dab !important;
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Prediction styling */
    .prediction-card {
        background: linear-gradient(135deg, #1652f0 0%, #0f3dab 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .prediction-metric {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    /* Container styling for model selection and prediction */
    div.row-widget.stButton {
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- Cache + Load model ----
@st.cache_resource(show_spinner=True)
def load_model_from_s3(model_name):
    try:
        s3 = boto3.client("s3")
        model_key = MODEL_PATHS[model_name]
        response = s3.get_object(Bucket=BUCKET, Key=model_key)
        model_bytes = response["Body"].read()
        model = joblib.load(BytesIO(model_bytes))
        logging.info(f"Successfully loaded model: {model_key}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model {model_name}: {e}\n{traceback.format_exc()}")
        st.error(f"Failed to load model: {e}")
        return None

# Load training and test data
@st.cache_data
def load_data():
    try:
        X_train = pd.read_csv('data/X_train_res.csv')
        y_train = pd.read_csv('data/y_train_res.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        return (X_train, y_train['y'].values,
                X_test, y_test['y'].values)
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None, None, None, None

try:
    X_train, y_train, X_test, y_test = load_data()
    
    # Debug information
    if X_train is not None:
        st.sidebar.info(f"Training data - X: {X_train.shape}, y: {y_train.shape}")
    if X_test is not None:
        st.sidebar.info(f"Test data - X: {X_test.shape}, y: {y_test.shape}")
    
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    X_train = None
    y_train = None
    X_test = None
    y_test = None

# Sidebar with modern styling
st.sidebar.markdown('<h2 style="color: #1652f0;">Navigation</h2>', unsafe_allow_html=True)
page = st.sidebar.selectbox(
    "",  # Empty label for cleaner look
    ["Prediction", "Model Performance", "Data Analysis"],
    format_func=lambda x: f"📊 {x}" if x == "Model Performance" 
                         else f"🔮 {x}" if x == "Prediction"
                         else f"📈 {x}"
)

if page == "Prediction":
    st.markdown('<h1>Bank Term Deposit Prediction</h1>', unsafe_allow_html=True)
    
    # Introduction card
    st.markdown("""
    <div class="styledCard">
        <p style="font-size: 1.1em; color: #666;">
            Make predictions about whether a client will subscribe to a term deposit based on various features. 
            Fill in the information below to get a prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main form container
    st.markdown('<div class="styledCard">', unsafe_allow_html=True)
    
    # Client Information Section
    st.markdown('<div class="section-header">📋 Client Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
    with col2:
        job = st.selectbox("Occupation", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician", "unemployed"
        ])
    with col3:
        education = st.selectbox("Education Level", [
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree"
        ])
    
    col4, col5 = st.columns(2)
    with col4:
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    with col5:
        default = st.selectbox("Has Credit Default?", ["no", "yes"])
    
    # Financial Status Section
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💰 Financial Status</div>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    with col7:
        loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    
    # Campaign Information Section
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📞 Campaign Information</div>', unsafe_allow_html=True)
    col8, col9, col10 = st.columns(3)
    with col8:
        contact = st.selectbox("Contact Method", ["telephone", "cellular"])
        campaign = st.number_input("Number of Contacts", min_value=1, value=1)
    with col9:
        month = st.selectbox("Contact Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        previous = st.number_input("Previous Contacts", min_value=0, value=0)
    with col10:
        day_of_week = st.selectbox("Contact Day", ["mon", "tue", "wed", "thu", "fri"])
        poutcome = st.selectbox("Previous Outcome", ["nonexistent", "failure", "success"])
    
    duration = st.slider("Last Contact Duration (seconds)", 0, 1000, 261)
    pdays = st.slider("Days Since Last Contact", 0, 999, 999)
    
    # Economic Indicators Section
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Economic Indicators</div>', unsafe_allow_html=True)
    col11, col12, col13 = st.columns(3)
    with col11:
        emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, format="%.2f")
        cons_price_idx = st.number_input("Consumer Price Index", value=93.994, format="%.3f")
    with col12:
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, format="%.1f")
        euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, format="%.3f")
    with col13:
        nr_employed = st.number_input("Number of Employees", value=5191.0, format="%.1f")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Selection and Prediction
    st.markdown('<div class="model-selection-container">', unsafe_allow_html=True)
    model_name = st.selectbox("Select Model for Prediction", list(MODEL_PATHS.keys()))
    predict_button = st.button("Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_button:
        # Prepare input data
        input_data = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day_of_week, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous,
            "poutcome": poutcome, "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m, "nr.employed": nr_employed
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the data
        input_df['default'] = input_df['default'].map({'yes': 1, 'no': 0})
        input_df['housing'] = input_df['housing'].map({'yes': 1, 'no': 0})
        input_df['loan'] = input_df['loan'].map({'yes': 1, 'no': 0})
        
        # One-hot encode categorical variables
        categorical_columns = ['job', 'marital', 'education', 'contact', 
                             'month', 'day_of_week', 'poutcome']
        input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=False)
        
        # Define expected feature columns in exact order from training
        expected_columns = [
            'age', 'housing', 'loan', 'duration', 'campaign', 'pdays',
            'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
            'nr.employed', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
            'job_management', 'job_retired', 'job_self-employed', 'job_services',
            'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
            'marital_married', 'marital_single', 'marital_unknown',
            'education_basic.6y', 'education_basic.9y', 'education_high.school',
            'education_illiterate', 'education_professional.course',
            'education_university.degree', 'education_unknown', 'contact_telephone',
            'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
            'month_may', 'month_nov', 'month_oct', 'month_sep', 'day_of_week_mon',
            'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed',
            'poutcome_nonexistent', 'poutcome_success'
        ]
        
        # Ensure all expected features are present
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data exactly
        input_df = input_df[expected_columns]

        # Load and make prediction with the selected model
        model = load_model_from_s3(model_name)
        if model is None:
            st.error("Failed to load model. Please check the logs for details.")
            st.stop()

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display prediction with styling
        st.markdown("""
        <div class="prediction-card">
            <h2 style="margin-bottom: 20px;">Prediction Results</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", "Will Subscribe" if prediction == 1 else "Won't Subscribe")
        with col2:
            st.metric("Probability", f"{probability:.1%}")
        with col3:
            st.metric("Confidence", f"{abs(probability - 0.5) * 2:.1%}")
            
        st.markdown("</div></div>", unsafe_allow_html=True)

elif page == "Model Performance":
    st.markdown('<h1>Model Performance Analysis</h1>', unsafe_allow_html=True)
    st.write("Compare the performance metrics of different models.")
    
    if all(v is not None for v in [X_train, y_train, X_test, y_test]) and len(y_train) > 0:
        try:
            # Calculate metrics for each model using both training and test data
            train_metrics = {}
            test_metrics = {}
            
            for model_name, model in MODEL_PATHS.items():
                # Training metrics
                y_train_pred = load_model_from_s3(model_name).predict(X_train)
                y_train_prob = load_model_from_s3(model_name).predict_proba(X_train)[:, 1]
                
                train_metrics[f"{model_name} (Train)"] = {
                    'Accuracy': accuracy_score(y_train, y_train_pred),
                    'Precision': precision_score(y_train, y_train_pred),
                    'Recall': recall_score(y_train, y_train_pred),
                    'F1': f1_score(y_train, y_train_pred)
                }
                
                # Test metrics
                y_test_pred = load_model_from_s3(model_name).predict(X_test)
                y_test_prob = load_model_from_s3(model_name).predict_proba(X_test)[:, 1]
                
                test_metrics[f"{model_name} (Test)"] = {
                    'Accuracy': accuracy_score(y_test, y_test_pred),
                    'Precision': precision_score(y_test, y_test_pred),
                    'Recall': recall_score(y_test, y_test_pred),
                    'F1': f1_score(y_test, y_test_pred)
                }
            
            # Combine metrics
            all_metrics = {**train_metrics, **test_metrics}
            
            # Model comparison bar chart
            metrics_df = pd.DataFrame(all_metrics).round(3)
            fig_metrics = px.bar(metrics_df, 
                                title="Model Performance Comparison (Train vs Test)",
                                barmode='group',
                                height=400)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # ROC curves
            st.write("### ROC Curves")
            fig_roc = go.Figure()
            
            for model_name, model in MODEL_PATHS.items():
                # Training ROC
                y_train_prob = load_model_from_s3(model_name).predict_proba(X_train)[:, 1]
                fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
                auc_train = auc(fpr_train, tpr_train)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr_train, y=tpr_train,
                    name=f'{model_name} (Train, AUC = {auc_train:.3f})',
                    mode='lines'
                ))
                
                # Test ROC
                y_test_prob = load_model_from_s3(model_name).predict_proba(X_test)[:, 1]
                fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
                auc_test = auc(fpr_test, tpr_test)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr_test, y=tpr_test,
                    name=f'{model_name} (Test, AUC = {auc_test:.3f})',
                    mode='lines',
                    line=dict(dash='dash')
                ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc.update_layout(
                title="ROC Curves (Train vs Test)",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Detailed metrics selector
            st.write("### Detailed Metrics")
            selected_metrics = st.selectbox("Select metrics to view", list(all_metrics.keys()))
            
            metric_cols = st.columns(4)
            for i, (metric, value) in enumerate(all_metrics[selected_metrics].items()):
                metric_cols[i].metric(metric, f"{value:.3f}")
                
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            st.write("Debug information:")
            st.write(f"Training data - X: {X_train.shape}, y: {y_train.shape}")
            st.write(f"Test data - X: {X_test.shape}, y: {y_test.shape}")
            st.write(f"Available models: {list(MODEL_PATHS.keys())}")
    else:
        st.error("Could not load data. Please check if all required files exist in the 'data' directory.")

elif page == "Data Analysis":
    st.markdown('<h1>Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="styledCard">
        <p>Explore patterns and insights from the bank marketing dataset.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if X_train is not None:
        # Age distribution
        st.markdown('<div class="styledCard">', unsafe_allow_html=True)
        st.markdown('<h3>Age Distribution of Clients</h3>', unsafe_allow_html=True)
        fig_age = px.histogram(
            X_train, x='age',
            title="Age Distribution",
            labels={'age': 'Age', 'count': 'Number of Clients'},
            nbins=30,
            color_discrete_sequence=['#1652f0'],
            height=400  # Set fixed height
        )
        fig_age.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=0, r=0, b=0),
            showlegend=False,
            yaxis_title="Number of Clients",
            xaxis_title="Age"
        )
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature distributions
        st.markdown('<div class="styledCard">', unsafe_allow_html=True)
        st.markdown('<h3>Feature Distributions</h3>', unsafe_allow_html=True)
        numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        selected_feature = st.selectbox("Select feature to visualize", numeric_cols)
        
        fig_dist = px.histogram(
            X_train, x=selected_feature,
            title=f"Distribution of {selected_feature}",
            labels={selected_feature: selected_feature.replace('_', ' ').title(), 'count': 'Number of Clients'},
            nbins=30,
            color_discrete_sequence=['#1652f0'],
            height=400  # Set fixed height
        )
        fig_dist.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=0, r=0, b=0),
            showlegend=False,
            yaxis_title="Number of Clients",
            xaxis_title=selected_feature.replace('_', ' ').title()
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature correlations
        st.markdown('<div class="styledCard">', unsafe_allow_html=True)
        st.markdown('<h3>Feature Correlations</h3>', unsafe_allow_html=True)
        corr_matrix = X_train[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto',
            height=600  # Larger height for correlation matrix
        )
        fig_corr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, l=0, r=0, b=0),
            xaxis={'side': 'bottom'},  # Move x-axis labels to bottom
            yaxis={'side': 'left'}     # Move y-axis labels to left
        )
        # Rotate x-axis labels for better readability
        fig_corr.update_xaxes(tickangle=45)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Target distribution
        if y_train is not None:
            st.markdown('<div class="styledCard">', unsafe_allow_html=True)
            st.markdown('<h3>Target Distribution</h3>', unsafe_allow_html=True)
            target_counts = pd.Series(y_train).value_counts()
            
            fig_target = px.pie(
                values=target_counts.values,
                names=['No Subscription', 'Subscription'],
                title="Distribution of Target Variable",
                color_discrete_sequence=['#ff4b4b', '#1652f0'],
                height=400  # Set fixed height
            )
            fig_target.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(t=40, l=0, r=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
            # Add summary statistics with improved layout
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Clients", f"{len(y_train):,}")
                st.metric("Subscription Rate", f"{(target_counts[1] / len(y_train) * 100):.1f}%")
            with col2:
                st.metric("Subscribed Clients", f"{target_counts[1]:,}")
                st.metric("Non-subscribed Clients", f"{target_counts[0]:,}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Could not load data. Please check if the data files exist in the 'data' directory.")

# Add some CSS to ensure proper spacing between cards
st.markdown("""
<style>
    .styledCard {
        margin-bottom: 2rem !important;
        padding: 1.5rem !important;
    }
    
    /* Improve plot container spacing */
    .js-plotly-plot {
        margin-bottom: 1rem !important;
    }
    
    /* Improve metric spacing */
    [data-testid="metric-container"] {
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">Bank Marketing Prediction App - Built with Streamlit</p>', unsafe_allow_html=True) 