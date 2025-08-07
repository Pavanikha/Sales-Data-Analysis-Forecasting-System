import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------
# Login Page
# -------------------------------
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials. Try again.")

# Check login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# -----------------------------
# Load your data
df = pd.read_csv(r"C:\Users\Pavanikha SN\Downloads\Dataset\cleaned_data.csv")  # Use raw string

# Reconstruct the order_date column
try:
    df['order_date'] = pd.to_datetime(df[['order_year', 'order_month', 'order_day']].astype(str).agg('-'.join, axis=1), errors='coerce')
except Exception as e:
    st.error(f"Error reconstructing date: {e}")

# Extract year and month if order_date exists
if 'order_date' in df.columns:
    df['year'] = pd.to_datetime(df['order_date'], errors='coerce').dt.year
    df['month'] = pd.to_datetime(df['order_date'], errors='coerce').dt.month

# Drop unnecessary columns
columns_to_drop = ["order_id", "customer_id", "product_id", "product_name", "customer_name"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# -----------------------------
# Page Setup
st.set_page_config(page_title="📊 Sales Dashboard", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("Choose Page", ["Revenue & Profit Overview", "Yearly Performance", "Monthly & Forecast"])

# ------------------------------------------
# Page 1: Revenue & Profit Overview
# ------------------------------------------
if page == "Revenue & Profit Overview":
    st.title("💰 Revenue & Profit Overview")

    total_revenue = df["sales"].sum()
    total_profit = df["sales"].sum()*0.6# Assuming profit = sales for now
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Profit", f"${total_profit:,.0f}")
    col3.metric("Profit Margin", f"{profit_margin:.2f}%")

    # Bar Chart by Category
    st.subheader("Revenue by Category (Bar Chart)")
    if "category" in df.columns:
        revenue_by_category = df.groupby("category")["sales"].sum().sort_values()
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=revenue_by_category.values, y=revenue_by_category.index, ax=ax1, palette="Greens_d")
        ax1.set_xlabel("Revenue")
        st.pyplot(fig1)

        # Pie Chart
        st.subheader("Revenue Distribution by Category (Pie Chart)")
        fig2, ax2 = plt.subplots()
        ax2.pie(revenue_by_category.values, labels=revenue_by_category.index, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        st.pyplot(fig2)
    else:
        st.warning("Column 'category' not found in dataset.")

# ------------------------------------------
# Page 2: Yearly Performance
# ------------------------------------------
elif page == "Yearly Performance":
    st.title("📅 Yearly Sales Performance")

    if 'order_date' not in df.columns or df['order_date'].isna().all():
        st.warning("❗ 'order_date' column is missing or invalid. Please check your date columns.")
    else:
        df = df.dropna(subset=['order_date'])  # Drop invalid dates
        yearly_sales = df.groupby("year")["sales"].sum().reset_index()

        if not yearly_sales.empty:
            # Year-over-year % change
            yearly_sales["change_%"] = yearly_sales["sales"].pct_change().fillna(0) * 100
            latest_year = yearly_sales["year"].max()
            latest_change = yearly_sales[yearly_sales["year"] == latest_year]["change_%"].values[0]

            st.markdown(f"📌 In **{latest_year}**, sales {'increased' if latest_change > 0 else 'decreased'} by **{abs(latest_change):.2f}%** compared to {latest_year - 1}.")

            # Bar Chart
            st.subheader("Year vs Revenue")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.barplot(data=yearly_sales, x="year", y="sales", palette="Blues_d", ax=ax3)
            ax3.set_ylabel("Revenue")
            ax3.set_xlabel("Year")

            # Add value labels
            for index, row in yearly_sales.iterrows():
                ax3.text(index, row["sales"], f"${row['sales']:,.0f}", ha='center', va='bottom', fontsize=8)

            st.pyplot(fig3)
        else:
            st.warning("Not enough valid data to show yearly performance.")

# ------------------------------------------
# Page 3: Monthly & Forecast
# ------------------------------------------
elif page == "Monthly & Forecast":
    st.title("📆 Monthly Sales & 📈 Forecast")

    if 'order_date' not in df.columns or df['order_date'].isna().all():
        st.warning("Date column missing or invalid.")
    else:
        df = df.dropna(subset=['order_date'])
        df['year_month'] = df['order_date'].dt.to_period('M').astype(str)

        # Monthly sales
        monthly_sales = df.groupby('year_month')['sales'].sum().reset_index()

        st.subheader("Monthly Sales Performance")
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=monthly_sales, x='year_month', y='sales', palette='mako', ax=ax4)
        ax4.set_ylabel("Sales")
        ax4.set_xlabel("Month")
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig4)

        # ------------------------
        # Future Forecast
        # ------------------------
        st.subheader("📈 Forecast: Next 12 Months (Linear Regression)")

        monthly_sales['date'] = pd.to_datetime(monthly_sales['year_month'])
        monthly_sales['timestamp'] = monthly_sales['date'].map(pd.Timestamp.toordinal)

        # Prepare data
        X = monthly_sales['timestamp'].values.reshape(-1, 1)
        y = monthly_sales['sales'].values

        # Model
        model = LinearRegression()
        model.fit(X, y)

        # Predict future
        future_months = pd.date_range(monthly_sales['date'].max() + pd.DateOffset(months=1), periods=12, freq='MS')
        future_ordinals = future_months.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_preds = model.predict(future_ordinals)

        forecast_df = pd.DataFrame({
            'date': future_months,
            'sales': future_preds
        })

        # Combine for display
        combined_df = pd.concat([
            monthly_sales[['date', 'sales']],
            forecast_df
        ]).reset_index(drop=True)

        combined_df['Label'] = ['Actual'] * len(monthly_sales) + ['Forecast'] * len(forecast_df)

        fig5, ax5 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=combined_df, x='date', y='sales', hue='Label', ax=ax5, palette=['#4c72b0', '#dd8452'])
        ax5.set_xlabel("Month")
        ax5.set_ylabel("Sales")
        ax5.set_xticklabels(combined_df['date'].dt.strftime('%Y-%m'), rotation=45, ha='right')
        st.pyplot(fig5)
