import pandas as pd
import plotly.express as px
import pycountry
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import streamlit as st


# Load Data Function (Cached for performance improvement)
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Year'] = data['Year'].astype(str).str.replace(',', '').astype(int)  # Clean the Year column
    return data


# Get Country Code Function
def get_country_code(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None


# Get Continent Function based on Country
def get_continent(country_name):
    continent_mapping = {
        "Asia": ["China", "India", "Japan"],
        "Europe": ["Germany", "France", "Italy"],
        "Africa": ["Nigeria", "South Africa", "Egypt"],
        "Oceania": ["Australia", "New Zealand"],
        "Americas": ["USA", "Canada", "Brazil"]
    }
    for continent, countries in continent_mapping.items():
        if country_name in countries:
            return continent
    return None


# Sidebar Filters
def sidebar_filters(data):
    st.sidebar.header("🔍 Filter Options")

    unique_status = data['Status'].unique()
    selected_status = st.sidebar.selectbox("Select Status", unique_status, index=0)
    filtered_data = data[data['Status'] == selected_status]

    unique_years = filtered_data['Year'].unique()
    year_range = st.sidebar.slider("Select Year Range", min(unique_years), max(unique_years),
                                   (min(unique_years), max(unique_years)))
    filtered_data = filtered_data[(filtered_data['Year'] >= year_range[0]) & (filtered_data['Year'] <= year_range[1])]

    unique_countries = filtered_data['Country'].unique()
    selected_country = st.sidebar.selectbox("Select Country", unique_countries)

    selected_columns = st.sidebar.multiselect("Select Columns for Analysis",
                                              ["Life expectancy", "GDP", "Schooling", "Alcohol", "Adult Mortality"],
                                              default=["Life expectancy", "GDP"])

    return selected_status, year_range, selected_country, selected_columns, filtered_data


# Build and Train the ML Model
def build_ml_model(data, selected_columns):
    features = selected_columns
    target = "Life expectancy"

    # Drop rows with missing values
    data_cleaned = data.dropna(subset=[target] + features)

    X = data_cleaned[features]
    y = data_cleaned[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regressor Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate the Model
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae, X_test, y_test, y_pred


# Main Function for Streamlit App
def main():
    st.set_page_config(page_title="Life Expectancy Dashboard", layout="wide", page_icon="🌍")
    st.sidebar.title("Life Expectancy Dashboard")
    st.sidebar.markdown("This dashboard visualizes global life expectancy data over time.")
    st.title("🌍 Global Life Expectancy Dashboard")
    st.markdown("Visualizing life expectancy trends across the globe over time.")

    # Load data
    file_path = r"C:\Users\uswah\Desktop\Life Expectancy Data.csv"
    df = load_data(file_path)

    # Add country code and continent to DataFrame
    df['Country Code'] = df['Country'].apply(get_country_code)
    df['Continent'] = df['Country'].apply(get_continent)

    # Sidebar filters
    selected_status, year_range, selected_country, selected_columns, filtered_data = sidebar_filters(df)

    # Global Life Expectancy Over Time
    st.subheader("Global Life Expectancy Over Time")
    color_scale = px.colors.sequential.Blues
    fig = px.choropleth(
        data_frame=df,
        locations="Country Code",
        color="Life expectancy",
        hover_name="Country",
        animation_frame="Year",
        color_continuous_scale=color_scale,
        range_color=(50, 90)
    )
    fig.update_layout(
        title_text="Global Life Expectancy Over Time",
        title_x=0.5,  # Center the title
        template="plotly",
        coloraxis_colorbar=dict(
            title="Life Expectancy",
            tickvals=[50, 60, 70, 80, 90],
            ticktext=["50", "60", "70", "80", "90"]
        ),
        font=dict(family="Arial", size=20, color="white"),
        geo=dict(
            lakecolor='rgb(255, 255, 255)',  # Make lakes white to enhance contrast
            projection_type='natural earth',  # Use a better map projection
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove margins around the map
        width=1000,  # Set the width to make the graph wider
        height=600  # Adjust the height accordingly
    )
    st.plotly_chart(fig, use_container_width=True)  # This will make the map stretch across the page



    # Radar Chart: Average Life Expectancy by Continent
    st.subheader("Radar Chart: Compare Life Expectancy Across Continents")
    radar_data = (
        filtered_data.groupby("Continent")["Life expectancy"]
        .mean()
        .reset_index()
    )
    radar_fig = px.line_polar(
        radar_data,
        r="Life expectancy",
        theta="Continent",
        line_close=True,
        title=f"Average Life Expectancy by Continent for {year_range[0]}-{year_range[1]}",
        color_discrete_sequence=["#636EFA"]
    )
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[50, 90])
        ),
        template="plotly_dark",
        font=dict(family="Arial", size=14, color="white")
    )
    st.plotly_chart(radar_fig, use_container_width=True)

    # Correlation Heatmap with Adjusted Values
    st.subheader("📊 Correlation Heatmap")
    numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_cols.corr()

    # Create the heatmap with a custom color palette
    fig, ax = plt.subplots(figsize=(16, 12))  # Set the figure size
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", annot_kws={"size": 14, "color": "white"},
                cbar_kws={"shrink": 0.8}, linewidths=0.5)

    # Adjust the appearance of the heatmap
    ax.set_facecolor("black")  # Set background color to black for contrast
    plt.xticks(color="white", fontsize=12)  # Adjust font size and color for x-ticks
    plt.yticks(color="white", fontsize=12)  # Adjust font size and color for y-ticks
    plt.title("Correlation Heatmap of Numeric Features", fontsize=16, color="white")

    # Remove extra space around the plot
    plt.tight_layout()  # This adjusts the plot to fit tightly within the figure

    # Display the heatmap
    st.pyplot(fig)

    # Disease Impact on Life Expectancy
    st.subheader("🦠 Disease Impact on Life Expectancy")
    disease_col = st.selectbox("Select Disease Metric", ['HIV/AIDS', 'Measles', 'Polio', 'Diphtheria'])
    disease_fig = px.scatter(
        filtered_data, x=disease_col, y="Life expectancy", color="Status", size="GDP",
        hover_name="Country", title=f"Life Expectancy vs {disease_col}",
        template="plotly_dark", size_max=30
    )
    st.plotly_chart(disease_fig, use_container_width=True)

    # Schooling vs Life Expectancy
    st.subheader("📚 Schooling vs Life Expectancy")
    schooling_fig = px.scatter(
        filtered_data, x="Schooling", y="Life expectancy", color="Status", trendline="ols",
        title="Schooling vs Life Expectancy by Status",
        hover_name="Country", template="seaborn"
    )
    st.plotly_chart(schooling_fig, use_container_width=True)

    # ML Model Prediction
    st.subheader("🔮 Life Expectancy Prediction")
    model, mae, X_test, y_test, y_pred = build_ml_model(filtered_data, selected_columns)

    # Prediction for Selected Country
    selected_data = filtered_data[filtered_data['Country'] == selected_country]
    X_new = selected_data[selected_columns]
    life_expectancy_pred = model.predict(X_new)
    # Display the predicted life expectancy with larger font size
    st.markdown(
        f"<h3 style='font-size: 24px; color: white;'>Predicted Life Expectancy for {selected_country}: {life_expectancy_pred[0]:.2f} years</h3>",
        unsafe_allow_html=True)

    # True vs Predicted Life Expectancy with smaller canvas
    fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size

    ax.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predictions")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("True Life Expectancy", fontsize=8)  # Reduced font size
    ax.set_ylabel("Predicted Life Expectancy", fontsize=8)  # Reduced font size
    ax.set_title("True vs Predicted Life Expectancy", fontsize=10)  # Reduced title size
    ax.legend(loc="upper left", fontsize=6)  # Smaller legend
    ax.grid(visible=True, alpha=0.5)

    # Adjust layout for tighter fitting
    plt.tight_layout(pad=2.0)

    # Display plot
    st.pyplot(fig)

    # Key Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Countries", filtered_data['Country'].nunique())
    with col2:
        st.metric("Average Life Expectancy", f"{filtered_data['Life expectancy'].mean():.2f}")
    with col3:
        st.metric("Max Life Expectancy", filtered_data['Life expectancy'].max())

    # View Dataset
    with st.expander("🔍 View Dataset"):
        st.subheader("Filtered Data")
        st.write(filtered_data.style.format({'Year': '{:.0f}'}))


if __name__ == "__main__":
    main()
