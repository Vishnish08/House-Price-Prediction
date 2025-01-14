# House-Price-Prediction
# California Housing Price Prediction Project  

This repository presents a comprehensive machine learning project focused on predicting **median house values** in California districts. The project leverages the **California Housing dataset**, a well-known dataset derived from the 1990 U.S. Census, containing demographic and geographic features. The analysis aims to explore data patterns, build predictive models, and derive insights into the factors influencing housing prices.


## Project Objectives  
1. **Data Understanding and Exploration**:  
   - Gain insights into the dataset's structure and attributes.  
   - Visualize relationships between key features and the target variable (`MedHouseVal`).  

2. **Data Preprocessing and Cleaning**:  
   - Ensure the dataset is free of missing or anomalous values.  
   - Prepare features for machine learning by scaling, transforming, and engineering where necessary.  

3. **Model Development**:  
   - Develop predictive models to estimate house prices using various regression techniques.  
   - Compare performance across models to select the best approach.  

4. **Model Evaluation and Insights**:  
   - Evaluate model performance using robust metrics.  
   - Analyze key factors influencing house prices to provide actionable insights.  


## Methodology  

### **1. Data Acquisition and Exploration**  
- **Dataset**: Loaded the California Housing dataset using Scikit-learn's `fetch_california_housing` function.  
- **Exploration**:  
  - Conducted descriptive statistical analysis to summarize feature distributions.  
  - Visualized correlations between features like median income, house age, and geographic location with house prices using Seaborn and Matplotlib.  

### **2. Data Preprocessing**  
- Checked for missing values (none were found in this dataset).  
- Addressed feature scaling for algorithms sensitive to feature magnitudes.  
- Split the dataset into training and testing sets to evaluate generalization performance.  

### **3. Model Development**  
- Implemented the following machine learning models:  
  - **Linear Regression**: A simple and interpretable baseline model.  
  - **Random Forest Regressor**: A robust ensemble method capable of capturing complex patterns in the data.  
- Tuned hyperparameters for the Random Forest model to optimize performance.  

### **4. Model Evaluation**  
- Assessed model performance using the following metrics:  
  - **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.  
  - **Mean Squared Error (MSE)**: Penalizes larger errors more heavily than MAE.  
  - **R-squared (R²)**: Indicates the proportion of variance explained by the model.  

### **5. Insights and Results**  
- Identified key predictors of house prices, with **median income** and **location (latitude and longitude)** emerging as the most significant.  
- Observed that the Random Forest Regressor outperformed Linear Regression, achieving lower error metrics and higher predictive accuracy.  



## Tools and Technologies  

- **Programming Language**: Python  
- **Libraries Used**:  
  - **Data Manipulation and Analysis**: Pandas, NumPy  
  - **Visualization**: Matplotlib, Seaborn  
  - **Machine Learning**: Scikit-learn  


## Dataset Overview  

The **California Housing dataset** consists of 20,640 samples with the following attributes:  
- **MedInc**: Median income in block group (normalized to population).  
- **HouseAge**: Median house age in block group.  
- **AveRooms**: Average number of rooms per household.  
- **AveBedrms**: Average number of bedrooms per household.  
- **Population**: Block group population.  
- **AveOccup**: Average household occupancy.  
- **Latitude** and **Longitude**: Geographic coordinates of the block group.  

The target variable is **`MedHouseVal`**, representing the median house value in units of $100,000.  



## Results and Insights  

1. **Model Performance**:  
   - **Random Forest Regressor**:  
     - Outperformed Linear Regression in terms of accuracy and error metrics.  
     - Demonstrated robustness in capturing non-linear relationships.  
   - **Linear Regression**:  
     - Provided a baseline for comparison with a simple, interpretable model.  

2. **Key Insights**:  
   - **Median Income** was the strongest predictor of house prices.  
   - Geographic location (latitude and longitude) significantly influenced house prices, with coastal regions generally having higher values.  

3. **Visualizations**:  
   - Correlation heatmaps to identify relationships between features.  
   - Scatter plots of house prices versus significant predictors.  



## Future Work  

- Experiment with advanced machine learning models, such as Gradient Boosting Machines or Neural Networks, for further performance improvement.  
- Perform feature engineering to create new variables that better capture complex interactions.  
- Extend the analysis to include external datasets, such as economic indicators or local amenities.  
