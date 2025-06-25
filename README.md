# Python-crime-detection
Step 1: Data Cleaning
Missing Values Handled
We identified key fields with high missing values such as weapon_code (nearly 130k values), cross_street, crime_code_2, crime_code_3, crime_code_4, etc. Also, demographic fields like victim_gender and victim_ethnicity has some missing data which has been handled.

Imputation
•	Categorical columns (cross_street, premise_description, victim_gender, victim_ethnicity) were filled with "Unknown".
•	Numerical columns (weapon_code, victim_age, crime_code, premise_code, etc.) were filled with 0. Though for age, it is irrelevant we excluded age value 0 while doing our analysis.

Duplicates
We checked and confirmed however, 0 duplicate rows found.

Standardization
We performed standardization techniques for different field values.
•	Victim Gender: Coded values were replaced:
M" → "Male", "F" → "Female", "X" → "Unknown", "H" → "Homosexual"
•	Victim Ethnicity: Mapped abbreviations to full terms to understand better (e.g., "W" → "White", "H" → "Hispanic"). "Rare" retained as-is.

Post-Cleaning Checks
We reviewed unique values to confirm cleaning and to understand all the values were replaced or not.
Step 2: Exploratory Data Analysis (EDA)
We exported the new cleaned dataset to a file named cleaned_crime_dataset.csv.

 1. Data Preparation
We checked lookup files crime_types.csv and weapon_types.csv were loaded to enrich the main dataset. These lookup tables were merged into the main dataset using crime_code and weapon_code.

3. Summary Statistics
•	Numeric Fields: Statistical summary showed distributions for columns like victim_age, weapon_code, latitude, and others.
•	Categorical Fields: Summarized values for fields such as area_name, victim_gender, victim_ethnicity, case_solved, etc.
o	Example: Most common victim gender was Male, and top ethnic group was Hispanic.

5. Data Understanding
In the data enrichment phase, a new binary column solved was created to represent case status, where "Solved" was mapped to 1 and "Not solved" to 0. This enabled a series of visual analyses comparing solve rates across various dimensions. A bar chart of the top 10 areas by solve rate showed that the Mission area had the highest percentage of solved cases, followed by West Valley and Foothill, highlighting which divisions are most effective. 
When analyzing crime types, offenses like Theft and Manslaughter showed the highest solve rates. A similar analysis by weapon type revealed that crimes involving weapons such as kitchen knives and glass were more frequently resolved. 
Gender-based analysis indicated that female victims were associated with higher solve rates compared to other gender categories.
Finally, examining victim ethnicity showed that Hispanic victims had the highest case resolution rate, pointing to potential disparities in justice outcomes. These visual insights help uncover patterns that influence law enforcement success.

Step 3: Data Visualization Summary
This phase focused on uncovering patterns, distributions, and relationships within the crime dataset through visual exploration. A bar chart was used to show the top 15 areas with the highest number of reported crimes, identifying Central as the most affected region, which can guide resource allocation and policy planning. 
By converting date_reported to day_of_week, it was observed that Fridays and weekdays saw the most crime activity, suggesting a temporal pattern. Further analysis by month revealed a seasonal trend, with July to December experiencing the highest crime rates, possibly linked to holidays or weather. A boxplot comparing victim age by crime type showed that serious crimes (Part 1) tend to involve slightly younger victims than less severe offenses. 
Gender distribution analysis indicated that males were most frequently victimized, followed by females. A histogram of victim ages (excluding missing or zero values) showed that ages 20–35 are the most common, highlighting young adults as the most affected group. A bar chart of the top 10 most frequent crime types, such as theft and assault, provides insights for crime prevention strategies. Lastly, a correlation heatmap revealed weak but informative relationships between numeric variables like victim_age, weapon_code, latitude, and longitude, suggesting some patterns especially geographic and weapon-related that could be further explored.

Step 4: Feature Engineering Summary
This step focused on transforming and preparing the dataset for machine learning by creating new features, encoding categorical data, and scaling numerical variables.
1. Temporal Feature Extraction
We extracted time-based features from the date_occurred column:
o	year_occurred
o	month_occurred
o	day_occurred
o	hour_occurred
These features help identify patterns based on time (e.g., seasonality, time-of-day effects).
 2. Categorical Encoding
We applied Label Encoding to convert the following text fields into numeric format:
o	victim_gender
o	victim_ethnicity
o	case_solved
The encoding enables the use of categorical data in ML models.
3. Normalization of Numerical Features
We used Min-Max Scaling to scale selected numerical features to a [0, 1] range:
o	victim_age
o	premise_code
o	weapon_code
o	latitude
o	longitude
This ensures that all numeric features are on a comparable scale for model training.
4. Processed Feature Preview
A cleaned and transformed dataset was prepared, including:
o	Demographics: victim_age, victim_gender, victim_ethnicity
o	Crime Details: case_solved, weapon_code, premise_code
o	Time Components: year_occurred, month_occurred, hour_occurred
o	Geolocation: latitude, longitude

Step 5: Model Building Summary
This phase focused on training and evaluating machine learning models to predict whether a crime case would be solved based on features extracted from the dataset.
1. Feature Selection & Splitting
•	Independent Variables (X) included:
o	victim_age, victim_gender, victim_ethnicity
o	year_occurred, month_occurred, hour_occurred
o	premise_code, weapon_code, latitude, longitude
•	Target Variable (y): case_solved (encoded as binary)
•	Dataset was split into training (80%) and testing (20%) sets using train_test_split.
2. Models Trained
Three classification models were built to predict the likelihood of case resolution:
Model	Description
Logistic Regression	A linear model for binary classification
Random Forest	An ensemble of decision trees for higher accuracy
Decision Tree	A simple, interpretable tree-based model
3. Evaluation Metrics
Each model was evaluated using:
•	Accuracy Score
•	Confusion Matrix
•	Classification Report (Precision, Recall, F1-score)
4. Model Comparison Results
Model	Accuracy
Logistic Regression-->79.3%
Random Forest-->82.4%
Decision Tree-->75.3%
Best Performing Model: From Random Forest we achieved the highest accuracy and is suitable for handling both linear and nonlinear patterns in the data.

Step 6: Model Evaluation
This step assessed the performance of three trained models using various classification metrics and visual diagnostics to determine their effectiveness in predicting whether a crime case is solved.
1. Metrics Used
For each model, the following evaluation metrics were calculated:
Metric	Description
Accuracy	Proportion of total predictions that were correct
Precision	Ratio of true positives to total predicted positives
Recall	Ratio of true positives to actual positives
F1-Score	Harmonic mean of Precision and Recall
AUC-ROC	Area Under the ROC Curve – a measure of model's ability to distinguish classes
2. Results Summary
Model	Accuracy	Precision	Recall	F1-Score	AUC-ROC
Logistic Regression	~79.3%	     -	    -	    - 	Moderate
Random Forest	~82.4%	High 	High	High	Best
Decision Tree	~75.3%	OK	OK	OK	fair
Observation:
Random Forest consistently outperformed the other models across all metrics, indicating strong predictive power and generalization.
3. Confusion Matrices
Visualized confusion matrices for each model show the distribution of:
•	True Positives (TP)
•	True Negatives (TN)
•	False Positives (FP)
•	False Negatives (FN)
This helps identify where the models are making errors, particularly in unbalanced datasets.
4. ROC Curve Analysis
ROC curves were plotted for all three models. The closer the curve is to the top-left corner, the better the model performance. Random Forest had the highest AUC (Area Under Curve), followed by Logistic Regression.
