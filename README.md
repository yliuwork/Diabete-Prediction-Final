### Diabetes Risk Prediction

    **Shirley Liu**

#### Executive summary
    This project aims to predict the risk of diabetes using survey data from the Behavioral Risk Factor Surveillance System (BRFSS) collected by the CDC in 2015. The dataset, sourced from Kaggle, contains responses from 253,680 individuals and features 21 variables. Through data preprocessing, exploratory data analysis, and machine learning modeling, we develop predictive models to classify individuals into three categories: no diabetes, prediabetes, and diabetes. The results identify key risk factors that are most predictive of diabetes risk, providing actionable insights for healthcare providers and public health officials.

#### Rationale
    Diabetes is one of the most prevalent chronic diseases in the United States, affecting millions and imposing a significant financial burden on the economy. Early diagnosis and risk assessment are crucial for preventing complications and improving patient outcomes. By leveraging survey data, this research aims to provide an efficient method for predicting diabetes risk, thus enabling early interventions and personalized care.

#### Research Question
    Can survey questions from the Behavioral Risk Factor Surveillance System (BRFSS) provide accurate predictions of whether an individual has diabetes or is at high risk of developing diabetes?

#### Data Sources
    The primary data source for this project is the diabetes-related dataset from Kaggle, which includes survey data from the Behavioral Risk Factor Surveillance System (BRFSS) collected by the CDC in 2015. This dataset contains responses from 253,680 individuals and features 21 variables. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

#### Methodology
    1. Data Preprocessing: Data cleaning and handling missing values, feature selection, and engineering.
    2. Exploratory Data Analysis (EDA): Statistical analysis and visualization of survey responses, analysis of class distribution and imbalance
       2.1 Analyze the distribution of the data of each feature using countplot
        2.1.1 Health Behaviors:Most individuals in the dataset engage in positive health behaviors, including regular physical activity, fruit and    vegetable consumption, and healthcare check-ups.The majority do not engage in heavy alcohol consumption, and the smoking status is nearly evenly split between smokers and non-smokers.
        2.1.2 Chronic Conditions:Conditions such as high blood pressure and high cholesterol are prevalent in a significant portion of the population.
        2.1.3 Stroke and heart disease/attack are less common but present in a notable minority.
        Socioeconomic Factors:There is a diverse distribution of education and income levels, with significant representation across all levels.
        Both lower and higher income and education brackets are well-represented in the dataset.
        2.1.4 Accessibility:Most individuals have access to healthcare, but a small subset avoids seeing a doctor due to cost concerns.
            Regular cholesterol checks are common among the population.
        2.1.5 Gender Distribution:
            The dataset shows a balanced gender distribution, providing an equitable basis for analysis across genders.

        2.2 Analyze data range of each feature using boxplot
         2.2.1 BMI and Health-Related Features:BMI shows a significant number of outliers, indicating a subset of individuals with considerably higher BMI values.Both mental and physical health variables exhibit a wide range and numerous outliers, suggesting varied health conditions across the population.
         2.2.2 Age and Socioeconomic Factors:The age distribution is fairly symmetric, encompassing a wide range of ages.
    Education and income levels also show wide ranges, highlighting the diversity in socioeconomic status within the population.
         2.2.3Binary Health Indicators:Many health-related features, such as high blood pressure (HighBP), high cholesterol (HighChol), and smoking status, are binary.These indicators clearly show the presence or absence of specific health conditions or behaviors.

        2.3 Analyze the relationship among a few critical features using pair plots and correlation matrix
         2.3.1 Correlation Matrix:Positive correlations between age and chronic conditions such as diabetes, high blood pressure, and high cholesterol.High BMI is correlated with higher likelihoods of diabetes and high blood pressure.
    General health is strongly related to both mental and physical health.
         2.3.2 Pair Plot Analysis:Shows relationships between features like HighBP, HighChol, BMI, Age, and Sex, highlighting clustering of individuals with diabetes in relation to these factors.

        2.4 Distribution of Diabetes:
    Highlights the imbalanced nature of the dataset, with most individuals not having diabetes and smaller portions having pre-diabetes or diabetes.

    3. Modeling:
        Training and evaluating multiple classification models were conducted, including Logistic Regression, Decision Trees, Random Forests, and Support Vector Machines (SVM). The process involved grid search for hyperparameter tuning and cross-validation for performance evaluation. The results of the training were as follows:
        Logistic Regression:
            Successfully trained and evaluated.
            Best parameters: {'C': 1}.
        Decision Trees:
            Successfully trained and evaluated.
            Best parameters: {'max_depth': 10}.
        Random Forests:
            Successfully trained and evaluated.
            Best parameters: {'n_estimators': 300}.
        Deep Neural Network:
             The neural network model consists of an input layer, two hidden layers, and an output layer designed for a multi-class classification task. Each hidden layer uses ReLU activation, batch normalization, and dropout to enhance training efficiency and prevent overfitting. The model is compiled with the Adam optimizer and categorical crossentropy loss function. Early stopping and learning rate reduction callbacks are employed to optimize the training process. The model is trained for up to 100 epochs with a batch size of 32.
             
    4. Evaluation:
        Performance metrics were used to evaluate and compare the models, including accuracy, precision, recall, F1-score, and ROC-AUC. The evaluation of the successfully trained models is summarized in the result section. 


#### Results
    The analysis aimed to determine if survey responses from the BRFSS can accurately predict diabetes risk. Various machine learning models were evaluated to assess their predictive capabilities. The results are summarized below:

    Logistic Regression:
        Best Parameters: {'C': 1}
        Accuracy: 84.57%
        Precision: 79.81%
        Recall: 84.57%
        F1-Score: 80.73%
        ROC AUC Score: 78.28%
        
    Decision Tree:
        Best Parameters: {'max_depth': 10}
        Accuracy: 84.56%
        Precision: 80.06%
        Recall: 84.56%
        F1-Score: 80.99%
        ROC AUC Score: 76.19%
    
    Random Forest:
        Best Parameters: {'n_estimators': 300}
        Accuracy: 84.33%
        Precision: 79.69%
        Recall: 84.33%
        F1-Score: 80.79%
        ROC AUC Score: 75.04%
       
    Deep Nerual Network:
        Best Parameters: Not applicable for neural networks in this context
        Training Accuracy: 84.55%
        Validation Accuracy: 84.84%
        Training Loss: 0.4057
        Validation Loss: 0.3973
        Precision: 80.05%
        Recall: 84.77%
        F1-Score: 80.13%
        ROC AUC Score: 82.15%

     Conclusion:
        The evaluation of four different models—Logistic Regression, Decision Tree, Random Forest, and Deep Neural Network—provides insights into their respective performances on the classification task.

        Logistic Regression:Exhibits a balanced performance with an accuracy of 84.57% and an F1-score of 80.73%. The ROC AUC score of 78.28% indicates a reasonable ability to distinguish between classes.
        
        Decision Tree:Shows slightly lower accuracy (84.56%) compared to Logistic Regression but with a higher F1-score (80.99%). Has a lower ROC AUC score (76.19%), suggesting potential overfitting and less effective classification of the minority class.
        
        Random Forest: Achieves an accuracy of 84.33% and an F1-score of 80.79%. The ROC AUC score of 75.04% is the lowest among the evaluated models, indicating challenges in minority class predictions despite overall robust performance.
        
        Deep Neural Network:Displays the highest validation accuracy (84.84%) and ROC AUC score (82.15%), indicating strong generalization capabilities. Precision (80.05%) and recall (84.77%) are balanced, with an F1-score of 80.13%, showing effective performance across classes.
    
    Summary:
        Best Overall Model: The Deep Neural Network stands out with the highest validation accuracy (84.84%) and ROC AUC score (82.15%), indicating superior performance in distinguishing between classes and generalization on unseen data.
        

#### Next steps
    Addressing Class Imbalance and Feature Engineering are crucial steps in building robust machine learning models. By applying techniques like SMOTE, ADASYN, and class weighting, we can ensure that our model performs well across all classes, including minority classes. Additionally, through feature selection and creation, we can enhance the predictive power of our model by focusing on the most important features and capturing complex relationships within the data.


#### Outline of project

- [https://github.com/yliuwork/diabetes_risk_predict/blob/main/diabetes_risk_prediction.ipynb]

##### Contact and Further Information
    yliuwork@gmail.com

