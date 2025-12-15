# American Express: Credit Score Classification 

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Jason Lei | @lei-jason | Data exploration, data preprocessing, data augmentation, model building |
| Wafa Berri | @wberri13 | Data exploration, data preprocessing, exploratory data analysis (EDA), feature selection|
| Allison Romero | @allisonr5002 | Model selection, decision tree, and neural network |
| Sheena Ansari | @Sheena-Ansari | Data exploration, feature engineering, Hyperparameter tuning |
| Jerry Lin | @NotJerwee | Data preprocessing, feature engineering, streamlit app
| Kareem Khusenov | @kareemx17 | Model finetuning, selection, neural network |
| Kashish Bhandari | @Kashish733 | Model selection, neural networks building and tuning |

---

## 🎯 **Project Highlights**

- Built a classification model to predict customer credit score categories using financial indicators 
- Implemented a complete Machine Learning pipeline, including data preprocessing, feature selection, model training, and evaluation to build a reliable predictive system.
- Achieved ~75% prediction accuracy, showing the model can correctly classify most credit score outcomes with the selected features.

---

## 👩🏽‍💻 **Setup and Installation**

Clone the repository
```
git clone `https://github.com/lei-json/amex-classification-1b.git`
cd amex-classification-1b
```
Create environment
```
conda create -n amex-classification
```
Activate environment
```
conda activate amex-classification
```

Install pip
```
conda install pip
```

Install dependencies
```
pip install -r requirement.txt
```

Run the streamlit app locally
```
streamlit run app.py
```

---

## 🏗️ **Project Overview**

- This project was made possible through the Break Through Tech AI Program, which provided us with a summer Machine Learning Foundations course where we learned core ML concepts, modeling techniques, and industry expectations. Based on our skills and interests from the coursework, we were matched with American Express to complete an AI Studio project. We’re grateful for the opportunity to apply what we learned in a real industry setting and expand our experience beyond the classroom.
- Our host company, American Express, challenged us to build a credit score classification model using a custom built dataset. The goal of the project was not only to create an accurate model, but to help us understand how the full Machine Learning pipeline is developed in industry, from data cleaning and feature engineering to model training, evaluation, and optimization. With guidance from our Challenge Advisor, Saurabh Gupta, we were able to explore real data considerations, choose meaningful features, handle skewed distributions, and evaluate our model responsibly. 
- Credit scoring plays a major role in determining who has access to financial services, credit card approval, loan offers, and interest rates. Improving how credit scores are analyzed and predicted can help companies better assess risk and extend fair, responsible credit to customers. By identifying key features that strongly influence credit categories—such as income, payment behavior, and occupation—our work supports the development of more data-driven decision tools. While our project is exploratory, it provides insights that can help refine future models, improve customer evaluation practices, and make financial assessments more transparent.


---

## 📊 **Data Exploration**

- The dataset contained financial variables (e.g., income, payment behavior, occupation) in tabular format. 
- Many numerical features were right-skewed, so we relied more on median based analysis to reduce the impact of outliers.
- Key EDA insight: income, occupation, and payment behavior showed strong relationships with credit score categories.
- Challenges: Handling skewed distributions, deciding which features to remove, and working with incomplete data.


---

## 🧠 **Model Development**

- Built the model using algorithms such as XGBoost, Catboost, LightGBM, Decision Tree, and Neural networks. 
- Selected features based on EDA patterns and contribution to prediction quality
- Avoided dropping too many features — only those with very poor data quality or little predictive relevance.
- Split data into training and testing sets (80/20) and evaluated performance using accuracy.

---

## 📈 **Results & Key Findings**

- Achieved ~75% accuracy in classifying customer credit score categories.
- Income, occupation, and payment behavior contributed most to model predictions.


---


## 🚀 **Next Steps**

- Giant notebook (~500+ cells) caused lag and slowed workflow.
- Hardware limits: Using only ~20% of the training set, each model run took ~5 minutes.
- Data quality uncertainty: Some entries may be incorrect or extreme outliers, requiring careful consideration.
- Visualization limits: Plotting 100k+ points affected clarity and analysis precision.
- Future work: With more time/resources, we would explore ways to increase model accuracy, try additional models or ensemble techniques, and enhance feature engineering to better capture patterns in the data.
---

## 🙏 **Acknowledgements**

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.

We would like to acknowledge our AI Studio Coach Jenna Hunte, Challenge Advisor Saurabh Gupta, Break Through Tech, and American Express for their support throughout this project. Their guidance, mentorship, and resources were instrumental in helping us complete the credit score classification project. 

