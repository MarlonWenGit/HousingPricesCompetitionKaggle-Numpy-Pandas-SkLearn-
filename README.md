# Ames Housing Price Prediction  

**Final score:** 0.16733 (using mean-square-root regression)  

---

## Description  

**Start here if…**  
You have some experience with R or Python and basic machine learning. This competition is perfect for data science students who have completed an online course in machine learning and want to expand their skills before tackling featured competitions.  

---

## 💡 Getting Started Notebook  

To get started quickly, you can use the [starter notebook](#) (link to your notebook).  

---

## Competition Description  

Ask a home buyer to describe their dream house, and they probably won't mention the height of the basement ceiling or proximity to an east-west railroad. But this competition’s dataset shows that much more influences price negotiations than just the number of bedrooms or a white-picket fence.  

With **79 explanatory variables** describing nearly every aspect of residential homes in Ames, Iowa, your task is to **predict the final price of each home**.  

---

## Practice Skills  

- Creative feature engineering  
- Advanced regression techniques such as Random Forest and Gradient Boosting  

---

## Acknowledgments  

The Ames Housing dataset was compiled by Dean De Cock for data science education. It’s a modernized and expanded alternative to the classic Boston Housing dataset.  

_Photo by Tom Thain on Unsplash._  

---

## Evaluation  

**Goal:**  
Predict the sales price for each house. For each Id in the test set, predict the value of the `SalePrice` variable.  

**Metric:**  
Submissions are evaluated using **Root Mean Squared Log Error (RMSLE)** between the logarithm of the predicted value and the logarithm of the actual sale price. Taking logs ensures that errors in predicting expensive and cheap houses affect the result equally.  
