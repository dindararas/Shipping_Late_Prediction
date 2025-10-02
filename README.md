# LATE DELIVERY PREDICTION

## 📦 PROBLEM STATEMENT
DataCo global company has been struggling with late deliveries. Out of 180K transactions over the period of 2015 - 2017, **55% orders were shipped late**. This issue led to customer dissatisfaction and loss revenue. This project will be focused on late delivery that arise in the company, including risk assessment and predicting delay probability. You can find streamlit version of this project [here](https://shipping-late.streamlit.app/)

## 🎯 OBJECTIVES
This project aims to :
1. Identify key risk factors influencing late delivery risk
2. Develop ML-based models to predict delay risk
3. Derive Actionable Insights

## 📊 DATASET OVERVIEW
Dataset for this project was downloaded from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis/data?select=DescriptionDataCoSupplyChain.csv). This supply chain dataset was used by DataCo Global company for their analysis which include detailed information about customer, shipping, and purchased products. This dataset was collected from January 2015 to September 2017

Features overview :
| Column Name | Description |
|------|------|
| `Type` | Type of transaction made  |
| `Days for shipping (real)` |  Actual shipping days of the purchased product  |
| `Days for shipment (scheduled)` |  Days of scheduled delivery of the purchased product |
| `Benefit per order` | Earnings per order placed  |
| `Sales per customer` | Total sales per customer made per customer  |
| `Delivery Status` |  Delivery status of orders: Advance shipping , Late delivery , Shipping canceled , Shipping on time  |
| `Late_delivery_risk` |  Categorical variable that indicates if sending is late (1), it is not late (0)  |
| `Category Id` | Product category code |
| `Category Name` |  Description of the product category  |
| `Customer City` | City where the customer made the purchase   |
| `Customer Country` | Country where the customer made the purchase  |
| `Customer Email` | Customer's email  |
| `Customer Fname` | Customer  first name|
| `Customer Id` | Customer ID |
| `Customer Lname` | Customer last name  |
| `Customer Password` |Masked customer key  |
| `Customer Segment` |  Types of Customers: Consumer , Corporate , Home Office  |
| `Customer State` | State to which the store where the purchase is registered|
| `Customer Street` | Street to which the store where the purchase is registered|
| `Customer Zipcode` |  Customer zipcode|
| `Department Id` |  Department code of store|
|`Department Name` |  Department name of store|
| `Latitude` |  Latitude corresponding to location of store |
| `Longitude` |    Longitude corresponding to location of store |
| `Market` |   Market to where the order is delivered : Africa , Europe , LATAM , Pacific Asia , USCA|
| `Order City` | Destination city of the order|
| `Order Country` |  Destination country of the order|
| `order date (DateOrders)` |  Date on which the order is made|
|`Order Id` | Order code|
| `Order Item Cardprod Id` |  Product code generated through the RFID reader |
| `Order Item Discount` |    Order item discount value |
| `Order Item Discount Rate` |  Order item discount percentage|
| `Order Item Id` | Order item code|
| `Order Item Product Price` | Price of products without discount |
| `Order Profit Ratio` | Order Item Profit Ratio  |
| `Order Item Quantity` |Number of products per order  |
| `Sales` |  Value in sales  |
| `Order Item Total` | Total amount per order|
| `Order Profit per Order` | Order Profit per Order|
| `Order Region` |  Region of the world where the order is delivered :  Southeast Asia ,South Asia ,Oceania ,Eastern Asia, West Asia , West of USA , US Center , West Africa, Central Africa ,North Africa ,Western Europe ,Northern , Caribbean , South America ,East Africa ,Southern Europe , East of USA ,Canada ,Southern Africa , Central Asia ,  Europe , Central America, Eastern Europe , South of  USA |
| `Order State` | State of the region where the order is delivered|
|`Order Status` |   Order Status : COMPLETE , PENDING , CLOSED , PENDING_PAYMENT ,CANCELED , PROCESSING ,SUSPECTED_FRAUD ,ON_HOLD ,PAYMENT_REVIEW|
| `Product Card Id` |  Product code |
| `Product Category Id` |    Product category code|
| `Product Description` |  Product Description|
| `Product Image` | Link of visit and purchase of the product|
| `Product Name` |  Product Name|
| `Product Price` |  Product Price
|`Product Status` |  Status of the product stock :If it is 1 not available , 0 the product is available |
| `Shipping date (DateOrders)` |  Exact date and time of shipment |
| `Shipping Mode` | The following shipping modes are presented : Standard Class , First Class , Second Class , Same Day |

## 🔎 KEY FINDINGS
### Shipping Mode
<img width="533" height="171" alt="image" src="https://github.com/user-attachments/assets/3db24e40-b5f9-4ce1-9d9c-f59250240ff8" />

**Insights** :
1. Surprisingly, First Class shipping had the lowest on-time rate (4.68%), while Standard Class achieved the highest on-time rate at 61.93% 
2. Second Class shipping had a low on-time rate of 23.37% with deliveries up to 4 days later than scheduled
3. With on-time rate of 54.26%, Same Day delivery were shipped either on schedule or delayed by one day

### Seasonality Analysis
<img width="297" height="193" alt="image" src="https://github.com/user-attachments/assets/024b02df-4691-4866-92b1-f8a1c97071e3" />

**Insights** : 
1. Overall, smaller quantity volume led to higher on-time rate, indicating that delayed delivery is influenced by shipment volume
2. On-time rate was quite stable at 44.5% - 46% from January to December with the lowest rate occurring in March and September (44.5%) and the highest rate occurring in April (45.9%)
3. December had both the lowest total quantity sold and relatively low on-time rate, indicating that shipped volume did not influence low on-time rate this month
Despite a high volume of shipped products, January still achieved the second-highest on-time rate

### Store Analysis
<img width="283" height="172" alt="image" src="https://github.com/user-attachments/assets/b84f4a66-4a3b-4f09-940e-d92a95ccd398" />

**Insights** :
On average, stores/warehouses in the e-commerce had an actual lead time of 3.5 days and an expected lead time of 2.9 days, making the shipping day gap at 0.6 days
More than half of total stores/warehouses (54.3%) shipped their products later than expected by more than 0.6 days, with the worst delay reaching 4 days

## 🤖 PREDICTION MODEL
### Model Comparison
<img width="722" height="221" alt="image" src="https://github.com/user-attachments/assets/52fcb641-aeda-4f41-8529-98edada5a504" />

**Insights** :
Random Forest and Decision Tree are overfitting to the training data, shown by all the metric scores of 1
Although Logistic Regression and XGBoost are good for data generalization, these models are possibly underfitting
There is no difference in model performance between original dan outlier handling

### Hyperparameter Tuning
<img width="252" height="197" alt="image" src="https://github.com/user-attachments/assets/8eaf21f7-353b-47dd-a368-6c950bf69364" />
<img width="212" height="146" alt="image" src="https://github.com/user-attachments/assets/38f6c099-ca95-4178-98c7-2fb361651b88" />

**Insights**:
1. Hyperparameter tuning has improved XGBoost performance, despite making it overfitting (all the metric scores on training data =1)
2. **False Positive Rate (0.08)** : About 8% of on-time delivery were incorrectly predicted as late delivery risk
3. **False Negative Rate (0.08)** : About 8% of late delivery were incorrectly predicted as on-time

### Feature Importance
Features importance from XGBoost model's prediction :
* `Days for shipment (scheduled)` : this feature has the highest importance (+1.76) for predicting late delivery risk. The result highlights that estimating the required shipping days is essential to reduce the probability of delayed delivery by allocating resources and time efficiently
* `Customer Street_encoded` : although customer street is not directly related to order destination, this feature ranks second for the highest importance (+0.92)
* `Shipping Mode_encoded` : this feature ranks third for the highest importance (+0.79). As shown in the EDA section, the choice of shipping mode influence the probability of delayed delivery
* `Order City_en_encoded` and `Type_encoded` : these features have relatively high importance (+0.62 and +0.66, respectively) which indicate that payment type and destination city influence late delivery risk. This makes sense since order destination is related to distance. Meanwhile, the duration of payment confirmation can also influence late delivery
* `shipping_month` and `shipping_hours` : some months/hours with high shipping volumes led to higher late delivery risk

## ✍ BUSINESS RECOMMENDATIONS
**1. Adjust Shipping Schedules**
* Develop model to estimate actual shipping days more accurately
* Extend shipping days to lower late delivery risk

**2. Optimize Warehouse Locations**
* Establish warehouses near regions with the highest number of orders
* **Cons** : need high cost to build new warehouses

**3. Route Optimization**
* Develop routing algorithm to make shipping efficient
* Cluster nearby regions, so deliveries can be completed faster and more efficiently

**4. Optimize Shipping Mode Performance**
* Evaluate First Class shipping mode and improve its performance
* Remove shipping modes with low on-time rate and high costs

**5. Plan Shipping During Peak Seasons/Hours**
* Allocate extra resources during peak months
* Prioritize early day shipping for high-risk orders

**6. Optimize Payment Process**
* Speed up payment confirmation to reduce delays

