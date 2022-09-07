# Credit_Risk_Analysis
<!-- Using Python and Scikit-learn / imbalanced-learn library to build and evaluate several machine learning models to predict credit risk -->
## Project Background
LendingClub, a peer-to-peer (P2P) lending service, wants to use machine learning to predict credit risk, with the goals of:
- Providing faster & more reliable loan experience (i.e. attracting customers)
- More accurate identification of good loan candidates (i.e. decreasing defaults)
<!-- Peer-to-peer (P2P) lending is a form of financial technology that allows people to lend or borrow money from one another without going through a bank. -->
Since the number of good loans significantly exceeds the number of risky loans, imbalanced classification machine learning models will be considered.  The models LendingClub would like evaluated with their provided credit card credit dataset are outlined in **Table 1**.

***Table 1: Classification Models for Evaluation***
| Case | Algorithm | Sampling Type |
| :---: | :---: | :---: |
| 1 | RandomOverSampler | Oversampling Minority |
| 2 | SMOTE | Oversampling Minority |
| 3 | ClusterCentroids | Undersampling Majority |
| 4 | SMOTEENN | Combinatorial / Hybrid | 
| 5 | BalancedRandomForestClassifier | Ensemble |
| 6 | EasyEnsembleClassifier | Ensemble |

## Purpose
<!-- The purpose of this analysis is well defined (4 pt) -->
This project is to train and evaluate the perfomance of six machine learning models for predicting credit risk.

## Resources
### Data Sources & Bespoke Code
1. [LoanStats_2019Q1.csv](Data/LoanStats_2019Q1.csv) [^1]
2. [credit_risk_resampling.ipynb](credit_risk_resampling.ipynb) [^2]
3. [credit_risk_ensemble.ipynb](credit_risk_ensemble.ipynb) [^2]

[^1]: 2019 Q1 Loan Data, provided by LendingClub  
[^2]: Jupyter Notebook

### Software & CDNs
***Table 2: Software & Library Versions***
| Software | Version |
| :--- | :---: |
| Python | 3.7.13 |
| NumPy | 1.21.5 |
| SciPy | 1.7.3 |
| Scikit-learn | 1.0.2 |
| imbalanced-learn | 0.7.0 |
| Visual Studio Code | 1.70.2 |

# Results 
<!-- There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt) -->
## Methodology
An ETL of review data for Amazon Video Games was performed using the Google CoLab code [Amazon_Reviews_ETL.ipynb](Amazon_Reviews_ETL.ipynb), uploading the cleaned data to an AWS PostgreSQL database.  The table *vine_table* was exported as a *.csv* file, and preliminary statistics were analyzed using the Jupyter Notebook code [Vine_Review_Analysis.ipynb](Vine_Review_Analysis.ipynb).

For the Vine Review analysis, the most useful Amazon Video Games reviews were filtered by:
- At least 20 total votes (65379 of 1785997 total reviews)
- At least 50% of votes were classified as helpful (40565 of 65379 reviews)

<details><summary>View Screenshot of vine_table ETL</summary>
  <p>
  <img src="images/example_ETL_df.png">
  </p>
</details>

<details><summary>View Screenshot of vine_table filtering</summary>
  <p>
  <img src="images/example_filtered_df.png">
  </p>
</details>

## Paid vs. Unpaid Reviews
Analysis of the filtered Amazon Video Games reviews showed:

***Table 2: Amazon Video Games Vine Review Statistics***

![paid_vs_unpaid.png](images/paid_vs_unpaid.png)


From ***Table 2***, it can be concluded:
- There are significantly more non-Vine reviews than Vine reviews, with Vine reviews only making up about 0.23% of the most useful Amazon Video Games reviews.
- 51.06% of the 94 Vine reviews were 5 stars, in comparison to 38.70% of the 40471 non-Vine reviews.

# Summary 
<!-- There is a summary of the results (2 pt) -->
<!-- There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt) -->
