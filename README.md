# Game_Application_Success_Prediction

The objective of the project is to prepare you to apply different machine
learning algorithms to real-world tasks. This will help you to increase your
knowledge about the workflow of the machine learning tasks. You will learn
how to clean your data, applying pre-processing, feature engineering,
regression, and classification methods.

## Exploring Dataset

- Dataset Shape
- Dataset Distribution
- Features Types
- Outliers Detection

## Preprocessing

- General Preprocessing
  - Drop null & Duplicate rows.
  - Train / Test split.
  - Drop ID, URL, Name columns.
  - Drop Subtitle column.
  - Drop Primary Genre column.
  - fill null values.
  - Convert date column to integers.
  - Encode age rating.
  - Extract game difficulty from description column.

- Train Dataset
  - Encode Languages and Genres.
  - Scaling.
  - Outliers removal.
  - Count Developer games.

- Features Selection
  - K-best.
  - set K = 100.

- Test Dataset
  - Encode Languages and Genres.
  - Scaling.
  - Count Developer games.
 
## Regression Models

- Random Forest.
- Decision Tree Regressor.
- Gradient Boosting Regressor.

## Classification Models

- SVC (RBF Kernal)
- SVC (Linear Kernal)
- Linear SVC
- SVC (with polynomial -degree 3- Kernal)
- Logistic Regression
- Decision Tree Classifier

## Conclusion

- Dataset
  - The lack of samples for some ratings/ classes makes it hard to distinguish whether an app will be given high
    or low ratings since we only have samples for the high ratings.
    Also, most of the features uniquely identifies the games which we could not use.

- Models
  - Regression Models: Random Forest Models and Gradient Boosting performed well in training,
    But Gradient Boosting performed better in testing.
  -  Classification Models: There is no big difference between the four models,
     But SVM with RBF kernel showed less overfitting behavior.

- Future Work
  - More feature engineering to extract meaningful features to improve the models learning.
  - More hyperparameters tuning.

## Developers :
- [**Abdelrhman Sayed**](https://github.com/Abdelrhman-Sayed70)
- [**Nour Ayman**](https://github.com/NourAyman10)
- [**Ruqaiyah Mohammed**](https://github.com/25Ruq)
- [**Heba Tarek**](https://github.com/hebatarekkamal) 
- [**Haneen Ibrahim**](https://github.com/HaneenIbrahim2)
- [**Mariam Ahmed**](https://github.com/MariamAhmeddd)

If you find any issues or have any suggestions, feel free to submit a pull request :)
