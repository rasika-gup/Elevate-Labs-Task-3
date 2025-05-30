import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("house_prices.csv")

# Step 2: Preprocess categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 3: Separate features and target
X = df_encoded.drop('price', axis=1)  # All columns except 'price'
y = df_encoded['price']              # Target: price

# Step 4: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("Model Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: Display model coefficients
print("\nModel Coefficients:")
coeff_df = pd.DataFrame(model.coef_, index=X.columns, columns=["Coefficient"])
print(coeff_df)

# Optional: Plot actual vs predicted prices
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
