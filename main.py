import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students.csv")



# ----------------------------
# 1. Basic Analysis (Pandas)
# ----------------------------
df["Average"] = df[["Math", "Science", "English", "Computer"]].mean(axis=1)

df["Result"] = df["Average"].apply(lambda x: "Pass" if x >= 50 else "Fail")

print("\n===== STUDENT DATA =====\n")
print(df)

print("\n===== CLASS STATISTICS =====\n")
print("Class Average:", df["Average"].mean())
print("Highest Average:", df["Average"].max())
print("Lowest Average:", df["Average"].min())

# ----------------------------
# 2. Simple AI (Rule-based)
# ----------------------------
def generate_insight(row):
    weak_subjects = []

    if row["Math"] < 50:
        weak_subjects.append("Math")
    if row["Science"] < 50:
        weak_subjects.append("Science")
    if row["English"] < 50:
        weak_subjects.append("English")
    if row["Computer"] < 50:
        weak_subjects.append("Computer")

    if len(weak_subjects) == 0:
        return "Excellent performance"
    else:
        return "Weak in " + ", ".join(weak_subjects)


df["Insight"] = df.apply(generate_insight, axis=1)

print("\n===== AI INSIGHTS =====\n")
for i in range(len(df)):
    print(df["Name"][i], "->", df["Insight"][i])

# ----------------------------
# 3. Machine Learning (Simple Model)
# ----------------------------
# (Very basic rule-based ML style classification)

def predict_pass(avg):
    return 1 if avg >= 50 else 0

df["Prediction"] = df["Average"].apply(predict_pass)

# ----------------------------
# 4. Data Visualization
# ----------------------------

# Bar Graph - Average marks
plt.figure()
plt.bar(df["Name"], df["Average"])
plt.title("Student Average Marks")
plt.xlabel("Students")
plt.ylabel("Average Marks")
plt.show()

# Histogram - Distribution
plt.figure()
plt.hist(df["Average"], bins=5)
plt.title("Marks Distribution")
plt.xlabel("Average Marks")
plt.ylabel("Frequency")
plt.show()

# Pie Chart - Pass/Fail
result_counts = df["Result"].value_counts()

plt.figure()
plt.pie(result_counts, labels=result_counts.index, autopct="%1.1f%%")
plt.title("Pass vs Fail Ratio")
plt.show()