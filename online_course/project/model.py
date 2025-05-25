import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
file_path = "online_course_recommendation_v2.xlsx"
df = pd.read_excel(file_path)

# Remove duplicate (user_id, course_id) pairs by taking the mean rating
df = df.groupby(["user_id", "course_id", "course_name"], as_index=False).mean(numeric_only=True)

# Filter out inactive users (rated < 5 courses) and unpopular courses (rated < 5 times)
active_users = df["user_id"].value_counts()
df = df[df["user_id"].isin(active_users[active_users >= 5].index)]

popular_courses = df["course_id"].value_counts()
df = df[df["course_id"].isin(popular_courses[popular_courses >= 5].index)]

# Reset index for consistent mapping
df = df.reset_index(drop=True)

# Convert the user-course matrix to a sparse format
user_course_matrix = df.pivot(index="user_id", columns="course_id", values="rating").fillna(0)
user_course_sparse = csr_matrix(user_course_matrix)

# Compute user similarity using sparse matrices
user_sim_matrix = cosine_similarity(user_course_sparse, dense_output=False)
user_sim_df = pd.DataFrame(user_sim_matrix.toarray(), index=user_course_matrix.index, columns=user_course_matrix.index)

# Create course features using course duration, feedback score, and enrollments
df["course_features"] = (
    df["course_id"].astype(str) + " " + 
    df["course_duration_hours"].astype(str) + " " + 
    df["feedback_score"].astype(str) + " " + 
    df["enrollment_numbers"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["course_features"])

# Map course_id to matrix index and course names
course_id_to_index = {course_id: idx for idx, course_id in enumerate(df["course_id"].unique())}
course_id_to_name = df.set_index("course_id")["course_name"].to_dict()
course_sim_matrix = cosine_similarity(tfidf_matrix)

# Hybrid Recommendation Function (Improved)
def hybrid_recommend(user_id, top_n=5):
    if user_id not in user_sim_df.index:
        return []

    # Get top 5 most similar users (higher similarity gets more weight)
    similar_users = user_sim_df.loc[user_id].sort_values(ascending=False)[1:6]

    recommended_courses = {}
    for sim_user, sim_score in similar_users.items():
        user_courses = user_course_matrix.loc[sim_user]
        top_courses = user_courses[user_courses > 3.0].index  # Only consider courses rated > 3.0

        for course in top_courses:
            if course not in course_id_to_index:
                continue  # Skip if course_id not found
            course_idx = course_id_to_index[course]
            
            # Use weighted similarity score
            content_score = np.mean(course_sim_matrix[course_idx])
            final_score = (0.7 * sim_score) + (0.3 * content_score)  # Hybrid scoring
            
            if course in recommended_courses:
                recommended_courses[course] += final_score
            else:
                recommended_courses[course] = final_score

    # Sort by highest scores
    sorted_courses = sorted(recommended_courses.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Return course names along with IDs
    return [(course_id, course_id_to_name.get(course_id, "Unknown Course")) for course_id, _ in sorted_courses]

# Precision & Recall Evaluation (Improved)
def evaluate_model(test_users, top_n=5):
    precision_list = []
    recall_list = []

    for user in test_users:
        actual_courses = set(df[df["user_id"] == user]["course_id"])
        recommended_courses = set([course[0] for course in hybrid_recommend(user, top_n)])  # Extract IDs

        if len(recommended_courses) == 0 or len(actual_courses) == 0:
            continue  # Skip users with no actual or recommended courses

        relevant_recommendations = actual_courses.intersection(recommended_courses)
        precision = len(relevant_recommendations) / max(len(recommended_courses), 1)
        recall = len(relevant_recommendations) / max(len(actual_courses), 1)

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    return avg_precision, avg_recall

# Run Evaluation on 100 Random Users
test_users = np.random.choice(df["user_id"].unique(), size=min(100, len(df["user_id"].unique())), replace=False)
precision, recall = evaluate_model(test_users)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Test Recommendation for a Sample User
sample_user = test_users[0]
recommendations = hybrid_recommend(sample_user, top_n=5)
print(f"\nRecommended Courses for User {sample_user}:")
for course_id, course_name in recommendations:
    print(f"- {course_name} (ID: {course_id})")