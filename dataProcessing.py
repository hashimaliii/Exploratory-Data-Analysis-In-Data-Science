
def extract_features(data):
    
    # Feature Engineering
    data["hour"] = data["datetime"].dt.hour
    data["day"] = data["datetime"].dt.day
    data["month"] = data["datetime"].dt.month
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    data["season"] = data["month"].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
    return data