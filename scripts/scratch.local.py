# %%
import pandas as pd
# %%
df = pd.read_csv("../nrel/routee/powertrain/resources/sample_routes/sample_route.csv")
# %%
df["grade_percent"] = df["grade_dec"] * 100
df["distance"] = df["miles"]
# %%
df[["speed_mph", "grade_percent", "distance"]].to_csv("../nrel/routee/powertrain/resources/sample_routes/sample_route.csv", index=False)
# %%
# %%
