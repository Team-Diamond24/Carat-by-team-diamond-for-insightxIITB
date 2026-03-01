import os
os.environ["USE_CSV"] = "true"
from engine import PandasAnalyst
analyst = PandasAnalyst()
res = analyst.analyze("What is the average transaction amount for the Food category?")
print("==== CODE ====")
print(res.get("code"))
print("\n==== DATA TYPE ====")
print(type(res.get("data")))
print("\n==== RESULT ====")
print(res.get("data")[:500] if isinstance(res.get("data"), str) else res.get("data"))
