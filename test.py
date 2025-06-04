import mlflow
print("Printing new tracking URI scheme below")
print(mlflow.get_tracking_uri())
# this gives output in files format though allowed tracing schemes are {'http', 'https'}. so it is bug in mlflow but it might be solved as of now.

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("Printing new tracking URI scheme below to get in http form")
print(mlflow.get_tracking_uri())
print("\n")