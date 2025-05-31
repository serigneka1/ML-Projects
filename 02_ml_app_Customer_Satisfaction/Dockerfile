FROM python:3.11-slim
WORKDIR /02_ml_app_Customer_Satisfaction
COPY requirements.txt .
COPY . .
RUN RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", ".py"]
