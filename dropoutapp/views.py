import os
import joblib
import pandas as pd
from django.shortcuts import render

from sktime.forecasting.base import ForecastingHorizon

# Load model once when server starts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'dropoutapp', 'model_files', 'dropout_forecaster_model.pkl')
forecaster = joblib.load(model_path)

# Load original index for forecasting
df = pd.read_csv(os.path.join(BASE_DIR, 'dropoutapp', 'model_files', 'cleaned_dropout_data.csv'))
df['StartYear'] = df['Year'].str[:4].astype(int)
yearly_avg = df.groupby('StartYear')['DropoutRate'].mean()
yearly_avg.index = pd.PeriodIndex(yearly_avg.index, freq='Y')

def forecast_dropout(request):
    if request.method == 'POST':
        start_year = int(request.POST.get('start_year'))
        end_year = int(request.POST.get('end_year'))

        # Generate forecast horizon
        forecast_index = pd.period_range(start=str(start_year), end=str(end_year), freq='Y')
        fh = ForecastingHorizon(forecast_index, is_relative=False)

        # Forecast
        forecast = forecaster.predict(fh=fh)

        # Prepare data for template
        forecast_df = forecast.reset_index()
        forecast_df.columns = ['Year', 'Predicted Dropout Rate']
        return render(request, 'forecast_result.html', {'forecast_df': forecast_df})
    
    return render(request, 'forecast_form.html')
