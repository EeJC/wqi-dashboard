from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from main_torchHYBRID import prepare_data, HybridModel, predict, get_pollutant_level, calculate_wqi

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'frontend'), static_url_path='/')
CORS(app)

# Global variables
model = None
df = None
scaler = None
features = None
lookback = None


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


def load_model_and_data():
    global model, df, scaler, features, lookback
    print("\n⚙️ Loading model and data...")
    try:
        data_path = os.path.join(BASE_DIR, 'water_quality_data.csv')
        model_path = os.path.join(BASE_DIR, 'model_d.pth')

        X_train, _, _, _, scaler, features, df, _ = prepare_data(
            data_path,
            use_climate=True,
            use_volcanic=True
        )

        lookback = X_train.shape[1]
        model = HybridModel(input_size=len(features), seq_length=lookback)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        print("✅ Model and data loaded.")
    except Exception as e:
        print(f"❌ Failed to load model/data: {str(e)}")
        raise


@app.route('/api/latest_params', methods=['GET'])
def get_latest_params():
    try:
        if df is None:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

        latest = df.iloc[-1]
        wqi = float(latest['WQI'])

        return jsonify({
            'status': 'success',
            'data': {
                'parameters': {
                    'ammonia': float(latest['Ammonia (mg/L)']),
                    'phosphate': float(latest['Phosphate (mg/L)']),
                    'nitrate': float(latest['Nitrate (mg/L)']),
                    'dissolvedOxygen': float(latest['Dissolved Oxygen (mg/L)']),
                    'pH': float(latest['pH Level']),
                    'temperature': float(latest['Surface Water Temp (°C)']),
                    'wqi': wqi,
                    'pollutantLevel': get_pollutant_level(wqi),
                },
                'date': latest.name.strftime('%Y-%m-%d'),
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    try:
        data = request.json
        date_str = data.get('date')
        param_choice = data.get('parameterSet', 'a').lower()

        if not date_str:
            return jsonify({'status': 'error', 'message': 'Date is required'}), 400

        date_obj = pd.to_datetime(date_str)
        if date_obj < df.index[0]:
            return jsonify({'status': 'error', 'message': f'Date must be after {df.index[0].strftime("%Y-%m-%d")}'}), 400

        use_climate = param_choice in ['b', 'd']
        use_volcanic = param_choice in ['c', 'd']

        data_path = os.path.join(BASE_DIR, 'water_quality_data.csv')
        model_path = os.path.join(BASE_DIR, f'model_{param_choice}.pth')

        _, _, _, _, scaler, features, updated_df, _ = prepare_data(
            data_path,
            use_climate=use_climate,
            use_volcanic=use_volcanic
        )

        model_config = HybridModel(input_size=len(features), seq_length=lookback)
        model_config.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model_config.eval()

        prediction = predict(model_config, updated_df, scaler, features, date_obj)
        if prediction is None:
            return jsonify({'status': 'error', 'message': 'Prediction failed - insufficient historical data'}), 400

        return jsonify({
            'status': 'success',
            'data': {
                'wqi': float(prediction['WQI']),
                'ammonia': float(prediction['Ammonia']),
                'nitrate': float(prediction['Nitrate']),
                'phosphate': float(prediction['Phosphate']),
                'pollutantLevel': get_pollutant_level(float(prediction['WQI'])),
                'date': date_obj.strftime('%Y-%m-%d'),
                'confidence': 0.85,
                'modelInfo': {
                    'type': 'Hybrid CNN-LSTM',
                    'parameterSet': param_choice.upper(),
                    'lookbackPeriod': f'{lookback} months',
                    'lastTrained': os.path.getmtime(model_path)
                }
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Prediction error: {str(e)}'}), 500


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'lookback_period': lookback,
        'server_time': datetime.now().isoformat()
    })


if __name__ == '__main__':
    load_model_and_data()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
