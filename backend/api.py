from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from main_torchHYBRID import prepare_data, HybridModel, predict, get_pollutant_level, calculate_wqi

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# Global variables for model and data
model = None
df = None
scaler = None
features = None
lookback = None  # Store the lookback period from the model


def load_model_and_data():
    """Load the trained model and preprocessed data"""
    global model, df, scaler, features, lookback

    print("⚙️ Loading data and model...")
    try:
        # Prepare data and load model
        X_train, y_train, X_test, y_test, scaler, features, df, test_dates = prepare_data(
            'water_quality_data.csv',
            use_climate=True,  # or True/False depending on what was used when training
            use_volcanic = True,
        )

        lookback = X_train.shape[1]  # Get the lookback period from training data

        model = HybridModel(input_size=len(features), seq_length=lookback)
        model.load_state_dict(torch.load('model_d.pth', map_location=torch.device('cpu')))
        model.eval()
        print(f"✅ Model and data loaded successfully (Lookback: {lookback} months)")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


@app.route('/<path:path>')
def serve_static(path):
    """Serve static frontend files (CSS, JS, etc.)"""
    return send_from_directory('../frontend', path)


@app.route('/api/latest_params', methods=['GET'])
def get_latest_params():
    """API endpoint to get latest water quality parameters"""
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
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/parameters', methods=['GET'])
def get_all_parameters():
    """API endpoint to get all parameters for dashboard"""
    try:
        if df is None:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

        # Get time range from query params
        time_range = request.args.get('range', '30days')

        if time_range == '30days':
            # Use the most recent data point
            last_row = df.iloc[-1]
            last_date = df.index[-1]

            # Create 4 weekly timestamps within the last month
            weekly_dates = pd.date_range(end=last_date, periods=4, freq='7D')

            # Function to divide the latest value into 4 equal parts
            def split_into_weeks(param):
                total = float(last_row[param])

                # Seed for consistent results per param/month
                seed_value = hash(str(last_date) + param) % 2 ** 32
                rng = np.random.default_rng(seed=seed_value)

                # Base weights favoring middle weeks (like a bell curve)
                base = np.array([1.1, 1, 1.1, 0.9])

                # Small randomness added to each (controlled and centered)
                noise = rng.normal(loc=1.0, scale=0.05, size=4)  # ±5% variation
                weights = base * noise
                weights = np.clip(weights, 0.75, 1.25)  # Avoid extreme values

                # Normalize so they sum to 1
                weights /= weights.sum()

                # Multiply by the total value
                values = (weights * total).tolist()
                return values

            # Construct a DataFrame simulating weekly data
            simulated_data = {
                'Ammonia (mg/L)': split_into_weeks('Ammonia (mg/L)'),
                'Phosphate (mg/L)': split_into_weeks('Phosphate (mg/L)'),
                'Dissolved Oxygen (mg/L)': split_into_weeks('Dissolved Oxygen (mg/L)'),
                'Nitrate (mg/L)': split_into_weeks('Nitrate (mg/L)'),
                'pH Level': split_into_weeks('pH Level'),
                'Surface Water Temp (°C)': split_into_weeks('Surface Water Temp (°C)')
            }

            filtered_df = pd.DataFrame(simulated_data, index=weekly_dates)
            cutoff = weekly_dates[0]

        elif time_range == '6months':
            cutoff = df.index[-1] - pd.DateOffset(months=6)
            filtered_df = df[df.index >= cutoff]
        elif time_range == '1year':
            cutoff = df.index[-1] - pd.DateOffset(years=1)
            filtered_df = df[df.index >= cutoff]
        else:  # 1year
            cutoff = df.index[-1] - pd.DateOffset(years=10)
            filtered_df = df[df.index >= cutoff]

        # Prepare data for each parameter
        parameters = {
            'ammonia': {
                'values': filtered_df['Ammonia (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Ammonia (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Ammonia (mg/L)'].mean())
            },
            'phosphate': {
                'values': filtered_df['Phosphate (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Phosphate (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Phosphate (mg/L)'].mean())
            },
            'dissolvedoxygen': {
                'values': filtered_df['Dissolved Oxygen (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Dissolved Oxygen (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Dissolved Oxygen (mg/L)'].mean())
            },
            'nitrate': {
                'values': filtered_df['Nitrate (mg/L)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'mg/L',
                'current': float(filtered_df['Nitrate (mg/L)'].iloc[-1]),
                'avg': float(filtered_df['Nitrate (mg/L)'].mean())
            },
            'ph': {
                'values': filtered_df['pH Level'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': 'pH',
                'current': float(filtered_df['pH Level'].iloc[-1]),
                'avg': float(filtered_df['pH Level'].mean())
            },
            'temperature': {
                'values': filtered_df['Surface Water Temp (°C)'].tolist(),
                'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
                'unit': '°C',
                'current': float(filtered_df['Surface Water Temp (°C)'].iloc[-1]),
                'avg': float(filtered_df['Surface Water Temp (°C)'].mean())
            }
        }

        return jsonify({
            'status': 'success',
            'data': parameters,
            'timeRange': time_range,
            'startDate': cutoff.strftime('%Y-%m-%d'),
            'endDate': df.index[-1].strftime('%Y-%m-%d'),
            'minDate': df.index[0].strftime('%Y-%m-%d'),
            'maxDate': (df.index[-1] + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500



@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    """API endpoint to make WQI predictions with configurable parameters"""
    try:
        data = request.json
        date_str = data.get('date')
        param_choice = data.get('parameterSet', 'a').lower()

        if not date_str:
            return jsonify({
                'status': 'error',
                'message': 'Date is required'
            }), 400

        # Validate and parse date
        try:
            date_obj = pd.to_datetime(date_str)
            min_date = pd.to_datetime(df.index[0])
            max_date = pd.to_datetime(df.index[-1] + pd.DateOffset(years=30))

            if date_obj < min_date:
                return jsonify({
                    'status': 'error',
                    'message': f'Date must be after {min_date.strftime("%Y-%m-%d")}'
                }), 400

            if date_obj > max_date:
                return jsonify({
                    'status': 'error',
                    'message': f'Date must be before {max_date.strftime("%Y-%m-%d")}'
                }), 400
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid date format: {str(e)}'
            }), 400

        # Decide feature configuration based on parameter set
        use_climate = False
        use_volcanic = False

        if param_choice == 'b':
            use_climate = True
        elif param_choice == 'c':
            use_volcanic = True
        elif param_choice == 'd':
            use_climate = True
            use_volcanic = True

        # Prepare data and features using selected parameter set
        try:
            _, _, _, _, scaler, features, updated_df, _ = prepare_data(
                'water_quality_data.csv',
                use_climate=use_climate,
                use_volcanic=use_volcanic
            )
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to prepare data: {str(e)}'
            }), 500

        # Create and load model with correct input size
        try:
            model_path = f'model_{param_choice}.pth'

            model_config = HybridModel(input_size=len(features), seq_length=lookback)
            model_config.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model_config.eval()
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Model loading failed: {str(e)}'
            }), 500

        # Run prediction
        prediction = predict(model_config, updated_df, scaler, features, date_obj)

        if prediction is None:
            return jsonify({
                'status': 'error',
                'message': 'Prediction failed - insufficient historical data'
            }), 400

        print("📊 Raw prediction:", prediction)

        wqi = float(prediction['WQI'])
        ammonia = float(prediction['Ammonia'])
        nitrate = float(prediction['Nitrate'])
        phosphate = float(prediction['Phosphate'])

        return jsonify({
            'status': 'success',
            'data': {
                'wqi': wqi,
                'ammonia': ammonia,
                'nitrate': nitrate,
                'phosphate': phosphate,
                'pollutantLevel': get_pollutant_level(wqi),
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
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}'
        }), 500



@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': df is not None,
        'lookback_period': lookback,
        'server_time': datetime.now().isoformat()
    })

load_model_and_data()

if __name__ == '__main__':
    # Load model and data when starting the server
    load_model_and_data()

    # Run the Flask app
    print("\n🌍 Server running at http://localhost:5000")
    print("🔌 API endpoints:")
    print("   - GET  /api/latest_params")
    print("   - GET  /api/parameters?range=30days|6months|1year")
    print("   - POST /api/predict")
    print("   - GET  /api/health\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
