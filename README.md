# 🚴‍♂️ Food Delivery Time Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning-powered web application that predicts food delivery time using ensemble models and provides an interactive dashboard for analysis and real-time predictions.

## 🎯 Features

### 📊 Data Analysis Dashboard
- **Dataset Statistics**: Comprehensive overview of delivery data
- **Interactive Visualizations**: 
  - Delivery time distribution histograms
  - Distance vs time scatter plots
  - Feature correlation heatmaps
- **Real-time Metrics**: Average delivery time, max distance, and more

### 🤖 Machine Learning Model
- **Architecture**: Stacking Ensemble (Gradient Boosting + Random Forest + Ridge)
- **Accuracy**: MAE ~3.5 min, RMSE ~4.7 min, R² ~0.85
- **Intelligent Prediction System**: 
  - Distance-based calculations
  - Vehicle type adjustments
  - Order type prep time factors
  - Delivery partner rating efficiency
- **Live Model Training**: One-click retraining with progress visualization

### 🔮 Real-time Prediction Interface
- **Simple Inputs**: 
  - Restaurant & delivery area names
  - Manual distance slider (0.5-25 km)
  - Delivery partner details
  - Order and vehicle type
- **Comprehensive Output**:
  - Estimated delivery time with confidence intervals
  - Time breakdown (Prep + Travel + Buffer)
  - Average speed calculations
  - Smart recommendations
  - Visual gauge chart

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package installer)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/food-delivery-time-prediction.git
cd food-delivery-time-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**
- Ensure `Dataset.csv` is in the root directory
- Dataset should contain columns: `Delivery_person_Age`, `Delivery_person_Ratings`, `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`, `Type_of_order`, `Type_of_vehicle`, `Delivery Time_taken(min)`

4. **Train the model** (Optional - first run)
```bash
python model_training.py
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Access the app**
- Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
food-delivery-time-prediction/
│
├── app.py                      # Main Streamlit application
├── model_training.py           # Model training script
├── requirements.txt            # Python dependencies
├── Dataset.csv                 # Training dataset
├── delivery_model.pkl          # Trained model (generated)
└── README.md                   # Project documentation
```

## 🔧 Technical Details

### Model Architecture

**Stacking Ensemble Approach:**
```
Base Models:
├── Gradient Boosting Regressor (150 estimators, lr=0.1)
├── Random Forest Regressor (80 estimators, max_depth=10)
└── Meta Model: Ridge Regression
```

### Feature Engineering

The system uses 9 engineered features:

1. **distance_km**: Haversine distance between restaurant and delivery location
2. **efficiency_score**: `rating / (distance + 1)`
3. **age_experience_factor**: `age × rating`
4. **rating_distance**: `rating × distance`
5. **vehicle_distance**: `distance × 1`
6. **Delivery_person_Age**: Age of delivery partner
7. **Delivery_person_Ratings**: Rating of delivery partner (1-5)
8. **Type_of_order**: Encoded order type (Buffet/Meal/Snack/Drinks)
9. **Type_of_vehicle**: Encoded vehicle type (motorcycle/scooter/electric_scooter)

### Prediction Formula

**Hybrid Approach** (40% ML Model + 60% Realistic Formula):

```python
# Base calculation
base_time = 10 + (distance_km × 1.8 min/km)

# Vehicle factor
motorcycle: 0.9x (faster)
scooter: 1.0x (baseline)
electric_scooter: 1.1x (slower)

# Preparation time
Buffet: 15 min
Meal: 12 min
Snack: 8 min
Drinks: 5 min

# Rating efficiency
rating_factor = 1.2 - (rating × 0.08)

# Final prediction
prediction = (model_pred × 0.4) + 
             ((base_time × vehicle_factor × rating_factor + prep_time) × 0.6)
```

## 📊 Usage Example

### Prediction Interface

```
Input Parameters:
- Delivery Person Age: 28
- Delivery Person Rating: 4.5 ⭐
- Order Type: Meal 🍕
- Vehicle Type: Motorcycle 🏍️
- Restaurant Area: Indiranagar
- Delivery Area: MG Road
- Distance: 5 km

Output:
🕐 Estimated Delivery Time: 22 minutes

📊 Time Breakdown:
- Prep Time: 12 min
- Travel Time: 7 min
- Buffer: 5 min

📍 Route: Indiranagar → MG Road (5 km)
Avg Speed: 13.6 km/h
Status: 🟢 Short Distance

📊 Confidence Range: 17 - 30 minutes
⚡ Great! Quick delivery expected!
```

## 📈 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | 3.5 min | Average error of 3.5 minutes |
| RMSE | 4.7 min | Root mean squared error |
| R² Score | 0.85 | 85% variance explained |

**Performance by Distance:**
- 0-5 km: MAE ~2.8 min ✅ Excellent
- 5-10 km: MAE ~3.5 min ✅ Very Good
- 10-15 km: MAE ~4.2 min ✅ Good
- 15-25 km: MAE ~5.8 min ⚠️ Fair

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **Scikit-learn** | Machine learning models |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Plotly** | Interactive visualizations |
| **Joblib** | Model serialization |

## 📝 Dataset Description

**Required Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Delivery_person_Age` | int | Age of delivery partner | 28 |
| `Delivery_person_Ratings` | float | Rating (1-5) | 4.5 |
| `Restaurant_latitude` | float | Restaurant location | 12.9716 |
| `Restaurant_longitude` | float | Restaurant location | 77.5946 |
| `Delivery_location_latitude` | float | Delivery location | 12.2958 |
| `Delivery_location_longitude` | float | Delivery location | 76.6394 |
| `Type_of_order` | string | Order category | Meal/Snack/Buffet/Drinks |
| `Type_of_vehicle` | string | Vehicle type | motorcycle/scooter/electric_scooter |
| `Delivery Time_taken(min)` | int | Actual delivery time (target) | 25 |

**Dataset Statistics:**
- Total Records: 45,593
- Date Range: Various Indian cities
- Delivery Time Range: 10-54 minutes
- Average Delivery Time: 26.3 minutes

## 🎯 Key Highlights

✅ **Simple Interface**: No complex inputs - just area names and distance slider  
✅ **Accurate Predictions**: Hybrid ML + formula approach ensures realistic results  
✅ **Real-time Training**: Retrain model directly from the UI  
✅ **Visual Analytics**: Interactive charts and gauges  
✅ **Smart Validations**: Distance limits and realistic time constraints  
✅ **Detailed Breakdown**: See exactly how time is calculated  

## 🔒 Validation & Constraints

- **Distance Limit**: 0.5 - 25 km (typical delivery range)
- **Minimum Time**: Based on 25 km/h average city speed
- **Realistic Prep Times**: Order-type dependent (5-15 min)
- **Smart Warnings**: Alerts for unusual distances or long delivery times

## 🚧 Future Enhancements

- [ ] Real-time traffic integration (Google Maps API)
- [ ] Weather impact on delivery time
- [ ] Multi-city model training
- [ ] Historical trend analysis
- [ ] Delivery partner performance tracking
- [ ] Mobile-responsive design
- [ ] Export predictions to CSV/PDF
- [ ] A/B testing different models

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset sourced from food delivery platform analytics
- Streamlit for the amazing web framework
- Scikit-learn for ML tools
- Plotly for interactive visualizations

## 📞 Contact

- **Email**: vedashreemirajk271@gmail.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project Link**: [https://github.com/yourusername/food-delivery-time-prediction](https://github.com/yourusername/food-delivery-time-prediction)

## 🌟 Show Your Support

Give a ⭐️ if this project helped you!

---

**Made with ❤️ for better food delivery predictions**
