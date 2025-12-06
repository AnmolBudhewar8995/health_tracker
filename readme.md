# 🏃‍♂️ Personal Health & Fitness Tracker — AI Insights

A comprehensive AI-powered health and fitness tracking application built with Streamlit, featuring machine learning models for personalized predictions and insights.

## ✨ Features

### 📊 Dashboard Overview
- **Real-time Metrics**: Track weight, BMI, steps, and activity levels
- **Delta Indicators**: See day-to-day changes in your health metrics
- **Quick Stats**: 7-day averages for steps, active minutes, and sleep

### 🎯 Goal Tracking
- **Animated Progress Bars**: Visual progress toward daily goals
- **Customizable Goals**: Set targets for steps (10,000), active minutes (30), and sleep (8 hours)
- **Achievement Celebrations**: Success messages when goals are met

### 📈 Advanced Analytics
- **Interactive Trend Charts**: Animated visualizations of weight, BMI, and activity data
- **Anomaly Detection**: Identify unusual patterns in your health data
- **Weekly Forecasting**: Predict weight trends for the next 4 weeks

### 🤖 AI Predictions
- **Calories Burned Estimator**: Predict daily calorie expenditure based on activity
- **BMI Change Forecasting**: 14-day BMI predictions based on current habits
- **Personalized Insights**: ML-driven recommendations

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd health_tracker
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install the required packages manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn joblib matplotlib plotly
   ```

## 📁 Project Structure

```
health_tracker/
├── data/
│   └── dataset.csv          # Generated health data
├── models/
│   ├── calories_model.joblib        # Calories prediction model
│   ├── bmi_change_14d_model.joblib  # BMI change prediction model
│   └── anomaly_detector.joblib      # Anomaly detection model
├── data_generator.py        # Synthetic data generation script
├── train_models.py          # Model training script
├── app.py                   # Main Streamlit application
└── README.md               # This file
```

## 🛠️ Usage

### 1. Generate Data (Optional)
If you want to generate new synthetic data:
```bash
python data_generator.py
```

### 2. Train Models
Train the machine learning models:
```bash
python train_models.py
```

### 3. Run the Application
Launch the Streamlit web app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## 🎨 UI Features

- **Modern Glassmorphism Design**: Frosted glass effects with mirror reflections
- **Gradient Background**: Beautiful purple-blue gradient theme
- **Interactive Animations**: Animated progress bars and chart visualizations
- **Responsive Layout**: Optimized for different screen sizes
- **Tabbed Interface**: Organized sections for Overview, Goals, Analysis, and Predictions

## 📊 Data Schema

The application uses health data with the following columns:
- `user_id`: Unique identifier for each user
- `date`: Date of the health record
- `age`: User's age
- `sex`: User's gender (M/F)
- `height_cm`: Height in centimeters
- `weight_kg`: Weight in kilograms
- `bmi`: Body Mass Index
- `steps`: Daily step count
- `active_min`: Active minutes per day
- `sleep_h`: Hours of sleep
- `calories_in`: Calories consumed
- `calories_burned`: Calories burned
- `drank_alcohol`: Alcohol consumption flag
- `smoked`: Smoking flag

## 🤖 Machine Learning Models

### Calories Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Age, sex, height, weight, steps, active minutes, sleep hours
- **Target**: Daily calories burned

### BMI Change Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Age, sex, height, weight, 7-day averages for steps, active minutes, calories in, sleep, habits
- **Target**: BMI change after 14 days

### Anomaly Detection Model
- **Algorithm**: Isolation Forest
- **Features**: All numeric health metrics
- **Purpose**: Identify unusual health patterns

## 🔧 Customization

### Modifying Goals
Edit the goal values in `app.py`:
```python
steps_goal = 10000  # Change daily steps goal
active_goal = 30    # Change active minutes goal
sleep_goal = 8      # Change sleep hours goal
```

### Adding New Features
The app is modular and can be extended with:
- New health metrics
- Additional ML models
- Custom visualizations
- Integration with wearables

## 📈 Model Performance

Current model performance metrics:
- **Calories Model RMSE**: ~26.36 kcal
- **BMI Change Model RMSE**: ~0.056

Models are trained on synthetic data and can be retrained with real data for better accuracy.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine Learning with [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/)
- Icons from [Emoji](https://emojipedia.org/)

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Stay healthy and track your fitness journey with AI-powered insights! 💪**
