# diagnose-faults-with-NeuralNetworks
Digital System Fault Diagnosis using Artificial Neural Networks

Project Overview:
It includes a complete pipeline from data preprocessing to neural network training and deployment through a Streamlit-based user interface.

Objectives:
- Analyze traffic logs from digital systems (routers, SDH, ATS) to detect anomalies.
- Train a neural network model to classify system faults in real-time.
- Develop a multi-label classification system to identify not only the fault but also its specific location.
- Build a user-friendly web interface for operators and engineers.

Key Features:
- Neural network model with 95%+ accuracy on test datasets.
- Real-time fault detection with multi-label outputs (e.g., out1–out5).
- Streamlit-based UI for loading CSV files and visualizing predictions.
- Dataset built on CICIDS 2017 and synthetic extensions for telecom use cases.
- Scalable architecture for future integration with real-time monitoring tools.

🛠️ Tech Stack:
- Python (NumPy, Pandas, Scikit-learn, TensorFlow)
- Streamlit for frontend UI
- Matplotlib & Seaborn for data visualization
- CSV-based logs for training and testing
