# Edunet_internship
Urban Heat Island Detection using CNN

This project detects Urban Heat Island (UHI) intensity — Low, Medium, or High — using a Convolutional Neural Network (CNN) trained on temperature maps and normal RGB city maps.

The model combines features from both image types to identify regions with higher heat concentration for sustainable urban planning.

📂 Dataset Structure
dataset/
├── temperature/   # Thermal or temperature maps
├── normal/        # Normal RGB satellite images
└── labels.csv     # image_name,label (low/medium/high)

⚙️ Requirements

Install required libraries:

pip install tensorflow numpy matplotlib scikit-learn opencv-python pandas

🚀 How to Run

Clone the repository

git clone https://github.com/<your-username>/urban-heat-island-cnn.git
cd urban-heat-island-cnn


Add your dataset inside the dataset/ folder.

Run the script:

python urban_heat_island_cnn.py


View accuracy plots and predictions in the console.

🛰️ Data Sources

NASA MODIS LST

Google Earth Engine

Sentinel Hub EO Browser

👩‍💻 Author

Akshaya Purushothaman
