![Banner Placeholder](https://via.placeholder.com/1200x300?text=Retina+AI+Project+Banner)

# Retina AI: Multimodal Student Dropout Predictor 🎓👁️

**One-Liner Pitch:** An intelligent Early Warning System that uses Multimodal Deep Learning (Tabular + Time-Series + NLP) to predict college dropout risks before they happen.

## ⚠️ The Problem it Solves
Every year, thousands of students drop out because their warning signs go unnoticed. A single metric like a GPA doesn't tell the whole story. If a student has decent grades but suddenly stops attending classes and is flagged by a counsellor for emotional distress, standard systems miss it until it's too late. This project solves that by looking at the *complete* student ecosystem to catch the warning signs early enough to trigger life-saving interventions.

## ✨ Key Features
*   **Multimodal Late-Fusion AI:** Seamlessly combines standard grades, sequential weekly attendance, and unstructured counsellor notes into a single predictive engine.
*   **3-Tier Risk Classification:** Accurately classifies students into `Low`, `Medium`, or `High` risk buckets using Focal Loss to prioritize the minority of at-risk students.
*   **Interactive Counsellor Dashboard:** A sleek, real-time UI that allows administration to view a student's risk profile instantly.
*   **Explainable AI (Mocked):** Provides visual insights (SHAP values) into exactly *why* a student was flagged, so counsellors aren't just trusting a black box.
*   **Automated Action Triggers:** Built-in workflow buttons to instantly assign alumni peer mentors, mandate check-ins, or provide free access to mental health platforms like YourDOST.

## 🛠️ Tech Stack Used
*   **Machine Learning:** PyTorch, Scikit-learn, Pandas, Numpy
*   **NLP:** HuggingFace Transformers (DistilBERT)
*   **Frontend UI:** Streamlit (100% Python)
*   **Environment:** Kaggle Notebooks (GPU P100/T4x2)

## 💻 How to Run/Install it

**1. Training the Model (Kaggle)**
*   Upload the `kaggle_solution.ipynb` file to Kaggle.
*   Ensure your Accelerator is set to `GPU`.
*   Click "Run All" to train the model and generate the `submission.csv`.

**2. Running the UI Dashboard (Local)**
```bash
# Clone the repository
git clone https://github.com/yourusername/retina-ai-dropout.git
cd retina-ai-dropout

# Install requirements
pip install streamlit pandas numpy torch transformers scikit-learn

# Run the app
streamlit run app.py
```

## 🚀 Future Scope
If I had more time, I would build out the following features:
*   **Real API Integrations:** Connect the YourDOST and Twilio APIs to send actual automated SMS nudges to High-Risk students.
*   **Advanced Data Tracking:** Integrate campus Wi-Fi GPS data to track physical attendance and library usage.
*   **Graph Neural Networks (GNNs):** Map out social networks (e.g., roommates, same branch) because dropout behaviors are often socially influenced by peer groups.


