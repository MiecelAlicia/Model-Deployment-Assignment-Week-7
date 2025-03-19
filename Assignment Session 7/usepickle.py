import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_trained_model(filepath):
    """Load a trained model from a pickle file."""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

def make_prediction(model, input_data):
    """Use the trained model to make a prediction based on user input."""
    return model.predict([input_data])[0]

def main():
    model_path = 'random_forest_model.pkl'  # Updated model filename
    model = load_trained_model(model_path)

    # Example user input data
    sample_input = [2, 1, 2, 3, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 3, 2, 0, 0, 2, 3, 26]
    
    # Get prediction
    result = make_prediction(model, sample_input)
    print(f"Predicted output: {result}")

if __name__ == "__main__":
    main()
