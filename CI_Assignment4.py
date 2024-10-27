import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'AirQualityUCI.xlsx'
data = pd.read_excel(file_path)

# Select relevant attributes (input: 3,6,8,10,11,12,13,14, output: 5)
selected_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'C6H6(GT)']
df_selected = data[selected_columns]

# Handle missing values by dropping rows with NaNs
df_selected = df_selected.dropna()

# Define input (X) and output (y)
X = df_selected[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']].values
y = df_selected['C6H6(GT)'].values

# Shift the output for 5-day predictions
y_5days = np.roll(y, -5)

# Remove the last 5 rows since they are shifted
X = X[:-5]
y_5days = y_5days[:-5]

# MLP forward propagation function
def MLP_forward(X, weights, layers):
    """
    Perform forward propagation for an MLP with given layers and weights.
    """
    input_layer = X
    for layer_weights in weights:
        input_layer = np.dot(input_layer, layer_weights)  # Linear transformation
        input_layer = np.tanh(input_layer)  # Activation function
    return input_layer

# PSO optimization function
def PSO_optimize(X_train, y_train, layers, num_particles=3, max_iter=20):
    """
    Particle Swarm Optimization for training MLP weights.
    """
    num_inputs = X_train.shape[1]
    swarm = []
    
    # Initialize particles with random weights
    for _ in range(num_particles):
        particle = {
            'position': [np.random.randn(num_inputs, layers[0])] + \
                        [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)],
            'velocity': [np.random.randn(num_inputs, layers[0])] + \
                        [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)],
            'best_position': None,
            'best_error': float('inf'),
        }
        swarm.append(particle)

    # PSO loop
    global_best_position = None
    global_best_error = float('inf')

    for iteration in range(max_iter):
        for particle in swarm:
            # Evaluate the particle's performance
            predictions = MLP_forward(X_train, particle['position'], layers)
            error = mean_absolute_error(y_train, predictions)

            # Update personal best
            if error < particle['best_error']:
                particle['best_error'] = error
                particle['best_position'] = particle['position']

            # Update global best
            if error < global_best_error:
                global_best_error = error
                global_best_position = particle['position']

        # Update velocities and positions of particles
        for particle in swarm:
            for i in range(len(particle['position'])):
                inertia = 0.5 * particle['velocity'][i]
                cognitive = 1.5 * random.random() * (particle['best_position'][i] - particle['position'][i])
                social = 1.5 * random.random() * (global_best_position[i] - particle['position'][i])
                particle['velocity'][i] = inertia + cognitive + social
                particle['position'][i] += particle['velocity'][i]
        print(f"Iteration {iteration + 1}/{max_iter}, Best Error: {global_best_error}")

    return global_best_position  # Return the best found weights

# Function to calculate Mean Absolute Error (MAE)
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Custom cross-validation function (10% cross-validation)
def cross_validate(X, y, layers, num_folds=10):
    fold_size = len(X) // num_folds
    errors = []
    best_weights = None
    lowest_error = float('inf')

    for i in range(num_folds):
        # Split data into training and testing sets
        start_test = i * fold_size
        end_test = start_test + fold_size

        X_test = X[start_test:end_test]
        y_test = y[start_test:end_test]

        X_train = np.concatenate((X[:start_test], X[end_test:]), axis=0)
        y_train = np.concatenate((y[:start_test], y[end_test:]), axis=0)

        # Train the model using PSO
        current_weights = PSO_optimize(X_train, y_train, layers)

        # Make predictions on the test set
        predictions = MLP_forward(X_test, current_weights, layers)

        # Calculate the MAE for this fold
        error = mean_absolute_error(y_test, predictions)
        errors.append(error)

        # Save the weights with the lowest error across folds
        if error < lowest_error:
            lowest_error = error
            best_weights = current_weights

        print(f"Fold {i + 1}/{num_folds}, MAE: {error}")

    # Calculate the mean error across all folds
    mean_error = np.mean(errors)

    # Plot mean error across all folds
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_folds + 1), errors, marker='o', label='MAE per fold')
    plt.axhline(y=mean_error, color='r', linestyle='--', label=f'Average MAE: {mean_error:.2f}')
    plt.title('Mean Absolute Error across folds')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()

    # Return the mean error and best weights
    return mean_error, best_weights


# New function to plot errors as bars
def plot_errors_as_bars(X, y, best_weights, layers):
    # Generate predictions
    predictions = MLP_forward(X, best_weights, layers)
    
    # Calculate errors
    errors = y - predictions.flatten()  # Ensure predictions are in the correct shape
    
    # Plotting the errors as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(errors)), errors, color='blue', alpha=0.6)
    
    # Adding grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Axis labels and title
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Error (True - Predicted)", fontsize=14)
    plt.title("Error (True Value - Predicted Value) per Sample", fontsize=16)
    
    # Show plot
    plt.tight_layout()
    plt.show()

# Example use case: Train the model and evaluate for 5-day prediction
layers = [8, 20, 3, 1]  
mean_error, best_weights = cross_validate(X, y_5days, layers)  # For 5-day prediction

# Plot errors as a bar chart
plot_errors_as_bars(X, y_5days, best_weights, layers)
