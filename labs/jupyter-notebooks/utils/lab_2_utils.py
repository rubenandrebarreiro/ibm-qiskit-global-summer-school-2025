# Import the Sparse Pauli Operator module from
# the Qiskit's Quantum Information module
from qiskit.quantum_info import SparsePauliOp

# Import the NumPy library
import numpy as np

# Import the Transpile module from the Qiskit library
from qiskit import transpile

# Import the Quantum Circuit module from the Qiskit library
from qiskit import QuantumCircuit

# Import the Curve Fit from the SciPy's Optimize module
from scipy.optimize import curve_fit

# Import the PyPlot module from
# the MatPlotLib library
import matplotlib.pyplot as plt

# Import the Generate Preset Pass Manager module
# from the Qiskit's Transpiler's Preset Pass Managers module
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager



# Function for a linear model for curve fitting
def linear_model(x, a, b):

    # Return a linear curve for the model
    return a * x + b


# Function for a quadratic model for curve fitting
def quadratic_model(x, a, b, c):

    # Return a quadratic curve for the model
    return a * x**2 + b * x + c
    

# Function for an exponential model for curve fitting
def exponential_model(x, a, b, c):

    # Return an exponential curve for the model
    return a * np.exp(-b * x) + c


# Function to perform Zero-Noise Extrapolation (ZNE)
# using an interpolation method for a given noisy
# scaling factors and noisy measurement results
def zne_method(method = "linear", xdata = [], ydata = []):

    # If the interpolation method is linear
    if method == "linear":
    
    	# Fit the a linear curve according to
    	# the given noisy scaling factors
    	# and noisy measurement results
        popt, _ = curve_fit(linear_model, xdata, ydata)
        
        # Compute the zero noise extrapolation
        zero_val = linear_model(0, *popt)
        
        # Define the curve fitting function
        fit_fn = linear_model

    # If the extrapolation method is quadratic
    elif method == "quadratic":
    
    	# Fit the a quadratic curve according to
    	# the given noisy scaling factors
    	# and noisy measurement results
        popt, _ = curve_fit(quadratic_model, xdata, ydata)
        
        # Compute the zero noise extrapolation
        zero_val = quadratic_model(0, *popt)
        
        # Define the curve fitting function
        fit_fn = quadratic_model
    
    # If the extrapolation method is exponential
    elif method == "exponential":
    
    	# Fit the an exponential curve according to
    	# the given noisy scaling factors
    	# and noisy measurement results
        popt, _ = curve_fit(exponential_model, xdata, ydata, p0=(1, 0.1, 0), maxfev=5000)
        
        # Compute the zero noise extrapolation
        zero_val = exponential_model(0, *popt)
        
        # Define the curve fitting function
        fit_fn = exponential_model
        
    # If the extrapolation method is none of the above
    else:
    
    	# Raise an Value Error for an unknown interpolation method
        raise ValueError(f"Unknown interpolation method '{method}'.\n"
        		 f"Use 'linear', 'quadratic', or 'exponential'.")

	
    # Return the zero noise extrapolation,
    # noisy measurement results, the extrapolation curve,
    # and the fitting function
    return zero_val, ydata, popt, fit_fn


# Function to plot the Zero-Noise Extrapolation (ZNE)
# according to the given noisy scaling factors,
# noisy measurement results, zero noise extrapolations,
# fitting function, fitting parameters, and interpolation method
def plot_zne(scales, values, zero_val, fit_fn, fit_params, method):

    # Generate a linear space of points within
    # the range of the noisy scaling factors
    x_plot = np.linspace(0, max(scales), 200)
    
    # Fit the curve according to the fitting function
    # and respective expected fitting parameters
    y_plot = fit_fn(x_plot, *fit_params)


    # Create a figure for plotting
    plt.figure(figsize = (8, 5))

    # Plot the noisy measurement results
    plt.plot(scales, values, "o", label = "Noisy Measurements")

    # Plot the fitting curve according to the extrapolation method used
    plt.plot(x_plot, y_plot, "-", label = f"{method.capitalize()} Fit")
   
   
    # Show a vertival line for the original x-axis value
    plt.axvline(0, linestyle = "--", color = "gray")
    
    # Show a horizontal line for the zero-noise
    # extrapolation estimate values
    plt.axhline(zero_val, linestyle = "--", color = "red",
                label = "Zero-Noise Extrapolation Estimate")
    
    
    # Define the label of the x-axis for
    # the plot of the Zero-Noise Extrapolation (ZNE)
    plt.xlabel("Noise Scaling Factor")
    
    # Define the label of the y-axis for
    # the plot of the Zero-Noise Extrapolation (ZNE)
    plt.ylabel("⟨Z⟩ Expectation Value")
    
    
    # Define the title for the plot of
    # the Zero-Noise Extrapolation (ZNE)
    plt.title(f"Zero-Noise Extrapolation ({method})")
    
    # Show the legend for the plot of
    # the Zero-Noise Extrapolation (ZNE)
    plt.legend()
    
    # Show the grid for the plot of
    # the Zero-Noise Extrapolation (ZNE)
    plt.grid(True)
    
    # Show the plot of the Zero-Noise Extrapolation (ZNE)
    plt.show()


# Function to plot the quantum gates' error rates
# and quantum gate counts of a Quantum Circuit executed
# in several noisy Quantum Simulators (Fake Quantum Backends)
def plot_backend_errors_and_counts(backends, errors_and_counts_list):

    # Unpack individually the quantum gates' error rates
    # and quantum gate counts from the respectively list
    ( acc_total_errors, acc_two_qubit_errors,
      acc_single_qubit_errors, acc_readout_errors,
      single_qubit_gate_counts, two_qubit_gate_counts ) = \
      np.array(errors_and_counts_list).T.tolist()
	
    # Retrieve the quantum gates' error rates
    errors = np.array( [ acc_total_errors,
            		 acc_two_qubit_errors, acc_single_qubit_errors,
            		 acc_readout_errors ] )
            		 
    # Define the list of labels for the previously
    # retrieved quantum gates' error rates
    error_labels = [ "Total Error", "Two-Qubit Error",
        	     "Single-Qubit Error", "Readout Error" ]
    
    
    # Retrieve the quantum gate counts
    counts = np.array([single_qubit_gate_counts, two_qubit_gate_counts])
    
    # Define the list of labels for
    # the previously retrieved quantum gate counts
    count_labels = ["Single-Qubit Gate Count", "Two-Qubit Gate Count"]

    
    # Transpose the quantum gates' error rates
    errors = errors.T
    
    # Transpose the previously retrieved quantum gate counts
    counts = counts.T
    	
    	
    # Generate a range of points according to
    # the number of all types of quantum gates' error rates
    x = np.arange(len(error_labels))

    # Define the width of the bars for the plot
    width = 0.2
    
    
    # Create a sub-figure for plotting
    # the quantum gates' error rates
    fig, ax = plt.subplots(figsize = (10, 5))
    
    
    # For each index of noisy Quantum Simulators (Fake Quantum Backends)
    for i in range(len(backends)):
    
    	# Plot a histogram bar according to the quantum gates' error rates
    	# extracted from the current noisy Quantum Simulator (Fake Quantum Backend)
        ax.bar(x + i * width, errors[i], width,
               label = backends[i].name)
    
    
    # Define the label for the x-axis of the plot
    ax.set_xlabel("Quantum Gate Error Rate Type")
    
    # Define the label for the y-axis of the plot
    ax.set_ylabel("Accumulated Error Rate")
    
    # Define the title of the plot
    ax.set_title("Accumulated Error Rates by\n" +\
    		 "Noisy Quantum Simulator (Fake Quantum Backend)")
    
    
    # Set the ticks for the x-axis of the plot
    ax.set_xticks(x + width)
    
    # Set the ticks for the y-axis of the plot
    ax.set_xticklabels(error_labels)
    
    
    # Show the legend of the plot
    ax.legend()
    
    # Show the final plot
    plt.show()

	
    # Plot for gate counts
    x = np.arange(len(count_labels))  # the label locations

    # Create a sub-figure for plotting
    # the quantum gate counts
    fig, ax = plt.subplots(figsize = (10, 5))
    
    
    # For each index of noisy Quantum Simulators (Fake Quantum Backends)
    for i in range(len(backends)):
    
    	# Plot a histogram bar according to the quantum gates' error rates
    	# extracted from the current noisy Quantum Simulator (Fake Quantum Backend)
        ax.bar(x + i * width, counts[i], width,
               label = backends[i].name)
    
    
    # Define the label for the x-axis of the plot
    ax.set_xlabel("Quantum Gate Type")
    
    # Define the label for the y-axis of the plot
    ax.set_ylabel("Quantum Gate Count")
    
    # Define the title of the plot
    ax.set_title("Quantum Gate Counts by\n" +\
    		 "Noisy Quantum Simulator (Fake Quantum Backend)")
    
    
    # Set the ticks for the x-axis of the plot
    ax.set_xticks(x + width)
    
    # Set the ticks for the y-axis of the plot
    ax.set_xticklabels(count_labels)
    
    
    # Show the legend of the plot
    ax.legend()
    
    # Show the final plot
    plt.show()
