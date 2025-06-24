import numpy as np
import matplotlib.pyplot as plt

def plot_common_functions():
    # Create a figure with 2x3 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Common Mathematical Functions', fontsize=16)
    
    # Generate x values
    x = np.linspace(-5, 5, 1000)
    
    # 1. Linear function: f(x) = x
    axs[0, 0].plot(x, x, 'b-')
    axs[0, 0].set_title('Linear: f(x) = x')
    axs[0, 0].grid(True)
    axs[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 2. Quadratic function: f(x) = x²
    axs[0, 1].plot(x, x**2, 'r-')
    axs[0, 1].set_title('Quadratic: f(x) = x²')
    axs[0, 1].grid(True)
    axs[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 3. Cubic function: f(x) = x³
    axs[0, 2].plot(x, x**3, 'g-')
    axs[0, 2].set_title('Cubic: f(x) = x³')
    axs[0, 2].grid(True)
    axs[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[0, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 4. Exponential function: f(x) = e^x
    axs[1, 0].plot(x, np.exp(x), 'c-')
    axs[1, 0].set_title('Exponential: f(x) = e^x')
    axs[1, 0].grid(True)
    axs[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 0].set_ylim(-1, 20)  # Limit y-axis for better visualization
    
    # 5. Logarithmic function: f(x) = ln(x)
    x_pos = np.linspace(0.1, 5, 1000)  # Avoid x=0 for logarithm
    axs[1, 1].plot(x_pos, np.log(x_pos), 'm-')
    axs[1, 1].set_title('Logarithmic: f(x) = ln(x)')
    axs[1, 1].grid(True)
    axs[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 6. Sine function: f(x) = sin(x)
    axs[1, 2].plot(x, np.sin(x), 'y-')
    axs[1, 2].set_title('Sine: f(x) = sin(x)')
    axs[1, 2].grid(True)
    axs[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 2].set_ylim(-1.5, 1.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_common_functions()