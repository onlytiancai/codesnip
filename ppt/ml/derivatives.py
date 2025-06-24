import numpy as np
import matplotlib.pyplot as plt

def plot_functions_with_derivatives():
    # Create 4 separate figures, one for each function
    plt.figure(figsize=(12, 10))
    
    # Function 1: f(x) = x^2
    plt.subplot(2, 2, 1)
    x1 = np.linspace(-3, 3, 1000)
    f1 = x1**2
    f1_prime = 2*x1
    f1_double_prime = np.ones_like(x1) * 2
    
    plt.plot(x1, f1, 'b-', label='f(x) = x²')
    plt.plot(x1, f1_prime, 'r-', label='f\'(x) = 2x')
    plt.plot(x1, f1_double_prime, 'g-', label='f\'\'(x) = 2')
    plt.title('Quadratic Function and Derivatives')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    
    # Function 2: f(x) = ln(x)
    plt.subplot(2, 2, 2)
    x2 = np.linspace(0.1, 5, 1000)  # Avoid x=0 for logarithm
    f2 = np.log(x2)
    f2_prime = 1/x2
    f2_double_prime = -1/(x2**2)
    
    plt.plot(x2, f2, 'b-', label='f(x) = ln(x)')
    plt.plot(x2, f2_prime, 'r-', label='f\'(x) = 1/x')
    plt.plot(x2, f2_double_prime, 'g-', label='f\'\'(x) = -1/x²')
    plt.title('Logarithmic Function and Derivatives')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.ylim(-5, 5)  # Limit y-axis for better visualization
    plt.legend()
    
    # Function 3: f(x) = x^3
    plt.subplot(2, 2, 3)
    x3 = np.linspace(-3, 3, 1000)
    f3 = x3**3
    f3_prime = 3*(x3**2)
    f3_double_prime = 6*x3
    
    plt.plot(x3, f3, 'b-', label='f(x) = x³')
    plt.plot(x3, f3_prime, 'r-', label='f\'(x) = 3x²')
    plt.plot(x3, f3_double_prime, 'g-', label='f\'\'(x) = 6x')
    plt.title('Cubic Function and Derivatives')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    
    # Function 4: f(x) = sin(x)
    plt.subplot(2, 2, 4)
    x4 = np.linspace(-2*np.pi, 2*np.pi, 1000)
    f4 = np.sin(x4)
    f4_prime = np.cos(x4)
    f4_double_prime = -np.sin(x4)
    
    plt.plot(x4, f4, 'b-', label='f(x) = sin(x)')
    plt.plot(x4, f4_prime, 'r-', label='f\'(x) = cos(x)')
    plt.plot(x4, f4_double_prime, 'g-', label='f\'\'(x) = -sin(x)')
    plt.title('Sine Function and Derivatives')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], 
               ['-2π', '-π', '0', 'π', '2π'])
    plt.legend()
    
    plt.tight_layout()
    plt.suptitle('Functions with Their Derivatives', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9)
    plt.show()

if __name__ == "__main__":
    plot_functions_with_derivatives()