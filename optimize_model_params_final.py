#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global parameter fitting for throttle & brake models (GPU-ready)
----------------------------------------------------------------
Loads all trips, concatenates them properly, and runs
Nelder–Mead optimization with real-time visualization.
Now includes actuator‐lag “look‐ahead”: we compute the required acceleration
τ seconds in the future as the target, and we fit throttle on throttle>0
and brake on brake>0, with no learned thresholds.
"""
import os
import random
import csv
import datetime
import glob     # For finding checkpoint files
import pickle  # For saving and loading checkpoints
import argparse  # For command line arguments
import copy     # For deep copying objects
import numpy as np
import torch
import scipy.optimize as opt
from scipy.signal import butter, filtfilt
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
plt.ion()  # Enable interactive mode for better GUI response

# Custom exception for early termination of optimization
class OptimizationTerminationException(Exception):
    """Exception raised when optimization should be terminated early"""
    pass

# ===============================
# --- INTEGRATED PLOTTING FUNCTIONS ---
# ===============================

def setup_plots():
    """Create a figure with the correct 3x2 layout"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    return fig, axes

def update_dynamic_plot(axes, spd, tp, bp, rt, rb, t, m_th, m_br, Fd, F_achieved, real_accel, achieved_accel, params=None):
    """
    Update the plot with the current simulation data in the 3x2 layout
    
    Row 1: Throttle and Brake
    Row 2: Velocity and Acceleration
    Row 3: Motor Voltage and External Forces
    
    Note: The Motor Voltage subplot will show voltage data if the following attributes are set:
        - update_dynamic_plot.Vlim: Voltage limit from the motor controller
        - update_dynamic_plot.backEMF: Back-EMF voltage from motor rotation
        - update_dynamic_plot.achieved_voltage: Applied voltage (V, limited by Vmax)
    """
    # Clear all axes first
    for row in axes:
        for ax in row:
            ax.cla()
    
    # -------------------- Row 1: Throttle and Brake --------------------
    
    # Throttle subplot (top left)
    axes[0, 0].plot(t, rt, 'b', label="Real Throttle")
    axes[0, 0].plot(t, tp, '--r', label="Sim Throttle")
    axes[0, 0].fill_between(t, 0, 100, where=m_th, facecolor='lightgreen', alpha=0.2)
    axes[0, 0].set_ylabel("Throttle %")
    axes[0, 0].set_title("Throttle Control")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend(loc="upper right")
    
    # Brake subplot (top right)
    axes[0, 1].plot(t, rb, 'g', label="Real Brake")
    axes[0, 1].plot(t, bp, '--m', label="Sim Brake")
    axes[0, 1].fill_between(t, 0, 100, where=m_br, facecolor='salmon', alpha=0.2)
    axes[0, 1].set_ylabel("Brake %")
    axes[0, 1].set_title("Brake Control")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].legend(loc="upper right")

    # -------------------- Row 2: Velocity and Acceleration --------------------
    
    # Velocity subplot (middle left)
    axes[1, 0].plot(t, spd, 'g', label="Actual Speed (m/s)")
    if achieved_accel is not None:
        speed_achieved = getattr(update_dynamic_plot, "speed_achieved", None)
        if speed_achieved is not None and len(speed_achieved) == len(t):
            axes[1, 0].plot(t, speed_achieved, '--m', label="Achieved Speed (m/s)")
    
    axes[1, 0].fill_between(t, 0, np.max(spd)*1.1, where=m_th, facecolor='lightgreen', alpha=0.2)
    axes[1, 0].fill_between(t, 0, np.max(spd)*1.1, where=m_br, facecolor='salmon', alpha=0.2)
    axes[1, 0].set_ylabel("Speed (m/s)")
    axes[1, 0].set_title("Vehicle Speed")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="upper right")
    
    # Acceleration subplot (middle right)
    axes[1, 1].plot(t, real_accel, 'g', label="Desired Acceleration")
    if achieved_accel is not None:
        axes[1, 1].plot(t, achieved_accel, '--m', label="Achieved Acceleration")
    
    axes[1, 1].fill_between(t, axes[1, 1].get_ylim()[0], axes[1, 1].get_ylim()[1], 
                         where=m_th, facecolor='lightgreen', alpha=0.2)
    axes[1, 1].fill_between(t, axes[1, 1].get_ylim()[0], axes[1, 1].get_ylim()[1], 
                         where=m_br, facecolor='salmon', alpha=0.2)
    axes[1, 1].set_ylabel("Acceleration (m/s²)")
    axes[1, 1].set_title("Vehicle Acceleration")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper right")
    
    # -------------------- Row 3: Motor Voltage and External Forces --------------------
    
    # Motor Voltage subplot (bottom left)
    # Check if we have the voltage data attributes
    has_voltage_data = (hasattr(update_dynamic_plot, "Vlim") and 
                       hasattr(update_dynamic_plot, "backEMF") and 
                       hasattr(update_dynamic_plot, "achieved_voltage"))
    
    if has_voltage_data:
        Vlim = update_dynamic_plot.Vlim
        backEMF = update_dynamic_plot.backEMF
        achieved_voltage = update_dynamic_plot.achieved_voltage
        
        # Plot voltage data
        axes[2, 0].plot(t, Vlim, 'r-', label="Vlim")
        axes[2, 0].plot(t, backEMF, 'g-', label="Back-EMF")
        axes[2, 0].plot(t, achieved_voltage, 'b-', label="Achieved Voltage")
    else:
        # Fallback to showing internal forces if voltage data is not available
        axes[2, 0].plot(t, Fd, 'k', label="Desired Internal Force")
        if F_achieved is not None:
            axes[2, 0].plot(t, F_achieved, 'b--', label="Achieved Internal Force")
    
    axes[2, 0].fill_between(t, axes[2, 0].get_ylim()[0], axes[2, 0].get_ylim()[1],
                         where=m_th, facecolor='lightgreen', alpha=0.2)
    axes[2, 0].fill_between(t, axes[2, 0].get_ylim()[0], axes[2, 0].get_ylim()[1],
                         where=m_br, facecolor='salmon', alpha=0.2)
    axes[2, 0].set_ylabel("Voltage (V)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_title("Motor Voltage")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(loc="upper right")
    
    # External Forces subplot (bottom right)
    # Calculate external forces using global parameters
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS
    Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m = GLOBAL_THROTTLE_PARAMS
    
    # Constants
    g = 9.81
    rho = 1.225
    # m is taken from GLOBAL_THROTTLE_PARAMS (learnable)
    cr1_th = 0.05
    
    # Use achieved speed if available, otherwise use original speed
    speed_to_use = getattr(update_dynamic_plot, "speed_achieved", spd)
    
    # Calculate external forces
    angle_to_use = getattr(update_dynamic_plot, "angle", np.zeros_like(t))
    Fg = m * g * np.sin(angle_to_use)  # Gravitational force along the incline
        
    # Use achieved speed for aerodynamic drag calculation
    Faero = 0.5 * rho * cd * A * speed_to_use ** 2  # Aerodynamic drag
    
    # Rolling resistance using achieved speed
    eff = 1 - np.exp(-(-np.log(1e-2)/cr1_th) * np.abs(speed_to_use))
    Fric = np.sign(speed_to_use) * cr1 * np.minimum(eff, 1.0) * m * g * np.cos(angle_to_use)
    
    # Plot each external force
    axes[2, 1].plot(t, Fg, 'r-', label="Gravitational Force")
    axes[2, 1].plot(t, Faero, 'g-', label="Aerodynamic Drag")
    axes[2, 1].plot(t, Fric, 'b-', label="Rolling Resistance")
    axes[2, 1].plot(t, Fg + Faero + Fric, 'k--', label="Total External Force")
    
    # Mark throttle and brake regions
    axes[2, 1].fill_between(t, axes[2, 1].get_ylim()[0], axes[2, 1].get_ylim()[1],
                         where=m_th, facecolor='lightgreen', alpha=0.2)
    axes[2, 1].fill_between(t, axes[2, 1].get_ylim()[0], axes[2, 1].get_ylim()[1],
                         where=m_br, facecolor='salmon', alpha=0.2)
    
    axes[2, 1].set_ylabel("Force (N)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_title("External Forces")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(loc="upper right")

def create_final_plot(data, params=None, save_path=None):
    """
    Create a final plot with the 3x2 layout using saved data
    
    Args:
        data: Dictionary containing the simulation data
        params: Model parameters for force calculations
        save_path: Path to save the figure, if None the figure is not saved
    """
    # Create a figure with 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    
    # Extract data
    t = data['time']
    spd = data['speed']
    rt = data['throttle']
    tp = data['throttle_sim']
    rb = data['brake']
    bp = data['brake_sim']
    m_th = data['throttle_mask']
    m_br = data['brake_mask']
    Fd = data['force_desired']
    F_achieved = data.get('force_achieved', None)
    speed_achieved = data.get('speed_achieved', None)
    
    # Calculate achieved acceleration if we have the force data
    achieved_accel = None
    if F_achieved is not None:
        m = GLOBAL_THROTTLE_PARAMS[-1]  # Vehicle mass (learnable)
        achieved_accel = F_achieved / m
    
    # Get real acceleration if available
    real_accel = data.get('acceleration', None)
    
    # Set speed_achieved as an attribute of update_dynamic_plot if available
    if speed_achieved is not None:
        update_dynamic_plot.speed_achieved = speed_achieved
        print(f"Setting speed_achieved in create_final_plot, shape: {speed_achieved.shape}")
    
    # Set voltage data as attributes if available
    if 'Vlim' in data and 'backEMF' in data and 'achieved_voltage' in data:
        update_dynamic_plot.Vlim = data['Vlim']
        update_dynamic_plot.backEMF = data['backEMF']
        update_dynamic_plot.achieved_voltage = data['achieved_voltage']
    
    # Update the plot with the data
    update_dynamic_plot(
        axes, spd, tp, bp, rt, rb, t, m_th, m_br, 
        Fd, F_achieved, real_accel, achieved_accel, params
    )
    
    # Overall figure adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for title if needed
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig, axes

# ---------------------------  Configuration  --------------------------- #
VEHICLE_MODEL    = 'ECentro'
VEHICLE_ID      = 'ECENTRO_HA_03'  # e.g., "NIRO_SJ_03" – set to None to disable filtering
DATA_FILE        = os.path.join('processed_data', VEHICLE_MODEL, VEHICLE_ID, 'all_trips_data.pt')
RESULTS_DIR      = 'optimization_results'  # Root directory for storing optimization results
RUN_TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR          = os.path.join(RESULTS_DIR, RUN_TIMESTAMP)  # Directory for this run's results
LOG_FILE         = os.path.join(RUN_DIR, 'optimization_log.csv')  # CSV log file
LOSS_PLOT_FILE   = os.path.join(RUN_DIR, 'loss_curves.png')  # Loss curves plot file
STATS_PLOT_FILE  = os.path.join(RUN_DIR, 'statistics.png')  # Statistics plot file

# Checkpoint configuration
CHECKPOINT_DIR   = os.path.join(RUN_DIR, 'checkpoints')  # Directory for checkpoint files
CHECKPOINT_FREQ  = 20  # Save checkpoint every CHECKPOINT_FREQ iterations if better than previous best
CHECKPOINT_FILE_THROTTLE = os.path.join(CHECKPOINT_DIR, 'checkpoint_throttle.pkl')  # Throttle optimization checkpoint
CHECKPOINT_FILE_BRAKE = os.path.join(CHECKPOINT_DIR, 'checkpoint_brake.pkl')  # Brake optimization checkpoint
RESUME_FROM_CHECKPOINT = True  # Set to True to resume from checkpoint
RESUME_CHECKPOINT_FILE = None  # Set this to the path of the checkpoint file to resume from
device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ACCEL_FILTER = False
np.random.seed(42)
random.seed(42)

# Create results directory structure
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create directory for checkpoints

# Initialize tracking structures for optimization history
throttle_losses = []
brake_losses = []
optimization_steps = {'throttle': 0, 'brake': 0}
checkpoint_counter = {'throttle': 0, 'brake': 0}
last_checkpoint_loss = {'throttle': np.inf, 'brake': np.inf}

# GLOBAL PARAMETER LISTS - COMPLETELY SEPARATE FOR THROTTLE AND BRAKE
# Throttle parameters: [Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m]
# Added mass `m` as a learnable throttle parameter (initial value 5200.0 kg)
GLOBAL_THROTTLE_PARAMS = [0.96, 0.21, 0.79, 0.044, 6.41, 0.0065, 7.8, 0.35, 0.50, 0.20, 750.0, 5200.0]
# Brake parameters: [b1, b2, b3, b4, tau_br, c0, c1, c2, c3]
# First 4 are inverse polynomial coefficients (force -> brake%)
# Next is the brake actuator lag 
# Last 4 are forward polynomial coefficients (brake% -> force)
# Initial forward polynomial coefficients set to make 100% brake ~10,000N force
GLOBAL_BRAKE_PARAMS = [1.65e-12, 1.121e-08, -5.843e-13, 8.013e-18, 0.20, 0.0, 10000.0, 0.0, 0.0]

print("=== INITIALIZED GLOBAL PARAMETER LISTS ===")
print(f"GLOBAL_THROTTLE_PARAMS: {GLOBAL_THROTTLE_PARAMS}")
print(f"GLOBAL_BRAKE_PARAMS: {GLOBAL_BRAKE_PARAMS}")
print("Throttle params address:", id(GLOBAL_THROTTLE_PARAMS))
print("Brake params address:", id(GLOBAL_BRAKE_PARAMS))

# Control flags for optimization
optimization_paused = False
finish_current_phase = False
display_best_params = False
current_optimization_phase = None  # 'throttle' or 'brake'
# control_fig removed - buttons are now in the main figure

# ---------------------------  Load & Prepare Data --------------------- #
print(f"Loading trips from {DATA_FILE}…")
ds        = torch.load(DATA_FILE, map_location=device, weights_only=False)
metadata  = ds.pop('metadata')
trip_ids  = list(ds.keys())
print(f"Found {len(trip_ids)} segments: {trip_ids}\n")

keys      = ['time','speed','acceleration','angle','throttle','brake']
lists_np  = {k: [] for k in keys}
lists_t   = {k: [] for k in keys}

for tid in trip_ids:
    entry = ds[tid]
    for k in keys:
        arr = entry.get(k, np.zeros_like(entry['time']))
        lists_np[k].append(arr.astype(np.float32))
        lists_t[k].append(torch.tensor(arr, dtype=torch.float32, device=device))

time_np_list = lists_np['time']
lengths      = [len(t) for t in time_np_list]
durations    = [t[-1] if len(t)>0 else 0.0 for t in time_np_list]
offset_time  = np.cumsum([0.0] + durations[:-1])
offset_idx   = np.cumsum([0] + lengths[:-1])
flat_time    = np.concatenate([t + offset_time[i] for i,t in enumerate(time_np_list)])

all_arrays = {k: torch.cat(lists_t[k], dim=0) for k in keys}

if torch.all(all_arrays['acceleration'] == 0):
    accs = []
    for i in range(len(trip_ids)):
        s = all_arrays['speed'][offset_idx[i]:offset_idx[i]+lengths[i]].cpu().numpy()
        t = time_np_list[i]
        if len(s) < 2:
            accs.append(torch.zeros(lengths[i], device=device))
            continue
        if USE_ACCEL_FILTER:
            fs = 1.0 / np.mean(np.diff(t))
            b, a = butter(3, 0.5/(0.5*fs), btype='low')
            sf   = filtfilt(b, a, s)
            ac   = np.gradient(sf, t)
        else:
            ac   = np.gradient(s, t)
        accs.append(torch.tensor(ac, dtype=torch.float32, device=device))
    all_arrays['acceleration'] = torch.cat(accs, dim=0)

valid_mask    = torch.zeros_like(all_arrays['throttle'], dtype=torch.bool)
for i,L in enumerate(lengths):
    valid_mask[offset_idx[i]:offset_idx[i]+L] = True
throttle_mask = (all_arrays['throttle'] > 0) & valid_mask
brake_mask    = (all_arrays['throttle'] == 0) & (all_arrays['brake'] > 0.004) & valid_mask

# # quick histograms
# plt.figure(figsize=(6,4))
# plt.hist(all_arrays['brake'][brake_mask].cpu().numpy(), bins=30)
# plt.title("Brake (non-zero)")
# plt.show()

# plt.figure(figsize=(6,4))
# plt.hist(all_arrays['throttle'][throttle_mask].cpu().numpy(), bins=30)
# plt.title("Throttle (non-zero)")
# plt.show()

# ==============================
# --- Model / command helper ---
# ==============================
def inverse_brake_poly(F, b1, b2, b3, b4):
    return b1*F + b2*F**2 + b3*F**3 + b4*F**4



def evaluate_forward_brake_polynomial(brake_pct, coeffs):
    """
    Evaluate the forward brake polynomial at given brake percentage(s)
    
    Args:
        brake_pct: Brake percentage (0-100) or array
        coeffs: Polynomial coefficients [c0, c1, c2, ...]
        
    Returns:
        force: Brake force in Newtons
    """
    brake_decimal = np.asarray(brake_pct) / 100.0
    force = np.zeros_like(brake_decimal)
    
    for i, coeff in enumerate(coeffs):
        force += coeff * (brake_decimal ** i)
    
    return np.maximum(force, 0.0)  # Ensure non-negative forces



def evaluate_forward_brake_polynomial(brake_pct, c0, c1, c2, c3):
    """
    Evaluate the forward brake polynomial at given brake percentage(s)
    
    Args:
        brake_pct: Brake percentage (0-100) or array
        c0, c1, c2, c3: Polynomial coefficients
        
    Returns:
        force: Brake force in Newtons
    """
    brake_decimal = np.asarray(brake_pct) / 100.0
    force = c0 + c1*brake_decimal + c2*brake_decimal**2 + c3*brake_decimal**3
    return np.maximum(force, 0.0)  # Ensure non-negative forces
    """
    Solve for brake force given brake percentage using Newton's method.
    
    We need to solve: brake_pct = b1*F + b2*F^2 + b3*F^3 + b4*F^4
    Rearranged as: f(F) = b1*F + b2*F^2 + b3*F^3 + b4*F^4 - brake_pct = 0
    
    Args:
        brake_pct: Brake percentage (0-100)
        b1, b2, b3, b4: Brake polynomial coefficients (inverse model)
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance
        debug: Print debug information
        
    Returns:
        F_brake: Brake force (N)
    """
    # Performance tracking - count total calls
    if not hasattr(forward_brake_poly_newton, 'call_count'):
        forward_brake_poly_newton.call_count = 0
        forward_brake_poly_newton.last_report = 0
        print("🔍 NEWTON PERFORMANCE: Initializing call counter")
    
    forward_brake_poly_newton.call_count += 1
    
    # Report every 100 calls to track performance impact
    if forward_brake_poly_newton.call_count - forward_brake_poly_newton.last_report >= 100:
        import traceback
        caller_info = traceback.extract_stack()[-2]  # Get the calling function
        caller_function = caller_info.name
        caller_line = caller_info.lineno
        print(f"🔍 NEWTON PERFORMANCE: Called {forward_brake_poly_newton.call_count} times total")
        print(f"   Last called from: {caller_function}() at line {caller_line}")
        forward_brake_poly_newton.last_report = forward_brake_poly_newton.call_count
    
    if brake_pct <= 0:
        if debug:
            print(f"DEBUG: brake_pct={brake_pct} <= 0, returning 0.0")
        return 0.0
        
    # Convert brake percentage to decimal (0-1 range)
    # The polynomial likely expects decimal values, not percentages
    brake_decimal = brake_pct / 100.0
    
    if debug:
        print(f"\nDEBUG Newton's method:")
        print(f"  Input brake_pct={brake_pct}%, brake_decimal={brake_decimal}")
        print(f"  Brake params: b1={b1:.6e}, b2={b2:.6e}, b3={b3:.6e}, b4={b4:.6e}")
    
    # Smart initial guess based on parameter magnitudes
    # Try multiple starting points since the polynomial might be ill-conditioned
    initial_guesses = []
    
    # Guess 1: Linear approximation if b1 is significant
    if abs(b1) > 1e-20:
        F_linear = brake_decimal / b1
        if F_linear > 0 and F_linear < 1e8:  # Reasonable force range
            initial_guesses.append(F_linear)
    
    # Guess 2: Typical vehicle brake forces
    initial_guesses.extend([0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0])

    # Guess 3: If quadratic term dominates
    if abs(b2) > 1e-15:
        F_quad = np.sqrt(abs(brake_decimal / b2)) if b2 > 0 else 1000.0
        if F_quad > 0 and F_quad < 1e8:
            initial_guesses.append(F_quad)
    
    if debug:
        print(f"  Trying initial guesses: {initial_guesses[:3]}")
    
    # Try each initial guess
    for guess_idx, F_init in enumerate(initial_guesses):
        F = F_init
        
        for i in range(max_iter):
            # f(F) = b1*F + b2*F^2 + b3*F^3 + b4*F^4 - brake_decimal
            f_val = b1*F + b2*F*F + b3*F*F*F + b4*F*F*F*F - brake_decimal
            
            # f'(F) = b1 + 2*b2*F + 3*b3*F^2 + 4*b4*F^3
            f_prime = b1 + 2*b2*F + 3*b3*F*F + 4*b4*F*F*F
            
            if debug and guess_idx == 0 and i < 3:  # Only print first guess, first few iterations
                print(f"  Guess {guess_idx}, Iter {i}: F={F:.2f}, f_val={f_val:.6e}, f_prime={f_prime:.6e}")
            
            # Check for zero or very small derivative
            if abs(f_prime) < 1e-20:
                if debug and guess_idx == 0:
                    print(f"  Zero/tiny derivative at iter {i}, trying next guess")
                break
                
            # Newton update with step limiting
            step = f_val / f_prime
            F_new = F - step
            
            # Keep force positive and reasonable
            F_new = max(F_new, 0.1)
            F_new = min(F_new, 1e7)  # Limit to reasonable brake force
            
            # Check convergence
            if abs(F_new - F) < tol or abs(f_val) < tol:
                if debug:
                    print(f"  Converged with guess {guess_idx} at iter {i}: F_final={F_new:.2f} N")
                    # Verify the result
                    check_val = b1*F_new + b2*F_new*F_new + b3*F_new*F_new*F_new + b4*F_new*F_new*F_new*F_new
                    print(f"  Verification: brake_poly({F_new:.2f}) = {check_val:.6f}, target = {brake_decimal:.6f}, error = {abs(check_val - brake_decimal):.2e}")
                return F_new
                
            F = F_new
    
    # If all Newton attempts fail, try a simple numerical search
    if debug:
        print(f"  Newton's method failed, trying numerical search...")
    
    # Simple brute force search over reasonable force range
    for F_test in np.linspace(0, 10000, 1000):
        poly_val = b1*F_test + b2*F_test*F_test + b3*F_test*F_test*F_test + b4*F_test*F_test*F_test*F_test
        if abs(poly_val - brake_decimal) < 0.001:  # Close enough
            if debug:
                print(f"  Numerical search found: F={F_test:.2f}N")
            return F_test
    
    # If everything fails, return 0
    if debug:
        print(f"  All methods failed, returning 0.0")
    return 0.0

def smooth_exp(s, th, eps=1e-2):
    k = -np.log(eps)/th
    return 1 - np.exp(-k*s)

def compute_internal_force(speed, angle, throttle, brake, time_vals=None):
    """
    Computes the internal force generated by the throttle/brake plus Fload,
    using the inverse of the mapping in compute_commands (i.e., throttle/brake to force).
    Uses global parameter lists only.
    
    NOW INCLUDES ACTUATOR DELAYS: Commands are delayed by tau_th and tau_br
    
    Arguments:
        speed: torch tensor of speed
        angle: torch tensor of road angle
        throttle: torch tensor of throttle values (%)
        brake: torch tensor of brake values (%)
        time_vals: torch tensor or numpy array of time values (needed for delays)
    Returns:
        F_internal: torch tensor of internal force (N)
    """
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS
    
    # Extract parameters from global lists (now includes mass `m` as last element)
    Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m = GLOBAL_THROTTLE_PARAMS
    b1, b2, b3, b4, tau_br, c0, c1, c2, c3 = GLOBAL_BRAKE_PARAMS  # Updated to include forward coefficients

    # Apply actuator delays if time information is provided
    if time_vals is not None:
        # Convert time to numpy for interpolation
        if isinstance(time_vals, torch.Tensor):
            time_np = time_vals.cpu().numpy()
        else:
            time_np = np.array(time_vals)
        
        # Apply delays: commands were issued tau seconds EARLIER to achieve current effect
        # So we look BACK in time by tau to find the command that affects current time
        time_delayed_th = time_np - tau_th
        time_delayed_br = time_np - tau_br
        
        # Interpolate to get delayed commands
        throttle_np = throttle.cpu().numpy() if isinstance(throttle, torch.Tensor) else np.array(throttle)
        brake_np = brake.cpu().numpy() if isinstance(brake, torch.Tensor) else np.array(brake)
        
        # Extrapolate with boundary values for times outside range
        throttle_delayed = np.interp(time_delayed_th, time_np, throttle_np)
        brake_delayed = np.interp(time_delayed_br, time_np, brake_np)
        
        # Convert back to tensors
        throttle = torch.tensor(throttle_delayed, dtype=throttle.dtype, device=throttle.device)
        brake = torch.tensor(brake_delayed, dtype=brake.dtype, device=brake.device)

    g = 9.81; rho = 1.225; cr1_th = 0.05
    
    # Ensure Vmax has a sensible value
    if Vmax < 100.0:
        Vmax = 750.0

    # static loads
    Fg    = m * g * torch.sin(angle)
    Faero = 0.5 * rho * cd * A * speed ** 2
    eff   = torch.tensor(smooth_exp(torch.abs(speed).cpu().numpy(), cr1_th),
                         device=speed.device)
    
    # Updated friction model - only rolling friction
    fric_coef = cr1 * torch.clamp(eff, max=1.0)
    additional_force = 0
    
    Fric  = torch.sign(speed) * fric_coef * m * g * torch.cos(angle)
    Fload = Faero + Fric + Fg + additional_force

    # --- Throttle: invert the Vlim/Vmax computation ---
    # throttle is in percent [0,100], so scale to [0,1]
    throttle_frac = throttle / 100.0
    # Use Vmax from parameters
    Vlim = throttle_frac * Vmax

    omega = speed * G / re
    # Tmotor = kt/Ra * (Vlim - kv*omega) - Jr*omega_dot
    # But we don't have omega_dot (acceleration), so ignore Jr*omega_dot for inversion
    # Twheel = G * Tmotor
    # Fdesired_th = Twheel / re
    # So, invert for Fdesired_th:
    Tmotor = kt / Ra * (Vlim - kv * omega)
    Twheel = G * Tmotor
    Fdesired_th = Twheel / re

    # --- Brake: use forward polynomial directly ---
    # Convert brake percentages to numpy for calculation
    brake_np = brake.cpu().numpy()
    # Calculate brake forces using forward polynomial
    Fbrake_forces = evaluate_forward_brake_polynomial(brake_np, c0, c1, c2, c3)
    # Convert back to tensor
    Fbrake_req_pos = torch.tensor(Fbrake_forces, dtype=torch.float32, device=brake.device)
    # Calculate desired brake force (negated because it opposes motion)
    Fdesired_br = -Fbrake_req_pos + Fload

    # Combine: use throttle when throttle > 0, brake when brake > 0
    F_internal = torch.where(throttle > 0, Fdesired_th, torch.where(brake > 0, Fdesired_br, Fload))
    
    return F_internal


def compute_delay_aware_comparison(speed, accel, angle, throttle_actual, brake_actual, time_vals):
    """
    Compute delay-aware predictions and achieved values for fair comparison.
    Accounts for the phase shift between predictions (look-ahead) and achieved (look-back).
    
    DELAY CONSISTENCY STRATEGY:
    - compute_commands(): Uses look-ahead acceleration (accel at t+tau) to predict commands at t
    - compute_internal_force(): Uses look-back commands (commands at t-tau) to compute forces at t  
    - This function: Time-aligns predictions by shifting them backward by tau for fair comparison
    
    This ensures that all loss calculations and statistics compare:
    "Predicted command for time t" vs "Actual command at time t"
    instead of the inconsistent:
    "Predicted command based on t+tau acceleration" vs "Actual command at time t"
    
    Args:
        speed: Vehicle speed tensor
        accel: Vehicle acceleration tensor  
        angle: Road angle tensor
        throttle_actual: Actual throttle commands tensor
        brake_actual: Actual brake commands tensor
        time_vals: Time values for temporal alignment
        
    Returns:
        thr_pred_aligned: Time-aligned throttle predictions
        br_pred_aligned: Time-aligned brake predictions  
        Fd: Desired force (unchanged)
        F_achieved: Achieved force with delays applied
        Vlim: Voltage limit (unchanged)
        backEMF: Back EMF (unchanged)
    """
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS
    Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m = GLOBAL_THROTTLE_PARAMS
    
    # Handle both old and new format of brake parameters
    if len(GLOBAL_BRAKE_PARAMS) == 9:  # New format with forward coefficients
        b1, b2, b3, b4, tau_br, c0, c1, c2, c3 = GLOBAL_BRAKE_PARAMS
    elif len(GLOBAL_BRAKE_PARAMS) == 5:  # Old format without forward coefficients
        b1, b2, b3, b4, tau_br = GLOBAL_BRAKE_PARAMS
        # Set default forward coefficients
        c0, c1, c2, c3 = 0.0, 500.0, 0.0, 0.0
        # Update global params to include forward coefficients
        GLOBAL_BRAKE_PARAMS = [b1, b2, b3, b4, tau_br, c0, c1, c2, c3]
    else:
        raise ValueError(f"Unexpected length of GLOBAL_BRAKE_PARAMS: {len(GLOBAL_BRAKE_PARAMS)}")
    
    # STEP 1: Compute predictions using look-ahead acceleration (as normal)
    # This already applies delays internally via time interpolation
    thr_pred, br_pred, Fd, Vlim, backEMF = compute_commands(
        speed, accel, angle, time_vals
    )
    
    # STEP 2: Compute achieved forces using delayed commands 
    # This applies delays to the commands before computing forces
    F_achieved = compute_internal_force(
        speed, angle, throttle_actual, brake_actual, time_vals
    )
    
    # STEP 3: TIME-ALIGN the predictions to account for phase shifts
    # The predictions are based on acceleration at time t+tau (look-ahead)
    # The achieved forces are based on commands at time t-tau (look-back)
    # We shift predictions backward by tau to align them temporally with achieved
    
    if isinstance(time_vals, torch.Tensor):
        time_np = time_vals.cpu().numpy()
    else:
        time_np = np.array(time_vals)
    
    # Shift predictions backward by their respective delays
    # This aligns "what we predicted for time t" with "what we achieved at time t"
    
    # For throttle: shift predictions back by tau_th
    time_shifted_th = time_np - tau_th
    thr_pred_aligned = torch.tensor(
        np.interp(time_shifted_th, time_np, thr_pred.cpu().numpy()),
        dtype=thr_pred.dtype, device=thr_pred.device
    )
    
    # For brake: shift predictions back by tau_br  
    time_shifted_br = time_np - tau_br
    br_pred_aligned = torch.tensor(
        np.interp(time_shifted_br, time_np, br_pred.cpu().numpy()),
        dtype=br_pred.dtype, device=br_pred.device
    )
    
    return thr_pred_aligned, br_pred_aligned, Fd, F_achieved, Vlim, backEMF


def compute_commands(speed, accel, angle, time_vals=None, skip_masks=False):
    """
    Compute throttle and brake commands based on the vehicle model
    Uses global parameter lists only.
    
    Args:
        speed: Vehicle speed tensor
        accel: Vehicle acceleration tensor
        angle: Road angle tensor
        time_vals: Time values for interpolation (optional)
        skip_masks: If True, return raw commands without applying throttle/brake masks
    
    Returns:
        thr: Throttle command
        brk: Brake command
        Fdesired_th: Desired force
        Vlim: Voltage limit
        backEMF: Back-EMF voltage
    """
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS

    # Extract parameters from global lists (including learnable mass `m`)
    Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m = GLOBAL_THROTTLE_PARAMS
    b1, b2, b3, b4, tau_br, c0, c1, c2, c3 = GLOBAL_BRAKE_PARAMS  # Updated to include forward coefficients

    g = 9.81; rho = 1.225; cr1_th = 0.05

    # static loads
    Fg    = m*g*torch.sin(angle)
    Faero = 0.5*rho*cd*A*speed**2
    eff   = torch.tensor(smooth_exp(torch.abs(speed).cpu().numpy(), cr1_th),
                         device=speed.device)
    
    # Updated friction model - only rolling friction
    fric_coef = cr1*torch.clamp(eff, max=1.0)
    additional_force = 0
    Fric  = torch.sign(speed) * fric_coef * m*g*torch.cos(angle)
    Fload = Faero + Fric + Fg + additional_force

    # look-ahead accel
    acc_np = accel.cpu().numpy()
    
    # Use provided time values or global flat_time
    if time_vals is not None:
        time_to_use = time_vals
    else:
        time_to_use = flat_time
    
    # Calculate time shifts for look-ahead accel
    ft_th = time_to_use + tau_th
    ft_br = time_to_use + tau_br
    
    # Interpolate acceleration values at shifted times
    acc_f_th = np.interp(ft_th, time_to_use, acc_np)
    acc_f_br = np.interp(ft_br, time_to_use, acc_np)
    
    acc_th = torch.tensor(acc_f_th, dtype=torch.float32, device=speed.device)
    acc_br = torch.tensor(acc_f_br, dtype=torch.float32, device=speed.device)

    # desired forces
    omega     = speed * G/re
    omega_dot = acc_th * G/re  # Using acceleration for omega_dot calculation
    Facc_th   = m * acc_th
        
    # Include Jr*omega_dot effect in the force calculation for greater accuracy
    # The force required to produce omega_dot includes motor inertia effects
    # F = m*a + Fload = Twheel/re = G*Tmotor/re
    # Tmotor = kt/Ra*(Vlim - kv*omega) - Jr*omega_dot
    # So the force including Jr*omega_dot is:
    # F = G*kt/(Ra*re)*(Vlim - kv*omega) - G*Jr*omega_dot/re + Fload
    # For Fdesired_th, we need the force that must be applied to achieve acc_th
    Fdesired_th = Facc_th + Fload
    
    Facc_br   = m * acc_br
    Fdesired_br = Facc_br + Fload
    
    Fbrake_req  = -Fdesired_br
    # 1) clamp negative
    Fbrake_req_pos = torch.clamp(Fbrake_req, min=0.0)

    # raw commands
    br_val = torch.clamp(
        inverse_brake_poly(Fbrake_req_pos, b1, b2, b3, b4),
        0.0, 1.0
    ) * 100.0

    Twheel = Fdesired_th * re
    Tmotor_load = Twheel / G  # Motor torque to handle the load
    
    # Include Jr*omega_dot in the motor torque calculation
    Tmotor_with_inertia = Tmotor_load + Jr*omega_dot  # Add inertia effect
    
    # Calculate required voltage considering both load and inertia
    Vreq   = (Ra/kt)*Tmotor_with_inertia + kv*omega
    Vlim   = torch.minimum(torch.tensor(Vmax, device=Vreq.device), Vreq)
    th_val = torch.clamp(Vlim/Vmax, 0.0, 1.0) * 100.0
    
    # Calculate back-EMF for the motor voltage plot
    backEMF = kv * omega

    # --- switch by actual driver mask, no thresholds ---
    if skip_masks:
        # Skip applying masks for plotting purposes
        thr = th_val
        brk = br_val
    else:
        # Apply masks for regular optimization
        thr = torch.where(throttle_mask, th_val, torch.zeros_like(th_val))
        brk = torch.where(brake_mask,  br_val, torch.zeros_like(br_val))

    # Return additional voltage information for plotting
    return thr, brk, Fdesired_th, Vlim, backEMF

# ===============================
# --- Optimization & Plotting ---
# ===============================
best_loss   = np.inf
best_params = None
is_plot     = True

# physical + brake + 2 lags + static friction params = 16 params
param_names = [
  "Ra","kt","kv","Jr","A","cr1","G","re","cd","tau_th","Vmax",
  "b1","b2","b3","b4","tau_br"
]

# File to save the final plot
FINAL_PLOT_FILE = os.path.join(RUN_DIR, 'final_plot.png')  # Final plot file

# update_control_panel_status function removed as control panel is no longer used

# setup_control_buttons function removed as buttons are now directly in the main figure

def log_params_to_csv(params, loss_total, loss_throttle, loss_brake, phase='all'):
    """
    Log the best parameters to a CSV file in the results directory
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(LOG_FILE, 'a', newline='') as csvfile:
        fieldnames = ['phase', 'timestamp', 'loss_total', 'loss_throttle', 'loss_brake'] + param_names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Create a dictionary with all parameter values
        row = {
            'phase': phase,
            'timestamp': timestamp,
            'loss_total': loss_total,
            'loss_throttle': loss_throttle,
            'loss_brake': loss_brake
        }
        
        # Add all parameter values
        for name, value in zip(param_names, params):
            row[name] = value
            
        writer.writerow(row)
        print(f"Parameters logged to {LOG_FILE}")

def save_checkpoint(params, optimizer_result, optimization_steps, loss_history, best_loss, phase='throttle'):
    """
    Save optimization checkpoint to allow resuming from this point
    
    Args:
        params: Current best parameters
        optimizer_result: The scipy.optimize result object
        optimization_steps: Dictionary of optimization step counts
        loss_history: List of losses
        best_loss: Current best loss value
        phase: 'throttle' or 'brake'
    """
    checkpoint_file = CHECKPOINT_FILE_THROTTLE if phase == 'throttle' else CHECKPOINT_FILE_BRAKE
    
    # Create checkpoint data dictionary
    checkpoint_data = {
        'params': params,
        'optimizer_result': optimizer_result,
        'optimization_steps': optimization_steps,
        'loss_history': loss_history,
        'best_loss': best_loss,
        'phase': phase,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to file
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved to {checkpoint_file}")
    return

def find_latest_checkpoint(phase=None):
    """
    Find the latest checkpoint file from all previous runs
    
    Args:
        phase: Optional filter for 'throttle' or 'brake' phase checkpoints
    
    Returns:
        Path to the latest checkpoint file, or None if no checkpoints found
    """
    # Find all optimization result directories
    results_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), 
                          key=os.path.getmtime, reverse=True)
    
    # Search for checkpoint files in each results directory
    for results_dir in results_dirs:
        checkpoint_dir = os.path.join(results_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            continue
        
        # Get all checkpoint files in this directory
        if phase == 'throttle':
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*throttle*.pkl'))
        elif phase == 'brake':
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*brake*.pkl'))
        else:
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pkl'))
        
        # Sort by modification time (most recent first)
        checkpoint_files = sorted(checkpoint_files, key=os.path.getmtime, reverse=True)
        
        if checkpoint_files:
            return checkpoint_files[0]  # Return the most recent checkpoint
    
    return None

def ensure_within_bounds(params, bounds):
    """
    Ensure that parameters are within the specified bounds
    
    Args:
        params: List of parameter values
        bounds: List of (lower, upper) bounds for each parameter
    
    Returns:
        List of parameters with values clamped to be within bounds
    """
    if len(params) != len(bounds):
        print(f"Warning: Parameter length ({len(params)}) doesn't match bounds length ({len(bounds)})")
        # Truncate to shorter length
        length = min(len(params), len(bounds))
        params = params[:length]
        bounds = bounds[:length]
    
    clamped_params = []
    for i, (param, (lower, upper)) in enumerate(zip(params, bounds)):
        if lower is not None and param < lower:
            print(f"Warning: Parameter {i} ({param}) is below lower bound ({lower}), clamping")
            clamped_params.append(lower)
        elif upper is not None and param > upper:
            print(f"Warning: Parameter {i} ({param}) is above upper bound ({upper}), clamping")
            clamped_params.append(upper)
        else:
            clamped_params.append(param)
    
    return clamped_params

def load_checkpoint(checkpoint_file):
    """
    Load optimization checkpoint from file
    
    Args:
        checkpoint_file: Path to the checkpoint file
    
    Returns:
        Dictionary with checkpoint data if successful, None otherwise
    """
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"Loaded checkpoint from {checkpoint_file}")
        print(f"Checkpoint timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
        print(f"Optimization phase: {checkpoint_data.get('phase', 'unknown')}")
        print(f"Best loss: {checkpoint_data.get('best_loss', 'unknown')}")
        
        return checkpoint_data
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None

# Create a 3x2 grid for plots with the specific layout using the setup_plots function:
# Row 1: Throttle (0,0) and Brake (0,1)
# Row 2: Velocity (1,0) and Acceleration (1,1)
# Row 3: Internal Forces (2,0) and External Forces (2,1)
fig, axes = setup_plots()

# Add small control buttons directly to the main figure
btn_width = 0.08
btn_height = 0.04
btn_y = 0.03  # Position buttons at the bottom
btn_pause_ax = plt.axes([0.30, btn_y, btn_width, btn_height], frameon=True)
btn_resume_ax = plt.axes([0.45, btn_y, btn_width, btn_height], frameon=True)
btn_finish_ax = plt.axes([0.60, btn_y, btn_width, btn_height], frameon=True)

# Create the buttons with smaller fontsize
btn_pause = Button(btn_pause_ax, 'Pause')
btn_resume = Button(btn_resume_ax, 'Resume')
btn_finish = Button(btn_finish_ax, 'Finish')

# Manually set the label font size
for button in [btn_pause, btn_resume, btn_finish]:
    button.label.set_fontsize(7)

# Define callback functions
def pause_callback(event):
    global optimization_paused
    optimization_paused = True
    print("Optimization paused. Press 'Resume' to continue.")

def resume_callback(event):
    global optimization_paused
    optimization_paused = False
    print("Optimization resumed.")

def finish_callback(event):
    global finish_current_phase, optimization_paused
    finish_current_phase = True
    # Also make sure we're not paused, as that could cause the loop to never check the finish flag
    optimization_paused = False
    print("Finishing current optimization phase... (please wait)")
    # Force immediate redraw to ensure UI is responsive
    plt.draw()

# Assign callbacks to buttons
btn_pause.on_clicked(pause_callback)
btn_resume.on_clicked(resume_callback)
btn_finish.on_clicked(finish_callback)

# Add a small label above the buttons
plt.figtext(0.5, 0.08, "Optimization Controls:", ha="center", fontsize=9, weight='bold')

# Store references to prevent garbage collection
fig._buttons = (btn_pause, btn_resume, btn_finish)

# Make sure the main plot window is positioned nicely
try:
    fig.canvas.manager.set_window_title("Vehicle Model Optimization")
    fig.canvas.manager.window.wm_geometry("+50+50")
except:
    # Some backends don't support this
    pass

def plot_loss_curves():
    """Plot the loss curves for throttle and brake optimization phases"""
    print("Generating loss curves plot...")
    plt_fig = plt.figure(figsize=(12, 10))
    
    # Add status info at the top
    status_text = f"Optimization Phase: {current_optimization_phase if current_optimization_phase else 'Not started'}"
    if optimization_paused:
        status_text += " (PAUSED)"
    status_text += f"\nBest Loss: {best_loss:.4f}" if best_loss != np.inf else "\nBest Loss: None yet"
    plt.figtext(0.5, 0.95, status_text, ha="center", fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.subplot(2, 1, 1)
    if throttle_losses:
        plt.plot(range(1, len(throttle_losses) + 1), throttle_losses)
        title = 'Throttle Optimization Loss'
        if current_optimization_phase == 'throttle':
            title += ' (Current Phase)'
        plt.title(title)
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.yscale('log')  # Often better to visualize on log scale
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No throttle optimization data available', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.subplot(2, 1, 2)
    if brake_losses:
        plt.plot(range(1, len(brake_losses) + 1), brake_losses)
        title = 'Brake Optimization Loss'
        if current_optimization_phase == 'brake':
            title += ' (Current Phase)'
        plt.title(title)
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.yscale('log')  # Often better to visualize on log scale
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No brake optimization data available', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_FILE)
    print(f"Loss curves saved to {LOSS_PLOT_FILE}")

def compute_and_plot_statistics(final_params):
    """Compute and plot delay-aware statistics for the optimized model"""
    print("Computing delay-aware statistics and generating plots...")
    
    # Convert params to tensor for use with model functions
    params_tensor = torch.tensor(final_params, device=device)
    
    # DELAY-AWARE COMPUTATION for statistics
    thr_pred_aligned, br_pred_aligned, Fd, F_achieved, Vlim, backEMF = compute_delay_aware_comparison(
        all_arrays['speed'], all_arrays['acceleration'], all_arrays['angle'], 
        all_arrays['throttle'], all_arrays['brake'], flat_time
    )
    
    # Calculate acceleration from force (F = ma) - now using delay-aware forces
    m = GLOBAL_THROTTLE_PARAMS[-1]  # Mass from the model (learnable)
    accel_achieved = F_achieved.detach().cpu().numpy() / m
    accel_desired = all_arrays['acceleration'].cpu().numpy()
    
    # Convert other relevant arrays to numpy for analysis
    speed = all_arrays['speed'].cpu().numpy()
    throttle = all_arrays['throttle'].cpu().numpy()
    brake = all_arrays['brake'].cpu().numpy()
    
    # Create a large figure with multiple subplots for different statistics
    plt.figure(figsize=(20, 16))
    
    # 1. Acceleration Error vs Speed and Desired Acceleration
    plt.subplot(2, 2, 1)
    
    # Create bins for speed and acceleration (increased resolution)
    speed_bins = np.linspace(0, 30, 50)  # 0-30 m/s in 50 bins
    accel_bins = np.linspace(-3, 3, 50)  # -3 to +3 m/s² in 50 bins
    
    # Use valid mask to filter out invalid data points
    valid_idx = valid_mask.cpu().numpy()
    
    # Create a 2D histogram of errors
    accel_error = accel_desired - accel_achieved
    
    # Create a 2D histogram using numpy
    H, xedges, yedges = np.histogram2d(
        speed[valid_idx], 
        accel_desired[valid_idx], 
        bins=[speed_bins, accel_bins], 
        weights=np.abs(accel_error[valid_idx])
    )
    
    # Count points in each bin for averaging
    counts, _, _ = np.histogram2d(
        speed[valid_idx], 
        accel_desired[valid_idx], 
        bins=[speed_bins, accel_bins]
    )
    
    # Avoid division by zero
    counts = np.maximum(counts, 1)
    
    # Average error in each bin
    avg_error = H / counts
    
    # Plot as a heatmap
    plt.imshow(avg_error.T, origin='lower', aspect='auto', 
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    plt.colorbar(label='Avg Acceleration Error (m/s²)')
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Desired Acceleration (m/s²)')
    plt.title('Acceleration Error vs Speed and Desired Acceleration')
    
    # 2. Throttle Model Error
    plt.subplot(2, 2, 2)
    # Only consider points with throttle activity - now using aligned predictions
    throttle_idx = throttle_mask.cpu().numpy()
    if np.any(throttle_idx):
        throttle_error = throttle[throttle_idx] - thr_pred_aligned.cpu().numpy()[throttle_idx]
        plt.hist(throttle_error, bins=50, alpha=0.7)
        plt.xlabel('Throttle Prediction Error (%) - Delay Aligned')
        plt.ylabel('Count')
        plt.title('Throttle Model Error Distribution (Delay Corrected)')
    else:
        plt.text(0.5, 0.5, 'No throttle data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 3. Brake Model Error
    plt.subplot(2, 2, 3)
    brake_idx = brake_mask.cpu().numpy()
    if np.any(brake_idx):
        brake_error = brake[brake_idx] - br_pred_aligned.cpu().numpy()[brake_idx]
        plt.hist(brake_error, bins=50, alpha=0.7)
        plt.xlabel('Brake Prediction Error (%) - Delay Aligned')
        plt.ylabel('Count')
        plt.title('Brake Model Error Distribution (Delay Corrected)')
    else:
        plt.text(0.5, 0.5, 'No brake data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # 4. Force Model Error vs Speed with Polynomial Fit
    plt.subplot(2, 2, 4)
    force_desired = all_arrays['acceleration'].cpu().numpy() * m  # F = ma
    # Get just the force error, not including the load forces
    Facc_th = all_arrays['acceleration'].cpu().numpy() * m
    force_error = Facc_th - F_achieved.detach().cpu().numpy()  # Convert tensor to numpy
    
    # Calculate average error by speed bin with more bins and ensuring speed=0 is included
    speed_bins = np.linspace(0, 30, 50)  # Increased from 30 to 50 bins for higher resolution
    avg_errors = []
    bin_centers = []
    
    # Make sure we include speed=0 in calculations
    zero_speed_mask = (speed == 0) & valid_idx
    if np.sum(zero_speed_mask) > 0:
        avg_errors.append(np.mean(np.abs(force_error[zero_speed_mask])))
        bin_centers.append(0.0)
    
    for i in range(len(speed_bins)-1):
        mask = (speed > speed_bins[i]) & (speed <= speed_bins[i+1]) & valid_idx
        if np.sum(mask) > 0:
            avg_errors.append(np.mean(np.abs(force_error[mask])))
            bin_centers.append((speed_bins[i] + speed_bins[i+1]) / 2)
    
    # Convert to numpy arrays for polynomial fitting
    bin_centers_array = np.array(bin_centers)
    avg_errors_array = np.array(avg_errors)
    
    # RANSAC polynomial fit (robust to outliers)
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    X = bin_centers_array.reshape(-1, 1)
    y = avg_errors_array
    poly = PolynomialFeatures(degree=3)
    ransac = make_pipeline(poly, RANSACRegressor())
    ransac.fit(X, y)
    x_fit = np.linspace(0, max(bin_centers_array), 100).reshape(-1, 1)
    y_fit = ransac.predict(x_fit)
    
    # Plot data points and RANSAC polynomial fit
    plt.plot(bin_centers_array, avg_errors_array, 'o', label='Data Points')
    plt.plot(x_fit, y_fit, 'r-', label='RANSAC Poly Fit')
    
    # Extract polynomial coefficients for display
    coefs = ransac.named_steps['ransacregressor'].estimator_.coef_ if hasattr(ransac.named_steps['ransacregressor'].estimator_, 'coef_') else None
    intercept = ransac.named_steps['ransacregressor'].estimator_.intercept_ if hasattr(ransac.named_steps['ransacregressor'].estimator_, 'intercept_') else None
    if coefs is not None and intercept is not None:
        poly_eq = f"f(v) = {coefs[3]:.6f}·v³ + {coefs[2]:.6f}·v² + {coefs[1]:.6f}·v + {intercept:.6f}"
        plt.text(0.05, 0.95, poly_eq, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top', horizontalalignment='left')
    
    plt.xlabel('Vehicle Speed (m/s)')
    plt.ylabel('Avg Force Error (N)')
    plt.title('Force Model Error vs Speed with Polynomial Fit')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(STATS_PLOT_FILE)
    print(f"Statistics plots saved to {STATS_PLOT_FILE}")

def update_plot(th_pred, br_pred, Fd, trip_i=None, params_to_use=None):
    """
    Update the plot with the current simulation data using the integrated 3x2 layout
    
    Parameters:
        th_pred: Predicted throttle values
        br_pred: Predicted brake values
        Fd: Desired internal force
        trip_i: Trip index to plot (optional)
        params_to_use: Model parameters to use for plotting (optional)
    """
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS
    
    print(f"=== UPDATE_PLOT CALLED ===")
    print(f"[update_plot] Reading GLOBAL_THROTTLE_PARAMS: {GLOBAL_THROTTLE_PARAMS[:5]}...")
    print(f"[update_plot] Reading GLOBAL_BRAKE_PARAMS: {GLOBAL_BRAKE_PARAMS}")
    
    # pick which trip to plot
    if trip_i is None:
        # find the longest trip that has throttle/brake activity
        longest_length = 0
        longest_idx = 0
        
        # Make sure we have at least one trip
        if len(trip_ids) == 0:
            print("Warning: No trips found. Using dummy data for plot.")
            i = 0
            off = 0
            L = len(all_arrays['speed']) if 'speed' in all_arrays else 0
        else:
            # Find the best trip to plot
            for i, (off, L) in enumerate(zip(offset_idx, lengths)):
                if i >= len(offset_idx) or off + L > len(all_arrays['throttle']):
                    continue  # Skip invalid indices
                    
                rt = all_arrays['throttle'][off:off+L].cpu().numpy()
                rb = all_arrays['brake'][off:off+L].cpu().numpy()
                
                if ((rt > 0).any() or (rb > 0.004).any()) and L > longest_length:
                    longest_length = L
                    longest_idx = i
                    
            # Use the longest trip found, or default to the first trip
            if longest_length > 0:
                i = longest_idx
            else:
                i = 0 if len(trip_ids) > 0 else 0
            
            # Double check that the index is valid
            if i >= len(offset_idx) or i >= len(lengths):
                print(f"Warning: Invalid trip index {i}. Using first trip instead.")
                i = 0
    else:
        # Use the provided trip index, but validate it
        i = trip_i
        if i >= len(offset_idx) or i >= len(lengths):
            print(f"Warning: Invalid trip index {i}. Using first trip instead.")
            i = 0 if len(trip_ids) > 0 else 0
    
    # Get offsets and lengths, with proper bounds checking
    if len(offset_idx) > i and len(lengths) > i:
        off, L = offset_idx[i], lengths[i]
    else:
        print("Warning: Using fallback values for offset and length")
        off, L = 0, min(100, len(all_arrays['speed']) if 'speed' in all_arrays else 0)

    # slice once
    t    = flat_time[off:off+L]
    rt   = all_arrays['throttle'][off:off+L].cpu().numpy()
    rb   = all_arrays['brake'   ][off:off+L].cpu().numpy()
    tp   = th_pred[off:off+L]
    bp   = br_pred[off:off+L]
    m_th = throttle_mask[off:off+L].cpu().numpy()
    m_br = brake_mask[off:off+L].cpu().numpy()
    spd  = all_arrays['speed'][off:off+L].cpu().numpy()
    
    # Get real acceleration data
    real_accel = all_arrays['acceleration'][off:off+L].cpu().numpy()
    
    # Calculate F_achieved using the global parameters and current internal force function
    F_achieved = compute_internal_force(
        all_arrays['speed'][off:off+L],
        all_arrays['angle'][off:off+L], 
        torch.tensor(rt, device=device),
        torch.tensor(rb, device=device),
        t  # Add time parameter for delay processing
    ).detach().cpu().numpy()
    
    # =================================================================
    # RECURSIVE FORWARD MODEL COMPUTATION
    # =================================================================
    # Compute speed, acceleration, force, and voltage using recursive integration
    # instead of the inverse approach used during optimization
    
    # Extract parameters from global lists (including learnable mass `m`)
    Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m = GLOBAL_THROTTLE_PARAMS
    
    # Handle both old and new format of brake parameters
    if len(GLOBAL_BRAKE_PARAMS) == 9:  # New format with forward coefficients
        b1, b2, b3, b4, tau_br, c0, c1, c2, c3 = GLOBAL_BRAKE_PARAMS
    elif len(GLOBAL_BRAKE_PARAMS) == 5:  # Old format without forward coefficients
        b1, b2, b3, b4, tau_br = GLOBAL_BRAKE_PARAMS
        # Set default forward coefficients
        c0, c1, c2, c3 = 0.0, 500.0, 0.0, 0.0
        # Update global params to include forward coefficients
        GLOBAL_BRAKE_PARAMS = [b1, b2, b3, b4, tau_br, c0, c1, c2, c3]
    
    # Pre-compute all brake forces using the forward polynomial
    import time
    fit_start_time = time.time()
    brake_forces_batch = evaluate_forward_brake_polynomial(rb, c0, c1, c2, c3)
    fit_end_time = time.time()
    print(f"🔧 Direct polynomial evaluation took {fit_end_time - fit_start_time:.3f}s for {len(rb)} points")
    print(f"🔧 Batch brake force range: {np.min(brake_forces_batch):.1f}N to {np.max(brake_forces_batch):.1f}N")
    
    # DEBUG: Test brake polynomial behavior
    print(f"\n=== BRAKE POLYNOMIAL DEBUG ===")
    print(f"Inverse params: b1={b1:.6e}, b2={b2:.6e}, b3={b3:.6e}, b4={b4:.6e}")
    print(f"Forward params: c0={c0:.6e}, c1={c1:.6e}, c2={c2:.6e}, c3={c3:.6e}")
    
    # Test with various brake percentages, full range
    print("\nTesting full brake range (0-100%):")
    test_points = [0, 1, 5, 10, 25, 50, 75, 100]
    print("Brake% -> Force -> Brake% (roundtrip)")
    for test_brake_pct in test_points:
        # Forward: brake% -> force (direct polynomial evaluation)
        F_forward = evaluate_forward_brake_polynomial(test_brake_pct, c0, c1, c2, c3)
        # Inverse: force -> brake% (what the fitted model does)
        brake_pct_check = inverse_brake_poly(F_forward, b1, b2, b3, b4) * 100
        print(f"  {test_brake_pct:3d}% -> {F_forward:7.1f}N -> {brake_pct_check:5.2f}% (error: {abs(test_brake_pct - brake_pct_check):6.2f}%)")
    
    # Test the reverse direction
    print("\nTesting from force to brake% to force (expected use case):")
    test_forces = [0, 100, 500, 1000, 2500, 5000, 7500, 10000]
    print("Force -> Brake% -> Force (roundtrip)")
    for F_test in test_forces:
        # Convert force to brake%
        brake_pct = inverse_brake_poly(F_test, b1, b2, b3, b4) * 100
        # Convert brake% back to force using the forward polynomial
        F_roundtrip = evaluate_forward_brake_polynomial(brake_pct, c0, c1, c2, c3)
        print(f"  {F_test:6.1f}N -> {brake_pct:5.2f}% -> {F_roundtrip:7.1f}N (error: {abs(F_test - F_roundtrip):6.1f}N)")
    
    print(f"\nActual brake data range: {np.min(rb):.1f}% to {np.max(rb):.1f}%")
    print("=== END BRAKE DEBUG ===")
    
    # Look at actual brake data range
    brake_data_min, brake_data_max = np.min(rb), np.max(rb)
    print(f"Actual brake data range: {brake_data_min:.1f}% to {brake_data_max:.1f}%")
    print("=== END BRAKE DEBUG ===\n")
    
    n = len(t)
    if n <= 1:
        # Handle edge case with insufficient data
        speed_achieved = spd.copy()
        achieved_accel = real_accel.copy()
        force_achieved = F_achieved.copy()
        # Create simple voltage data for edge case
        Vlim = np.array([rt[0] / 100.0 * Vmax]) if len(rt) > 0 else np.array([0.0])
        backEMF = np.array([kv * spd[0] * G / re]) if len(spd) > 0 else np.array([0.0])
        V_applied = np.minimum(Vlim, Vmax)  # Applied voltage (limited by Vmax)
        achieved_voltage = V_applied  # Applied voltage, not resistive drop
        voltage_data = {'Vlim': Vlim, 'backEMF': backEMF, 'achieved_voltage': achieved_voltage}
    else:
        # Vehicle parameters (mass `m` comes from GLOBAL_THROTTLE_PARAMS)
        g = 9.81    # Gravitational acceleration
        rho = 1.225 # Air density

        # Get angle data
        angle = all_arrays['angle'][off:off+L].cpu().numpy()

        # Create time steps for integration
        dt = np.diff(t)
        dt = np.append(dt[0], dt) if len(dt) > 0 else np.array([0.1])  # Prepend first dt

        # Initialize arrays
        speed_achieved = np.zeros(n)
        achieved_accel = np.zeros(n)
        force_achieved = np.zeros(n)
        omega_achieved = np.zeros(n)
        Vlim_achieved = np.zeros(n)

        # Initialize Butterworth IIR filter for F_net (minimal delay filtering)
        # Cutoff based on maximum reasonable acceleration (2 m/s²) * mass
        max_reasonable_accel = 2.0  # m/s² (conservative for this heavy vehicle)
        F_cutoff_force = max_reasonable_accel * m  # mass `m` is learnable

        # Design Butterworth filter for minimal delay
        avg_dt = np.mean(dt) if len(dt) > 0 else 0.01  # Average time step
        fs = 1.0 / avg_dt  # Sampling frequency

        # Cutoff frequency: high enough to preserve dynamics, low enough to filter noise
        # Use frequency corresponding to reasonable acceleration changes
        cutoff_freq = 3.0  # Hz - allows rapid acceleration changes but filters noise
        nyquist_freq = fs / 2.0
        normalized_cutoff = cutoff_freq / nyquist_freq

        # Ensure cutoff is within valid range
        normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.95)
        
        # Design 2nd order Butterworth filter (minimal delay while effective)
        filter_order = 2
        butter_b, butter_a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
        
        # Debug: Print filter parameters
        print(f"Butterworth Filter: fs={fs:.1f}Hz, cutoff={cutoff_freq:.1f}Hz, order={filter_order}")
        print(f"Filter coefficients - b: {butter_b}, a: {butter_a}")
        
        # Initialize filter state variables for real-time filtering
        # For 2nd order filter, we need to track 2 previous inputs and 2 previous outputs
        filter_x_hist = np.zeros(filter_order + 1)  # Input history [x[n], x[n-1], x[n-2]]
        filter_y_hist = np.zeros(filter_order + 1)  # Output history [y[n], y[n-1], y[n-2]]
        
        def apply_butterworth_filter(x_new, b_coeffs, a_coeffs, x_hist, y_hist):
            """
            Apply Butterworth filter in real-time using difference equation.
            For 2nd order: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
            """
            # Shift history and add new input
            x_hist[1:] = x_hist[:-1]  # Shift right
            x_hist[0] = x_new
            
            # Compute filtered output using difference equation
            y_new = 0.0
            for i in range(len(b_coeffs)):
                y_new += b_coeffs[i] * x_hist[i]
            for i in range(1, len(a_coeffs)):
                y_new -= a_coeffs[i] * y_hist[i-1]
            
            # Shift output history and add new output
            y_hist[1:] = y_hist[:-1]  # Shift right
            y_hist[0] = y_new
            
            return y_new
        backEMF_achieved = np.zeros(n)
        voltage_achieved = np.zeros(n)
        
        # Start with initial measured speed
        speed_achieved[0] = spd[0]
        omega_achieved[0] = speed_achieved[0] * G / re
        
        # COMPUTE ACTUATOR DELAYS IN TIME STEPS
        # Calculate average time step for delay conversion
        avg_dt = np.mean(dt) if len(dt) > 0 else 0.01
        delay_steps_th = max(1, int(tau_th / avg_dt))  # Throttle delay in time steps
        delay_steps_br = max(1, int(tau_br / avg_dt))  # Brake delay in time steps
        
        print(f"🔧 Actuator delays: tau_th={tau_th:.3f}s ({delay_steps_th} steps), tau_br={tau_br:.3f}s ({delay_steps_br} steps)")
        
        # Recursive forward integration
        for i in range(1, n):
            # Previous state
            v = speed_achieved[i-1]
            w = omega_achieved[i-1]
            
            # DELAYED CONTROL INPUTS (physically realistic)
            # Apply actuator delays: commands take time to produce force
            throttle_idx = max(0, i-1-delay_steps_th)  # Throttle with delay
            brake_idx = max(0, i-1-delay_steps_br)     # Brake with delay
            
            throttle_pct = rt[throttle_idx]
            brake_pct = rb[brake_idx]
            
            if i < 5:  # Debug first few steps
                print(f"Step {i}: current_idx={i-1}, throttle_idx={throttle_idx}, brake_idx={brake_idx}")
                print(f"  Delays: throttle={throttle_pct:.1f}%, brake={brake_pct:.1f}%")
            
            # === THROTTLE FORCE ===
            # Calculate back-EMF from motor speed (not vehicle speed during braking)
            back_emf = kv * w
            
            # Calculate voltage and current based on operating mode
            # Motor is decoupled whenever throttle is zero (pure braking OR coasting)
            if throttle_pct == 0:
                # When throttle is zero: motor is decoupled, input voltage is 0
                Vlim = 0  # No input voltage when not throttling
                V = 0
                I = -back_emf / Ra if Ra > 0 else 0  # Current only from back-EMF (can be negative)
                T = kt * I  # Motor torque (can be negative, opposing motion)
                F_throttle = 0  # Motor contributes NO force to vehicle when decoupled
            else:
                # During throttling: normal motor operation
                Vlim = (throttle_pct / 100.0) * Vmax
                V = min(Vlim, Vmax)
                I = (V - back_emf) / Ra if Ra > 0 else 0
                T = kt * I
                
                # Apply throttle force
                F_throttle = T * G / re
                
            # === BRAKE FORCE ===
            # Use pre-computed batch brake force (FAST, replaces Newton's method)
            F_brake_newton = brake_forces_batch[i]
            
            # Only apply brake force if brake_pct > 0
            if brake_pct > 0:
                F_brake = max(F_brake_newton, 0.0)  # Ensure non-negative
                if i < 5:  # Debug first few steps
                    print(f"Step {i}: brake_pct={brake_pct:.1f}%, F_brake_batch={F_brake_newton:.2f}N, F_brake_applied={F_brake:.2f}N")
            else:
                F_brake = 0
                
            # === EXTERNAL FORCES ===
            # Gravitational force along incline
            F_g = m * g * np.sin(angle[i-1])
            
            # Aerodynamic drag
            F_aero = 0.5 * rho * cd * A * v * v * (1 if v >= 0 else -1)
            
            # Rolling resistance with speed-dependent effect
            # Use separate threshold parameter for efficiency calculation
            cr1_th = 0.05  # Speed threshold parameter for rolling resistance efficiency
            eff = 1 - np.exp(-(-np.log(1e-2) / cr1_th) * np.abs(v))
            # Use cr1 from global parameters as the actual rolling resistance coefficient
            F_roll = np.sign(v) * cr1 * min(eff, 1.0) * m * g * np.cos(angle[i-1]) if v != 0 else 0
            
            # Total external force
            F_ext = F_g + F_aero + F_roll
            
            # === NET DYNAMICS ===
            # Net force (unfiltered)
            F_net_raw = F_throttle - F_brake - F_ext
            
            # Apply Butterworth filter to F_net for numerical stability and realistic dynamics
            # Filters out unrealistic high-frequency force oscillations while maintaining
            # responsiveness (minimal delay). Uses 2nd order Butterworth low-pass filter.
            F_net = apply_butterworth_filter(F_net_raw, butter_b, butter_a, filter_x_hist, filter_y_hist)
            
            # Debug: Show filtering effect for first few steps
            if i < 5:
                print(f"  Step {i}: F_net_raw={F_net_raw:.1f}N → F_net_filtered={F_net:.1f}N (Δ={F_net_raw-F_net:.1f}N)")
            
            # Acceleration from filtered force
            a = F_net / m
            
            # Debug force balance for first few steps with braking
            if i < 5 and brake_pct > 0:
                print(f"  Force balance: F_throttle={F_throttle:.2f}N, F_brake={F_brake:.2f}N, F_ext={F_ext:.2f}N")
                print(f"  F_net={F_net:.2f}N, accel={a:.3f}m/s², speed: {v:.2f}→{max(0, v + a * dt[i]):.2f}m/s")
            
            # Update speed using forward Euler integration
            speed_achieved[i] = max(0, v + a * dt[i])  # Prevent negative speed
            
            # Update motor angular velocity - DECOUPLE when throttle is zero
            if throttle_pct == 0:
                # When throttle is zero (braking OR coasting): motor is decoupled from drivetrain
                # Motor speed evolves based on its own electrical dynamics only
                # Motor torque equation: T_motor = J_r * omega_dot + T_load
                # When decoupled: T_load = 0, so omega_dot = T_motor / J_r
                T_motor = kt * I  # Motor torque from electrical dynamics
                omega_dot_motor = T_motor / Jr if Jr > 0 else 0
                
                # Update motor speed independently
                omega_achieved[i] = max(0, w + omega_dot_motor * dt[i])  # Motor can't go negative
                
                if i < 5:  # Debug decoupling
                    mode = "braking" if brake_pct > 0 else "coasting"
                    print(f"  DECOUPLED ({mode}): throttle=0%, motor_omega={omega_achieved[i]:.2f}, vehicle_speed={speed_achieved[i]:.2f}")
            else:
                # During throttling: motor is coupled to drivetrain
                omega_achieved[i] = speed_achieved[i] * G / re
            
            # Store results for this step
            achieved_accel[i-1] = a  # Store acceleration from previous step
            force_achieved[i-1] = F_net  # Store net force (includes external forces)
            
            # Store voltage information
            Vlim_achieved[i-1] = Vlim
            backEMF_achieved[i-1] = back_emf
            voltage_achieved[i-1] = V  # Applied voltage (not resistive drop)
            
            # Debug braking decoupling for first few steps
            if i < 5 and throttle_pct == 0:
                mode = "braking" if brake_pct > 0 else "coasting"
                print(f"  MOTOR DECOUPLING ({mode}): V={V:.2f}V, back_emf={back_emf:.2f}V, applied_voltage={V:.2f}V")
                print(f"  Motor current={I:.3f}A, Motor torque={T:.2f}Nm, Vehicle force={F_throttle:.2f}N")
        
        # Calculate final step values
        if n > 1:
            # Calculate final acceleration from filtered force (consistent with main loop)
            # Don't override with finite difference - use the filtered force approach
            final_accel = force_achieved[-1] / m  # Use filtered force for final acceleration
            achieved_accel[-1] = final_accel
            
            # For the last point, use the same approach as the previous point
            v = speed_achieved[-1]
            w = omega_achieved[-1]
            
            # Control inputs for final step
            throttle_pct = rt[-1]
            brake_pct = rb[-1]
            
            # Calculate throttle force with proper decoupling logic
            back_emf = kv * w
            
            if throttle_pct == 0:
                # When throttle is zero: motor decoupled
                Vlim = 0
                V = 0  
                I = -back_emf / Ra if Ra > 0 else 0
                F_throttle = 0  # No vehicle force from motor when decoupled
            else:
                # During throttling: normal operation
                Vlim = (throttle_pct / 100.0) * Vmax
                V = min(Vlim, Vmax)
                I = (V - back_emf) / Ra if Ra > 0 else 0
                T = kt * I
                
                F_throttle = T * G / re
                
            # Calculate brake force for final step using pre-computed batch values
            F_brake_newton = brake_forces_batch[-1]  # Use last element for final step
            
            if brake_pct > 0:
                F_brake = max(F_brake_newton, 0.0)
            else:
                F_brake = 0
                
            # Calculate external forces for final step (same as in main loop)
            F_g = m * g * np.sin(angle[-1])
            F_aero = 0.5 * rho * cd * A * v * v * (1 if v >= 0 else -1)
            cr1_th = 0.05  # Speed threshold parameter for rolling resistance efficiency
            eff = 1 - np.exp(-(-np.log(1e-2) / cr1_th) * np.abs(v))
            F_roll = np.sign(v) * cr1 * min(eff, 1.0) * m * g * np.cos(angle[-1]) if v != 0 else 0
            F_ext = F_g + F_aero + F_roll
            
            # Calculate net force for final step (unfiltered)
            F_net_final_raw = F_throttle - F_brake - F_ext
            
            # Apply same Butterworth filter to final F_net for consistency
            F_net_final = apply_butterworth_filter(F_net_final_raw, butter_b, butter_a, filter_x_hist, filter_y_hist)
            force_achieved[-1] = F_net_final  # Store filtered net force
            
            # Store final voltage information
            Vlim_achieved[-1] = Vlim
            backEMF_achieved[-1] = back_emf
            voltage_achieved[-1] = V  # Applied voltage (not resistive drop)
        
        # Package voltage data
        voltage_data = {
            'Vlim': Vlim_achieved,
            'backEMF': backEMF_achieved, 
            'achieved_voltage': voltage_achieved
        }
    
    # Store results for plotting
    update_dynamic_plot.speed_achieved = speed_achieved
    
    # Override the original calculated values with recursive results
    F_achieved = force_achieved
    # achieved_accel already contains filtered values from recursive computation
    
    # Store angle for external forces calculation
    update_dynamic_plot.angle = all_arrays['angle'][off:off+L].cpu().numpy()
    
    # Use voltage data from recursive computation
    update_dynamic_plot.Vlim = voltage_data['Vlim']
    update_dynamic_plot.backEMF = voltage_data['backEMF']
    update_dynamic_plot.achieved_voltage = voltage_data['achieved_voltage']

    # Use the update_dynamic_plot function
    update_dynamic_plot(
        axes, spd, tp, bp, rt, rb, t, m_th, m_br, 
        Fd[off:off+L], F_achieved, real_accel, achieved_accel, None
    )
    
    # Add title with trip information - with bounds checking
    if 0 <= i < len(trip_ids) and i < len(durations) and i < len(lengths):
        plt.suptitle(f"Trip {i+1}/{len(trip_ids)}, Duration: {durations[i]:.1f}s, Length: {lengths[i]} samples", 
                    fontsize=12, y=0.995)
    else:
        plt.suptitle(f"Vehicle Model Optimization", fontsize=12, y=0.995)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.12)  # Adjust for the title and make room for buttons
    plt.pause(0.01)

    # Store the last plotted data for saving the final plot
    data_dict = {
        'time': t,
        'speed': spd,
        'speed_achieved': speed_achieved,
        'throttle_real': rt,
        'throttle_pred': tp,
        'brake_real': rb,
        'brake_pred': bp,
        'force_desired': Fd[off:off+L],
        'force_achieved': F_achieved,
        'throttle_mask': m_th,
        'brake_mask': m_br,
        'trip_idx': i if 0 <= i < len(trip_ids) else 0,
        'Vlim': voltage_data['Vlim'],
        'backEMF': voltage_data['backEMF'],
        'achieved_voltage': voltage_data['achieved_voltage'],
        'real_acceleration': real_accel,
        'achieved_accel': achieved_accel
    }
        
    update_plot.last_plot_data = data_dict




# ===============================
# --- SEPARATE LOSS FUNCTIONS ---
# ===============================

def throttle_loss_fn(x):
    """
    THROTTLE-ONLY loss function - completely separate from brake optimization
    Updates GLOBAL_THROTTLE_PARAMS only, reads GLOBAL_BRAKE_PARAMS only
    """
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS
    global throttle_losses, optimization_steps, current_optimization_phase
    global best_loss, is_plot
    
    current_optimization_phase = 'throttle'
    
    # Handle pause button - wait until resumed
    if optimization_paused:
        plt.pause(0.1)  # Give time for UI updates
        while optimization_paused and not finish_current_phase:
            plt.pause(0.1)  # Wait for resume button or finish button
    
    # Print finish message only once per phase
    if finish_current_phase:
        if not hasattr(throttle_loss_fn, "finish_printed") or not throttle_loss_fn.finish_printed:
            print(f"Finishing throttle optimization phase as requested...")
            throttle_loss_fn.finish_printed = True
        # Force the optimizer to exit by raising an exception
        raise OptimizationTerminationException("Throttle optimization terminated by user")
    else:
        throttle_loss_fn.finish_printed = False
    
    # UPDATE GLOBAL THROTTLE PARAMETERS ONLY
    # x should be [Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m]
    if len(x) == 12:
        GLOBAL_THROTTLE_PARAMS[:] = list(x)  # Update in place to maintain reference
    else:
        return 1e10
    
    # DELAY-AWARE COMPUTATION for fair comparison
    thr_pred_aligned, br_pred_aligned, Fd, F_achieved, Vlim, backEMF = compute_delay_aware_comparison(
        all_arrays['speed'], all_arrays['acceleration'], all_arrays['angle'], 
        all_arrays['throttle'], all_arrays['brake'], flat_time
    )
    
    # Only compute throttle loss using temporally aligned predictions
    t_np = all_arrays['throttle'].cpu().numpy()
    m_t  = throttle_mask.cpu().numpy()
    
    if np.sum(m_t) == 0:
        return 1e10
        
    # Use aligned predictions for fair comparison
    Lth = np.sum((t_np[m_t] - thr_pred_aligned.cpu().numpy()[m_t])**2)
    
    # Track optimization steps
    optimization_steps['throttle'] += 1
    
    # Track losses for plotting
    throttle_losses.append(Lth)
    
    # Update best loss tracking
    if Lth < best_loss:
        improvement = 0
        if best_loss != np.inf:
            improvement = (best_loss - Lth) / best_loss * 100
        best_loss = Lth
        
        print(f"NEW BEST THROTTLE LOSS: {Lth:.2f}")
        if improvement > 0:
            print(f"Improvement: {improvement:.2f}%")
        
        # Log the current parameters
        log_params_to_csv(GLOBAL_THROTTLE_PARAMS + GLOBAL_BRAKE_PARAMS, Lth, Lth, 0.0, 'throttle')
    
    # Plotting (if enabled) - use original predictions for visual consistency
    if is_plot:
        thr_pred_orig, br_pred_orig, _, _, _ = compute_commands(
            all_arrays['speed'], all_arrays['acceleration'], all_arrays['angle'], flat_time
        )
        update_plot(thr_pred_orig.detach().cpu().numpy(),
                   br_pred_orig.detach().cpu().numpy(),
                   Fd.detach().cpu().numpy(),
                   trip_i=None,
                   params_to_use=None)  # Use global params via compute_commands
    
    return Lth


def brake_loss_fn(x):
    """
    BRAKE-ONLY loss function - completely separate from throttle optimization
    Updates GLOBAL_BRAKE_PARAMS only, reads GLOBAL_THROTTLE_PARAMS only
    Now includes optimization of both inverse and forward brake polynomials
    """
    global GLOBAL_THROTTLE_PARAMS, GLOBAL_BRAKE_PARAMS
    global brake_losses, optimization_steps, current_optimization_phase
    global best_loss, is_plot
    
    current_optimization_phase = 'brake'
    
    # Handle pause button - wait until resumed
    if optimization_paused:
        plt.pause(0.1)  # Give time for UI updates
        while optimization_paused and not finish_current_phase:
            plt.pause(0.1)  # Wait for resume button or finish button
    
    # Print finish message only once per phase
    if finish_current_phase:
        if not hasattr(brake_loss_fn, "finish_printed") or not brake_loss_fn.finish_printed:
            print(f"Finishing brake optimization phase as requested...")
            brake_loss_fn.finish_printed = True
        # Force the optimizer to exit by raising an exception
        raise OptimizationTerminationException("Brake optimization terminated by user")
    else:
        brake_loss_fn.finish_printed = False
    
    # UPDATE GLOBAL BRAKE PARAMETERS ONLY
    # x should be [b1, b2, b3, b4, tau_br, c0, c1, c2, c3]
    if len(x) == 9:  # Updated to include forward polynomial coefficients
        GLOBAL_BRAKE_PARAMS[:] = list(x)  # Update in place to maintain reference
    else:
        return 1e10
    
    # DELAY-AWARE COMPUTATION for fair comparison
    thr_pred_aligned, br_pred_aligned, Fd, F_achieved, Vlim, backEMF = compute_delay_aware_comparison(
        all_arrays['speed'], all_arrays['acceleration'], all_arrays['angle'], 
        all_arrays['throttle'], all_arrays['brake'], flat_time
    )
    
    # Only compute brake loss using temporally aligned predictions
    b_np = all_arrays['brake'].cpu().numpy()
    m_b  = brake_mask.cpu().numpy()
    
    if np.sum(m_b) == 0:
        return 1e10
    
    # Calculate brake command loss using aligned predictions
    Lbr_cmd = np.sum((b_np[m_b] - br_pred_aligned.cpu().numpy()[m_b])**2)
    
    # Extract parameters for consistency check
    b1, b2, b3, b4, tau_br, c0, c1, c2, c3 = GLOBAL_BRAKE_PARAMS
    
    # Calculate consistency loss between inverse and forward polynomials
    # Sample points across the brake range
    brake_samples = np.linspace(0.01, 1.0, 40)  # brake % in decimal form, more samples
    
    # Forward: brake% -> force
    forces_fwd = c0 + c1*brake_samples + c2*brake_samples**2 + c3*brake_samples**3
    forces_fwd = np.maximum(forces_fwd, 0.0)  # Non-negative forces
    
    # Inverse: force -> brake%
    brake_inv = b1*forces_fwd + b2*forces_fwd**2 + b3*forces_fwd**3 + b4*forces_fwd**4
    
    # Consistency loss (should map back to original brake values)
    Lbr_consistency = np.sum((brake_samples - brake_inv)**2)
    
    # Add realistic force scale penalty
    # Ensure brake forces are in a realistic range (e.g., 50% brake should be around 5000N)
    realistic_force_targets = np.array([0.0, 1000.0, 2500.0, 5000.0, 8000.0]) 
    brake_test_points = np.array([0.0, 0.1, 0.25, 0.5, 0.75])
    forces_test = c0 + c1*brake_test_points + c2*brake_test_points**2 + c3*brake_test_points**3
    forces_test = np.maximum(forces_test, 0.0)
    Lbr_force_scale = np.sum((forces_test - realistic_force_targets)**2) / 1e6  # Scale down to match other losses
    
    # Combine losses with weighting
    # Higher weight on command accuracy, moderate weight on consistency, small weight on force scale
    Lbr = Lbr_cmd + 0.5 * Lbr_consistency + 0.2 * Lbr_force_scale
    
    # Track optimization steps
    optimization_steps['brake'] += 1
    
    # Track losses for plotting
    brake_losses.append(Lbr)
    
    # Update best loss tracking
    if Lbr < best_loss:
        improvement = 0
        if best_loss != np.inf:
            improvement = (best_loss - Lbr) / best_loss * 100
        best_loss = Lbr
        
        print(f"NEW BEST BRAKE LOSS: {Lbr:.2f} (Command: {Lbr_cmd:.2f}, Consistency: {Lbr_consistency:.2f}, Force Scale: {Lbr_force_scale:.2f})")
        if improvement > 0:
            print(f"Improvement: {improvement:.2f}%")
        
        # Report the forces generated by each brake percentage for debugging
        test_percentages = [0, 10, 25, 50, 75, 100]
        test_decimals = [p/100.0 for p in test_percentages]
        test_forces = c0 + c1*np.array(test_decimals) + c2*np.array(test_decimals)**2 + c3*np.array(test_decimals)**3
        print("Forward polynomial force mapping:")
        for i, p in enumerate(test_percentages):
            print(f"  {p}% brake -> {test_forces[i]:.1f}N force")
        
        # Log the current parameters
        log_params_to_csv(GLOBAL_THROTTLE_PARAMS + GLOBAL_BRAKE_PARAMS, Lbr, 0.0, Lbr, 'brake')
    
    # Plotting (if enabled) - use original predictions for visual consistency
    if is_plot:
        thr_pred_orig, br_pred_orig, _, _, _ = compute_commands(
            all_arrays['speed'], all_arrays['acceleration'], all_arrays['angle'], flat_time
        )
        update_plot(thr_pred_orig.detach().cpu().numpy(),
                   br_pred_orig.detach().cpu().numpy(),
                   Fd.detach().cpu().numpy(),
                   trip_i=None,
                   params_to_use=None)  # Use global params via compute_commands
    
    return Lbr

if __name__=="__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize vehicle model parameters')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume optimization from a checkpoint (uses latest if --checkpoint not specified)')
    parser.add_argument('--checkpoint', type=str, 
                        help='Path to specific checkpoint file to resume from (used with --resume)')
    parser.add_argument('--checkpoint-freq', type=int, default=CHECKPOINT_FREQ, 
                        help=f'Save checkpoint every N iterations when loss improves (default: {CHECKPOINT_FREQ})')
    parser.add_argument('--no-auto-load', action='store_true', 
                        help='Start with default parameters instead of automatically loading the latest checkpoint')
    parser.add_argument('--plot-only', action='store_true', 
                        help='Load existing results and generate plots only (no optimization)')
    parser.add_argument('--run-dir', type=str,
                        help='Specific run directory to load results from (e.g., 20250820_022318)')
    # Control panel argument removed as buttons are now directly embedded in the main figure
    args = parser.parse_args()
    
    # By default, we'll try to load the latest checkpoint if one exists
    AUTO_LOAD_LATEST = not args.no_auto_load
    
    # Control panel is now embedded in the main figure
    print("Control buttons added to main figure. You can pause/resume optimization.")
    
    # Update checkpoint settings from command line arguments
    if args.resume:
        RESUME_FROM_CHECKPOINT = True
        if args.checkpoint:
            RESUME_CHECKPOINT_FILE = args.checkpoint
    
    if args.checkpoint_freq:
        CHECKPOINT_FREQ = args.checkpoint_freq
    
    # Check if we should only generate plots
    if args.plot_only:
        print("\n=== PLOT-ONLY MODE: Loading existing results ===\n")
        
        # Determine which run directory to use
        if args.run_dir:
            target_run_dir = os.path.join(RESULTS_DIR, args.run_dir)
            if not os.path.exists(target_run_dir):
                print(f"Error: Run directory not found: {target_run_dir}")
                exit(1)
        else:
            # Find the most recent run directory
            results_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), 
                                  key=os.path.getmtime, reverse=True)
            if not results_dirs:
                print("Error: No run directories found.")
                exit(1)
            target_run_dir = results_dirs[0]
        
        print(f"Loading results from: {target_run_dir}")
        
        # Look for checkpoint files in the target run
        target_checkpoint_dir = os.path.join(target_run_dir, 'checkpoints')
        if not os.path.exists(target_checkpoint_dir):
            print(f"Error: No checkpoints directory found in {target_run_dir}")
            exit(1)
        
        # Find throttle and brake checkpoints
        throttle_checkpoint = os.path.join(target_checkpoint_dir, 'checkpoint_throttle.pkl')
        brake_checkpoint = os.path.join(target_checkpoint_dir, 'checkpoint_brake.pkl')
        
        if not (os.path.exists(throttle_checkpoint) and os.path.exists(brake_checkpoint)):
            print(f"Error: Missing checkpoint files in {target_checkpoint_dir}")
            print(f"Throttle checkpoint exists: {os.path.exists(throttle_checkpoint)}")
            print(f"Brake checkpoint exists: {os.path.exists(brake_checkpoint)}")
            exit(1)
        
        # Load throttle parameters
        throttle_data = load_checkpoint(throttle_checkpoint)
        if throttle_data and throttle_data.get('params'):
            params = throttle_data['params']
            if len(params) >= 12:
                if len(params) >= 17:
                    # Full parameter vector - extract throttle params (first 12)
                    GLOBAL_THROTTLE_PARAMS[:] = params[:12]
                else:
                    # Throttle params only
                    GLOBAL_THROTTLE_PARAMS[:] = params[:12]
                print(f"Loaded throttle params: {GLOBAL_THROTTLE_PARAMS}")
        
        # Load brake parameters  
        brake_data = load_checkpoint(brake_checkpoint)
        if brake_data and brake_data.get('params'):
            params = brake_data['params']
            if len(params) >= 17:
                # Extract brake params from full parameter vector
                # Full param order: [Ra, kt, kv, Jr, A, cr1, G, re, cd, tau_th, Vmax, m, b1, b2, b3, b4, tau_br]
                brake_params = [params[12], params[13], params[14], params[15], params[16]]
                GLOBAL_BRAKE_PARAMS[:] = brake_params
                print(f"Loaded brake params: {GLOBAL_BRAKE_PARAMS}")
            elif len(params) >= 5:
                # Brake params only
                GLOBAL_BRAKE_PARAMS[:] = params[:5]
                print(f"Loaded brake params: {GLOBAL_BRAKE_PARAMS}")
        
        # Load loss histories for plotting
        throttle_losses = throttle_data.get('loss_history', []) if throttle_data else []
        brake_losses = brake_data.get('loss_history', []) if brake_data else []
        
        # Override output directories to save in the target run directory
        original_run_dir = RUN_DIR
        RUN_DIR = target_run_dir
        LOG_FILE = os.path.join(RUN_DIR, 'optimization_log.csv')
        LOSS_PLOT_FILE = os.path.join(RUN_DIR, 'loss_curves_regenerated.png')
        STATS_PLOT_FILE = os.path.join(RUN_DIR, 'statistics_regenerated.png')
        FINAL_PLOT_FILE = os.path.join(RUN_DIR, 'final_plot_regenerated.png')
        
        # Create mock results for final plot titles
        class MockResult:
            def __init__(self, fun):
                self.fun = fun
        
        result_throttle = MockResult(throttle_losses[-1] if throttle_losses else 0.0)
        result_brake = MockResult(brake_losses[-1] if brake_losses else 0.0)
        
        # Generate final plots and statistics
        combined_params = GLOBAL_THROTTLE_PARAMS + GLOBAL_BRAKE_PARAMS
        
        print("\n=== Generating plots and statistics ===")
        
        # Generate a sample plot using current parameters
        thr_pred_aligned, br_pred_aligned, Fd, F_achieved, Vlim, backEMF = compute_delay_aware_comparison(
            all_arrays['speed'], all_arrays['acceleration'], all_arrays['angle'], 
            all_arrays['throttle'], all_arrays['brake'], flat_time
        )
        
        update_plot(thr_pred_aligned.detach().cpu().numpy(),
                   br_pred_aligned.detach().cpu().numpy(), 
                   Fd.detach().cpu().numpy())
        
        # Plot loss curves
        plot_loss_curves()
        
        # Generate statistical analysis
        compute_and_plot_statistics(combined_params)
        
        # Save final plot if available
        if hasattr(update_plot, 'last_plot_data'):
            print("\nSaving final plot...")
            data = update_plot.last_plot_data
            
            plot_data = {
                'time': data['time'],
                'speed': data['speed'],
                'throttle': data['throttle_real'],
                'throttle_sim': data['throttle_pred'],
                'brake': data['brake_real'],
                'brake_sim': data['brake_pred'],
                'force_desired': data['force_desired'],
                'throttle_mask': data['throttle_mask'],
                'brake_mask': data['brake_mask']
            }
            
            # Add optional data if available
            if 'speed_achieved' in data:
                plot_data['speed_achieved'] = data['speed_achieved']
            if 'force_achieved' in data:
                plot_data['force_achieved'] = data['force_achieved']
            if 'real_acceleration' in data:
                plot_data['acceleration'] = data['real_acceleration']
            
            # Create final plot
            final_fig, final_axes = create_final_plot(plot_data, combined_params, FINAL_PLOT_FILE)
            
            run_name = args.run_dir if args.run_dir else os.path.basename(target_run_dir)
            title = f"Regenerated Results from {run_name}\nThrottle Loss: {result_throttle.fun:.2f}, Brake Loss: {result_brake.fun:.2f}"
            plt.suptitle(title, fontsize=14, y=0.998)
            plt.tight_layout()
            plt.subplots_adjust(top=0.94)
            
            plt.savefig(FINAL_PLOT_FILE, dpi=300, bbox_inches='tight')
            print(f"Final plot saved to {FINAL_PLOT_FILE}")
        
        print(f"\n=== PLOT-ONLY MODE COMPLETE ===")
        print(f"Regenerated results saved to: {RUN_DIR}")
        print(f"Files created:")
        print(f"  - {LOSS_PLOT_FILE}")
        print(f"  - {STATS_PLOT_FILE}")
        print(f"  - {FINAL_PLOT_FILE}")
        plt.show()
        exit(0)
    
    # Results directory is already created in the configuration section
    
    # Initialize the CSV log file with headers
    print(f"Saving optimization results to: {RUN_DIR}")
    with open(LOG_FILE, 'w', newline='') as csvfile:
        fieldnames = ['phase', 'timestamp', 'loss_total', 'loss_throttle', 'loss_brake'] + param_names
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Check if we should resume from a checkpoint
    if RESUME_FROM_CHECKPOINT:
        # If a specific checkpoint file was provided, use it
        if RESUME_CHECKPOINT_FILE and os.path.exists(RESUME_CHECKPOINT_FILE):
            checkpoint_file = RESUME_CHECKPOINT_FILE
        else:
            # First try to find latest throttle checkpoint (preferred order)
            checkpoint_file = find_latest_checkpoint(phase='throttle')
            if not checkpoint_file:
                # If no throttle checkpoint, try brake checkpoint
                checkpoint_file = find_latest_checkpoint(phase='brake')
                
            if checkpoint_file:
                print(f"No specific checkpoint file provided, using latest: {checkpoint_file}")
            else:
                print("No checkpoint files found. Starting from scratch.")
                checkpoint_file = None
                RESUME_FROM_CHECKPOINT = False
        
        # Load the checkpoint if we found one
        if checkpoint_file:
            checkpoint_data = load_checkpoint(checkpoint_file)
            if checkpoint_data:
                # Get the phase from the checkpoint
                checkpoint_phase = checkpoint_data.get('phase')
                
                # Restore optimization state from checkpoint
                best_loss = checkpoint_data.get('best_loss', np.inf)
                optimization_steps = checkpoint_data.get('optimization_steps', {'throttle': 0, 'brake': 0})
                
                # Define default bounds for both phases to check parameters are within bounds
                bounds_throttle_default = [
                    (0.01, 10.0), (0.1, 1.0), (0.1, 1.0), (0.01, 10.0),
                    (6.0, 8.0), (0.006, 0.015), (1.0, 20.0), (0.34, 0.36), (0.4, 0.6),
                    (0.0, 1.0), (300.0, 750.0), (4700.0, 5500.0)
                ]
                bounds_brake_default = [
                    (1e-15, 1e-10), (1e-11, 1e-7), (-1e-11, 1e-11), (-1e-16, 1e-16), (0.0, 1.0)
                ]
                
                # Load parameters into global lists based on checkpoint phase
                checkpoint_params = checkpoint_data.get('params')
                if checkpoint_params is not None:
                    if checkpoint_phase == 'throttle':
                        throttle_losses = checkpoint_data.get('loss_history', [])
                        print("\n=== Resuming THROTTLE optimization from checkpoint ===\n")
                        
                        # Extract and update throttle parameters
                        if len(checkpoint_params) >= 12:
                            loaded_throttle_params = checkpoint_params[:12]  # First 12 are throttle params (including mass)
                            GLOBAL_THROTTLE_PARAMS[:] = ensure_within_bounds(loaded_throttle_params, bounds_throttle_default)
                            print(f"Loaded throttle params from checkpoint: {GLOBAL_THROTTLE_PARAMS}")
                        
                        # Keep brake parameters unchanged (will be default values)
                        print(f"Using default brake params: {GLOBAL_BRAKE_PARAMS}")
                        
                    elif checkpoint_phase == 'brake':
                        brake_losses = checkpoint_data.get('loss_history', [])
                        print("\n=== Resuming BRAKE optimization from checkpoint ===\n")
                        
                        # Extract and update brake parameters
                        if len(checkpoint_params) >= 17:
                            # Full vector: first 12 are throttle params, last 5 are brake params
                            loaded_brake_params = [checkpoint_params[12], checkpoint_params[13], checkpoint_params[14], checkpoint_params[15], checkpoint_params[16]]
                            GLOBAL_BRAKE_PARAMS[:] = ensure_within_bounds(loaded_brake_params, bounds_brake_default)
                            print(f"Loaded brake params from checkpoint: {GLOBAL_BRAKE_PARAMS}")
                            
                            # Also load throttle parameters from checkpoint (first 12)
                            loaded_throttle_params = checkpoint_params[:12]
                            GLOBAL_THROTTLE_PARAMS[:] = ensure_within_bounds(loaded_throttle_params, bounds_throttle_default)
                            print(f"Loaded throttle params from checkpoint: {GLOBAL_THROTTLE_PARAMS}")
                        
                print(f"=== CHECKPOINT LOADED - GLOBAL PARAMS UPDATED ===")
                print(f"GLOBAL_THROTTLE_PARAMS: {GLOBAL_THROTTLE_PARAMS}")
                print(f"GLOBAL_BRAKE_PARAMS: {GLOBAL_BRAKE_PARAMS}")
                print(f"Best loss from checkpoint: {best_loss}")
    
    # --- Throttle optimization (FIRST) ---
    print("\n=== Optimizing THROTTLE parameters only ===\n")
    best_loss = np.inf if not RESUME_FROM_CHECKPOINT else best_loss
    # Only optimize throttle params and tau_th, fix tau_br
    fixed_tau_br = 0.20

    # Default initial parameters
    best_tau_th = 0.35

    # Try to load latest throttle checkpoint if auto-load is enabled
    if not RESUME_FROM_CHECKPOINT:  # Only auto-load if we're not already resuming from a specific checkpoint
        latest_throttle_checkpoint_file = find_latest_checkpoint(phase='throttle')
        if latest_throttle_checkpoint_file:
            print(f"Found latest throttle checkpoint: {latest_throttle_checkpoint_file}")
            checkpoint_data = load_checkpoint(latest_throttle_checkpoint_file)
            if checkpoint_data and checkpoint_data.get('params') is not None:
                checkpoint_params = checkpoint_data.get('params')
                if len(checkpoint_params) >= 16:
                    # Full parameter vector: extract only the 11 throttle parameters we need
                    loaded_params = checkpoint_params[:11]
                    GLOBAL_THROTTLE_PARAMS[:] = loaded_params
                    print(f"Using parameters from latest throttle checkpoint as initial values")
                    print(f"Updated GLOBAL_THROTTLE_PARAMS: {GLOBAL_THROTTLE_PARAMS}")
                    print(f"Initial loss: {checkpoint_data.get('best_loss', 'unknown')}")
                elif len(checkpoint_params) >= 11:
                    # Throttle params only
                    loaded_params = checkpoint_params[:11]
                    GLOBAL_THROTTLE_PARAMS[:] = loaded_params
                    print(f"Using parameters from latest throttle checkpoint as initial values")
                    print(f"Updated GLOBAL_THROTTLE_PARAMS: {GLOBAL_THROTTLE_PARAMS}")
                    print(f"Initial loss: {checkpoint_data.get('best_loss', 'unknown')}")
                else:
                    print(f"Checkpoint format not compatible (not enough parameters), using current global parameters")
        else:
            print("Using current global throttle parameters")

    # Define bounds for throttle parameters
    bounds_throttle = [
        (0.01, 10.0),   # Ra
        (0.05, 10.0),   # kt
        (0.05, 10.0),   # kv
        (0.01, 10.0),   # Jr
        (6.0, 8.0),     # A
        (0.006, 0.015), # cr1
        (1.0, 20.0),    # G
        (0.34, 0.36),   # re
        (0.4, 0.6),     # cd
        (0.0, 1.0),     # tau_th
        (300.0, 750.0)  # Vmax
    ]

    # Add mass bounds (learnable parameter)
    bounds_throttle.append((4700.0, 5500.0))  # m (kg)

    # Log parameters before optimization
    print("\nThrottle parameters before optimization:")
    param_names_throttle = ["Ra", "kt", "kv", "Jr", "A", "cr1", "G", "re", "cd", "tau_th", "Vmax", "m"]
    for i, (name, value, bound) in enumerate(zip(param_names_throttle, GLOBAL_THROTTLE_PARAMS, bounds_throttle)):
        lower, upper = bound
        within_bounds = (lower is None or value >= lower) and (upper is None or value <= upper)
        status = "OK" if within_bounds else "OUT OF BOUNDS"
        print(f"{i}: {name} = {value:.6g} (bounds: {lower:.6g} to {upper:.6g}) - {status}")

    # Define a callback to check for early termination
    def callback_throttle(xk):
        # Check if the finish flag has been set by the button
        return finish_current_phase

    # Run throttle optimization
    try:
        result_throttle = opt.minimize(
            throttle_loss_fn,
            x0=GLOBAL_THROTTLE_PARAMS,
            method='Nelder-Mead',
            bounds=bounds_throttle,
            callback=callback_throttle,
            options={'maxiter': 200000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True}
        )
    except OptimizationTerminationException as e:
        print(f"Throttle optimization terminated early: {e}")
        # Create a mock result object for consistency
        class MockResult:
            def __init__(self, x, fun, message):
                self.x = x
                self.fun = fun
                self.message = message
                self.success = True
        
        result_throttle = MockResult(
            x=GLOBAL_THROTTLE_PARAMS[:], 
            fun=throttle_losses[-1] if throttle_losses else 0.0,
            message="Terminated by user request"
        )
    
    print("\n=== THROTTLE OPTIMIZATION COMPLETE ===")
    print(f"Total optimization steps: {optimization_steps['throttle']}")
    print(result_throttle)
    
    # Update global throttle parameters with optimized values
    optimized_throttle_params = ensure_within_bounds(result_throttle.x, bounds_throttle)
    GLOBAL_THROTTLE_PARAMS[:] = optimized_throttle_params
    
    print(f"Optimized GLOBAL_THROTTLE_PARAMS: {GLOBAL_THROTTLE_PARAMS}")
    print(f"Final throttle loss: {result_throttle.fun}")

    # Save throttle checkpoint
    combined_params = GLOBAL_THROTTLE_PARAMS + GLOBAL_BRAKE_PARAMS
    save_checkpoint(combined_params, result_throttle, optimization_steps, 
                   throttle_losses, result_throttle.fun, phase='throttle')

    # Reset the finish_current_phase flag for the next optimization phase
    finish_current_phase = False

    # --- Brake optimization (SECOND) ---
    print("\n=== Optimizing BRAKE parameters only ===\n")
    best_loss = np.inf if not RESUME_FROM_CHECKPOINT else best_loss

    # Define bounds for brake parameters - now includes forward polynomial coefficients
    bounds_brake = [
        (1e-15, 1e-10),  # b1 - realistic bounds based on scale
        (1e-11, 1e-7),   # b2 - realistic bounds based on scale
        (-1e-11, 1e-11), # b3 - allow both positive and negative values
        (-1e-16, 1e-16), # b4 - allow both positive and negative values
        (0.0, 1.0),      # tau_br
        (0.0, 100.0),    # c0 - constant term (offset) for forward polynomial (small offset)
        (1000.0, 15000.0), # c1 - linear term for forward polynomial (main contribution)
        (-5000.0, 5000.0), # c2 - quadratic term for forward polynomial
        (-5000.0, 5000.0)  # c3 - cubic term for forward polynomial
    ]

    # Try to load latest brake checkpoint if auto-load is enabled
    if not RESUME_FROM_CHECKPOINT:  # Only auto-load if we're not already resuming from a specific checkpoint
        latest_brake_checkpoint_file = find_latest_checkpoint(phase='brake')
        if latest_brake_checkpoint_file:
            print(f"Found latest brake checkpoint: {latest_brake_checkpoint_file}")
            checkpoint_data = load_checkpoint(latest_brake_checkpoint_file)
            if checkpoint_data and checkpoint_data.get('params') is not None:
                checkpoint_params = checkpoint_data.get('params')
                if len(checkpoint_params) >= 5:
                    # Extract only the 5 brake parameters we need
                    if len(checkpoint_params) >= 17:
                        # Full vector format: extract brake params from full parameter vector
                        loaded_params = [checkpoint_params[12], checkpoint_params[13], checkpoint_params[14], checkpoint_params[15], checkpoint_params[16]]
                    else:
                        # Brake params only
                        loaded_params = checkpoint_params[:5]
                    GLOBAL_BRAKE_PARAMS[:] = ensure_within_bounds(loaded_params, bounds_brake)
                    print(f"Using parameters from latest brake checkpoint as initial values")
                    print(f"Updated GLOBAL_BRAKE_PARAMS: {GLOBAL_BRAKE_PARAMS}")
                    print(f"Initial loss: {checkpoint_data.get('best_loss', 'unknown')}")
                else:
                    print(f"Checkpoint format not compatible (not enough parameters), using current global parameters")
        else:
            print("Using current global brake parameters")

    # Log parameters before optimization
    print("\nBrake parameters before optimization:")
    param_names_brake = ["b1", "b2", "b3", "b4", "tau_br"]
    for i, (name, value, bound) in enumerate(zip(param_names_brake, GLOBAL_BRAKE_PARAMS, bounds_brake)):
        lower, upper = bound
        within_bounds = (lower is None or value >= lower) and (upper is None or value <= upper)
        status = "OK" if within_bounds else "OUT OF BOUNDS"
        print(f"{i}: {name} = {value:.6g} (bounds: {lower:.6g} to {upper:.6g}) - {status}")

    # Define a callback to check for early termination
    def callback_brake(xk):
        # Check if the finish flag has been set by the button
        return finish_current_phase

    # Run brake optimization
    try:
        result_brake = opt.minimize(
            brake_loss_fn,
            x0=GLOBAL_BRAKE_PARAMS,
            method='Nelder-Mead',
            bounds=bounds_brake,
            callback=callback_brake,
            options={'maxiter': 200000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': True}
        )
    except OptimizationTerminationException as e:
        print(f"Brake optimization terminated early: {e}")
        # Create a mock result object for consistency
        class MockResult:
            def __init__(self, x, fun, message):
                self.x = x
                self.fun = fun
                self.message = message
                self.success = True
        
        result_brake = MockResult(
            x=GLOBAL_BRAKE_PARAMS[:], 
            fun=brake_losses[-1] if brake_losses else 0.0,
            message="Terminated by user request"
        )

    print("\n=== BRAKE OPTIMIZATION COMPLETE ===")
    print(f"Total optimization steps: {optimization_steps['brake']}")
    print(result_brake)

    # Update global brake parameters with optimized values
    optimized_brake_params = ensure_within_bounds(result_brake.x, bounds_brake)
    GLOBAL_BRAKE_PARAMS[:] = optimized_brake_params

    print(f"Optimized GLOBAL_BRAKE_PARAMS: {GLOBAL_BRAKE_PARAMS}")
    print(f"Final brake loss: {result_brake.fun}")

    # Save brake checkpoint
    combined_params = GLOBAL_THROTTLE_PARAMS + GLOBAL_BRAKE_PARAMS
    save_checkpoint(combined_params, result_brake, optimization_steps, 
                   brake_losses, result_brake.fun, phase='brake')

    # Reset the finish_current_phase flag
    finish_current_phase = False

else:
    best_tau_th = 0.20

# Create full parameter vector combining best throttle and brake params
print("\n=== Final combined best parameters ===")
combined_params = GLOBAL_THROTTLE_PARAMS + GLOBAL_BRAKE_PARAMS
param_names = ["Ra", "kt", "kv", "Jr", "A", "cr1", "G", "re", "cd", "tau_th", "Vmax", "m", "b1", "b2", "b3", "b4", "tau_br", "c0", "c1", "c2", "c3"]
for name, val in zip(param_names, combined_params):
    print(f"{name}: {val}")

# Log the final combined parameters
log_params_to_csv(combined_params, 0, 0, 0, 'combined')

# Plot loss curves
plot_loss_curves()

# Generate statistical analysis
compute_and_plot_statistics(combined_params)# Save the final plot if available
if hasattr(update_plot, 'last_plot_data'):
            print("\nSaving final plot...")
            data = update_plot.last_plot_data
            
            # Convert data keys to match expected input for create_final_plot function
            plot_data = {
                'time': data['time'],
                'speed': data['speed'],
                'throttle': data['throttle_real'],
                'throttle_sim': data['throttle_pred'],
                'brake': data['brake_real'],
                'brake_sim': data['brake_pred'],
                'force_desired': data['force_desired'],
                'throttle_mask': data['throttle_mask'],
                'brake_mask': data['brake_mask']
            }
            
            # Add optional data if available
            if 'speed_achieved' in data:
                plot_data['speed_achieved'] = data['speed_achieved']
                # Print confirmation that speed_achieved is being included in final plot
                print(f"Including achieved speed data in final plot, shape: {data['speed_achieved'].shape}")
            if 'force_achieved' in data:
                plot_data['force_achieved'] = data['force_achieved']
            if 'real_acceleration' in data:
                plot_data['acceleration'] = data['real_acceleration']
                
            # Use the trip index to get angle data for external forces
            trip_idx = data['trip_idx']
            # Get the offset to fetch angle data
            if 0 <= trip_idx < len(offset_idx) and 0 <= trip_idx < len(lengths):
                off = offset_idx[trip_idx]
                L = len(data['time'])
                
                # Add title information
                title = None
                if 0 <= trip_idx < len(trip_ids) and trip_idx < len(durations):
                    title = (f"Trip {trip_idx+1}/{len(trip_ids)}, Duration: {durations[trip_idx]:.1f}s\n"
                            f"Final Optimization Results - Throttle Loss: {result_throttle.fun:.2f}, Brake Loss: {result_brake.fun:.2f}")
                else:
                    title = f"Final Optimization Results\nThrottle Loss: {result_throttle.fun:.2f}, Brake Loss: {result_brake.fun:.2f}"
            
            # Create the final plot using the function from plot_fixed_3x2
            final_fig, final_axes = create_final_plot(plot_data, combined_params, FINAL_PLOT_FILE)
            
            # Add title with optimization results
            plt.suptitle(title, fontsize=14, y=0.998)
            plt.tight_layout()
            plt.subplots_adjust(top=0.94)  # Adjust for the title
            
            # Save plot if not already saved by create_final_plot
            plt.savefig(FINAL_PLOT_FILE, dpi=300, bbox_inches='tight')
            print(f"Final plot saved to {FINAL_PLOT_FILE}")

plt.show()

