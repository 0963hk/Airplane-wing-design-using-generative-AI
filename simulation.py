import os
import numpy as np
from pathlib import Path
import json

def read_airfoil_dat(filepath):
    data = np.loadtxt(filepath)
    if data.shape[1] != 2:
        raise ValueError("Airfoil file must have exactly 2 columns")
    
    x = data[:, 0]
    y = data[:, 1]
    
    if len(x) < 10:
        raise ValueError(f"Airfoil file has too few points: {len(x)}")
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    if x_max - x_min < 1e-6:
        raise ValueError("Airfoil has zero chord length")
    
    if abs(y_max - y_min) < 1e-6:
        raise ValueError("Airfoil has zero thickness")
    
    return x, y

def validate_and_fix_airfoil(x, y):
    x = np.array(x)
    y = np.array(y)
    
    tolerance = 1e-10
    
    if len(x) != len(y):
        raise ValueError("x and y arrays have different lengths")
    
    x_unique = []
    y_unique = []
    
    for i in range(len(x)):
        if i == 0:
            x_unique.append(x[i])
            y_unique.append(y[i])
        else:
            dx = x[i] - x_unique[-1]
            dy = y[i] - y_unique[-1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > tolerance:
                x_unique.append(x[i])
                y_unique.append(y[i])
    
    x_fixed = np.array(x_unique)
    y_fixed = np.array(y_unique)
    
    if len(x_fixed) < 10:
        raise ValueError(f"After removing duplicates, too few points: {len(x_fixed)}")
    
    first_point = np.array([x_fixed[0], y_fixed[0]])
    last_point = np.array([x_fixed[-1], y_fixed[-1]])
    dist_to_first = np.linalg.norm(last_point - first_point)
    
    if dist_to_first > tolerance:
        x_fixed = np.append(x_fixed, x_fixed[0])
        y_fixed = np.append(y_fixed, y_fixed[0])
    
    return x_fixed, y_fixed

def create_panels(x, y):
    n = len(x) - 1
    panels = []
    
    for i in range(n):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i+1], y[i+1]
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1e-10:
            continue
        
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        theta = np.arctan2(dy, dx)
        
        nx = -np.sin(theta)
        ny = np.cos(theta)
        
        panels.append({
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'xc': xc, 'yc': yc,
            'length': length,
            'theta': theta,
            'nx': nx,
            'ny': ny
        })
    
    return panels

def compute_vortex_influence(x, y, x1, y1, x2, y2):
    r1_sq = (x - x1)**2 + (y - y1)**2
    r2_sq = (x - x2)**2 + (y - y2)**2
    
    if r1_sq < 1e-12 or r2_sq < 1e-12:
        return 0.0, 0.0
    
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)
    
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)
    
    if length < 1e-12:
        return 0.0, 0.0
    
    cos_theta = dx / length
    sin_theta = dy / length
    
    u_local = (x - x1) * cos_theta + (y - y1) * sin_theta
    v_local = -(x - x1) * sin_theta + (y - y1) * cos_theta
    
    r1_local = np.sqrt(u_local**2 + v_local**2)
    r2_local = np.sqrt((u_local - length)**2 + v_local**2)
    
    if r1_local < 1e-12 or r2_local < 1e-12:
        return 0.0, 0.0
    
    theta1 = np.arctan2(v_local, u_local)
    theta2 = np.arctan2(v_local, u_local - length)
    
    dtheta = theta2 - theta1
    
    if abs(v_local) < 1e-12:
        u_vel = 0.0
        v_vel = 0.0
    else:
        u_vel = (dtheta) / (2.0 * np.pi)
        v_vel = (np.log(r2_local / r1_local)) / (2.0 * np.pi)
    
    u_global = u_vel * cos_theta - v_vel * sin_theta
    v_global = u_vel * sin_theta + v_vel * cos_theta
    
    return u_global, v_global

def compute_influence_matrix(panels, alpha):
    n = len(panels)
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(n):
        panel_i = panels[i]
        xi = panel_i['xc']
        yi = panel_i['yc']
        ni_x = panel_i['nx']
        ni_y = panel_i['ny']
        
        for j in range(n):
            panel_j = panels[j]
            u, v = compute_vortex_influence(
                xi, yi, panel_j['x1'], panel_j['y1'], 
                panel_j['x2'], panel_j['y2']
            )
            A[i, j] = u * ni_x + v * ni_y
        
        freestream_u = np.cos(alpha)
        freestream_v = np.sin(alpha)
        b[i] = -(freestream_u * ni_x + freestream_v * ni_y)
    
    if n > 0:
        A[n-1, :] = 0.0
        A[n-1, 0] = 1.0
        A[n-1, n-1] = 1.0
        b[n-1] = 0.0
    
    return A, b

def compute_pressure_coefficient(panels, gamma, alpha, V_inf=1.0):
    n = len(panels)
    cp = np.zeros(n)
    
    for i in range(n):
        panel = panels[i]
        theta = panel['theta']
        xi = panel['xc']
        yi = panel['yc']
        
        u_total = V_inf * np.cos(alpha)
        v_total = V_inf * np.sin(alpha)
        
        for j in range(n):
            panel_j = panels[j]
            u, v = compute_vortex_influence(
                xi, yi, panel_j['x1'], panel_j['y1'], 
                panel_j['x2'], panel_j['y2']
            )
            u_total += gamma[j] * u
            v_total += gamma[j] * v
        
        V_tangent = u_total * (-np.sin(theta)) + v_total * np.cos(theta)
        V_normal = u_total * np.cos(theta) + v_total * np.sin(theta)
        
        V_total_sq = V_tangent**2 + V_normal**2
        
        cp[i] = 1.0 - (V_total_sq / V_inf**2)
    
    return cp

def compute_forces(panels, cp, alpha):
    n = len(panels)
    
    lift = 0.0
    drag = 0.0
    
    for i in range(n):
        panel = panels[i]
        length = panel['length']
        nx = panel['nx']
        ny = panel['ny']
        
        pressure_coeff = cp[i]
        
        fx = -pressure_coeff * length * nx
        fy = -pressure_coeff * length * ny
        
        lift += -fx * np.sin(alpha) + fy * np.cos(alpha)
        drag += fx * np.cos(alpha) + fy * np.sin(alpha)
    
    return lift, drag

def compute_coefficients_with_viscous_correction(panels, cp, alpha, reynolds, mach):
    n = len(panels)
    
    lift, drag_inviscid = compute_forces(panels, cp, alpha)
    
    x_coords = []
    for panel in panels:
        x_coords.append(panel['x1'])
        x_coords.append(panel['x2'])
    x_coords = np.array(x_coords)
    chord = np.max(x_coords) - np.min(x_coords)
    
    if chord < 1e-6:
        chord = 1.0
    
    total_panel_length = sum(panel['length'] for panel in panels)
    
    if total_panel_length > 0:
        cl_inviscid = 2.0 * lift / chord
        cd_inviscid = 2.0 * drag_inviscid / chord
    else:
        cl_inviscid = 0.0
        cd_inviscid = 0.0
    
    if reynolds > 0:
        if reynolds < 5e5:
            cf = 1.328 / np.sqrt(reynolds)
        else:
            cf = 0.455 / ((np.log10(reynolds))**2.58)
        
        wetted_area = 0.0
        for panel in panels:
            wetted_area += panel['length']
        
        cd_viscous = cf * wetted_area / chord
        
        cd_total = cd_inviscid + cd_viscous
    else:
        cd_total = cd_inviscid
    
    cl = cl_inviscid
    
    if mach > 0.3:
        beta = np.sqrt(max(1.0 - mach**2, 0.01))
        cl = cl_inviscid / beta
        cd_total = cd_total / beta
    
    if np.isnan(cl) or np.isnan(cd_total) or np.isinf(cl) or np.isinf(cd_total):
        cl = -np.inf
        cd_total = np.inf
    
    l_d = cl / cd_total if cd_total > 0 and not np.isinf(cd_total) else float('inf')
    
    if np.isnan(l_d) or l_d > 300:
        l_d = -1.0
    
    return cl, cd_total, l_d

def calculate_airfoil_coefficients_panel_method(airfoil_dir, output_dir, reynolds=1.8e6, mach=0.01, alpha=0.0):
    airfoil_dir = Path(airfoil_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dat_files = list(airfoil_dir.glob('*.dat'))
    
    if not dat_files:
        print(f"No .dat files found in {airfoil_dir}")
        return {}
    
    results = {}
    
    for dat_file in dat_files:
        print(f"Processing {dat_file.name}...")
        
        try:
            x, y = read_airfoil_dat(dat_file)
            print(f"  Loaded {len(x)} points")
            
            x, y = validate_and_fix_airfoil(x, y)
            print(f"  After validation: {len(x)} points")
            
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            chord = x_max - x_min
            thickness = y_max - y_min
            
            print(f"  Chord: {chord:.6f}, Max thickness: {thickness:.6f}")
            
            alpha_rad = np.deg2rad(alpha)
            
            panels = create_panels(x, y)
            print(f"  Created {len(panels)} panels")
            
            A, b = compute_influence_matrix(panels, alpha_rad)
            
            try:
                gamma = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                print(f"  Warning: Matrix singular, using least squares")
                gamma = np.linalg.lstsq(A, b, rcond=1e-10)[0]
            
            if np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
                print(f"  Warning: Invalid gamma values, setting to zero")
                gamma = np.zeros_like(gamma)
            
            cp = compute_pressure_coefficient(panels, gamma, alpha_rad)
            
            lift_force, drag_force = compute_forces(panels, cp, alpha_rad)
            
            cl, cd, l_d = compute_coefficients_with_viscous_correction(
                panels, cp, alpha_rad, reynolds, mach
            )
            
            if np.isnan(cl) or np.isinf(cl) or np.isnan(cd) or np.isinf(cd):
                print(f"  Warning: Invalid coefficients, using fallback values")
                cl = -np.inf
                cd = np.inf
                l_d = -1.0
            
            results[dat_file.name] = {
                'CL': float(cl),
                'CD': float(cd),
                'L/D': float(l_d)
            }
            
            if not (np.isinf(cl) or np.isinf(cd)):
                print(f"  {dat_file.name}: CL={cl:.6f}, CD={cd:.6f}, L/D={l_d:.6f}")
                print(f"    Lift force: {lift_force:.6e}, Drag force: {drag_force:.6e}")
                print(f"    Max CP: {np.max(cp):.6f}, Min CP: {np.min(cp):.6f}")
            else:
                print(f"  {dat_file.name}: Computation failed (CL={cl}, CD={cd})")
            
        except Exception as e:
            print(f"Error processing {dat_file.name}: {e}")
            results[dat_file.name] = {
                'CL': float(-np.inf),
                'CD': float(np.inf),
                'L/D': float(-1.0)
            }
            continue
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Airfoil Analysis Results (Panel Method)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Reynolds Number: {reynolds:.2e}\n")
        f.write(f"Mach Number: {mach}\n")
        f.write(f"Angle of Attack: {alpha} degrees\n")
        f.write("=" * 50 + "\n\n")
        for filename, coeffs in results.items():
            f.write(f"{filename}:\n")
            f.write(f"  CL: {coeffs['CL']:.6f}\n")
            f.write(f"  CD: {coeffs['CD']:.6f}\n")
            f.write(f"  L/D: {coeffs['L/D']:.6f}\n\n")
    
    return results

if __name__ == "__main__":
    airfoil_dir = r"D:\Project\IP\DCGAN_result\airfoil_coordinates\Analysis"
    output_dir = r"D:\Project\IP\Simulation"
    reynolds = 1.8e6
    mach = 0.01
    alpha = 0.0
    
    results = calculate_airfoil_coefficients_panel_method(
        airfoil_dir, output_dir, reynolds, mach, alpha
    )
    
    if results:
        print("\n" + "="*80)
        print("FINAL SUMMARY - AIRFOIL ANALYSIS RESULTS")
        print("="*80)
        print(f"{'Airfoil':<30} {'CL':>12} {'CD':>12} {'CL/CD':>12}")
        print("-"*80)
        
        valid_results = []
        invalid_results = []
        
        for filename, coeffs in results.items():
            cl = coeffs['CL']
            cd = coeffs['CD']
            l_d = coeffs['L/D']
            
            if not (np.isinf(cl) or np.isinf(cd) or np.isnan(cl) or np.isnan(cd)):
                print(f"{filename:<30} {cl:>12.6f} {cd:>12.6f} {l_d:>12.6f}")
                valid_results.append((filename, cl, cd, l_d))
            else:
                print(f"{filename:<30} {'FAILED':>12} {'FAILED':>12} {'FAILED':>12}")
                invalid_results.append(filename)
        
        print("-"*80)
        
        if valid_results:
            cl_values = [r[1] for r in valid_results]
            cd_values = [r[2] for r in valid_results]
            l_d_values = [r[3] for r in valid_results]
            
            print(f"\nStatistics (Valid Results: {len(valid_results)}):")
            print(f"  CL:  Min={min(cl_values):.6f},  Max={max(cl_values):.6f},  Mean={np.mean(cl_values):.6f}")
            print(f"  CD:  Min={min(cd_values):.6f},  Max={max(cd_values):.6f},  Mean={np.mean(cd_values):.6f}")
            print(f"  L/D: Min={min(l_d_values):.6f},  Max={max(l_d_values):.6f},  Mean={np.mean(l_d_values):.6f}")
        
        if invalid_results:
            print(f"\nFailed airfoils ({len(invalid_results)}): {', '.join(invalid_results)}")
        
        print("="*80)
        print(f"\nResults saved to {output_dir}")
        print(f"  - results.json: JSON format")
        print(f"  - summary.txt: Text format")
