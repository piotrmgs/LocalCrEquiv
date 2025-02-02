import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.utils.data import DataLoader
from datetime import datetime
from scipy.stats import f_oneway, kruskal
from datetime import datetime
import time
import torch
import os

from dataset_preprocessing import load_cifar10_data
from train_cifar_cnn import SimpleCifarCNN

###############################################################################
# Configuration / Logging
###############################################################################
torch.manual_seed(42)
np.random.seed(42)

model_path = "[...]/model_mlp_california.pt" #path of .pt file
output_path = "[...]"#path of output
os.makedirs(output_path, exist_ok=True)

def get_basic_info():
    info = {
        "GPU_available": torch.cuda.is_available(),
        "GPU_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    }
    try:
        if torch.cuda.is_available():
            info["GPU_memory_allocated"] = torch.cuda.memory_allocated(0)/1024**2  # MB
            info["GPU_memory_cached"] = torch.cuda.memory_reserved(0)/1024**2  # MB
    except:
        info["GPU_memory_allocated"] = "Not available"
        info["GPU_memory_cached"] = "Not available"
    return info

#logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s - %(levelname)s - %(message)s',
#    handlers=[
#        logging.FileHandler(os.path.join(output_path, 'verification_cifar.log')),
#        logging.StreamHandler()
#    ]
#)

###############################################################################
# Load the trained CNN and data
###############################################################################
train_loader, test_loader = load_cifar10_data(batch_size=64)
model = SimpleCifarCNN(num_classes=10)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

params = [p for p in model.parameters() if p.requires_grad]
theta_star = torch.cat([p.data.view(-1) for p in params])

criterion = nn.CrossEntropyLoss()

###############################################################################
# Flatten/Unflatten Utilities
###############################################################################
def flatten_model_params(model):
    flat_list = []
    for p in model.parameters():
        flat_list.append(p.view(-1))
    return torch.cat(flat_list, dim=0).detach()

def unflatten_model_params(model, flat_tensor):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        with torch.no_grad():
            p.copy_(flat_tensor[idx: idx+numel].view(p.shape))
        idx += numel

###############################################################################
# Hessian-Vector Product for Cross Entropy
###############################################################################
def compute_gradient_hvp(model, data, targets, v):
    model.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad = torch.cat([g.view(-1) for g in grad])
    
    grad_v_prod = torch.sum(grad * v)
    Hv_list = torch.autograd.grad(grad_v_prod, params, retain_graph=True)
    
    
    Hv = torch.cat([h.contiguous().view(-1) for h in Hv_list])
   
    
    return grad.detach(), Hv.detach()


###############################################################################
# Verifying the Local Equivalence Inequality
###############################################################################
def verify_inequality(model, data, targets, base_params_flat, grad_f, hessian_f, scales):
    """
    For each perturbation scale, define delta, build a local polynomial for g,
    measure f(theta^* + delta) with cross entropy, compare (g-f) to ||grad_f||^2.
    """
    with torch.no_grad():
        old_params_flat = base_params_flat.clone()
        unflatten_model_params(model, base_params_flat)
        # measure f(theta^*) = Cross Entropy over 'data, targets'
        outputs = model(data)
        f_star_val = criterion(outputs, targets).item()
    
    grad_norm = torch.norm(grad_f)
    ratios = []
    for scale in scales:
        delta = torch.randn_like(grad_f) * scale
        # local expansion: g = f_star_val + grad_f^T delta + 0.5 delta^T H delta
        first_order = torch.dot(grad_f, delta).item()
        second_order = 0.5 * torch.sum(delta * hessian_f * delta).item()
        g_pert_val = f_star_val + first_order + second_order

        # measure f(theta^* + delta)
        param_pert = base_params_flat.clone() + delta
        with torch.no_grad():
            unflatten_model_params(model, param_pert)
            out_pert = model(data)
            f_pert_val = criterion(out_pert, targets).item()

        # difference: (g - f)(theta^* + delta)
        gf_diff = abs(g_pert_val - f_pert_val)

        if grad_norm > 1e-6:
            ratio_val = gf_diff / (grad_norm**2)  # r=2 => denominator=||grad||^2
            ratios.append(ratio_val)

        # revert model
        unflatten_model_params(model, old_params_flat)

    return ratios

###############################################################################
# Gather ~100 samples from the train_loader
###############################################################################
data_samples, target_samples = [], []
for batch in train_loader:
    data_samples.append(batch[0])
    target_samples.append(batch[1])
    if len(data_samples) >= 100:
        break

data_samples = torch.cat(data_samples[:100], dim=0)
target_samples = torch.cat(target_samples[:100], dim=0)

# we can do a sub-batch approach or just use the entire 100
logging.info(f"Using {len(data_samples)} CIFAR-10 samples for local check")

###############################################################################
# Iterations for local checks
###############################################################################
start_time = time.time()
iteration_times = []
basic_info = get_basic_info()
results = []
num_checks = 100
k = 5

perturb_scales = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

start_time = time.time()
for i in range(num_checks):
    iter_start = time.time()
    ratio_values_iter = []
    for j in range(k):
        v_j = torch.randn_like(theta_star)
        grad_f, hv_f = compute_gradient_hvp(model, data_samples, target_samples, v_j)
        ratio_list = verify_inequality(
            model, data_samples, target_samples,
            theta_star, grad_f, hv_f,
            perturb_scales
        )
        ratio_values_iter.append(ratio_list)
    results.append(ratio_values_iter)
    iter_time = time.time() - iter_start
    iteration_times.append(iter_time)
    
   
    if i % 10 == 0:
        elapsed = time.time() - start_time
        estimated_total = elapsed / (i + 1) * num_checks
        remaining = estimated_total - elapsed
        print(f"Iteration {i}/{num_checks}")
        print(f"Elapsed: {elapsed/60:.1f}min, ETA: {remaining/60:.1f}min")
        print(f"Last iteration time: {iter_time:.2f}s")
total_time = time.time() - start_time
execution_time = time.time() - start_time
logging.info(f"Local check finished in {execution_time:.2f} seconds")

###############################################################################
# Convert results to numpy, flatten, outlier filter
###############################################################################
results_np = np.array(results)  # shape (num_checks, k, n_scales)
results_2d = results_np.reshape(-1, results_np.shape[-1])  # (num_checks*k, n_scales)

Q1 = np.percentile(results_2d, 25, axis=0)
Q3 = np.percentile(results_2d, 75, axis=0)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#filtered = np.clip(results_2d, lower_bound, upper_bound)
# Limit outlier values
filtered_results = np.clip(results_2d, lower_bound, upper_bound)
filtered_results = np.array(filtered_results)
mean_ratios = np.nanmean(filtered_results, axis=0)
std_ratios  = np.nanstd(filtered_results, axis=0)

# The rest of advanced stats/plots 
def statistical_tests(results):
    """Statistical tests, including normality tests and confidence intervals."""
    # Random sampling for normality test
    max_samples = 5000
    results_flat = results.flatten()
    if results.size > max_samples:
        random_indices = np.random.choice(results.size, size=max_samples, replace=False)
        sampled_data = results.flatten()[random_indices]
        stat, p_value = stats.shapiro(sampled_data)
        logging.info(f"Shapiro-Wilk normality test on {max_samples} samples")
    else:
        sampled_data = results_flat  # fallback on entire results
        stat, p_value = stats.shapiro(sampled_data)
        logging.info("Shapiro-Wilk normality test on the entire result set")

        stat, p_value = stats.shapiro(results.flatten())
    
    # Additional normality test (no sample size limitations)
    dagostino_stat, dagostino_p = stats.normaltest(results.flatten())
    
    confidence_interval = stats.norm.interval(0.95, loc=np.mean(results), scale=stats.sem(results))
    effect_size = np.mean(results) / np.std(results)
    anova_stat, anova_p = f_oneway(*results.T)  # Analysis of variance
    kruskal_stat, kruskal_p = kruskal(*results.T)  # Kruskal-Wallis test
    
    logging.info(f"Shapiro-Wilk normality test: stat={stat}, p={p_value}")
    logging.info(f"D'Agostino-Pearson normality test: stat={dagostino_stat}, p={dagostino_p}")
    logging.info(f"95% confidence interval: {confidence_interval}")
    logging.info(f"Effect size: {effect_size}")
    logging.info(f"ANOVA: stat={anova_stat}, p={anova_p}")
    logging.info(f"Kruskal-Wallis test: stat={kruskal_stat}, p={kruskal_p}")

    # Confidence intervals visualization
    plt.figure(figsize=(10, 6))

    # Data
    mean_values = np.mean(results, axis=0)
    ci_lower = np.abs(confidence_interval[0] - mean_values)
    ci_upper = np.abs(confidence_interval[1] - mean_values)

    # Plot with enhanced styling
    plt.errorbar(perturb_scales, mean_values, 
                yerr=[ci_lower, ci_upper],
                fmt='o-',  # line connecting points
                color='blue',
                ecolor='lightblue',  # error bar color
                capsize=5,
                capthick=1.5,
                elinewidth=1.5,
                markersize=8,
                label='Mean value')

    # Add confidence interval region
    plt.fill_between(perturb_scales, 
                    mean_values - ci_lower,
                    mean_values + ci_upper,
                    color='blue',
                    alpha=0.1,
                    label='95% confidence interval')

    # Logarithmic scales
    plt.xscale('log')
    plt.yscale('log')

    # Formatting
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.xlabel('Perturbation scale')
    plt.ylabel('Inequality coefficient')
    plt.title('Confidence intervals for inequality coefficient\nwith 95% confidence level')

    # Legend
    plt.legend()

    # Add value annotations
    for i, (x, y, yerr_low, yerr_up) in enumerate(zip(perturb_scales, mean_values, ci_lower, ci_upper)):
        plt.annotate(f'{y:.2e}\n±{yerr_up:.2e}', 
                    xy=(x, y), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confidence_intervals.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Kruskal-Wallis test results visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate ranks for the entire dataset
    all_data = results.flatten()
    ranks = stats.rankdata(all_data).reshape(results.shape)
    ranks_by_group = np.mean(ranks, axis=0)

    # Mean ranks plot
    ax1.plot(range(len(ranks_by_group)), ranks_by_group, 'o-', color='blue')
    ax1.set_xlabel('Perturbation scale index')
    ax1.set_ylabel('Mean rank')
    ax1.set_title('Mean Ranks')
    ax1.grid(True)

    # Boxplot with logarithmic scale
    ax2.boxplot([results[:, i] for i in range(results.shape[1])], 
                showfliers=True)  # show outliers
    ax2.set_yscale('log')  # logarithmic scale
    ax2.set_xlabel('Perturbation scale index')
    ax2.set_ylabel('Coefficient value (log scale)')
    ax2.set_title('Value Distribution')
    ax2.grid(True)

    plt.suptitle(f'Kruskal-Wallis Test\nH-statistic={kruskal_stat:.2f}, p-value<{kruskal_p:.2e}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "kruskal_results.png"))
    plt.close()

    # ANOVA results and normality tests visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Mean values plot
    means = np.mean(results, axis=0)
    sems = stats.sem(results, axis=0)
    ax1.errorbar(range(len(means)), means, yerr=sems, fmt='o-', capsize=5)
    ax1.set_yscale('log')
    ax1.set_xlabel('Perturbation scale index')
    ax1.set_ylabel('Mean value (log scale)')
    ax1.set_title('Mean Values with Standard Error')
    ax1.grid(True)

    # Residuals distribution plot
    group_means = np.mean(results, axis=0)
    residuals = results - group_means[np.newaxis, :]
    residuals_normalized = (residuals - np.mean(residuals)) / np.std(residuals)
    ax2.hist(residuals_normalized.flatten(), bins=30, density=True, alpha=0.7)
    ax2.set_xlabel('Normalized residuals')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Normalized Residuals')
    x = np.linspace(-4, 4, 100)
    p = stats.norm.pdf(x, 0, 1)
    ax2.plot(x, p, 'r--', lw=2, label='Normal distribution')
    ax2.legend()
    ax2.grid(True)

    # Q-Q plot for sample used in Shapiro-Wilk test
    stats.probplot(sampled_data, dist="norm", plot=ax3)
    ax3.set_title('Q-Q plot (5000 samples)')
    ax3.grid(True)

    # Sample histogram
    ax4.hist(sampled_data, bins=30, density=True, alpha=0.7)
    x = np.linspace(min(sampled_data), max(sampled_data), 100)
    p = stats.norm.pdf(x, np.mean(sampled_data), np.std(sampled_data))
    ax4.plot(x, p, 'r--', lw=2, label='Normal distribution')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Sample Histogram')
    ax4.legend()
    ax4.grid(True)

    plt.suptitle(f'Analysis of Variance (ANOVA) and Normality Tests\n' + 
                f'F-statistic={anova_stat:.2f}, p-value<{anova_p:.2e}\n' +
                f'Shapiro-Wilk (n=5000): p={p_value:.2e}, D\'Agostino: p={dagostino_p:.2e}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "anova_and_normality.png"))
    plt.close()

    # Visualization of dependency on perturbation scale with standard deviation
    plt.figure(figsize=(10, 6))
    plt.errorbar(perturb_scales, mean_ratios, yerr=std_ratios, 
                fmt='o-', capsize=5, capthick=1.5, elinewidth=1.5, 
                markersize=8, color='blue', label='Mean value')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Perturbation scale')
    plt.ylabel('Inequality coefficient')
    plt.title('Inequality Coefficient vs Perturbation Scale')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ratios_vs_scale_with_std.png"), dpi=300)
    plt.close()

    # Boxplot visualization
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot([filtered_results[:, i] for i in range(filtered_results.shape[1])], 
                    labels=[f'{s:.1e}' for s in perturb_scales],
                    patch_artist=True)

    # Boxplot settings
    for box in bp['boxes']:
        box.set(facecolor='lightblue', alpha=0.7)
    plt.yscale('log')  # logarithmic scale for better visualization
    plt.xticks(rotation=45)
    plt.xlabel('Perturbation scale')
    plt.ylabel('Inequality coefficient (log scale)')
    plt.title('Distribution of Inequality Coefficient for Different Perturbation Scales')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ratios_boxplot.png"), dpi=300)
    plt.close()


    return (stat, p_value, confidence_interval, effect_size, anova_stat, anova_p, 
            kruskal_stat, kruskal_p, dagostino_stat, dagostino_p)


###############################################################################
# Simple text output with final ratio means/stdev
###############################################################################
# Save results to text file
output_file = os.path.join(output_path, "ratios_results.txt")
#stat_res = statistical_tests(results)
stat_res = statistical_tests(filtered_results)

with open(output_file, "w") as f:
    # Header
    f.write("=" * 80 + "\n")
    f.write("STATISTICAL ANALYSIS RESULTS\n")
    f.write("=" * 80 + "\n\n")

    # Execution information
    f.write("-" * 40 + "\n")
    f.write("GENERAL INFORMATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Execution time: {execution_time:.2f} seconds\n")
    f.write(f"Number of samples: {results_np.shape[0]}\n")
    f.write(f"Number of perturbation scales: {results_np.shape[1]}\n\n")

    # Coefficients
    f.write("-" * 40 + "\n")
    f.write("INEQUALITY COEFFICIENTS\n")
    f.write("-" * 40 + "\n")
    f.write("Perturbation scale | Mean ± Standard deviation\n")
    f.write("-" * 40 + "\n")
    for i, (mean, std, scale) in enumerate(zip(mean_ratios, std_ratios, perturb_scales)):
        f.write(f"{scale:.1e} | {mean:.4e} ± {std:.4e}\n")
    f.write("\n")

    # Statistical tests
    f.write("-" * 40 + "\n")
    f.write("STATISTICAL TESTS\n")
    f.write("-" * 40 + "\n")
    
    # Normality tests
    f.write("\n1. Normality tests:\n")
    f.write("\na) Shapiro-Wilk test:\n")
    f.write(f"   Statistic: {float(stat_res[0]):.4f}\n")
    f.write(f"   p-value: {float(stat_res[1]):.4e}\n")
    f.write(f"   Conclusion: {'Normal distribution' if float(stat_res[1]) > 0.05 else 'Not a normal distribution'}\n")
    
    f.write("\nb) D'Agostino-Pearson test:\n")
    f.write(f"   Statistic: {float(stat_res[8]):.4f}\n")
    f.write(f"   p-value: {float(stat_res[9]):.4e}\n")
    f.write(f"   Conclusion: {'Normal distribution' if float(stat_res[9]) > 0.05 else 'Not a normal distribution'}\n")

    # Confidence intervals
    f.write("\n2. Confidence intervals (95%):\n")
    f.write(f"   Interval: [{stat_res[2][0]}, {stat_res[2][1]}]\n")

    # Effect size
    f.write("\n3. Effect size:\n")
    f.write(f"   Value: {float(stat_res[3]):.4f}\n")

    # ANOVA
    f.write("\n4. Analysis of Variance (ANOVA):\n")
    f.write(f"   F-statistic: {float(stat_res[4]):.4f}\n")
    f.write(f"   p-value: {float(stat_res[5]):.4e}\n")
    f.write(f"   Conclusion: {'Significant differences between groups' if float(stat_res[5]) < 0.05 else 'No significant differences between groups'}\n")

    # Kruskal-Wallis test
    f.write("\n5. Kruskal-Wallis test:\n")
    f.write(f"   H-statistic: {float(stat_res[6]):.4f}\n")
    f.write(f"   p-value: {float(stat_res[7]):.4e}\n")
    f.write(f"   Conclusion: {'Significant differences between groups' if float(stat_res[7]) < 0.05 else 'No significant differences between groups'}\n")
    
    # Summary
    f.write("\n" + "-" * 40 + "\n")
    f.write("SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write("1. Distribution normality: ")
    if float(stat_res[1]) > 0.05 or float(stat_res[9]) > 0.05:
        f.write("At least one test indicates normal distribution\n")
    else:
        f.write("Both tests indicate non-normal distribution\n")
    
    f.write("2. Differences between groups: ")
    if float(stat_res[7]) < 0.05:
        f.write("Statistically significant (Kruskal-Wallis test)\n")
    else:
        f.write("Not statistically significant\n")

    # Footer
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n")

    f.write("=== Runtime Details ===\n")
    f.write(f"Date: {datetime.now()}\n\n")
    
    f.write("Basic Information:\n")
    for key, value in basic_info.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\nExecution Statistics:\n")
    f.write(f"Number of checks: {num_checks}\n")
    f.write(f"Directions per check (k): {k}\n")
    f.write(f"Total execution time: {total_time/60:.2f} minutes\n")
    f.write(f"Average iteration time: {np.mean(iteration_times):.2f} seconds\n")
    f.write(f"Std dev iteration time: {np.std(iteration_times):.2f} seconds\n")
    
    f.write("\nIteration Time Distribution:\n")
    percentiles = np.percentile(iteration_times, [25, 50, 75])
    f.write(f"25th percentile: {percentiles[0]:.2f} seconds\n")
    f.write(f"Median: {percentiles[1]:.2f} seconds\n")
    f.write(f"75th percentile: {percentiles[2]:.2f} seconds\n")
    
plt.figure(figsize=(10, 5))
plt.plot(iteration_times)
plt.title('Iteration Times')
plt.xlabel('Iteration')
plt.ylabel('Time (s)')
plt.grid(True)
plt.savefig(os.path.join(output_path, "iteration_times.png"))
plt.close()    
    

print(f"Results saved to {output_file}")
