import json
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

def calculate_weighted_majority_voting_metrics(data, save_dir, category, json_file_path, N_max=128):
    '''
    Calculate Weighted Majority Voting metrics by aggregating RM rewards across identical responses.
    Save metrics and plots.

    Args:
        json_file_path (str): The path to the JSON file.
    
    Returns:
        dict: A dictionary containing the metrics for Weighted Majority Voting.
    '''

    # Prepare the output directory
    output_dir = os.path.join(save_dir, 'weighted_majority_voting_metrics', os.path.basename(json_file_path).split('.js')[0], category)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for Weighted Majority Voting
    max_samples = N_max  # Maximum CoT solutions per problem
    sample_powers = [2 ** i for i in range(int(math.log(N_max, 2)) + 1)]
    aggregation_methods = ['last', 'mean', 'min']
    metrics = {method: {} for method in aggregation_methods}  # Store metrics for each method

    for method in aggregation_methods:
        sampling_results = {n: [] for n in sample_powers}  # Store results for Weighted Majority Voting

        for n in sample_powers:
            if n > max_samples:
                break

            # Repeat sampling 5 times for each size `n`
            for seed in range(10):
                random.seed(seed)  # Set random seed for reproducibility
                correct_count = 0

                # Loop over each question
                for question in data:
                    # Get all CoT solutions and their RM rewards
                    cot_solutions = question['chain_of_thoughts']
                    weighted_scores = {}

                    # Calculate RM reward for each solution based on the aggregation method
                    for cot in cot_solutions:
                        prm_rewards = cot['prm_reward']
                        if method == 'last':
                            rm_reward = prm_rewards[-1]  # Use the last step's prm_reward
                        elif method == 'mean':
                            rm_reward = np.mean(prm_rewards)  # Use the mean of all steps' prm_reward
                        elif method == 'min':
                            rm_reward = np.min(prm_rewards)  # Use the minimum of all steps' prm_reward

                        answer = cot['parsed_answer']
                        if answer not in weighted_scores:
                            weighted_scores[answer] = 0
                        weighted_scores[answer] += rm_reward

                    # Sample N solutions randomly
                    sampled_answers = random.sample(cot_solutions, n)

                    # Aggregate RM rewards for sampled answers
                    sampled_weighted_scores = {}
                    for cot in sampled_answers:
                        prm_rewards = cot['prm_reward']
                        if method == 'last':
                            rm_reward = prm_rewards[-1]
                        elif method == 'mean':
                            rm_reward = np.mean(prm_rewards)
                        elif method == 'min':
                            rm_reward = np.min(prm_rewards)

                        answer = cot['parsed_answer']
                        if answer not in sampled_weighted_scores:
                            sampled_weighted_scores[answer] = 0
                        sampled_weighted_scores[answer] += rm_reward

                    # Select the answer with the highest weighted score
                    best_weighted_answer = max(sampled_weighted_scores.items(), key=lambda x: x[1])[0]

                    # Check correctness of the selected answer
                    for cot in question['chain_of_thoughts']:
                        if cot['parsed_answer'] == best_weighted_answer:
                            if cot['parsed_answer_correctness']:
                                correct_count += 1
                            break

                # Calculate accuracy for this sampling
                accuracy = correct_count / len(data)
                sampling_results[n].append(accuracy)

        # Aggregate results (mean, max, min) for each sampling size
        metrics[method] = {
            n: {
                'mean': np.mean(sampling_results[n]),
                'max': np.max(sampling_results[n]),
                'min': np.min(sampling_results[n]),
                'all': sampling_results[n]
            }
            for n in sampling_results
        }

        # Save results for this method to a JSON file
        metrics_file_path = os.path.join(output_dir, f'metrics_{method}.json')
        with open(metrics_file_path, 'w', encoding='utf-8') as file:
            json.dump(metrics[method], file, indent=4)

        # Plot the results
        x = list(metrics[method].keys())
        y_mean = [metrics[method][n]['mean'] * 100 for n in x]  # Convert to percentages
        y_max = [metrics[method][n]['max'] * 100 for n in x]
        y_min = [metrics[method][n]['min'] * 100 for n in x]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y_mean, '-o', label='Mean Accuracy', color='blue')
        plt.fill_between(x, y_min, y_max, color='blue', alpha=0.2, label='Range (Min-Max)')
        plt.xscale('log', base=2)
        plt.xticks(x, labels=[f'{n}' for n in x])
        plt.xlabel('Number of sampled CoT solutions')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Weighted Majority Voting Accuracy ({method.capitalize()} RM Reward Aggregation)')
        plt.legend()
        plt.grid(True)

        # Save the plot for this method
        plot_file_path = os.path.join(output_dir, f'accuracy_plot_{method}.png')
        plt.savefig(plot_file_path)
        plt.close()

    return metrics

def calculate_best_of_n_metrics(data, save_dir, category, json_file_path, N_max=128):
    '''
    Calculate Best-of-N metrics for choosing the most plausible answer using RM rewards.
    Use three RM reward aggregation methods: last, mean, and min.
    Save metrics and plots for each aggregation method.

    Args:
        json_file_path (str): The path to the JSON file.
    
    Returns:
        dict: A dictionary containing metrics for all RM reward aggregation methods.
    '''

    # Prepare the output directory
    output_dir = os.path.join(save_dir, 'best_of_n_metrics', os.path.basename(json_file_path).split('.js')[0], category)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for Best-of-N
    max_samples = N_max  # Maximum CoT solutions per problem
    sample_powers = [2 ** i for i in range(int(math.log(N_max, 2)) + 1)]
    aggregation_methods = ['last', 'mean', 'min']
    metrics = {method: {} for method in aggregation_methods}  # Store metrics for each method

    for method in aggregation_methods:
        # Store results for this aggregation method
        sampling_results = {n: [] for n in sample_powers}

        # Loop through different values of N (2^0 to 2^8)
        for n in sample_powers:
            if n > max_samples:
                break

            # Repeat sampling 5 times for each size `n`
            for seed in range(10):
                random.seed(seed)  # Set random seed for reproducibility
                correct_count = 0

                # Loop over each question
                for question in data:
                    # Get all CoT solutions and their prm_reward
                    cot_solutions = question['chain_of_thoughts']
                    rewards = []

                    # Calculate RM reward for each solution based on the aggregation method
                    for cot in cot_solutions:
                        prm_rewards = cot['prm_reward']
                        if method == 'last':
                            rm_reward = prm_rewards[-1]  # Use the last step's prm_reward
                        elif method == 'mean':
                            rm_reward = np.mean(prm_rewards)  # Use the mean of all steps' prm_reward
                        elif method == 'min':
                            rm_reward = np.min(prm_rewards)  # Use the minimum of all steps' prm_reward
                        rewards.append((cot['parsed_answer'], rm_reward))

                    # Sample N solutions randomly
                    sampled_rewards = random.sample(rewards, n)

                    # Select the solution with the highest RM reward
                    best_answer = max(sampled_rewards, key=lambda x: x[1])[0]

                    # Check correctness of the selected answer
                    for cot in question['chain_of_thoughts']:
                        if cot['parsed_answer'] == best_answer:
                            if cot['parsed_answer_correctness']:
                                correct_count += 1
                            break

                # Calculate accuracy for this sampling
                accuracy = correct_count / len(data)
                sampling_results[n].append(accuracy)

        # Aggregate results (mean, max, min) for each sampling size
        metrics[method] = {
            n: {
                'mean': np.mean(sampling_results[n]),
                'max': np.max(sampling_results[n]),
                'min': np.min(sampling_results[n]),
                'all': sampling_results[n]
            }
            for n in sampling_results
        }

        # Save results for this method to a JSON file
        metrics_file_path = os.path.join(output_dir, f'metrics_{method}.json')
        with open(metrics_file_path, 'w', encoding='utf-8') as file:
            json.dump(metrics[method], file, indent=4)

        # Plot the results
        x = list(metrics[method].keys())
        y_mean = [metrics[method][n]['mean'] * 100 for n in x]  # Convert to percentages
        y_max = [metrics[method][n]['max'] * 100 for n in x]
        y_min = [metrics[method][n]['min'] * 100 for n in x]

        plt.figure(figsize=(8, 6))
        plt.plot(x, y_mean, '-o', label='Mean Accuracy', color='blue')
        plt.fill_between(x, y_min, y_max, color='blue', alpha=0.2, label='Range (Min-Max)')
        plt.xscale('log', base=2)
        plt.xticks(x, labels=[f'{n}' for n in x])
        plt.xlabel('Number of sampled CoT solutions')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Best-of-N Accuracy ({method.capitalize()} RM Reward Aggregation)')
        plt.legend()
        plt.grid(True)

        # Save the plot for this method
        plot_file_path = os.path.join(output_dir, f'accuracy_plot_{method}.png')
        plt.savefig(plot_file_path)
        plt.close()

    return metrics

def calculate_majority_voting_metrics_with_sampling(data, save_dir, category, json_file_path, N_max=128):
    '''
    Calculate metrics for majority voting accuracy by sampling CoT solutions with sizes 2^0 to 2^8.
    For each sampling size, repeat the sampling 5 times with different random seeds.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing sampled accuracies (mean, max, min) and overall metrics.
    '''

    # Prepare the output directory
    output_dir = os.path.join(save_dir, 'majority_voting_metrics', os.path.basename(json_file_path).split('.js')[0], category)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables for sampling metrics
    max_samples = N_max  # Maximum CoT solutions per problem
    sample_powers = [2 ** i for i in range(int(math.log(N_max, 2)) + 1)]
    sampling_results = {n: [] for n in sample_powers}

    # Outer loop for each sampling size (2^0, 2^1, ..., 2^8)
    for n in sample_powers:
        if n > max_samples:
            break

        # Repeat sampling 5 times for each size `n`
        for seed in range(10):
            random.seed(seed)  # Set random seed for reproducibility
            correct_count = 0

            # Loop over each question
            min_len = 128
            for question in data:
                # Get all parsed answers and their correctness
                if len(question['chain_of_thoughts']) < min_len:
                    min_len = len(question['chain_of_thoughts'])
                parsed_answers = [cot['parsed_answer'] for cot in question['chain_of_thoughts']]
                correctness_list = [cot['parsed_answer_correctness'] for cot in question['chain_of_thoughts']]

                # Sample `n` solutions randomly
                sampled_indices = random.sample(range(len(parsed_answers)), n)
                sampled_answers = [parsed_answers[i] for i in sampled_indices]
                sampled_correctness = [correctness_list[i] for i in sampled_indices]

                # Perform majority voting on the sampled solutions
                answer_counter = Counter(sampled_answers)
                sampled_majority_answer, _ = answer_counter.most_common(1)[0]
                

                # Check correctness of the sampled majority answer
                sampled_majority_correctness = None
                for i in sampled_indices:
                    if parsed_answers[i] == sampled_majority_answer:
                        sampled_majority_correctness = correctness_list[i]
                        break

                # Update correct count based on majority correctness
                if sampled_majority_correctness:
                    correct_count += 1
            pass
            # Calculate accuracy for this sampling
            accuracy = correct_count / len(data)
            sampling_results[n].append(accuracy)

    # Aggregate results (mean, max, min) for each sampling size
    aggregated_results = {
        n: {
            'mean': np.mean(sampling_results[n]),
            'max': np.max(sampling_results[n]),
            'min': np.min(sampling_results[n]),
            'all': sampling_results[n]
        }
        for n in sampling_results
    }

    # Save results to a JSON file
    metrics_file_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file_path, 'w', encoding='utf-8') as file:
        json.dump(aggregated_results, file, indent=4)

    # Plot the results
    x = list(aggregated_results.keys())
    y_mean = [aggregated_results[n]['mean'] * 100 for n in x]  # Convert to percentages
    y_max = [aggregated_results[n]['max'] * 100 for n in x]
    y_min = [aggregated_results[n]['min'] * 100 for n in x]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_mean, '-o', label='Mean Accuracy', color='blue')
    plt.fill_between(x, y_min, y_max, color='blue', alpha=0.2, label='Range (Min-Max)')
    plt.xscale('log', base=2)
    plt.xticks(x, labels=[f'{n}' for n in x])
    plt.xlabel('Number of sampled CoT solutions')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Number of Sampled CoT Solutions')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(plot_file_path)
    plt.close()

    return aggregated_results

def compare_results(file_basename, save_dir, majority_voting_folder, best_of_n_folder, weighted_majority_voting_folder):
    '''
    Compare the results of Majority Voting, Best-of-N, and Weighted Majority Voting
    and plot them on the same graph for each RM reward aggregation method (last, mean, min).
    
    Args:
        majority_voting_folder (str): Folder name of Majority Voting results.
        best_of_n_folder (str): Folder name of Best-of-N results.
        weighted_majority_voting_folder (str): Folder name of Weighted Majority Voting results.
    '''
    # Define the output directory
    output_dir = os.path.join(save_dir, 'comparison', file_basename)
    os.makedirs(output_dir, exist_ok=True)

    # Define RM reward aggregation methods
    aggregation_methods = ['last', 'mean', 'min']

    # Define file paths for each method
    majority_voting_path = os.path.join(save_dir, majority_voting_folder, file_basename)
    best_of_n_path = os.path.join(save_dir, best_of_n_folder, file_basename)
    weighted_majority_voting_path = os.path.join(save_dir, weighted_majority_voting_folder, file_basename)

    for method in aggregation_methods:
        # Load metrics for Majority Voting
        majority_metrics_file = os.path.join(majority_voting_path, 'metrics.json')
        with open(majority_metrics_file, 'r', encoding='utf-8') as file:
            majority_metrics = json.load(file)

        # Load metrics for Best-of-N
        best_of_n_metrics_file = os.path.join(best_of_n_path, f'metrics_{method}.json')
        with open(best_of_n_metrics_file, 'r', encoding='utf-8') as file:
            best_of_n_metrics = json.load(file)

        # Load metrics for Weighted Majority Voting
        weighted_majority_voting_metrics_file = os.path.join(weighted_majority_voting_path, f'metrics_{method}.json')
        with open(weighted_majority_voting_metrics_file, 'r', encoding='utf-8') as file:
            weighted_majority_voting_metrics = json.load(file)

        # Extract data for plotting
        x = list(map(int, best_of_n_metrics.keys()))  # Sampling sizes (2^0, 2^1, ..., 2^8)
        majority_y = [majority_metrics[str(n)]['mean'] * 100 for n in x]  # Convert to percentages
        best_of_n_y = [best_of_n_metrics[str(n)]['mean'] * 100 for n in x]
        weighted_majority_voting_y = [weighted_majority_voting_metrics[str(n)]['mean'] * 100 for n in x]

        # Plot the results
        plt.figure(figsize=(8, 8))
        plt.plot(x, majority_y, '-o', label='Majority Voting', color='blue')
        plt.plot(x, best_of_n_y, '-o', label='Best-of-N', color='orange')
        plt.plot(x, weighted_majority_voting_y, '-o', label='Weighted Majority Voting', color='green')
        plt.xscale('log', base=2)
        plt.xticks(x, labels=[f'{n}' for n in x])
        plt.xlabel('Number of sampled CoT solutions (log scale)')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Comparison of Voting Methods ({method.capitalize()} RM Reward Aggregation)')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file_path = os.path.join(output_dir, f'comparison_{method}.png')
        plt.savefig(plot_file_path)
        plt.close()

    print(f'Comparison plots saved to {output_dir}')

MATH_ADJACENT_DOMAINS = ['chemistry', 'computer science', 'engineering', 'physics']
NON_MATH_ADJACENT_DOMAINS = ['biology', 'health', 'psychology', 'business', 'economics', 'law', 'history', 'philosophy', 'other']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rewards_dir', type=str, required=True, help='Directory of the prm rewards files')
    parser.add_argument('--save_dir', type=str, default='results_by_category')
    parser.add_argument('--prm_name', type=str, required=True)
    parser.add_argument('--N_max', type=int, default=128)
    args = parser.parse_args()

    file_dir = args.rewards_dir
    file_list = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f)) and os.path.join(file_dir, f).endswith('rewards.json')]

    for filename in file_list:
        file_path = os.path.join(file_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        data_by_category = {'all': [], 'all_except_math': [], 'math_adjacent': [], 'non_math_adjacent': []}

        idx_abstain = []

        for i, obj in enumerate(data):

            ### Padding the number chain of thoughts to N_max, for cases where there are incompletely generated chains of thoughts due to timeout or max token length limitation.
            if len(obj['chain_of_thoughts']) < args.N_max:

                for j in range(len(obj['chain_of_thoughts']), args.N_max):
                    obj['chain_of_thoughts'].append({'steps': [''],
                                                    'parsed_answer': 'NOANSWER_{}'.format(j),
                                                    'parsed_answer_correctness': False,
                                                    'cot_id': str(j),
                                                    'prm_reward': [0]
                                                      })
                    
            data_by_category['all'].append(obj)
            if obj['metadata']['category'] != 'math':
                 data_by_category['all_except_math'].append(obj)

            if obj['metadata']['category'] in MATH_ADJACENT_DOMAINS:
                data_by_category['math_adjacent'].append(obj)

            if obj['metadata']['category'] in NON_MATH_ADJACENT_DOMAINS:
                data_by_category['non_math_adjacent'].append(obj)

            if obj['metadata']['category'] not in data_by_category:
                data_by_category[obj['metadata']['category']] = [obj]
            else:
                data_by_category[obj['metadata']['category']].append(obj)
                        
        assert len(data_by_category['all_except_math']) + len(data_by_category['math']) == len(data_by_category['all'])
        assert len(data_by_category['math_adjacent']) + len(data_by_category['non_math_adjacent']) == len(data_by_category['all_except_math'])

        acc_values = {'best-of-128 (min)': {},
                      'weighted majority voting (min)': {},
                      'majority voting (128)': {}}

        for category in data_by_category:

            majority_voting_metrics = calculate_majority_voting_metrics_with_sampling(data_by_category[category], args.save_dir, category, file_path, args.N_max)
            best_of_n_metrics = calculate_best_of_n_metrics(data_by_category[category], args.save_dir, category, file_path, args.N_max)
            weighted_majority_voting_metrics = calculate_weighted_majority_voting_metrics(data_by_category[category], args.save_dir, category, file_path, args.N_max)


            acc_values['majority voting (128)'][category] = majority_voting_metrics[args.N_max]['mean']
            acc_values['best-of-128 (min)'][category] = best_of_n_metrics['min'][args.N_max]['mean']
            acc_values['weighted majority voting (min)'][category] = weighted_majority_voting_metrics['min'][args.N_max]['mean']


            compare_results(file_basename = os.path.join(os.path.basename(file_path).split('.js')[0], category),
                            save_dir=args.save_dir, 
                            majority_voting_folder='majority_voting_metrics',
                            best_of_n_folder='best_of_n_metrics',
                            weighted_majority_voting_folder='weighted_majority_voting_metrics'
                            )
            
            print('model {} category {} done.'.format(args.prm_name, category))

        os.makedirs(os.path.join(args.save_dir, 'mean_acc_values'), exist_ok=True)
        with open(os.path.join(args.save_dir, 'mean_acc_values', '{}_mean_acc_values.json'.format(args.prm_name)), 'w', encoding='utf-8') as file:
            json.dump(acc_values, file, indent=4)