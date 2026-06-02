from vdmarl.eval_results import get_raw_dict_from_multirun_folder, Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    runs_folder = "/home/jlcg/projects/vdmarl/scripts/runs"
    
    print(f"Loading JSON logs from {runs_folder}...")
    raw_dict = get_raw_dict_from_multirun_folder(multirun_folder=runs_folder)
    
    if not raw_dict:
        print("No marl-eval JSON files found. Did you set experiment_config.create_json = True?")
        exit(1)
        
    print("Processing data...")
    processed_data = Plotting.process_data(raw_dict)
    
    env_name = list(processed_data.keys())[0] # Usually 'smacv2'
    
    (
        environment_comparison_matrix,
        sample_efficiency_matrix,
    ) = Plotting.create_matrices(processed_data, env_name=env_name)

    print("Generating aggregate scores...")
    Plotting.aggregate_scores(
        environment_comparison_matrix=environment_comparison_matrix
    )
    plt.savefig(os.path.join(runs_folder, "aggregate_scores.png"))
    plt.close()

    print("Generating performance profile...")
    Plotting.performance_profile_figure(
        environment_comparison_matrix=environment_comparison_matrix
    )
    plt.savefig(os.path.join(runs_folder, "performance_profile.png"))
    plt.close()

    print("Generating sample efficiency curves...")
    Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix
    )
    plt.savefig(os.path.join(runs_folder, "sample_efficiency.png"))
    plt.close()
    
    print(f"Plots saved to {runs_folder}")
