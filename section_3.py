from model import LeNet5
from util import get_config, plot_result
from train import train_model

exp_dict = {
    "multi_step_gamma_0.1": {
        "Learning_rate": 0.1,
        "Scheduler": "MultiStepLR",
        "Scheduler_param": {
            "milestones": [10, 20],
            "gamma": 0.1,
        },
    },
    "multi_step_gamma_0.5": {
        "Learning_rate": 0.1,
        "Scheduler": "MultiStepLR",
        "Scheduler_param": {
            "milestones": [10, 20],
            "gamma": 0.5,
        },
    },
    "multi_step_dense_milestones": {
        "Learning_rate": 0.1,
        "Scheduler": "MultiStepLR",
        "Scheduler_param": {
            "milestones": [5, 10, 15, 20],
            "gamma": 0.5,
        },
    },

    "linear_decay_slow": {
        "Learning_rate": 0.1,
        "Scheduler": "IncreaseDecayLR",
        "Scheduler_param": {
            "total_epochs": 50,
        },
    },
    "linear_decay_fast": {
        "Learning_rate": 0.1,
        "Scheduler": "IncreaseDecayLR",
        "Scheduler_param": {
            "total_epochs": 20,
        },
    },
    "linear_decay_very_fast": {
        "Learning_rate": 0.1,
        "Scheduler": "IncreaseDecayLR",
        "Scheduler_param": {
            "total_epochs": 10,
        },
    },

    "warmup_short_gamma_0.95": {
        "Learning_rate": 0.1,
        "Scheduler": "CycleLR",
        "Scheduler_param": {
            "warmup_epochs": 3,
            "gamma": 0.95,
        },
    },
    "warmup_long_gamma_0.95": {
        "Learning_rate": 0.1,
        "Scheduler": "CycleLR",
        "Scheduler_param": {
            "warmup_epochs": 10,
            "gamma": 0.95,
        },
    },
    "warmup_short_gamma_0.9": {
        "Learning_rate": 0.1,
        "Scheduler": "CycleLR",
        "Scheduler_param": {
            "warmup_epochs": 3,
            "gamma": 0.9,
        },
    },    

    "cyclic_short_cycle": {
        "Learning_rate": 0.1,
        "Scheduler": "CycleLR",
        "Scheduler_param": {
            "cycle_size": 6,    
            "min_lr_factor": 0.2  
        },
    },

    "cyclic_medium_cycle": {
        "Learning_rate": 0.1,
        "Scheduler": "CycleLR",
        "Scheduler_param": {
            "cycle_size": 10,    
            "min_lr_factor": 0.1     
        },
    },
    "cyclic_long_cycle": {
        "Learning_rate": 0.1,
        "Scheduler": "CycleLR",
        "Scheduler_param": {
            "cycle_size": 20,       
            "min_lr_factor": 0.05   
        },
    },
}


def main():
    exp_result_dict = {}
    
    for exp_name, exp_config in exp_dict.items():
        print(f"Running experiment: {exp_name}")
        config = get_config(exp_config)
        model = LeNet5()
        
        history = train_model(model, config)
        exp_result_dict[exp_name] = history
    plot_result(exp_result_dict)
    
if __name__ == "__main__":
    main()