import os

class setupFolders:
    def __init__(self, args):
        main_result_path    = "./training_results/"      
        self.trained_model_path     = f"{main_result_path}{args.experimentName}/checkpoint/"
        self.result_path    = f"{main_result_path}{args.experimentName}/validation_results/"
        self.log_path       = f"{main_result_path}{args.experimentName}/"
        self.tb_path    = f"{main_result_path}/tensorboard_summary/" + args.experimentName
            

        # Create directories if they don't exist
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.trained_model_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.tb_path, exist_ok=True)