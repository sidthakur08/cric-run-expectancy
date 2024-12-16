import subprocess

def run_pipeline():
    '''
    Runs the data generation and model training scripts in sequence.
    '''
    
    print("Running data generation script...")
    subprocess.run(['python', 'data_gen.py'])  # Run data generation script

    print("Running model training script...")
    subprocess.run(['python', 'model.py'])  # Run model training script

if __name__ == '__main__':
    run_pipeline()