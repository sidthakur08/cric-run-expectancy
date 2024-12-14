import subprocess

def run_pipeline():
    subprocess.run(['python','data_gen.py'])
    subprocess.run(['python','model.py'])

if __name__ == '__main__':
    run_pipeline()