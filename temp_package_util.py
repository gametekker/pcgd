import os

def install_packages():
    os.system('pip install scipy')
    os.system('pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    os.system('pip install pandas')
    os.system('pip install numpy')
    os.system('pip install pettingzoo[mpe]')
    os.system('pip install matplotlib')

if __name__ == "__main__":
    install_packages()
