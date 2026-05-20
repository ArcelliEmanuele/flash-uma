#go to project root
PROJECT_ROOT="$HOME/FlashUMA"
cd "$PROJECT_ROOT"

#remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

#create new venv
python -m venv .venv

#activate venv
source .venv/bin/activate

#upgrade pip
pip install --upgrade pip

#install fairchem-core
if [ ! -d "fairchem" ]; then
    echo "Cloning fairchem..."
    git clone https://github.com/facebookresearch/fairchem.git
fi
echo "Installing fairchem-core..."
pip install -e fairchem/packages/fairchem-core[dev]

#install triton
pip install triton