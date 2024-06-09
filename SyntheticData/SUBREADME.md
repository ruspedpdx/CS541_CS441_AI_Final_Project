# How to use
- Create the virtual environment:
`python -m venv synth-env`
- Activate the virtual environment:
`source synth-env/Scripts/activate` # for windows
- Install the dependencies
`pip install -r requirements.txt`
- Add virtual environment to Jupyter
`python -m ipykernel install --user --name=synth-env --display-name "Python (synth-env)"`
- Start Jupyter Notebook
`jupyter notebook`


# Sources
Obtained the GREP questions and input files from the excellently curated GREP course
found at the repository:
https://github.com/learnbyexample/TUI-apps/tree/main/GrepExercises

These questions were the initial basis for getting a baseline for Gemini-1.5-flash-latest's
ability to generate grep answers and then to create a larger dataset from the well
functioning agent.