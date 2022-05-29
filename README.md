# voice2voice
Testing methods to transform voices.  
It will probably use some kind of "voice" encoder to extract the "voice" features for further processing, together with a model to use these features to alter the given input voiceline.

The voice encoder will likely be trained using multiple samples from an individual person to generate an embedding containing the required voice features.

## 1. Install Requirements
```
# create env
python -m venv .venv
source .venv/bin/activate #or .\.venv\Scripts\Activate.ps1

# pytorch+torchaudio version of your choice
pip install torch torchvision torchaudio


# some basic libs (you never know...)
pip install tqdm notebook ipywidgets matplotlib scikit-learn numpy soundfile
``` 

