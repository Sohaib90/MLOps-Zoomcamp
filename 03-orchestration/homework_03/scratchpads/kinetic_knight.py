"""
NOTE: Scratchpad blocks are used only for experimentation and testing out code.
The code written here will not be executed as part of the pipeline.
"""
from mage_ai.data_preparation.variable_manager import get_variable


model = get_variable('energetic_bird', 'iridescent_oracle', 'output_2')
print(model.intercept_)
