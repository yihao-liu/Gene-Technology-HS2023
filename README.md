# Gene Technology HS2023 - Deep Metric Learning for Plasmid Origin Prediction

Welcome to the project repository for the lecture 535-0810-00L Gene Technology HS2023. This project is dedicated to showcasing the power of machine learning in solving pharmaceutical problems, in our case, prediction of the lab-of-origin, and serves as an educational tool for students. We've adapted the source code from "Deep Metric Learning Improves the Genetically Modified Plasmid Origin Prediction Laboratory" to further the state-of-the-art in predicting the origins of genetically modified plasmids. Additionally, we include more detailed explanations of the code and the underlying concepts, as well as a showcase of the results.

## Getting Started

These instructions will help you set up and run the project on your local machine or directly on Moodle using Jupyter Hub for educational and development purposes.

### Option1 : Using Jupyter Hub on Moodle

1. Open Jupyter Hub on Moodle.
2. Use git to clone the project repository:
   ```
   git clone https://github.com/yihao-liu/Gene-Technology-HS2023.git
    ```
3. Download the Data: 
   Before running the notebook, you need to download the data folder required for this project. Get the data from [this link](https://codeocean.com/capsule/3003146/tree/v1). Make sure to place it in the appropriate directory within your project folder. The data folder should be placed directly under the **Gene-Technology-HS2023** folder.


### Option 2: Local Setup
#### Prerequisites

1. **Python 3.6 or higher**: You can download it from [here](https://www.python.org/downloads/).
2. **Conda**: We recommend using Conda as a package and environment manager. Install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).


#### Installation

1. Clone this repository:
    ```
    git clone https://github.com/yihao-liu/Gene-Technology-HS2023.git
    cd Gene-Technology-HS2023
    ```

2. Set up a Conda environment using the provided `environment.yml` file:
    ```
    conda env create -f environment.yml
    ```

3. Activate the Conda environment:
    ```
    conda activate your-env-name
    ```
Replace `your-env-name` with the name of the environment specified in the `environment.yml` file.


#### Usage

1. Download the Data: 
   Before running the notebook, you need to download the data folder required for this project. Get the data from [this link](https://codeocean.com/capsule/3003146/tree/v1). Make sure to place it in the appropriate directory within your project folder.

2. Launch Jupyter Notebook:
    ```
    jupyter notebook
    ```
This will open a new page in your web browser with a list of files in your current directory.

3. Navigate to the `Gene_Technology_showcase.ipynb` file and click on it to open.

4. Once inside the notebook, you can run each cell by clicking on it and pressing `Shift+Enter`. This will execute the code inside the cell and move to the next one.

For a more detailed guide on using Jupyter Notebooks, please refer to the [official Jupyter documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html).

## To-Do

- [ ] Verify the `environment.yml` file.
- [ ] Clean up and organize the files contained in the project.

## Acknowledgments

- The original [source code](https://codeocean.com/capsule/3003146/tree/v1) from "Deep Metric Learning Improves the Genetically Modified Plasmid Origin Prediction Laboratory".
- Prof. Dr. Klaus Eyer and Dr. Ines Lüchtefeld for their guidance and support.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE.md](LICENSE.md) file.
