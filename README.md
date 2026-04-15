# GeoMechLab

Repository of the second guided project of the course "Scientific Computing Tools for Advanced Mathematical Modelling" AY 2025/2026.

# Conda environment Setup

Follow these steps to create and configure the conda environment for the project.

1. Create the environment

    conda create -n poresct4amm2526 python=3.13
    conda activate poresct4amm2526

2. Install PorePy

Clone the repository:

    git clone https://github.com/pmgbergen/porepy.git

cd porepy

Switch to the stable version:

    git checkout main

Install PorePy and its dependencies:

    pip install .

3. Install PyGeon

Clone the repository:

git clone https://github.com/compgeo-mox/pygeon.git

    cd pygeon

Install PyGeon and its dependencies:

    pip install .


$ conda create -n poresct4amm2526 python=3.13

Then we are going to install porepy. To do that we need to clone the following repository:

    git clone https://github.com/pmgbergen/porepy.git

    cd porepy

The default branch is `develop`. To get the stable version:

    git checkout main

Install PorePy and the needed requirements

    pip install .

The same should be repeat for pygeon:

    git clone https://github.com/compgeo-mox/pygeon.git

    cd pygeon 

Install pygeon and the needed requirements

    pip install .


# Rules

Each group needs to create its branch of the main branch (name_of_your_new_branch=Surname1Surname2Surname3). 

Create the branch on your local machine and switch in this branch :

    git checkout -b [name_of_your_new_branch]

Then push the branch on GitHub :

    git push origin [name_of_your_new_branch]

Please upload only the code required for project evaluation and avoid including unnecessary material (e.g., images or additional datasets). Add an appropriate .gitignore file to your branch.

The data and code provided must be used exclusively within the scope of this course. For any other use, please notify the lecturer and request permission. Implementations and reports derived from course materials must not be published online or used for job applications or interviews without prior permission.
