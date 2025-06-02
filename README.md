# Gruppenprojekt - Big Data

## Project Description
This repository contains an implementation of a user-based recommendation system in Python. The system predicts the rating of an item by a user based on the ratings of similar users.


## Project Setup
For dependency management, the project uses Poetry. As explained in [the Poetry documentation](https://python-poetry.org/docs/), you can install Poetry in a specific virtual environment or globally. 
After Poetry is installed, you can use it to download all dependencies for the application by running:

```bash
poetry install
```

in the repository root directory.

### Adding Dependencies in the Python Backend
To add new dependencies, use (e.g., for the package `pandas`):

```bash
poetry add pandas
```

This will add the package to the `pyproject.toml` file and to the `poetry.lock` file, and install it in the virtual environment.


## License
This project is under the MIT License.