# Building Documentation for scikit-dsp-comm

This guide explains how to build the Sphinx documentation for the `scikit-dsp-comm` project.

## Prerequisites

Before building the documentation, ensure you have the following installed:

- Python 3.7+ (as specified in [setup.py](setup.py))
- pip (Python package manager)

## Setting Up the Environment

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install the required packages:
   The `scikit-dsp-comm` package and its dependencies need to be installed for auto-generating the API documentation.
   ```bash
   pip install . # scikit-dsp-comm
   pip install -r docs/requirements.txt
   ```

## Building the Documentation

### Local Build

1. Navigate to the docs directory:
   ```bash
   cd docs
   ```

2. Build the HTML documentation:
   ```bash
   make html
   ```

3. View the documentation by opening `docs/build/html/index.html` in your web browser.

### Alternative Build Methods

- Build PDF documentation (requires LaTeX):
  ```bash
  make latexpdf
  ```
  The PDF will be available at `docs/build/latex/scikit-dsp-comm.pdf`

- Build EPUB:
  ```bash
  make epub
  ```

- Clean the build directory:
  ```bash
  make clean
  ```

## Documentation Structure

- `source/` - Contains all source files for the documentation
  - `conf.py` - Sphinx configuration file
  - `index.rst` - Main documentation landing page
  - `nb_examples/` - Jupyter notebooks

## Writing Documentation and Examples

1. **ReStructuredText (.rst)**: Main format for documentation
1. **Jupyter Notebooks**: Supported through `nbsphinx`

### Adding New Documentation

1. Create a new `.rst` file in the appropriate directory
1. Add it to the relevant `toctree` directive in `index.rst` or another appropriate file
1. Clear all outputs from Jupyter notebooks (ensures that the latest outputs are generated)
1. Build and test your changes locally
1. Submit a pull request with your changes

## Documentation Style Guide

- Follow [NumPy documentation style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for API documentation

## Common Issues

- Images don't update: Rebuild the docs.
  ```bash
  make clean
  make html
  ```

## Continuous Integration

The documentation is automatically built and deployed on commits to the main branch using [Read the Docs](https://readthedocs.org/).


[![Documentation Status](https://readthedocs.org/projects/scikit-dsp-comm/badge/?version=latest)](https://scikit-dsp-comm.readthedocs.io/en/latest/?badge=latest)

---

## License

This documentation is licensed under the same terms as the scikit-dsp-comm project.

---
*Last updated: June 2025*
