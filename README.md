# Chat with a Movie Recommender System

This project is a simple chat made with ChainLit using a Movie Recommender System. It allows users to select a movie and receive recommendations for similar movies along with their posters.

## Features

- **Interactive UI**: The project uses Streamlit to provide an intuitive user interface.
- **Movie Recommendations**: Based on a selected movie, the system recommends five similar movies.
- **Movie Posters**: The system fetches and displays posters of the recommended movies using The Movie Database (TMDb) API.

-**Chainlit chat**: GPT4 interactive chat using Azure OpenAI Service, where the movie recommener system is added as a tool

## Requirements

To run this project, you'll need to have the following Python packages installed:

```bash
pip install streamlit pandas requests
```

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/carloshm-experiments/movie-recommender-system/
   cd movie-recommender-system
   ```

2. **Ensure the Required Files are Present (you should run the notebook and create similarity.pkl (too large for Github repo))**:
   - `movies_dict.pkl`: A pickle file containing the movie data.
   - `similarity.pkl`: A pickle file containing the precomputed similarity matrix.
   - `movies.pkl`: A pickle file containing additional movie information used in the project.
   - `movie_recommender_system.ipynb`: A Jupyter Notebook that likely contains data analysis, model training, or exploratory work related to the recommender system.

3. **Run the Application**:

   Start the Streamlit app using the following command:

   ```bash
   streamlit run app.py
   ```

4. **Use the Application**:
   - Open the local URL provided by Streamlit in your browser.
   - Select a movie from the dropdown list.
   - Click on the "Recommend" button to get movie recommendations along with their posters.

## Project Structure

- **app.py**: The main application file that contains the logic for fetching movie recommendations and displaying the UI.
- **movie_recommender_system.ipynb**: A Jupyter Notebook that provides additional insights or model training code related to the recommender system.
- **movies_dict.pkl**: A pickle file containing the movie data.
- **movies.pkl**: A pickle file containing additional movie information.
- **similarity.pkl**: A pickle file containing the similarity matrix used for recommendations.

## API Usage

The application uses The Movie Database (TMDb) API to fetch movie posters. Ensure you have a valid API key and update the `fetch_poster` function in `app.py` with your API key if necessary.
