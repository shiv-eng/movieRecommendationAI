# Movie Recommendation Flask App

Welcome to the Movie Recommendation Flask App! This application provides movie recommendations based on user-preferred genres using collaborative filtering with the Surprise library.

## Getting Started

### Prerequisites

- Python installed on your machine.
- [Pip](https://pip.pypa.io/en/stable/installation/) (Python package installer).

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/MovieRecommendationFlaskApp.git
    ```

2. **Navigate to the project folder:**

    ```bash
    cd MovieRecommendationFlaskApp
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask app:**

    ```bash
    python app.py
    ```

5. **Open your browser and visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to see the home page.**

## API Endpoints

### Get Recommendations by Genre

```bash
GET /recommend_by_genre?genre=Action
```
