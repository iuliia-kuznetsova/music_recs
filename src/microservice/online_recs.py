'''
    Online recommendations service

    This microservice is used to generate online recommendations for a given track.
    It uses the similarity-based recommendations to generate recommendations for a given track.

    Usage examples:
    python3 -m src.microservice.online_recs # launch online recommendations service
'''

# ---------- Imports ---------- #
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, Request

# ---------- Load environment variables ---------- #
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

online_recs_path = os.getenv('ONLINE_RECS_PATH')

# ---------- Similar tracks class ---------- #
class SimilarTracks:
    '''
        Class for displaying online recommendations (similarity-based recommendations).
    '''

    def __init__(self) -> None:
        self._similar_tracks: pl.DataFrame | None = None

    def load(self, path: str, **kwargs):
        '''
            Loads similarity-based recommendations.

            Args:
            - path - path to the recommendations file
            - **kwargs - additional arguments to pass to the Polars read_parquet function
        '''

        # Load similarity-based recommendations
        self._similar_tracks = pl.read_parquet(path, **kwargs)
        # Ensure required columns exist
        expected_cols = {'track_id', 'similar_track_id', 'similarity_score'}
        missing = expected_cols - set(self._similar_tracks.columns)
        if missing:
            raise ValueError(f'Missing required columns in similarity-based recommendations: {missing}')

    def get(self, track_id: int, k: int = 10):
        '''
            Retrieves first k online recommendations.
        '''

        # Check if similarity-based recommendations are loaded
        if self._similar_tracks is None:
            return {'similar_track_id': [], 'similarity_score': []}

        # Check if k is greater than the number of recommendations for the given track
        available_count = self._similar_tracks.filter(pl.col('track_id') == track_id).height
        if k > available_count:
            k = available_count

        # Filter rows where track_id == track_id, sort by similarity_score, take top k
        i2i = (
            self._similar_tracks
                .filter(pl.col('track_id') == track_id)
                .sort('similarity_score', descending=True)
                .select(['similar_track_id', 'similarity_score'])
                .head(k)
        )

        # Check if no recommendations were found
        if i2i.height == 0:
            return {'similar_track_id': [], 'similarity_score': []}
        else:
            return {
                'similar_track_id': i2i.get_column('similar_track_id').to_list(),
                'similarity_score': i2i.get_column('similarity_score').to_list(),
            }

# ---------- API Lifespan ---------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
        Loads data on application start-up.
    '''

    similar_items_store = SimilarTracks()
    similar_items_store.load(path=online_recs_path)
    # Attach the similar items store to app state via lifespan context
    yield {'similar_items_store': similar_items_store}

# ---------- Initialize an app ---------- #
app = FastAPI(title='rec_online', lifespan=lifespan)

# ---------- Add endpoints for the service ---------- #
# Health check
@app.get('/healthy')
async def healthy():
    '''
        Health check endpoint.

        Returns:
        - dictionary with status 'healthy'
    '''
    return {'status': 'healthy'}

# Get recommendations
@app.post('/similar_tracks')
async def similar_tracks(request: Request, track_id: int, k: int):
    '''
        Generate online recommendations.

        Args:
        - request - FastAPI request instance
        - track_id - track id to generate recommendations for
        - k - number of recommendations to generate
    '''

    # Get object with loaded similarity data from app state
    similar_items_store = request.state.similar_items_store
    # Get recommendations
    i2i = similar_items_store.get(track_id, k)
    
    return i2i
