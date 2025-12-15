'''
    Service for outputting online recommendations (track similarity) using Polars.
'''

# ---------- Imports ---------- #
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI, Request

from src.logging_set_up import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('uvicorn.error')

# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

online_recs_path = os.getenv('ONLINE_RECS_PATH')

# ---------- Similar tracks class ---------- #
class SimilarTracks:
    '''
        Class for displaying online recommendations (similarity-based recommendations).
    '''

    def __init__(self) -> None:
        # Will hold a Polars DataFrame with similarity data
        self._similar_tracks: pl.DataFrame | None = None

    def load(self, path: str, **kwargs):
        '''
            Loads similarity-based recommendations.

            Args:
            - path - path to the recommendations file
            - **kwargs - additional arguments to pass to the Polars read_parquet function
        '''

        logger.info(f'Loading similarity-based recommendations from {path}')
        # Load similarity-based recommendations
        self._similar_tracks = pl.read_parquet(path, **kwargs)
        # Ensure required columns exist
        expected_cols = {'track_id_1', 'track_id_2', 'score'}
        missing = expected_cols - set(self._similar_tracks.columns)
        if missing:
            logger.error(f'Missing required columns in similarity-based recommendations: {missing}')
            raise ValueError(f'Missing required columns in similarity-based recommendations: {missing}')
        logger.info(f'DONE: Similarity-based recommendations loaded from {path}')

    def get(self, track_id: int, k: int = 10):
        '''
            Retrieves first k online recommendations.
        '''

        logger.info(f'Getting {k} recommendations for track {track_id}')
        # Check if similarity-based recommendations are loaded
        if self._similar_tracks is None:
            logger.error('Similarity-based recommendations not loaded')
            return {'track_id_2': [], 'score': []}

        # Check if k is greater than the number of recommendations for the given track
        if k > self._similar_tracks.filter(pl.col('track_id_1') == track_id).height:
            logger.warning(f'Requested {k} recommendations for track {track_id}, but only {self._similar_tracks.filter(pl.col('track_id_1') == track_id).height} recommendations available')
            k = self._similar_tracks.filter(pl.col('track_id_1') == track_id).height

        logger.info(f'Filtering rows where track_id_1 == {track_id} and sorting by score in descending order')   
        # Filter rows where track_id_1 == track_id, sort by score, take top k
        i2i = (
            self._similar_tracks
                .filter(pl.col('track_id_1') == track_id)
                .sort('score', descending=True)
                .select(['track_id_2', 'score'])
                .head(k)
        )

        # Check if no recommendations were found
        if i2i.height == 0:
            logger.error('ERROR: No recommendations found')
            return {'track_id_2': [], 'score': []}
        else:
            logger.info(f'DONE: Found {i2i.height} recommendations for track {track_id}')
            return {
                'track_id_2': i2i.get_column('track_id_2').to_list(),
                'score': i2i.get_column('score').to_list(),
            }

# ---------- API Lifespan ---------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
        Loads data on application start-up.
    '''

    logger.info('Loading similarity-based recommendations')
    sim_items_store = SimilarTracks()
    sim_items_store.load(path=online_recs_path)
    logger.info('DONE: Similarity-based recommendations loaded from {online_recs_path}')
    # Attach the store to app state via lifespan context
    yield {'sim_items_store': sim_items_store}

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

    logger.info(f'Getting {k} recommendations for track {track_id}')
    # Get object with loaded similarity data from app state
    sim_items_store = request.state.sim_items_store
    # Get recommendations
    i2i = sim_items_store.get(track_id, k)
    logger.info(f'DONE: Found {len(i2i)} recommendations for track {track_id}')
    
    return i2i
