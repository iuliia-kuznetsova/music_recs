'''
    Microservice for generating offline recommendations.

    This microservice is used to generate offline recommendations for a given user.
    It uses the personal and top-popular recommendations to generate recommendations for a given user.
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

personal_recs_path = os.getenv('PERSONAL_RECS_PATH')
default_recs_path = os.getenv('DEFAULT_RECS_PATH')

# ---------- Recommender class ---------- #
class Recommender:
    def __init__(self):
        self._recs = {
            'personal': None,
            'default': None,
        }
        self._stats = {
            'request_personal_count': 0,
            'request_default_count': 0,
        }

    def load(self, rec_type, path, **kwargs):
        '''
            Load offline recommendations.

            Args:
            - rec_type - type of recommendations to load: 'personal' or 'default'
            - path - path to the recommendations file
            - **kwargs - additional arguments to pass to the Polars read_parquet function
        '''
        logger.info(f'Loading recommendations: {rec_type}')

        self._recs[rec_type] = pl.read_parquet(path, **kwargs)
        if rec_type == 'personal':
            # Ensure the personal recs contain user_id for filtering
            if 'user_id' not in self._recs[rec_type].columns:
                logger.error(f'No 'user_id' column in {rec_type} recommendations file')
                return None

        logger.info(f'DONE: {rec_type} recommendations loaded')

        return None

    def get(self, user_id: int, k: int = 10) -> List[int]:
        '''
            Generate k offline recommendations for a user.

            Args:
            - user_id - user id to generate recommendations for
            - k - number of recommendations to generate

            Returns:
            - list of recommended track_ids
        '''

        logger.info(f'Getting {k} recommendations for user {user_id}')
        if user_id in self._recs['personal']['user_id'].to_list():
            recs = (
                self._recs['personal']
                    .filter(pl.col('user_id') == user_id)
                    .select('track_id')
                    .head(k)
                    .to_series()
                    .to_list())

            logger.info(f'DONE: Found {len(recs)} recommendations for user {user_id}')
            return recs

        elif user_id not in self._recs['personal']['user_id'].to_list():
            recs = (
                self._recs['default']
                    .filter(pl.col('user_id') == user_id)
                    .select('track_id')
                    .head(k)
                    .to_series()
                    .to_list())

            logger.info(f'DONE: Found {len(recs)} top-popular recommendations for user {user_id}')
            return recs

        else:
            logger.error(f'ERROR: Recommendations for user {user_id} not found in personal or top-popular recommendations')
            return None

# ---------- API Lifespan ---------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
        Load data on application start-up.

        Args:
        - app - FastAPI app instance

        Returns:
        - dictionary of rec_store instance
    '''
    # ---------- Load data on application start-up ---------- #
    logger.info('Starting application start-up')
    logger.info(f'Loading personal and top-popular recommendations from {personal_recs_path} and {default_recs_path}')
    rec_store = Recommender()
    rec_store.load(rec_type='personal', path=personal_recs_path)
    rec_store.load(rec_type='default', path=default_recs_path)

    # ---------- Yield rec_store instance ---------- #
    yield {'rec_store': rec_store}

    logger.info('DONE: Data loaded on application start-up')
    return None

# ---------- Initialize an app ---------- #
app = FastAPI(title='rec_offline', lifespan=lifespan)

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

# Endpoint: display status message
@app.get('/status')
async def status():
    '''
        Display status message.

        Returns:
        - dictionary with status message
    '''

    logger.info('Displaying status message')
    logger.info(f'DONE: rec_offline is running')
    return {'status': 'rec_offline is running'}

# Endpoint: display statistics
@app.get('/stats')
async def stats():
    '''
        Display statistics.

        Returns:
        - dictionary with statistics
    '''
    logger.info('Displaying statistics')
    rec_store = request.state.rec_store
    logger.info(f'DONE: Displayed statistics: {rec_store._stats}')
    return {'stats': rec_store._stats}

# Endpoint: get recommendations
@app.post('/get_recs')
async def recommendations(request: Request, user_id: int, k: int):
    '''
        Generate offline recommendations.

        Args:
        - request - FastAPI request instance
        - user_id - user id to generate recommendations for
        - k - number of recommendations to generate
    '''

    logger.info(f'Getting {k} recommendations for user {user_id}')
    rec_store = request.state.rec_store
    recs = rec_store.get(user_id, k)
    logger.info(f'DONE: Found {len(recs)} recommendations for user {user_id}')

    return recs