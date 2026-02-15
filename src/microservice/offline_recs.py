'''
    Offline recommendations service

    This microservice is used to generate offline recommendations for a given user.
    It uses the als_based and popularity_based recommendations to generate recommendations 
    for a given user.

    Usage examples:
    python3 -m src.microservice.offline_recs # launch offline recommendations service
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

        self._recs[rec_type] = pl.read_parquet(path, **kwargs)
        if rec_type == 'personal' and 'user_id' not in self._recs[rec_type].columns:
            raise ValueError(f"No 'user_id' column in {rec_type} recommendations file")

    def get(self, user_id: int, k: int = 10):
        '''
            Generate k offline recommendations for a user.

            Args:
            - user_id - user id to generate recommendations for
            - k - number of recommendations to generate

            Returns:
            - list of recommended track_ids
        '''

        # Try to get personal recommendations for this user
        user_recs = self._recs['personal'].filter(pl.col('user_id') == user_id)
        
        if user_recs.height > 0:
            # User has personal recommendations
            recs = (
                user_recs
                    .select('track_id')
                    .head(k)
                    .to_series()
                    .to_list()
            )
            self._stats['request_personal_count'] += 1
            return recs
        else:
            # Fall back to top popular tracks (no user_id filter needed)
            recs = (
                self._recs['default']
                    .select('track_id')
                    .head(k)
                    .to_series()
                    .to_list()
            )
            self._stats['request_default_count'] += 1
            return recs

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
    # Load data on application start-up
    rec_store = Recommender()
    rec_store.load(rec_type='personal', path=personal_recs_path)
    rec_store.load(rec_type='default', path=default_recs_path)

    # Yield rec_store instance
    yield {'rec_store': rec_store}

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

# Endpoint: display statistics
@app.get('/stats')
async def stats(request: Request):
    '''
        Display statistics.

        Returns:
        - dictionary with statistics
    '''
    rec_store = request.state.rec_store
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

    rec_store = request.state.rec_store
    recs = rec_store.get(user_id, k)

    return recs