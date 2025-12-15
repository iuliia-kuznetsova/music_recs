
'''
    Service for adding/retrieving events history.

    This service is used to add/retrieve online events for a given user.
'''

# ---------- Imports ---------- #
from fastapi import FastAPI
import polars as pl
from src.logging_set_up import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('uvicorn.error')

# ---------- Events storage class ---------- #
class EventStore:
    '''
        Class for adding/retrieving online events of a user.
    '''

    def __init__(self, max_events_per_user: int = 10):
        # Empty DataFrame to store events: one row per (user_id, track_id, position)
        self.events = pl.DataFrame(
            {
                'user_id': pl.Series([], dtype=pl.Int64),
                'track_id': pl.Series([], dtype=pl.Int64),
                'position': pl.Series([], dtype=pl.Int64),
            }
        )
        self.max_events_per_user = max_events_per_user

    def put(self, user_id: int, track_id: int):
        '''
            Adds a new event for a user to the online history.
        '''

        # All existing events for the given user
        user_events = (
            self.events
                .filter(pl.col('user_id') == user_id)
                .sort('position')
                .collect()
        )

        # Shift existing events for the given user
        if user_events.height > 0:
            # Shift existing positions by +1 as the new event is the most recent
            shifted = (
                user_events
                    .with_columns((pl.col('position') + 1).alias('position'))
                    .filter(pl.col('position') <= self.max_events_per_user)
            )
        else:
            # No existing events
            shifted = user_events  

        # Add a new event at position 0
        new_event = pl.DataFrame(
            {
                'user_id': [user_id],
                'track_id': [track_id],
                'position': [0],
            }
        )
        updated_user_events = pl.concat([new_event, shifted])

        # Remove events of the given user from global events
        remaining = self.events.filter(pl.col('user_id') != user_id)

        # Combine remaining events with updated user events
        self.events = pl.concat([remaining, updated_user_events])

    def get(self, user_id: int, k: int = 5):
        '''
            Retrieves events of the given user.

            Args:
            - user_id - the id of the user
            - k - the number of events to retrieve

            Returns:
            - list of track ids
        '''

        user_events = (
            self.events
                .filter(pl.col('user_id') == user_id)
                .sort('position')
                .select('track_id')
                .head(k)
        )

        return user_events.get_column('track_id').to_list()

# ---------- Initialize an object ---------- #
event_store = EventStore()

# ---------- Initialize an app ---------- #
app = FastAPI(title='events')

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

# Endpoint: put
@app.post('/put')
async def put(user_id: int, track_id: int):
    '''
        Add an event (track listening) to the history.

        Args:
        - user_id - the id of the user
        - track_id - the id of the track

        Returns:
        - dictionary with result
    '''

    logger.info(f'Adding event for user {user_id} and track {track_id}')
    event_store.put(user_id, track_id)
    logger.info(f'DONE: Added event for user {user_id} and track {track_id}')

    return {'result': 'OK'}

# Endpoint: get
@app.post('/get')
async def get(user_id: int, k: int):
    '''
        Retrieves user events from the history.

        Args:
        - user_id - the id of the user
        - k - the number of events to retrieve

        Returns:
        - dictionary with events
    '''

    logger.info(f'Getting events for user {user_id} and k {k}')
    events = event_store.get(user_id, k)
    logger.info(f'DONE: Found {len(events)} events for user {user_id}')

    return {'events': events}
